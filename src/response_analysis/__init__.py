"""Independent, fixed-grid display analysis. No design, UI or persistence imports."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from threading import RLock

import numpy as np
from scipy import signal

VERSION = "display-3-shared-target-boundary"
MULTIPLIERS = (1, 2, 4)


def multiplier(value):
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        return 1
    return value if value in MULTIPLIERS else 1


def fft_length(fs, scale=1):
    if int(fs) <= 0:
        raise ValueError("sample rate must be positive")
    n = int(fs) * multiplier(scale)
    return n if n % 2 else n + 1


def digest(*values):
    h = hashlib.sha256(VERSION.encode())
    for value in values:
        if isinstance(value, np.ndarray):
            a = np.ascontiguousarray(value)
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.view(np.uint8))
        else:
            h.update(repr(value).encode())
        h.update(b"\0")
    return h.hexdigest()


def frozen(values, dtype=None):
    a = np.array(values, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class ResponseBundle:
    role: str
    fs: int
    n: int
    frequency: np.ndarray
    response: np.ndarray
    key: str
    stages: tuple = ()
    estimated: bool = False


class AnalysisCache:
    """Owned by one caller/session; bounded immutable results, no pickle reload."""
    def __init__(self, max_bytes=64*1024*1024, max_entries=48):
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.size = 0
        self.lock = RLock()

    def get(self, key, compute):
        with self.lock:
            if key in self.entries:
                value, size = self.entries.pop(key)
                self.entries[key] = value, size
                return value
        value = compute()
        size = (value.frequency.nbytes + value.response.nbytes if isinstance(value, ResponseBundle)
                else value.nbytes if isinstance(value, np.ndarray) else 0)
        with self.lock:
            if key in self.entries:
                return self.entries[key][0]
            if size <= self.max_bytes:
                self.entries[key] = value, size
                self.size += size
                while self.size > self.max_bytes or len(self.entries) > self.max_entries:
                    _, (_, removed) = self.entries.popitem(last=False)
                    self.size -= removed
        return value


def bundle(role, fs, n, response, stages=(), estimated=False):
    h = np.asarray(response, dtype=complex)
    if fs <= 0 or n < 3 or n % 2 != 1 or h.shape != (n//2+1,) or not np.all(np.isfinite(h)):
        raise ValueError("invalid odd-grid response")
    if abs(h[0].imag) > 1e-10:
        raise ValueError("DC must be real")
    return ResponseBundle(role, int(fs), int(n), frozen(np.fft.rfftfreq(n, 1/fs)),
                          frozen(h), digest(role, fs, n, h, stages, estimated), tuple(stages), estimated)


def project_measurement(x, gain, phase, frequency, fs, *, speaker=True):
    """Linear-Hz in-band values; shared P0/Speaker boundary estimation.

    None phase means unavailable, never an implicit zero-phase measurement.
    Invalid/gapped samples are rejected, not silently removed/interpolated.
    """
    x, gain = np.asarray(x, float), np.asarray(gain, float)
    f = np.asarray(frequency, float)
    if (not np.isfinite(fs) or fs <= 0 or f.ndim != 1
            or not np.all(np.isfinite(f)) or np.any(f < 0) or np.any(f > fs/2)):
        raise ValueError("表示周波数軸が有効範囲外です。")
    if phase is None:
        raise ValueError("位相がないため時間応答を表示できません。")
    phase = np.asarray(phase, float)
    if (x.ndim != 1 or len(x) < 2 or gain.shape != x.shape or phase.shape != x.shape
            or not np.all(np.isfinite(x+gain+phase)) or np.any(np.diff(x) <= 0)
            or x[0] < 0):
        raise ValueError("周波数・Gain・位相の有効な連続データが必要です。")
    from response_completion import complete_response
    completed_gain, completed_phase, _ = complete_response(x, gain, phase, f)
    h = 10**(completed_gain/20)*np.exp(1j*np.deg2rad(completed_phase))
    if len(f) and f[0] == 0:
        h[0] = 0 if speaker else h[0].real
    if not np.isfinite(h).all():
        raise ValueError("応答の補完結果が有限値ではありません。")
    return h


def measurement(cache, role, fs, n, x, gain, phase, *, speaker=True):
    x, gain = np.asarray(x,float), np.asarray(gain,float)
    phase = None if phase is None else np.asarray(phase,float)
    key = digest(role,fs,n,x,gain,phase,speaker)
    return cache.get(key, lambda: bundle(role,fs,n,project_measurement(
        x,gain,phase,np.fft.rfftfreq(n,1/fs),fs,speaker=speaker), estimated=True))


def fir_response(cache, coefficients, fs, n):
    b = np.asarray(coefficients,float)
    if b.ndim != 1 or len(b) > n or len(b) == 0 or not np.all(np.isfinite(b)):
        raise ValueError("FIR長が解析長を超えるか、係数が無効です。表示倍率を確認してください。")
    key = digest("fir",fs,n,b)
    def compute():
        f = np.fft.rfftfreq(n,1/fs)
        h = np.fft.rfft(b,n=n)*np.exp(2j*np.pi*f*(len(b)-1)/2/fs)
        return bundle("filter",fs,n,h,(digest(b),))
    return cache.get(key,compute)


def sos_response(cache, sos, fs, n):
    sos = np.asarray(sos,float)
    key = digest("sos",fs,n,sos)
    return cache.get(key,lambda:bundle("iir",fs,n,
        np.ones(n//2+1,complex) if sos.size == 0 else signal.sosfreqz(
        sos,worN=np.fft.rfftfreq(n,1/fs),fs=fs)[1],(digest(sos),)))


def multiply(cache, left, right):
    if (left.fs,left.n)!=(right.fs,right.n):
        raise ValueError("response grid mismatch")
    key=digest("multiply",left.key,right.key)
    return cache.get(key,lambda:bundle("acoustic",left.fs,left.n,left.response*right.response,
        left.stages+right.stages,left.estimated or right.estimated))


def derive(cache, source, view):
    key=digest(source.key,view)
    def compute():
        if view == "impulse":
            return frozen(np.fft.fftshift(np.fft.irfft(source.response,n=source.n)))
        if view == "step":
            return frozen(np.cumsum(derive(cache,source,"impulse")))
        if view == "gain":
            return frozen(20*np.log10(np.maximum(abs(source.response),1e-12)))
        if view == "phase":
            return frozen(np.unwrap(np.angle(source.response)))
        if view == "gd":
            p=derive(cache,source,"phase")
            g=-1000*np.gradient(p,2*np.pi*source.frequency)
            g[abs(source.response)<1e-12]=np.nan
            return frozen(g)
        raise ValueError("unknown view")
    return cache.get(key,compute)
