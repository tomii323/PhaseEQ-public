"""NTM4 analog prototype and prewarped SOS, independent of application state."""
from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.optimize import brentq

DEFAULT_LOBE_DB = -36.0
MIN_LOBE_DB = -80.0
MAX_LOBE_DB = -20.0


def validate_lobe_db(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not MIN_LOBE_DB <= value <= MAX_LOBE_DB:
        raise ValueError("NTM4 Lobe Height must be within -80..-20 dB")
    return value


@lru_cache(maxsize=601)
def k_squared(lobe_db: float = DEFAULT_LOBE_DB) -> float:
    """Solve the post-notch LP peak (HP is reciprocal in analog frequency)."""
    amplitude = 10.0 ** (validate_lobe_db(lobe_db) / 20.0)

    def peak(t):
        d = np.sqrt(1.0 - t * t)
        # Stable form of (1-d)/(2*d), including small requested lobes.
        return t * t / (2.0 * d * (1.0 + d))

    return float(brentq(lambda t: peak(t) - amplitude, 1e-9, 0.99, xtol=1e-15))


def allpass_q(lobe_db: float = DEFAULT_LOBE_DB) -> float:
    return float(1.0 / np.sqrt(2.0 * (1.0 - k_squared(lobe_db))))


def ntm4_sos(kind: str, frequency_hz: float, sample_rate_hz: int,
             lobe_db: float = DEFAULT_LOBE_DB) -> np.ndarray:
    """Return two LP/HP Biquads or one sum-AP using the same prototype."""
    fc, fs = float(frequency_hz), float(sample_rate_hz)
    if not np.isfinite(fs) or not fs > 0 or not np.isfinite(fc) or not 0 < fc < fs / 2:
        raise ValueError("NTM4 frequency must be within 0..Nyquist")
    t = k_squared(lobe_db)
    if kind == "allpass":
        from .alignment import AllPassSection, allpass_sos
        return allpass_sos(AllPassSection(fc, allpass_q(lobe_db)), sample_rate_hz)
    alpha = np.sqrt(2.0 * (1.0 - t))
    # Transform each second-order factor separately to avoid fourth-order
    # polynomial root conditioning at low Fc / high sample rates.
    warp = 2.0 * fs * np.tan(np.pi * fc / fs)
    denominator = [1.0, alpha * warp, warp * warp]
    numerators = {
        "lowpass": ([t, 0.0, warp * warp], [0.0, 0.0, warp * warp]),
        "highpass": ([1.0, 0.0, t * warp * warp], [1.0, 0.0, 0.0]),
    }
    if kind not in numerators:
        raise ValueError("NTM4 kind must be lowpass, highpass or allpass")
    rows = []
    for numerator in numerators[kind]:
        b, a = signal.bilinear(numerator, denominator, fs=fs)
        rows.append(np.r_[np.pad(b, (0, 3-len(b))), a])
    return np.asarray(rows)


def cascaded_way_sos(way_index, methods, frequencies, overlaps, lobes, fs):
    """Split the remaining HP bus; structural APs belong to alignment state.

    Used only by NTM4-containing banks. Legacy banks retain their topology.
    FIR/Through boundaries contribute no IIR; flat-sum qualification applies
    to zero-overlap banks composed entirely of complementary IIR boundaries.
    """
    from .iir import IIRCrossoverConfig, crossover_edges_hz, iir_crossover_sos
    rows = []
    for index, (method, fc, overlap, lobe) in enumerate(zip(methods, frequencies, overlaps, lobes)):
        if index > way_index or method not in {"NTM4", "LR2", "LR4"}:
            continue
        kind = "highpass" if index < way_index else "lowpass"
        cutoff = crossover_edges_hz(fc, overlap)[1 if kind == "highpass" else 0]
        if method == "NTM4":
            rows.append(ntm4_sos(kind, cutoff, fs, lobe))
        else:
            rows.append(iir_crossover_sos(IIRCrossoverConfig(
                enabled=True, order=int(method[2:]),
                highpass_hz=cutoff if kind == "highpass" else 0,
                lowpass_hz=cutoff if kind == "lowpass" else 0), fs))
    return np.vstack(rows) if rows else np.empty((0, 6))


def cascaded_allpass_sections(ways, frequencies, methods, lobes=(), *, block_limit):
    """Plan later-boundary sum APs for the earlier LP outputs of an NTM4 bank."""
    from .alignment import AllPassSection
    lobes = tuple(lobes) or (DEFAULT_LOBE_DB,) * len(methods)
    if len(lobes) != len(methods):
        raise ValueError("NTM4 Lobe Height count does not match boundaries")
    sections = {way: [] for way in ways}
    recipients = []
    for index, (fc, method, lobe) in enumerate(zip(frequencies, methods, lobes)):
        current = []
        if method in {"NTM4", "LR2", "LR4"}:
            for way in ways[:index]:
                if len(sections[way]) >= block_limit:
                    continue
                q = allpass_q(lobe) if method == "NTM4" else 0.0 if method == "LR2" else 1/np.sqrt(2)
                sections[way].append(AllPassSection(float(fc), q, order=1 if method == "LR2" else 2))
                current.append(way)
        recipients.append(tuple(current))
    return {way: tuple(values) for way, values in sections.items()}, tuple(recipients)
