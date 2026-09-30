# -*- coding: utf-8 -*-
"""
slope.py  (v1.0.0)

Baffle Step FIR with Log Slope & Auto Heuristics
- make_baffle_step_fir(): 対数スロープ（dB/oct）でのBSC FIR設計
- estimate_baffle_pivot_freq(): バッフル幅 W[m] から支点周波数を推定
- log_tilt_curve(): firwin2 へ渡せる log 傾斜ターゲットの生成
- auto_soft_knee(): 予測リンギング/群遅延の簡易評価による soft_knee 自動化

推奨:
- 奇数 taps（線形位相の中心を定義しやすい）
- gain_max_db は 3〜7dB で運用（BSC 想定）
"""

from __future__ import annotations
import math
from typing import Tuple, Sequence, Optional, Union

import numpy as np
from scipy.signal import firwin2, group_delay, freqz

Number = Union[int, float]

# -----------------------------
# 基本ユーティリティ
# -----------------------------
def odd_number(n: Number) -> int:
    """与えられた n を切り下げた後、奇数へ調整して返す。"""
    n = int(math.floor(n))
    return n if (n % 2 == 1) else (n + 1)


def estimate_baffle_pivot_freq(baffle_width_m: float, c: float = 343.0) -> float:
    """
    バッフル幅 W[m] から支点周波数 f_b ≈ c / (π·W) を推定する。

    Parameters
    ----------
    baffle_width_m : float
        バッフル正味幅 [m]（有効バッフル径の近似でも可）
    c : float, default 343.0
        音速 [m/s]（20°C・無風の近似）

    Returns
    -------
    float
        推定支点周波数 [Hz]
    """
    W = max(float(baffle_width_m), 1e-3)
    return float(c / (math.pi * W))


# -----------------------------
# 対数スロープのターゲット生成
# -----------------------------
def log_tilt_curve(
    fs: float,
    n_grid: int,
    f_pivot: float,
    slope_db_per_oct: float = 6.0,
    gain_max_db: float = 6.0,
    f_lo: float = 20.0,
    f_hi: Optional[float] = None,
    soft_knee: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    対数軸上の傾斜（dB/oct）で定義したターゲット曲線を返す。
    戻り値は firwin2 にそのまま渡せる (w, amp)。

    Notes
    -----
    - 低域側 (f < f_pivot) が +、高域側で − となる符号系（BSC向け）
    - 0Hz は f_lo 相当値でホールド
    """
    nyq = fs * 0.5
    if f_hi is None:
        f_hi = nyq
    n_grid = max(512, int(n_grid))

    # 0を含む単調増加の周波数列（firwin2要件）
    f_grid = np.geomspace(max(f_lo, 1e-3), f_hi, n_grid - 1)
    f_grid = np.concatenate(([0.0], f_grid))
    w = f_grid / nyq

    eps = 1e-12
    ff = np.maximum(f_grid, max(f_lo, eps))
    gdB = - slope_db_per_oct * np.log2(ff / max(f_pivot, eps))
    if soft_knee is not None and soft_knee > 0:
        gdB = gain_max_db * np.tanh(gdB / (gain_max_db * soft_knee))
    gdB = np.clip(gdB, -gain_max_db, +gain_max_db)
    gdB[0] = gdB[1]

    amp = 10.0 ** (gdB / 20.0)

    # firwin2 の端点条件を満たす（1.0 を末尾に追加）
    if w[-1] < 1.0:
        w = np.append(w, 1.0)
        amp = np.append(amp, amp[-1])

    return w, amp


# -----------------------------
# ソフトニー自動化（ヒューリスティック + クイック評価）
# -----------------------------
def _window_factor(window: Union[str, Sequence, Tuple]) -> float:
    """
    窓の鋭さをざっくりスコア化（大きいほど鋭い→リンギング増の傾向）。
    Kaiser: 1 + beta/10,  Hamming/Hann: ~1.0,  Blackman: ~1.2
    """
    if isinstance(window, tuple):
        wname = str(window[0]).lower()
        if wname == "kaiser":
            beta = float(window[1]) if len(window) > 1 else 12.0
            return 1.0 + beta / 10.0
        return 1.1  # 既知以外はやや強め
    else:
        wname = str(window).lower()
        if "blackman" in wname:
            return 1.2
        if "hann" in wname or "hanning" in wname or "hamm" in wname:
            return 1.0
        if "kaiser" in wname:
            return 1.8  # beta 未指定の仮
    return 1.1


def _pre_post_ringing_ratio(ir: np.ndarray, center: int, width: int = 32) -> float:
    """中心から前後 ±width サンプルのエネルギー比（pre / post）。"""
    pre = ir[center - width:center]
    post = ir[center + 1:center + 1 + width]
    pre_e = float(np.sum(pre ** 2) + 1e-18)
    post_e = float(np.sum(post ** 2) + 1e-18)
    return pre_e / post_e


def _gd_ripple_metric(fs: float, h: np.ndarray, n_fft: int = 4096) -> float:
    """
    群遅延の簡易揺らぎメトリクス（線形位相 FIR でも端部/遷移で数値的揺らぎは残る）。
    値が大きいほど“落ち着きが悪い”とみなす。
    """
    w, H = freqz(h, worN=n_fft, fs=fs)
    # 位相 unwrap → 差分から群遅延（サンプル）を数値近似
    phase = np.unwrap(np.angle(H))
    # dphi / dw (rad/Hz) をサンプル遅延に換算: gd = - (dphi/dw) * fs / (2π)
    dphi = np.gradient(phase, w + 1e-9)
    gd = - dphi * fs / (2.0 * np.pi)
    # 通過域（例: 50Hz〜f_pivot*2）だけで粗めに評価したいが f_pivot 未知のため、中央 10%〜80%で評価
    i0, i1 = int(0.10 * len(gd)), int(0.80 * len(gd))
    sel = gd[i0:i1]
    if sel.size < 16:
        return 0.0
    # 変動のRMS / 中央値で正規化（無次元）
    med = float(np.median(sel))
    rms = float(np.sqrt(np.mean((sel - med) ** 2)))
    denom = max(abs(med), 1e-9)
    return rms / denom


def auto_soft_knee(
    fs: float,
    taps: int,
    slope_db_per_oct: float,
    gain_max_db: float,
    f_pivot: float,
    f_lo: float,
    f_hi: float,
    window: Union[str, Tuple] = ("kaiser", 12.0),
    target_pr_ratio: float = 0.35,
    target_gd_ripple: float = 0.08,
    max_iter: int = 3,
) -> float:
    """
    soft_knee を自動推定する。
    1) ヒューリスティック初期値 → 2) 仮設計→評価→必要なら段階的に増加。

    Returns
    -------
    float
        推奨 soft_knee （例: 0.8〜2.0）
    """
    wfac = _window_factor(window)
    # ヒューリスティック初期値：強い傾斜/長いtap/鋭い窓ほど knee を強く
    slope_norm = max(abs(slope_db_per_oct), 1e-6) / 6.0  # 6dB/oct 基準化
    taps_norm = max(taps, 1) / 512.0
    knee0 = 0.8 + 0.5 * slope_norm + 0.4 * (taps_norm) + 0.2 * (wfac - 1.0)
    knee = float(np.clip(knee0, 0.7, 1.8))

    # クイック評価で微調整（最大 max_iter 回）
    nyq = fs * 0.5
    for _ in range(max_iter):
        w, amp = log_tilt_curve(
            fs=fs, n_grid=max(512, 2 * taps), f_pivot=f_pivot,
            slope_db_per_oct=slope_db_per_oct, gain_max_db=gain_max_db,
            f_lo=f_lo, f_hi=f_hi, soft_knee=knee
        )
        h = firwin2(numtaps=odd_number(taps), freq=w, gain=amp, window=window, antisymmetric=False)
        c = len(h) // 2
        pr = _pre_post_ringing_ratio(h, c, width=min(64, max(16, len(h) // 16)))
        gd_r = _gd_ripple_metric(fs, h)

        # いずれかが閾値を超えたら knee を少し強める
        if (pr > target_pr_ratio) or (gd_r > target_gd_ripple):
            knee *= 1.15
            knee = float(np.clip(knee, 0.7, 2.2))
        else:
            break

    return float(np.clip(knee, 0.7, 2.2))


# -----------------------------
# メイン：BSC FIR 設計
# -----------------------------
def make_baffle_step_fir(
    fs: float,
    taps: int,
    *,
    # 傾斜定義（logスロープ）
    slope_db_per_oct: float = 6.0,
    gain_max_db: float = 6.0,
    # 支点周波数：直接指定 or バッフル幅から推定
    f_pivot: Optional[float] = None,
    baffle_width_m: Optional[float] = None,
    sound_speed: float = 343.0,
    # 有効帯域
    f_lo: float = 20.0,
    f_hi: Optional[float] = None,
    # 窓
    window: Union[str, Tuple] = ("kaiser", 12.0),
    # ソフトニー
    soft_knee: Optional[float] = None,
    auto_soft_knee_flag: bool = True,
    # グリッド密度
    grid_mul: int = 2,
    ensure_odd_taps: bool = True,
) -> np.ndarray:
    """
    バッフルステップ補正のための対数スロープ FIR を設計する（線形位相）。

    Parameters
    ----------
    fs : float
        サンプリング周波数 [Hz]
    taps : int
        フィルタ長（奇数推奨）
    slope_db_per_oct : float, default 6.0
        1オクターブあたりの傾斜量 [dB/oct]。正で低域ブースト。
    gain_max_db : float, default 6.0
        クリップ上限（±dB）
    f_pivot : float | None
        支点周波数 [Hz]。None の場合、baffle_width_m から自動推定。
    baffle_width_m : float | None
        バッフル幅 [m]。f_pivot 未指定時に使用。
    sound_speed : float, default 343.0
        音速 [m/s]（支点推定用）
    f_lo : float, default 20.0
        有効帯域の下限 [Hz]（log 定義下限。0Hzはこの値でホールド）
    f_hi : float | None
        有効帯域の上限 [Hz]。None は Nyquist。
    window : window spec, default ("kaiser", 12.0)
        firwin2 に渡す窓。
    soft_knee : float | None
        クリップ前の tanh ソフトクランプ係数。None かつ auto_soft_knee_flag=True で自動推定。
    auto_soft_knee_flag : bool, default True
        True で soft_knee 自動化を有効。
    grid_mul : int, default 2
        周波数グリッドの密度倍率（taps*grid_mul 以上に自動で底上げ）
    ensure_odd_taps : bool, default True
        True なら tap を奇数へ丸める。

    Returns
    -------
    np.ndarray
        FIR 係数（線形位相）
    """
    if ensure_odd_taps and (taps % 2 == 0):
        taps = odd_number(taps)

    nyq = fs * 0.5
    if f_hi is None:
        f_hi = nyq

    # --- 自動 f_pivot ---
    if f_pivot is None:
        if baffle_width_m is not None:
            f_pivot = estimate_baffle_pivot_freq(baffle_width_m, c=sound_speed)
        else:
            # バックアップ（一般的な小型SPの目安 500〜800Hz）
            f_pivot = 600.0

    # --- soft_knee 自動化 ---
    if (soft_knee is None) and auto_soft_knee_flag:
        soft_knee = auto_soft_knee(
            fs=fs, taps=taps, slope_db_per_oct=slope_db_per_oct,
            gain_max_db=gain_max_db, f_pivot=f_pivot, f_lo=f_lo, f_hi=f_hi,
            window=window
        )

    # --- ターゲット生成 ---
    n_grid = max(512, int(taps * max(2, grid_mul)))
    w, amp = log_tilt_curve(
        fs=fs, n_grid=n_grid, f_pivot=f_pivot,
        slope_db_per_oct=slope_db_per_oct, gain_max_db=gain_max_db,
        f_lo=f_lo, f_hi=f_hi, soft_knee=soft_knee
    )

    # --- FIR 設計 ---
    h = firwin2(numtaps=taps, freq=w, gain=amp, window=window, antisymmetric=False)
    return h


# -----------------------------
# 簡易 self-test（モジュール直実行で動作確認）
# -----------------------------
if __name__ == "__main__":
    fs = 48000
    taps = 511
    # 例1：W=0.22m の書棚スピーカー程度
    h = make_baffle_step_fir(
        fs=fs, taps=taps,
        slope_db_per_oct=6.0, gain_max_db=6.0,
        baffle_width_m=0.22,   # → 自動で f_pivot 推定
        f_lo=20.0, f_hi=None,
        window=("kaiser", 12.0),
        soft_knee=None,        # 自動
        auto_soft_knee_flag=True
    )
    print("FIR length:", len(h), "sum:", float(np.sum(h)))
