# processing/fractional_centering.py
from __future__ import annotations
import numpy as np

try:
    import pandas as pd  # 任意依存。無い場合は最低限で動作
except Exception:
    pd = None

__all__ = ["auto_center_fractional"]

def _coerce_ir(h) -> np.ndarray:
    """
    任意の h を 1次元 float64 ndarray に正規化する。
    受理：np.ndarray / list / tuple / pd.Series / 単一列の pd.DataFrame /
         dict の場合は 'ir' | 'y' | 'data' のいずれかを探索

    失敗時：ValueError（受理可能な型と推奨対処を明記）
    """
    # 直接 ndarray
    if isinstance(h, np.ndarray):
        arr = h
    # list / tuple
    elif isinstance(h, (list, tuple)):
        arr = np.asarray(h, dtype=object)  # 一旦 object
    # pandas 系
    elif pd is not None:
        if isinstance(h, pd.Series):
            arr = h.to_numpy()
        elif isinstance(h, pd.DataFrame):
            if h.shape[1] == 1:
                arr = h.iloc[:, 0].to_numpy()
            else:
                # 多列なら先頭列（慣例）。明示したい場合は呼び出し側で列選択を
                arr = h.iloc[:, 0].to_numpy()
        else:
            arr = None
    # dict（よくある包み）
    elif isinstance(h, dict):
        for key in ("ir", "y", "data"):
            if key in h:
                return _coerce_ir(h[key])
        arr = None
    else:
        arr = None

    if arr is None:
        raise ValueError(
            "auto_center_fractional: unsupported type for h. "
            "Accepts ndarray/list/tuple/pandas(Series/DataFrame single column)/dict with keys ['ir','y','data']."
        )

    # 文字列が混在する場合への対処
    if arr.dtype.kind in ("U", "S", "O"):
        # 文字列ベース（例: ['0.1','-0.2', ...] や ['Low', ...]）
        if pd is not None:
            s = pd.Series(arr)
            s_num = pd.to_numeric(s, errors="coerce")
            if s_num.isna().any():
                bad = s[s_num.isna()].unique()[:5]
                raise ValueError(
                    f"auto_center_fractional: h contains non-numeric tokens {list(bad)} "
                    "→ 信号データではなくラベル文字列（例:'Low'）を渡していませんか？"
                )
            arr = s_num.to_numpy(dtype=np.float64)
        else:
            # pandas が無い場合の厳格路線
            try:
                arr = np.array([float(x) for x in arr], dtype=np.float64)
            except Exception as e:
                raise ValueError(
                    "auto_center_fractional: h contains non-numeric data. "
                    "数値以外の文字列（例:'Low'）が含まれています。"
                ) from e
    else:
        arr = np.asarray(arr, dtype=np.float64)

    # 2D → 1D（先頭チャンネル）
    if arr.ndim == 2:
        if 1 in arr.shape:
            arr = arr.reshape(-1,)
        else:
            arr = arr[:, 0]
    elif arr.ndim > 2:
        raise ValueError("auto_center_fractional: h must be 1D or 2D (single channel).")

    if arr.size == 0:
        raise ValueError("auto_center_fractional: empty impulse response.")

    return arr


def _estimate_shift_samples(h: np.ndarray, mode: str, resolution: int) -> float:
    """
    分数サンプルのシフト量（サンプル単位, 実数）を推定する軽量ダミー実装。
    実装詳細は既存コードのロジックを流用してください。
    """
    # === 例示：エネルギー重心 vs 群遅延基準（簡易版）===
    if mode == "centroid":
        n = np.arange(len(h))
        centroid = (np.sum(n * (h ** 2)) / np.sum(h ** 2))
        shift = (len(h) // 2) - centroid
    else:  # "groupdelay"
        # 実装簡易化。実機では unwrap 位相→群遅延→ピーク等で推定
        # ここでは“ほぼ中央”に寄せる小調整ダミー
        shift = 0.0
    return float(shift)


def _fractional_delay_fir(shift_samples: float, taps: int = 63) -> np.ndarray:
    """
    Lagrange 近似などで分数ディレイ FIR を生成する簡易版。
    既存実装があればそちらを優先してください（ここは雛形）。
    """
    n = np.arange(taps) - (taps - 1) / 2.0
    # Sinc 近似（簡便）。本来は最適化や窓を選択
    eps = 1e-12
    h = np.sinc(n - shift_samples + eps)
    # 窓（例：cosine-taper）を軽く適用（端リンギング低減）
    w = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(taps) / (taps - 1))))
    h = h * w
    # 正規化
    h /= np.sum(h) if np.sum(h) != 0 else 1.0
    return h.astype(np.float64)


def auto_center_fractional(
    h,
    resolution: int = 4,
    mode: str = "groupdelay",
    smart_threshold: float = 0.10,
    fir_taps: int = 63,
) -> np.ndarray:
    """
    IRを「分数サンプル精度」でセンタリング（小数センタリング）。

    Parameters
    ----------
    h : array-like | pandas | dict
        IR。任意の包みを _coerce_ir() で 1D float64 に正規化。
    resolution : int, default 4
        分数シフトの解像度（4なら 1/4 サンプル精度相当の探索・丸め）。
    mode : {"groupdelay", "centroid"}, default "groupdelay"
        基準の選択。表記は大文字小文字を無視。
    smart_threshold : float, default 0.10
        推定シフト量がこの閾値未満なら適用スキップ（“スマート適用”）。
        単位はサンプル。
    fir_taps : int, default 63
        分数ディレイ FIR のタップ数。

    Returns
    -------
    np.ndarray
        センタリング後の IR（1D, float64）
    """
    arr = _coerce_ir(h)

    # mode 正規化（大文字小文字無視、別名も吸収）
    mode_norm = (mode or "").strip().lower()
    if mode_norm in ("gd", "group_delay"):
        mode_norm = "groupdelay"
    if mode_norm not in ("groupdelay", "centroid"):
        raise ValueError("auto_center_fractional: mode must be 'groupdelay' or 'centroid'.")

    # シフト推定
    raw_shift = _estimate_shift_samples(arr, mode=mode_norm, resolution=resolution)

    # 解像度に基づき丸め（例：1/4 サンプル刻み）
    step = 1.0 / max(1, int(resolution))
    shift_q = np.round(raw_shift / step) * step

    # スマート適用：微小ならスキップ
    if abs(shift_q) < float(smart_threshold):
        return arr  # 無変更

    # 分数ディレイ FIR でシフト適用（畳み込み）
    fd = _fractional_delay_fir(shift_q, taps=fir_taps)
    out = np.convolve(arr, fd, mode="same").astype(np.float64)
    return out
