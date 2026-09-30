from __future__ import annotations

import numpy as np


def built_in_target_magnitude_db(
    frequency_hz: np.ndarray,
    *,
    level_db: float = 0.0,
    low_corner_hz: float = 20.0,
    high_corner_hz: float = 20_000.0,
    low_slope_db_per_oct: float = 0.0,
    high_slope_db_per_oct: float = 0.0,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=float)
    safe = np.maximum(frequency, 1e-9)
    gain = np.full(frequency.shape, float(level_db), dtype=float)
    if low_corner_hz > 0.0:
        gain += np.where(
            safe < float(low_corner_hz),
            float(low_slope_db_per_oct) * np.log2(float(low_corner_hz) / safe),
            0.0,
        )
    if high_corner_hz > 0.0:
        gain += np.where(
            safe > float(high_corner_hz),
            float(high_slope_db_per_oct) * np.log2(safe / float(high_corner_hz)),
            0.0,
        )
    gain[frequency <= 0.0] = gain[1] if gain.size > 1 else float(level_db)
    return gain


def target_response_from_magnitude_db(
    magnitude_db: np.ndarray,
    *,
    minimum_phase: bool,
) -> np.ndarray:
    magnitude = 10.0 ** (np.asarray(magnitude_db, dtype=float) / 20.0)
    if not minimum_phase or magnitude.size < 2:
        return magnitude.astype(np.complex128)
    log_half = np.log(np.maximum(magnitude, 1e-15))
    full_log = np.concatenate((log_half, log_half[-2:0:-1]))
    cepstrum = np.fft.ifft(full_log).real
    minimum_cepstrum = np.zeros_like(cepstrum)
    minimum_cepstrum[0] = cepstrum[0]
    half = cepstrum.size // 2
    minimum_cepstrum[1:half] = 2.0 * cepstrum[1:half]
    if cepstrum.size % 2 == 0:
        minimum_cepstrum[half] = cepstrum[half]
    return np.exp(np.fft.fft(minimum_cepstrum)[:magnitude.size]).astype(np.complex128)


def target_impulse_response(response: np.ndarray, fft_size: int) -> np.ndarray:
    impulse = np.fft.irfft(np.asarray(response, dtype=complex), n=int(fft_size))
    return np.roll(impulse, int(fft_size) // 2)
