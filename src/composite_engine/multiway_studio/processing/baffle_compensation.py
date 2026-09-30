from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy import optimize, signal

from .filter import make_baffle_step_fir


@dataclass(frozen=True)
class BaffleCompensationPlan:
    mode: Literal["off", "fir", "iir"]
    reason: str
    gain_db: float = 0.0
    frequency_hz: float = 0.0
    q: float = 0.0
    sos: tuple[tuple[float, ...], ...] = ()
    fit_rms_db: float = 0.0
    fit_max_error_db: float = 0.0

    @property
    def display_mode(self) -> str:
        return {
            "off": "OFF",
            "fir": "Linear-phase FIR",
            "iir": "Minimum-phase IIR High Shelf",
        }[self.mode]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "display_mode": self.display_mode,
            "reason": self.reason,
            "gain_db": self.gain_db,
            "frequency_hz": self.frequency_hz,
            "q": self.q,
            "sos": [list(row) for row in self.sos],
            "fit_rms_db": self.fit_rms_db,
            "fit_max_error_db": self.fit_max_error_db,
        }


def resolve_baffle_compensation_mode(
    enabled_bands: Sequence[str],
    fir_states: Mapping[str, Mapping[str, object]],
    *,
    enabled: bool,
) -> Literal["off", "fir", "iir"]:
    """Use one common realization so baffle correction cannot add inter-Way phase."""
    if not enabled:
        return "off"
    bands = tuple(dict.fromkeys(str(value) for value in enabled_bands if str(value)))
    if bands and all(bool(fir_states.get(band, {}).get("enabled", False)) for band in bands):
        return "fir"
    return "iir"


def _shelf_response(
    frequency_hz: np.ndarray, sample_rate_hz: int, gain_db: float,
    center_hz: float, q: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_rate = int(sample_rate_hz)
    frequency = float(center_hz)
    quality = float(q)
    if not 0.0 < frequency < sample_rate / 2.0 or quality <= 0.0:
        raise ValueError("High Shelf frequency and Q are invalid")
    amplitude = 10.0 ** (float(gain_db) / 40.0)
    angular = 2.0 * np.pi * frequency / sample_rate
    cosine = np.cos(angular)
    alpha = np.sin(angular) / (2.0 * quality)
    root_a = np.sqrt(amplitude)
    coefficients = np.asarray((
        amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + 2.0 * root_a * alpha),
        -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
        amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - 2.0 * root_a * alpha),
        (amplitude + 1.0) - (amplitude - 1.0) * cosine + 2.0 * root_a * alpha,
        2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
        (amplitude + 1.0) - (amplitude - 1.0) * cosine - 2.0 * root_a * alpha,
    ), dtype=float)
    coefficients /= coefficients[3]
    sos = coefficients[None, :]
    _, response = signal.sosfreqz(sos, worN=frequency_hz, fs=float(sample_rate_hz))
    return np.asarray(sos, dtype=float), np.asarray(response, dtype=np.complex128)


@lru_cache(maxsize=64)
def design_baffle_iir_shelf(
    sample_rate_hz: int,
    upper_frequency_hz: float,
    correction_db: float,
    lower_frequency_hz: float = 170.0,
) -> BaffleCompensationPlan:
    """Fit one RBJ High Shelf to the realized legacy linear-phase BSC magnitude."""
    sample_rate = int(sample_rate_hz)
    f0 = max(20.0, float(lower_frequency_hz))
    f1 = min(max(f0 + 1.0, float(upper_frequency_hz)), sample_rate / 2.0 - 1.0)
    amount = max(0.0, float(correction_db))
    if amount <= 0.0:
        return BaffleCompensationPlan("off", "補正量が0 dBです。")

    reference_fir = make_baffle_step_fir(
        sample_rate, f0=f0, f1=f1, max_atten_db=amount,
    )
    frequency = np.geomspace(20.0, min(sample_rate * 0.45, max(10_000.0, f1 * 8.0)), 768)
    _, reference = signal.freqz(reference_fir, worN=frequency, fs=float(sample_rate))
    reference_db = 20.0 * np.log10(np.maximum(np.abs(reference), 1e-12))

    high_band = frequency >= min(f1 * 4.0, frequency[-1])
    initial_gain = float(np.median(reference_db[high_band])) if np.any(high_band) else -amount / 2.0
    initial = np.asarray([
        np.clip(initial_gain, -amount, -0.01),
        np.sqrt(f0 * f1),
        0.6,
    ])
    lower = np.asarray([-max(0.1, amount), max(20.0, f0 * 0.5), 0.1])
    upper = np.asarray([-1e-4, min(sample_rate * 0.45, f1 * 2.0), 1.0])

    def residual(parameters: np.ndarray) -> np.ndarray:
        _sos, response = _shelf_response(
            frequency, sample_rate, parameters[0], parameters[1], parameters[2],
        )
        realized_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
        return realized_db - reference_db

    fitted = optimize.least_squares(
        residual, initial, bounds=(lower, upper),
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200,
    )
    gain_db, center_hz, q = (float(value) for value in fitted.x)
    sos, _response = _shelf_response(frequency, sample_rate, gain_db, center_hz, q)
    errors = residual(fitted.x)
    return BaffleCompensationPlan(
        "iir",
        "FIRなしChannelを含むため、全Channelを共通IIR Shelfで補正します。",
        gain_db=gain_db,
        frequency_hz=center_hz,
        q=q,
        sos=tuple(tuple(float(value) for value in row) for row in sos),
        fit_rms_db=float(np.sqrt(np.mean(np.square(errors)))),
        fit_max_error_db=float(np.max(np.abs(errors))),
    )


def baffle_plan(
    sample_rate_hz: int,
    upper_frequency_hz: float,
    correction_db: float,
    enabled_bands: Sequence[str],
    fir_states: Mapping[str, Mapping[str, object]],
    *,
    enabled: bool,
) -> BaffleCompensationPlan:
    mode = resolve_baffle_compensation_mode(
        enabled_bands, fir_states, enabled=enabled,
    )
    if mode == "off":
        return BaffleCompensationPlan("off", "バッフル補正はOFFです。")
    if mode == "fir":
        return BaffleCompensationPlan(
            "fir", "対象の全ChannelにFIRがあるため、全Channelを共通FIRで補正します。",
        )
    return design_baffle_iir_shelf(
        int(sample_rate_hz), float(upper_frequency_hz), float(correction_db),
    )


def common_baffle_response(
    plan: BaffleCompensationPlan,
    frequency_hz: np.ndarray,
    sample_rate_hz: int,
    *,
    fir: np.ndarray | None = None,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=float)
    if plan.mode == "iir" and plan.sos:
        _, response = signal.sosfreqz(
            np.asarray(plan.sos, dtype=float), worN=frequency,
            fs=float(sample_rate_hz),
        )
        return np.asarray(response, dtype=np.complex128)
    if plan.mode == "fir" and fir is not None:
        values = np.asarray(fir, dtype=float)
        _, response = signal.freqz(values, worN=frequency, fs=float(sample_rate_hz))
        angular = 2.0 * np.pi * frequency / float(sample_rate_hz)
        return np.asarray(response, dtype=np.complex128) * np.exp(
            1j * angular * ((values.size - 1) / 2.0)
        )
    return np.ones(frequency.shape, dtype=np.complex128)
