from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .output_pipeline import DSPSystemOutput
from .regeneration import DSPWayResult
from .signal_path import final_way_complex_response


@dataclass(frozen=True)
class ReverseNullView:
    """User-facing crossover comparison without a pass/fail judgement."""

    frequency: np.ndarray
    lower_response: np.ndarray
    upper_response: np.ndarray
    normal_sum: np.ndarray
    reverse_sum: np.ndarray
    crossover_frequency_hz: float
    phase_difference_deg: float
    reverse_null_db: float
    reverse_null_frequency_hz: float


def build_reverse_null_view(
    lower: DSPWayResult,
    upper: DSPWayResult,
    output: DSPSystemOutput,
    crossover_frequency_hz: float,
    *,
    input_id: str | None = None,
) -> ReverseNullView:
    """Evaluate normal and one-way reversed sums on the exported signal path."""

    sample_rate = max(int(lower.device.sample_rate), int(upper.device.sample_rate))
    fft_size = max(
        2,
        int(lower.analysis_fft_size),
        int(upper.analysis_fft_size),
        int(lower.device.taps),
        int(upper.device.taps),
    )
    if fft_size % 2:
        fft_size += 1
    frequency = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    lower_response = final_way_complex_response(lower, output, frequency, input_id=input_id)
    upper_response = final_way_complex_response(upper, output, frequency, input_id=input_id)
    normal = lower_response + upper_response
    reverse = lower_response - upper_response

    fc = float(crossover_frequency_hz)
    index = int(np.argmin(np.abs(frequency - fc)))
    phase_delta = np.rad2deg(np.angle(lower_response[index] / (upper_response[index] + 1e-30)))
    phase_delta = float(abs((phase_delta + 180.0) % 360.0 - 180.0))
    band = (frequency >= fc / 2.0) & (frequency <= fc * 2.0)
    lower_active = np.abs(lower_response) >= np.max(np.abs(lower_response)) * 1e-2
    upper_active = np.abs(upper_response) >= np.max(np.abs(upper_response)) * 1e-2
    candidates = np.flatnonzero(band & lower_active & upper_active)
    if not len(candidates):
        candidates = np.array([index])
    null_index = int(candidates[np.argmin(np.abs(reverse[candidates]))])
    null_db = float(20.0 * np.log10(max(float(np.abs(reverse[null_index])), 1e-12)))
    return ReverseNullView(
        frequency=frequency,
        lower_response=lower_response,
        upper_response=upper_response,
        normal_sum=normal,
        reverse_sum=reverse,
        crossover_frequency_hz=fc,
        phase_difference_deg=phase_delta,
        reverse_null_db=null_db,
        reverse_null_frequency_hz=float(frequency[null_index]),
    )
