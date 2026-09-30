from __future__ import annotations

from dataclasses import dataclass

from scipy.fft import next_fast_len

from .config import SUPPORTED_ANALYSIS_FFT_SIZES


DEFAULT_ANALYSIS_FFT_SIZE = 16_384


@dataclass(frozen=True)
class AnalysisPolicy:
    """Application-wide FFT policy used for plots and response evaluation.

    FIR length remains owned by the DSP Device.  Engines may request a larger
    transform, but a Speaker Package never owns or changes this policy.
    """

    base_fft_size: int = DEFAULT_ANALYSIS_FFT_SIZE

    def normalized(self) -> "AnalysisPolicy":
        requested = int(self.base_fft_size)
        nearest = min(
            SUPPORTED_ANALYSIS_FFT_SIZES,
            key=lambda value: abs(int(value) - requested),
        )
        return AnalysisPolicy(base_fft_size=int(nearest))

    def effective_fft_size(
        self,
        *,
        required_length: int = 1,
        engine_minimum: int = 1,
    ) -> int:
        normalized = self.normalized()
        required = max(
            int(normalized.base_fft_size),
            int(required_length),
            int(engine_minimum),
            2,
        )
        effective = int(next_fast_len(required, real=True))
        return effective if effective % 2 == 0 else effective + 1


def normalize_analysis_fft_size(value: int) -> int:
    return AnalysisPolicy(int(value)).normalized().base_fft_size
