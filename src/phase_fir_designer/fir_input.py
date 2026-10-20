"""IIR application boundary for FIR calculation and its input display.

Callers supply the original, projected S, never an already corrected response.
Interpolation/extension remains with the caller; this stage owns S -> S * H.
No derived input is written into persisted measurement or IIR settings.
"""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import IIRFilter


@dataclass(frozen=True)
class FIRInputTransfer:
    response: np.ndarray

    def complex_response(self, source: np.ndarray) -> np.ndarray:
        return np.asarray(source, dtype=complex) * self.response

    def gain_db(self, source: np.ndarray) -> np.ndarray:
        return np.asarray(source, dtype=float) + 20 * np.log10(np.maximum(abs(self.response), 1e-12))

    def phase_deg(self, source: np.ndarray) -> np.ndarray:
        return np.asarray(source, dtype=float) + np.rad2deg(np.unwrap(np.angle(self.response)))


def fir_input_transfer(filters: Iterable["IIRFilter"], sample_rate: int, frequency: np.ndarray) -> FIRInputTransfer:
    from .iir import iir_frequency_response
    return FIRInputTransfer(iir_frequency_response(filters, sample_rate, frequency))
