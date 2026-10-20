"""Definition and materialization of the generated, non-measured input."""
from dataclasses import dataclass

import numpy as np

from .config import SpeakerResponse


IDEAL_INPUT_ID = "phaseeq_builtin_ideal_speaker_v1"
IDEAL_INPUT_NAME = "理想スピーカー特性（平坦・0 dB／0°）"


def ideal_input_definition() -> dict:
    return {"kind": "ideal_speaker", "version": 1}


def normalize_input_definition(value) -> dict | None:
    if value is None:
        return None
    if (not isinstance(value, dict) or value != ideal_input_definition()
            or type(value.get("version")) is not int):
        raise ValueError("理想スピーカー特性の定義が不正です。")
    return ideal_input_definition()


def ideal_speaker_response(sample_rate: int, analysis_fft_size: int) -> SpeakerResponse:
    if int(sample_rate) <= 0 or int(analysis_fft_size) < 4:
        raise ValueError("ideal input requires a positive sample rate and FFT size >= 4")
    frequency = np.fft.rfftfreq(int(analysis_fft_size), d=1.0 / int(sample_rate))
    # Speaker assets use strictly positive frequencies; retain the full upper
    # range independently of the FIR tap count and of the library preview axis.
    frequency = np.r_[min(2.0, float(frequency[1]) / 2), frequency[1:]]
    zeros = np.zeros_like(frequency).tolist()
    from .response_support import generated_support
    return generated_support(SpeakerResponse(frequency.tolist(), zeros, list(zeros)))


@dataclass(frozen=True)
class InputSelection:
    kind: str
    definition: dict | None = None


def plan_input_selection(*, saved_state_exists: bool, requested_kind: str | None = None) -> InputSelection:
    """An explicit empty state is saved state, never a request for a default."""
    if requested_kind is not None:
        if requested_kind not in {"measured", "ideal", "empty"}:
            raise ValueError("unknown input selection")
        kind = requested_kind
    else:
        kind = "restore" if saved_state_exists else "ideal"
    return InputSelection(kind, ideal_input_definition() if kind == "ideal" else None)
