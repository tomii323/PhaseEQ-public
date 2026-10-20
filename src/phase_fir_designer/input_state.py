"""Pure input classification and operation preconditions.

An empty saved input is valid state. Availability never selects a replacement
input and never modifies persisted settings or the supplied responses.
"""
from dataclasses import dataclass

import numpy as np

from .config import SpeakerResponse
from timing_provenance import normalize_timing_provenance


@dataclass(frozen=True)
class Availability:
    allowed: bool
    reason: str = ""


@dataclass(frozen=True)
class InputState:
    kind: str
    gain_available: bool = False
    phase_available: bool = False
    absolute_timing_available: bool = False


def classify_input(response, *, definition=None, timing_provenance=None, phase_origin=None) -> InputState:
    if response is None:
        return InputState("empty")
    kind = "ideal" if definition is not None else "measured"
    try:
        frequency = np.asarray(response.frequency, dtype=float)
        gain = np.asarray(response.gain_db, dtype=float)
        if (frequency.ndim != 1 or frequency.size < 2
                or gain.shape != frequency.shape
                or not np.all(np.isfinite(frequency)) or np.any(frequency < 0)
                or not np.all(np.diff(frequency) > 0) or not np.all(np.isfinite(gain))):
            return InputState("invalid")
        phase_available = response.phase_deg is not None
        if phase_available:
            phase = np.asarray(response.phase_deg, dtype=float)
            if phase.shape != frequency.shape or not np.all(np.isfinite(phase)):
                return InputState("invalid")
    except (AttributeError, TypeError, ValueError):
        return InputState("invalid")
    provenance = normalize_timing_provenance(timing_provenance)
    absolute = (kind == "measured" and phase_available
                and phase_origin in {None, "measured"}
                and provenance is not None and provenance.trusted)
    return InputState(kind, True, phase_available, absolute)


def input_availability(state: InputState, operation: str, *, target=None,
                       acoustic_target: bool = False) -> Availability:
    """Return one operation's requirements, without imposing a global gate."""
    if operation in {"manual_eq", "standalone_export"}:
        return Availability(True)
    if operation not in {"speaker_preview", "auto_iir", "fir_auto_gain",
                          "fir_auto_phase", "multiway_return", "absolute_timing"}:
        raise ValueError(f"unknown input operation: {operation}")
    if not state.gain_available:
        return Availability(False, "invalid_input" if state.kind == "invalid" else "empty_input")
    if (operation in {"fir_auto_phase", "absolute_timing"}
            or (operation == "multiway_return" and acoustic_target)) and not state.phase_available:
        return Availability(False, "missing_phase")
    if operation == "absolute_timing" and not state.absolute_timing_available:
        return Availability(False, "missing_absolute_timing")
    if operation in {"auto_iir", "fir_auto_gain", "fir_auto_phase"}:
        target_state = classify_input(target)
        if not target_state.gain_available:
            return Availability(False, "missing_target")
    return Availability(True)


@dataclass(frozen=True)
class ProcessedInputContext:
    owner: object
    generation: str
    state: InputState
    speaker: SpeakerResponse | None
    target: SpeakerResponse | None

    def availability(self, operation, *, owner, generation=None, acoustic_target=False):
        if self.owner != owner or (generation is not None and self.generation != generation):
            return Availability(False, "context_mismatch")
        return input_availability(self.state, operation, target=self.target,
                                  acoustic_target=acoustic_target)


def clear_processed_input_context(state):
    """Discard derived responses at an input/owner restore boundary only."""
    for key in ("_processed_input_context", "_current_processed_speaker_response",
                "_current_effective_target_response", "_current_input_pipeline_result",
                "_current_target_pipeline_result"):
        state.pop(key, None)
