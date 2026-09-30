"""Public UI-independent Target generation facade."""

from phase_fir_designer.target_pipeline import (
    TargetProcessingResult,
    TargetProcessingSettings,
    flat_target_response,
    process_target,
)

__all__ = [
    "TargetProcessingResult",
    "TargetProcessingSettings",
    "flat_target_response",
    "process_target",
]
