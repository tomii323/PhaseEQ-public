from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import DesignConfig
from .design import DesignResult, design_phase_fir
from .iir import iir_frequency_response
from .phase_curves import build_target_curves, manual_gain_eq_db, manual_phase_eq_deg


@dataclass(frozen=True)
class CorrectionResult:
    config: DesignConfig
    frequency: np.ndarray
    iir_response: np.ndarray
    manual_gain_db: np.ndarray
    manual_phase_deg: np.ndarray
    auto_gain_db: np.ndarray
    auto_phase_deg: np.ndarray
    design_result: DesignResult

    @property
    def fir(self) -> np.ndarray:
        return self.design_result.fir


def build_correction(
    config: DesignConfig,
    *,
    apply_linear_fir: bool = False,
    apply_linear_fir_mask: bool | None = None,
    compute_group_delay: bool = True,
) -> CorrectionResult:
    effective = config.normalized()
    mask_enabled = apply_linear_fir if apply_linear_fir_mask is None else bool(apply_linear_fir_mask)
    if not apply_linear_fir and not mask_enabled:
        effective = replace(effective, linear_fir_filters=[])
    design_result = design_phase_fir(
        effective,
        compute_group_delay=compute_group_delay,
        apply_linear_fir_overlay=apply_linear_fir,
    )
    return correction_from_design_result(
        effective,
        design_result,
        apply_linear_fir_mask=mask_enabled,
    )


def correction_from_design_result(
    config: DesignConfig,
    design_result: DesignResult,
    *,
    apply_linear_fir_mask: bool = True,
) -> CorrectionResult:
    """Attach the existing IIR, manual EQ and Auto EQ reports to a FIR result."""

    effective = config.normalized()
    frequency = np.asarray(design_result.frequency, dtype=float)
    manual_gain = manual_gain_eq_db(
        effective,
        frequency,
        apply_linear_fir_mask=bool(apply_linear_fir_mask),
    )
    manual_phase = manual_phase_eq_deg(
        effective,
        frequency,
        apply_linear_fir_mask=bool(apply_linear_fir_mask),
    )
    auto_off = replace(
        effective.auto_eq,
        gain_enabled=False,
        phase_enabled=False,
        sections=[replace(item, gain_enabled=False, phase_enabled=False) for item in effective.auto_eq.sections],
        gain_sections=[replace(item, enabled=False, gain_enabled=False) for item in effective.auto_eq.gain_sections],
        phase_sections=[replace(item, enabled=False, phase_enabled=False) for item in effective.auto_eq.phase_sections],
    )
    without_auto_gain, without_auto_phase = build_target_curves(
        replace(effective, auto_eq=auto_off),
        frequency,
    )
    with_auto_gain, with_auto_phase = build_target_curves(effective, frequency)
    auto_gain = with_auto_gain - without_auto_gain
    auto_phase = with_auto_phase - without_auto_phase
    return CorrectionResult(
        config=effective,
        frequency=frequency,
        iir_response=iir_frequency_response(effective.iir_filters, effective.sample_rate, frequency),
        manual_gain_db=manual_gain,
        manual_phase_deg=manual_phase,
        auto_gain_db=auto_gain,
        auto_phase_deg=auto_phase,
        design_result=design_result,
    )
