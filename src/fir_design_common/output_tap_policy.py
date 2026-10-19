"""Shared range planning, real-coefficient crop, taper and timing policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import hashlib

import numpy as np

from .adaptive_crop import (AdaptiveCropResult, plan_adaptive_crop, execute_crop, crop_result, AdaptiveCropPlan, _metrics)
from fir_output_window import output_window
from .validation import FIRResponseValidation, FIRValidationError, validate_final_fir, validate_fixed_fir
from .fixed_frame import FixedFIRFramePlan, plan_fixed_fir_frame, execute_fixed_fir_frame
from .validation import validate_fixed_frame


COSINE_TAPER_KAISER_RATIO_NUMERATOR = 11
COSINE_TAPER_KAISER_RATIO_DENOMINATOR = 10
FixedWindowMode = Literal["protected", "direct"]


@dataclass(frozen=True)
class FinalCropRange:
    """Step ⑤ decision only: requested length is not the realized length."""

    start: int
    stop: int
    requested_taps: int
    edge_limited: bool

    @property
    def taps(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class FinalOutputTapResult:
    """Authoritative final coefficients and their absolute crop time origin."""

    coefficients: np.ndarray
    mode: Literal["fixed", "adaptive"]
    configured_taps: int
    input_taps: int
    final_taps: int
    adaptive: AdaptiveCropResult | None = None
    taper_reference_taps: int | None = None
    taper_frame_taps: int | None = None
    taper_source_limit_taps: int | None = None
    taper_padding_taps: int = 0
    crop_delay_samples: float = 0.0
    window_applied: bool = False
    validation: FIRResponseValidation | None = None
    crop_range: FinalCropRange | None = None
    window_mode: FixedWindowMode = "protected"
    frame_plan: FixedFIRFramePlan | None = None


def _validated_fir(values: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("FIR coefficients must be real")
    source = np.asarray(values, dtype=float)
    if source.ndim != 1 or source.size < 1 or not np.isfinite(source).all():
        raise ValueError("FIR coefficients must be a nonempty finite vector")
    return source.copy()


def decide_final_crop_range(start: int, stop: int, input_taps: int,
                            enabled: bool, *, require_odd: bool = True) -> FinalCropRange:
    """Decide length/range without reading, cropping or changing coefficients."""
    if not 0 <= start < stop <= input_taps:
        raise ValueError("invalid Adaptive range")
    if not enabled:
        return FinalCropRange(start, stop, stop-start, False)
    taps = stop - start
    if require_odd and taps % 2 == 0:
        raise FIRValidationError(
            "cosine taper requires an odd Adaptive support; no precision-safe odd "
            "range was found in the real input coefficients; output was not adopted")
    desired = (11 * taps + 9) // 10
    desired += 1 - desired % 2
    side = min((desired - taps + 1) // 2, start, input_taps - stop)
    return FinalCropRange(start-side, stop+side, desired,
                          stop-start+2*side < desired)


def _execute_final_plan(source, plan, decision, *, mode, configured_taps,
                         output_window_enabled, window_mode: FixedWindowMode = "protected"):
    """Steps ⑥–⑨ in one ordered executor; publish only after validation."""
    output = execute_crop(source, plan, start=decision.start, stop=decision.stop)
    window = output_window(output.size, output_window_enabled,
                           protected_center_taps=plan.taps if window_mode == "protected" else None)
    if output_window_enabled:
        output *= window
    if window_mode == "direct":
        validation = validate_fixed_fir(source, plan, decision.start, decision.stop, output,
                                        window=window)
    else:
        validation = validate_final_fir(source, plan, decision.start, decision.stop, output,
                                        window=window,
                                        enforce_precision=mode == "adaptive" and plan.adopted)
    crop = (crop_result(source, plan, decision.start, decision.stop, output,
                        window_enabled=output_window_enabled) if mode == "adaptive" else None)
    # Immutable bytes prevent a consumer from changing the authoritative FIR.
    output = np.frombuffer(np.asarray(output, dtype="<f8").tobytes(), dtype="<f8")
    if crop is not None:
        from dataclasses import replace
        crop = replace(crop, coefficients=output)
    return FinalOutputTapResult(
        output, mode, configured_taps, source.size, output.size, adaptive=crop,
        taper_reference_taps=plan.taps if output_window_enabled and window_mode == "protected" else None,
        taper_frame_taps=output.size if output_window_enabled and window_mode == "protected" else None,
        taper_source_limit_taps=source.size if output_window_enabled and window_mode == "protected" else None,
        crop_delay_samples=float(decision.start), window_applied=True,
        validation=validation, crop_range=decision, window_mode=window_mode,
    )


def finalize_adaptive_fir(coefficients: np.ndarray, *, minimum_taps: object = 1, maximum_taps: int | None = None,
                          linear_phase: bool | None = None,
                          output_window_enabled: bool = False) -> FinalOutputTapResult:
    """Plan, expand, execute once, taper once, validate; never pad or redesign."""
    source = _validated_fir(coefficients)
    plan = plan_adaptive_crop(source, minimum_taps=minimum_taps, maximum_taps=maximum_taps, linear_phase=linear_phase)
    decision = decide_final_crop_range(plan.start, plan.stop, source.size, output_window_enabled)
    return _execute_final_plan(source, plan, decision, mode="adaptive",
                               configured_taps=int(maximum_taps or source.size),
                               output_window_enabled=output_window_enabled)


def apply_final_output_tap_policy(
    coefficients: np.ndarray, output_taps: int, *, adaptive: bool = False,
    minimum_taps: object = 1,
    output_window_enabled: bool = False, kaiser_band_split_active: bool = False,
    fixed_window_mode: FixedWindowMode = "protected",
) -> FinalOutputTapResult:
    """Finalize a manual exact frame or the existing protected/Adaptive range."""
    source = _validated_fir(coefficients)
    configured = int(output_taps)
    if configured < 1:
        raise ValueError("resolved output taps must be positive")
    if fixed_window_mode not in ("protected", "direct"):
        raise ValueError("unknown fixed FIR window mode")
    if adaptive:
        return finalize_adaptive_fir(source, minimum_taps=minimum_taps, maximum_taps=configured,
                                     output_window_enabled=output_window_enabled)
    if fixed_window_mode == "direct":
        return _finalize_fixed_frame(source, configured, output_window_enabled)
    core_taps = min(source.size, configured)
    start = (source.size - core_taps) // 2
    stop = start + core_taps
    plan = AdaptiveCropPlan(source.size, start, stop, False, _metrics(source, source),
                            input_digest=hashlib.sha256(
                                np.asarray(source, dtype="<f8").tobytes()).hexdigest())
    decision = decide_final_crop_range(start, stop, source.size,
                                       output_window_enabled and fixed_window_mode == "protected",
                                       require_odd=False)
    return _execute_final_plan(source, plan, decision, mode="fixed", configured_taps=configured,
                               output_window_enabled=output_window_enabled, window_mode=fixed_window_mode)


def _finalize_fixed_frame(source, configured, window_enabled):
    plan = plan_fixed_fir_frame(source.size, configured)
    framed = execute_fixed_fir_frame(source, plan)
    window = output_window(configured, window_enabled)
    output = framed * window
    validation = validate_fixed_frame(source, plan, output, window=window)
    output = np.frombuffer(np.asarray(output, dtype="<f8").tobytes(), dtype="<f8")
    return FinalOutputTapResult(
        coefficients=output, mode="fixed", configured_taps=configured,
        input_taps=source.size, final_taps=configured,
        crop_delay_samples=float(plan.origin_samples), window_applied=True,
        validation=validation, window_mode="direct", frame_plan=plan,
        crop_range=FinalCropRange(plan.source_start, plan.source_stop, configured, False),
    )
