from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .config import IIRFilter, clamp_iir_q


@dataclass(frozen=True)
class AutoIIRResolutionPolicy:
    """Validated multi-resolution policy for Auto IIR.

    Candidate topology detection remains on the engine's 36 points/octave
    correction axis. Only the iterative evaluation axis is reduced here.
    """

    candidate_points_per_octave: float = 28.0
    candidate_minimum_points: int = 224
    candidate_maximum_points: int = 448
    validation_points_per_octave: float = 48.0
    validation_minimum_points: int = 384
    validation_maximum_points: int = 768
    fc_step_ratio: float = 0.005
    gain_step_db: float = 0.05
    q_step: float = 0.1
    twelfth_rms_tolerance_db: float = 0.01
    third_abs_tolerance_db: float = 0.01
    perceptual_peak_tolerance_db: float = 0.01
    positive_peak_area_tolerance_db_oct: float = 0.02


@dataclass(frozen=True)
class AutoIIRResolutionMetrics:
    twelfth_rms_db: float
    third_abs_db: float
    perceptual_peak_db: float
    positive_peak_area_db_oct: float
    headroom_db: float
    headroom_ok: bool
    filter_count: int
    shelf_kinds: tuple[str, ...]


@dataclass(frozen=True)
class AutoIIRResolutionValidation:
    passed: bool
    reasons: tuple[str, ...]
    degradation_twelfth_rms_db: float
    degradation_third_abs_db: float
    degradation_perceptual_peak_db: float
    degradation_positive_peak_area_db_oct: float


def quantize_auto_iir_filters(
    filters: Iterable[IIRFilter],
    *,
    f_min: float,
    f_max: float,
    policy: AutoIIRResolutionPolicy = AutoIIRResolutionPolicy(),
) -> list[IIRFilter]:
    """Quantize editable Auto IIR values without changing their topology."""

    lower = max(float(f_min), 1e-9)
    upper = max(float(f_max), lower)
    frequency_step = np.log1p(max(float(policy.fc_step_ratio), 1e-9))
    gain_step = max(float(policy.gain_step_db), 1e-9)
    q_step = max(float(policy.q_step), 1e-9)
    result: list[IIRFilter] = []
    for item in filters:
        ratio = max(float(item.fc), lower) / lower
        fc = lower * np.exp(round(np.log(ratio) / frequency_step) * frequency_step)
        q = clamp_iir_q(item.kind, round(float(item.q) / q_step) * q_step)
        gain_db = round(float(item.gain_db) / gain_step) * gain_step
        result.append(
            replace(
                item,
                fc=float(np.clip(fc, lower, upper)),
                q=float(q),
                gain_db=float(gain_db),
            )
        )
    return result


def validate_auto_iir_resolution_candidate(
    candidate: AutoIIRResolutionMetrics,
    reference: AutoIIRResolutionMetrics,
    *,
    policy: AutoIIRResolutionPolicy = AutoIIRResolutionPolicy(),
) -> AutoIIRResolutionValidation:
    """Validate final quantization on the full-resolution evaluation axis."""

    degradation = {
        "twelfth": float(candidate.twelfth_rms_db - reference.twelfth_rms_db),
        "third": float(candidate.third_abs_db - reference.third_abs_db),
        "peak": float(candidate.perceptual_peak_db - reference.perceptual_peak_db),
        "area": float(
            candidate.positive_peak_area_db_oct - reference.positive_peak_area_db_oct
        ),
    }
    values = (
        candidate.twelfth_rms_db,
        candidate.third_abs_db,
        candidate.perceptual_peak_db,
        candidate.positive_peak_area_db_oct,
        candidate.headroom_db,
    )
    reasons: list[str] = []
    if not all(np.isfinite(value) for value in values):
        reasons.append("non_finite_metric")
    if degradation["twelfth"] > float(policy.twelfth_rms_tolerance_db):
        reasons.append("twelfth_rms")
    if degradation["third"] > float(policy.third_abs_tolerance_db):
        reasons.append("third_abs")
    if degradation["peak"] > float(policy.perceptual_peak_tolerance_db):
        reasons.append("perceptual_peak")
    if degradation["area"] > float(policy.positive_peak_area_tolerance_db_oct):
        reasons.append("positive_peak_area")
    if not candidate.headroom_ok:
        reasons.append("headroom")
    if int(candidate.filter_count) != int(reference.filter_count):
        reasons.append("filter_count")
    if tuple(candidate.shelf_kinds) != tuple(reference.shelf_kinds):
        reasons.append("shelf_topology")
    return AutoIIRResolutionValidation(
        passed=not reasons,
        reasons=tuple(reasons),
        degradation_twelfth_rms_db=degradation["twelfth"],
        degradation_third_abs_db=degradation["third"],
        degradation_perceptual_peak_db=degradation["peak"],
        degradation_positive_peak_area_db_oct=degradation["area"],
    )
