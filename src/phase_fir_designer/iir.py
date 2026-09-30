from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import io
import json
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy import signal
from scipy.optimize import least_squares

from .matched_allpass import design_hi_matched_allpass, design_lo_matched_allpass

from .auto_iir_objective import (
    AutoIIRObjectiveReference,
    AutoIIRObjectiveWeights,
    PositivePeakMetrics,
    auto_iir_objective_reference,
    evaluate_auto_iir_objective,
    fractional_octave_smooth,
    positive_peak_metrics,
)
from .auto_iir_optimizer import (
    OptimizationEvaluation,
    SearchParameter,
    SobolOptunaResult,
    SobolOptunaSettings,
    optimize_sobol_optuna,
)
from .auto_iir_perceptual import (
    AutoIIRPerceptualProfile,
    perceptual_level_on_axis,
    relative_audibility_weights,
    weighted_peak_error,
)
from .auto_iir_resolution import (
    AutoIIRResolutionMetrics,
    AutoIIRResolutionPolicy,
    quantize_auto_iir_filters,
    validate_auto_iir_resolution_candidate,
)
from .config import (
    IIR_PEQ_Q_MAX,
    IIR_SHELF_Q_MAX,
    IIRFilter,
    SpeakerResponse,
    clamp_iir_q,
    quantize_iir_q,
)


IIR_FILTER_TYPES = (
    "peq",
    "high_shelf",
    "low_shelf",
    "allpass",
    "high_pass",
    "low_pass",
)
IIR_CROSSOVER_FAMILIES = ("linkwitz_riley", "bessel", "butterworth", "variable_q", "elliptic")
IIR_CROSSOVER_ORDERS = (1, 2, 3, 4, 6, 8)
IIR_LINKWITZ_RILEY_ORDERS = (2, 4, 6, 8)
# Increment when AutoIIRDiagnostics gains fields that the Streamlit UI reads.
# phaseeq.py uses this to refresh a stale imported module after a hot reload.
AUTO_IIR_DIAGNOSTICS_SCHEMA_VERSION = 15
AUTO_IIR_FILTER_HEADROOM_DB = 12.0
# Finite bounds are required by Sobol and keep malformed imported responses
# numerically stable. This is an internal search bound, not a user correction
# limit; the audible safety condition is the combined +12 dB headroom below.
AUTO_IIR_NUMERIC_GAIN_BOUND_DB = 120.0


@dataclass(frozen=True)
class AutoIIRSettings:
    f_min: float = 20.0
    f_max: float = 20_000.0
    max_filters: int = 6
    include_peq: bool = True
    # Programmatic migration aid for callers created before ``max_filters``.
    # The UI and persisted settings use the total filter budget exclusively.
    max_peq: int | None = None
    # Deprecated compatibility inputs. Auto IIR no longer applies individual
    # Max Boost / Max Cut limits; combined filter headroom is used instead.
    max_boost_db: float | None = None
    max_cut_db: float | None = None
    filter_headroom_limit_db: float = AUTO_IIR_FILTER_HEADROOM_DB
    min_correction_db: float = 0.50
    include_low_shelf: bool = True
    include_high_shelf: bool = True
    max_peq_q: float = 5.0
    max_shelf_q: float = IIR_SHELF_Q_MAX
    sensitivity_weight: float = 0.18
    sensitivity_center_hz: float = 3_000.0
    sensitivity_width_oct: float = 1.5
    smoothness_weight: float = 0.10
    boost_penalty: float = 0.035
    cut_penalty: float = 0.010
    high_q_penalty: float = 0.025
    selection_regularization_scale: float = 0.25
    peq_fc_limit_oct: float = 0.50
    shelf_smoothing_oct: float = 1.0 / 3.0
    peq_smoothing_oct: float = 1.0 / 12.0
    shelf_min_area_improvement_db_oct: float = 0.15
    peq_min_area_improvement_db_oct: float = 0.04
    close_opposite_spacing_oct: float = 1.0 / 3.0
    close_opposite_strong_area_db_oct: float = 0.80
    reliable_f_min_hz: float | None = None
    reliable_f_max_hz: float | None = None
    independent_resolution_hz: float | None = None
    guard_rms_tolerance_db: float = 0.10
    guard_p95_tolerance_db: float = 0.20
    guard_area_tolerance_ratio: float = 0.005
    guard_area_tolerance_db_oct: float = 0.50
    candidate_shortlist_size: int = 4
    candidate_shortlist_max: int = 4
    candidate_shortlist_tie_ratio: float = 0.02
    # Relative-level priority only. This is deliberately not an SPL hearing
    # threshold because imported responses are commonly normalized.
    audibility_threshold_db: float = -40.0
    audibility_transition_db: float = 20.0
    audibility_min_weight: float = 0.20
    audibility_reference_percentile: float = 100.0
    target_mask_enabled: bool = True
    target_mask_limit_db: float = -40.0
    target_mask_smoothing_oct: float = 1.0 / 3.0
    wavelet_peak_smoothing_oct: float = 1.0 / 6.0
    wavelet_peak_context_oct: float = 1.0
    wavelet_peak_min_prominence_db: float = 0.50
    wavelet_peak_max_q: float = 5.0
    wavelet_peak_priority_weight: float = 0.50
    wavelet_peak_priority_max_factor: float = 1.50
    positive_peak_residual_db: float = 0.30
    positive_peak_max_reduction_ratio: float = 0.70
    positive_peak_level_weight: float = 0.65
    positive_peak_under_target_tradeoff: float = 1.25
    wavelet_response_agreement_min_db: float = 0.25
    # Deprecated compatibility input. Positive Auto PEQ gain is no longer
    # capped separately from the combined filter-headroom condition.
    broad_boost_max_db: float | None = None
    broad_boost_max_q: float = 1.20
    under_target_area_tolerance_db_oct: float = 0.05
    under_target_peak_tolerance_db: float = 0.25
    prune_objective_tolerance: float = 0.015
    selection_area_weight: float = 1.0
    selection_rms_weight: float = 0.30
    selection_peak_weight: float = 0.30
    objective_area_weight: float = 0.08
    objective_peak_weight: float = 0.10
    optimizer_mode: str = "sobol_optuna"
    optimizer_sobol_trials: int = 32
    optimizer_optuna_trials: int = 64
    optimizer_seed: int = 0
    optimizer_timeout_seconds: float | None = 2.0
    optimizer_top_k: int = 3
    objective_one_octave_abs_weight: float = 0.60
    objective_third_octave_abs_weight: float = 0.40
    objective_perceptual_peak_weight: float = 0.45
    objective_positive_peak_area_weight: float = 0.25
    objective_center_shift_weight: float = 0.10
    adaptive_resolution_enabled: bool = False
    evaluation_points_per_octave: float = 48.0
    evaluation_minimum_points: int = 384
    evaluation_maximum_points: int = 768


@dataclass(frozen=True)
class AutoIIRDiagnostics:
    requested_f_min_hz: float
    requested_f_max_hz: float
    correction_f_min_hz: float
    correction_f_max_hz: float
    reliable_f_min_hz: float
    reliable_f_max_hz: float
    evaluation_f_min_hz: float
    evaluation_f_max_hz: float
    independent_resolution_hz: float
    before_weighted_rms_db: float
    after_weighted_rms_db: float
    before_raw_weighted_rms_db: float
    after_raw_weighted_rms_db: float
    before_error_area_db_oct: float
    after_error_area_db_oct: float
    before_outside_error_area_db_oct: float
    after_outside_error_area_db_oct: float
    before_p95_abs_db: float
    after_p95_abs_db: float
    before_p99_abs_db: float
    after_p99_abs_db: float
    before_max_abs_db: float
    after_max_abs_db: float
    before_one_octave_abs_db: float
    after_one_octave_abs_db: float
    before_third_octave_abs_db: float
    after_third_octave_abs_db: float
    before_contour_abs_db: float
    after_contour_abs_db: float
    before_perceptual_peak_db: float
    after_perceptual_peak_db: float
    before_positive_peak_area_db_oct: float
    after_positive_peak_area_db_oct: float
    before_under_target_area_db_oct: float
    after_under_target_area_db_oct: float
    before_under_target_peak_db: float
    after_under_target_peak_db: float
    audibility_threshold_db: float
    audibility_min_weight: float
    perceptual_profile_source: str
    max_peq_q: float
    wavelet_peak_smoothing_oct: float
    optimizer_mode: str
    optimizer_seed: int
    optimizer_sobol_trials: int
    optimizer_optuna_trials: int
    optimizer_completed_trials: int
    optimizer_feasible_trials: int
    optimizer_best_score: float
    optimizer_center_shift_oct: float
    optimizer_final_candidate_count: int
    optimizer_selected_source: str
    filter_headroom_db: float
    filter_headroom_limit_db: float
    filter_headroom_ok: bool
    total_filter_budget: int
    generated_filter_count: int
    peq_count: int
    shelf_count: int
    low_shelf_allowed: bool
    high_shelf_allowed: bool
    low_shelf_selected: bool
    high_shelf_selected: bool
    shelf_topology_candidate_count: int
    optimization_seconds: float
    stop_reason: str
    resolution_mode: str = "full_48"
    resolution_candidate_points_per_octave: float = 48.0
    resolution_validation_points_per_octave: float = 48.0
    resolution_guard_passed: bool = True
    resolution_fallback_used: bool = False
    resolution_guard_reasons: tuple[str, ...] = ()
    resolution_candidate_seconds: float = 0.0


@dataclass(frozen=True)
class AutoIIRResult:
    filters: list[IIRFilter]
    diagnostics: AutoIIRDiagnostics


@dataclass(frozen=True)
class IIRStageResponse:
    """One coherent before/filter/after/target response set for IIR result views."""

    frequency: np.ndarray
    before_complex: np.ndarray
    filter_complex: np.ndarray
    after_complex: np.ndarray
    target_complex: np.ndarray


@dataclass(frozen=True)
class _AutoIIRCandidate:
    item: IIRFilter
    prominence_db: float
    priority: float


@dataclass(frozen=True)
class _AutoIIRFinalCandidate:
    source: str
    filters: list[IIRFilter]
    score: float
    one_octave_abs_db: float
    third_octave_abs_db: float
    contour_abs_db: float
    perceptual_peak_db: float
    positive_peak_area_db_oct: float
    under_target_area_db_oct: float
    under_target_peak_db: float
    center_shift_oct: float
    error_area_db_oct: float
    weighted_rms_db: float
    p95_abs_db: float


def _clip_cutoff(fc: float, sample_rate: int) -> float:
    return min(max(float(fc), 1e-3), float(sample_rate) / 2.0 - 1e-3)


def _normalize_section(coefficients: Iterable[float]) -> np.ndarray:
    b0, b1, b2, a0, a1, a2 = (float(value) for value in coefficients)
    if not np.isfinite(a0) or abs(a0) < 1e-15:
        raise ValueError("IIR coefficient a0 must be finite and non-zero")
    section = np.asarray([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])
    if not np.all(np.isfinite(section)):
        raise ValueError("IIR coefficients must be finite")
    return section


def _rbj_section(
    kind: str,
    fc: float,
    q: float,
    gain_db: float,
    sample_rate: int,
) -> np.ndarray:
    fc = _clip_cutoff(fc, sample_rate)
    q = max(float(q), 1e-4)
    gain_db = float(gain_db)
    omega = 2.0 * np.pi * fc / float(sample_rate)
    cosine = np.cos(omega)
    sine = np.sin(omega)
    alpha = sine / (2.0 * q)
    amplitude = 10.0 ** (gain_db / 40.0)

    if kind in {"low_pass", "high_pass"}:
        sign = 1.0 if kind == "low_pass" else -1.0
        numerator = (1.0 - sign * cosine) / 2.0
        coefficients = (numerator, sign * 2.0 * numerator, numerator,
                        1.0 + alpha, -2.0 * cosine, 1.0 - alpha)
    elif kind == "peq":
        coefficients = (
            1.0 + alpha * amplitude,
            -2.0 * cosine,
            1.0 - alpha * amplitude,
            1.0 + alpha / amplitude,
            -2.0 * cosine,
            1.0 - alpha / amplitude,
        )
    elif kind == "allpass":
        coefficients = (
            1.0 - alpha,
            -2.0 * cosine,
            1.0 + alpha,
            1.0 + alpha,
            -2.0 * cosine,
            1.0 - alpha,
        )
    elif kind == "low_shelf":
        root_a = np.sqrt(amplitude)
        coefficients = (
            amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine + 2.0 * root_a * alpha),
            2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
            amplitude * ((amplitude + 1.0) - (amplitude - 1.0) * cosine - 2.0 * root_a * alpha),
            (amplitude + 1.0) + (amplitude - 1.0) * cosine + 2.0 * root_a * alpha,
            -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
            (amplitude + 1.0) + (amplitude - 1.0) * cosine - 2.0 * root_a * alpha,
        )
    elif kind == "high_shelf":
        root_a = np.sqrt(amplitude)
        coefficients = (
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + 2.0 * root_a * alpha),
            -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - 2.0 * root_a * alpha),
            (amplitude + 1.0) - (amplitude - 1.0) * cosine + 2.0 * root_a * alpha,
            2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
            (amplitude + 1.0) - (amplitude - 1.0) * cosine - 2.0 * root_a * alpha,
        )
    else:
        raise ValueError(f"unsupported RBJ IIR filter type: {kind}")
    return _normalize_section(coefficients)


def _rbj_sos(item: IIRFilter, sample_rate: int) -> np.ndarray:
    return _rbj_section(
        item.kind,
        item.fc,
        item.q,
        item.gain_db,
        sample_rate,
    )[None, :]


def _first_order_allpass_sos(fc: float, sample_rate: int) -> np.ndarray:
    """Return a prewarped first-order all-pass with -90 degrees at ``fc``."""
    fc = _clip_cutoff(fc, sample_rate)
    tangent = np.tan(np.pi * fc / float(sample_rate))
    coefficient = (tangent - 1.0) / (tangent + 1.0)
    return np.asarray(
        [[coefficient, 1.0, 0.0, 1.0, coefficient, 0.0]],
        dtype=float,
    )


def design_iir_sos(item: IIRFilter, sample_rate: int) -> np.ndarray:
    """Return SciPy SOS rows using H(z)=B(z)/(1+a1*z^-1+a2*z^-2)."""
    if not item.enabled:
        return np.empty((0, 6), dtype=float)
    if item.kind == "allpass" and item.allpass_preset != "manual":
        preset = str(item.allpass_preset)
        if preset == "lr2_hi":
            return design_hi_matched_allpass(sample_rate, item.fc, 2).sos
        order = {"lr2_lo": 2, "lr4": 4, "lr8": 8}.get(preset)
        if order is None:
            raise ValueError(f"unsupported IIR All Pass preset: {preset}")
        return design_lo_matched_allpass(sample_rate, item.fc, order).sos
    if item.kind == "allpass" and int(item.order) == 1:
        return _first_order_allpass_sos(item.fc, sample_rate)
    if item.kind == "allpass" and int(item.order) != 2:
        raise ValueError("manual IIR All Pass order must be 1 or 2")
    if item.kind in {"peq", "high_shelf", "low_shelf", "allpass"}:
        return _rbj_sos(item, sample_rate)
    if item.kind not in {"high_pass", "low_pass"}:
        raise ValueError(f"unsupported IIR filter type: {item.kind}")

    btype = "highpass" if item.kind == "high_pass" else "lowpass"
    family = str(item.family)
    order = int(item.order)
    fc = _clip_cutoff(item.fc, sample_rate)
    if family == "variable_q":
        if item.order != 2:
            raise ValueError("Variable-Q requires order 2")
        return _rbj_sos(item, sample_rate)
    if family == "elliptic":
        if item.order not in range(1, 9):
            raise ValueError("Elliptic order must be 1..8")
        return np.asarray(signal.ellip(order, item.ripple_db, item.stop_db, fc,
                                      btype=btype, fs=sample_rate, output="sos"), dtype=float)
    if family == "linkwitz_riley":
        if order not in IIR_LINKWITZ_RILEY_ORDERS:
            raise ValueError("Linkwitz-Riley order must be 2, 4, 6, or 8")
        half_order = order // 2
        base = signal.butter(half_order, fc, btype=btype, fs=sample_rate, output="sos")
        return np.vstack([base, base]).astype(float)
    if order not in IIR_CROSSOVER_ORDERS:
        raise ValueError("Butterworth/Bessel order must be 1, 2, 3, 4, 6, or 8")
    if family == "butterworth":
        return np.asarray(signal.butter(order, fc, btype=btype, fs=sample_rate, output="sos"), dtype=float)
    if family == "bessel":
        return np.asarray(
            signal.bessel(order, fc, btype=btype, norm="phase", fs=sample_rate, output="sos"),
            dtype=float,
        )
    raise ValueError(f"unsupported IIR crossover family: {family}")


def cascade_iir_sos(filters: Iterable[IIRFilter], sample_rate: int) -> np.ndarray:
    sections = [design_iir_sos(item, sample_rate) for item in filters if item.enabled]
    sections = [item for item in sections if item.size]
    if not sections:
        return np.empty((0, 6), dtype=float)
    result = np.vstack(sections).astype(float)
    validate_iir_sos(result)
    return result


def validate_iir_sos(sos: np.ndarray) -> None:
    sections = np.asarray(sos, dtype=float)
    if sections.size == 0:
        return
    if sections.ndim != 2 or sections.shape[1] != 6:
        raise ValueError("IIR SOS must have shape (sections, 6)")
    for index, section in enumerate(sections):
        poles = np.roots([section[3], section[4], section[5]])
        if np.any(np.abs(poles) >= 1.0 - 1e-10):
            raise ValueError(f"IIR section {index + 1} is unstable")


def iir_frequency_response(
    filters: Iterable[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=float)
    sos = cascade_iir_sos(filters, sample_rate)
    if sos.size == 0:
        return np.ones_like(frequency, dtype=complex)
    clipped = np.clip(frequency, 0.0, float(sample_rate) / 2.0)
    _frequency, response = signal.sosfreqz(sos, worN=clipped, fs=sample_rate)
    return np.asarray(response, dtype=complex)


def iir_stage_response(
    frequency: np.ndarray,
    before_gain_db: np.ndarray,
    before_phase_deg: np.ndarray,
    target_gain_db: np.ndarray,
    target_phase_deg: np.ndarray,
    filters: Iterable[IIRFilter],
    sample_rate: int,
) -> IIRStageResponse:
    """Build every IIR EQ graph from the same complex-response relationship."""

    frequency = np.asarray(frequency, dtype=float)
    arrays = [
        np.asarray(before_gain_db, dtype=float),
        np.asarray(before_phase_deg, dtype=float),
        np.asarray(target_gain_db, dtype=float),
        np.asarray(target_phase_deg, dtype=float),
    ]
    if any(values.shape != frequency.shape for values in arrays):
        raise ValueError("IIR stage response arrays must match the frequency axis")

    before_complex = 10.0 ** (arrays[0] / 20.0) * np.exp(1j * np.deg2rad(arrays[1]))
    target_complex = 10.0 ** (arrays[2] / 20.0) * np.exp(1j * np.deg2rad(arrays[3]))
    filter_complex = iir_frequency_response(filters, sample_rate, frequency)
    return IIRStageResponse(
        frequency=frequency,
        before_complex=before_complex,
        filter_complex=filter_complex,
        after_complex=before_complex * filter_complex,
        target_complex=target_complex,
    )


def apply_iir_to_speaker_response(
    response: SpeakerResponse | None,
    filters: Iterable[IIRFilter],
    sample_rate: int,
) -> SpeakerResponse | None:
    if response is None:
        return None
    enabled = [item for item in filters if item.enabled]
    if not enabled:
        return response
    frequency = np.asarray(response.frequency, dtype=float)
    gain = np.asarray(response.gain_db, dtype=float)
    transfer = iir_frequency_response(enabled, sample_rate, frequency)
    result_gain = gain + 20.0 * np.log10(np.maximum(np.abs(transfer), 1e-12))
    result_phase: list[float] | None = None
    if response.phase_deg is not None:
        source_phase = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(response.phase_deg, dtype=float))))
        filter_phase = np.rad2deg(np.unwrap(np.angle(transfer)))
        result_phase = (source_phase + filter_phase).tolist()
    return SpeakerResponse(
        frequency=frequency.tolist(),
        gain_db=result_gain.tolist(),
        phase_deg=result_phase,
    )


def _response_gain_on_log_axis(response: SpeakerResponse, frequency: np.ndarray) -> np.ndarray:
    source_frequency = np.asarray(response.frequency, dtype=float)
    source_gain = np.asarray(response.gain_db, dtype=float)
    valid = np.isfinite(source_frequency) & np.isfinite(source_gain) & (source_frequency > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Auto IIR requires at least two finite response points")
    order = np.argsort(source_frequency[valid])
    source_frequency = source_frequency[valid][order]
    source_gain = source_gain[valid][order]
    source_frequency, unique = np.unique(source_frequency, return_index=True)
    source_gain = source_gain[unique]
    requested = np.asarray(frequency, dtype=float)
    tolerance = max(float(source_frequency[-1]), 1.0) * 1e-10
    if (
        float(np.min(requested)) < float(source_frequency[0]) - tolerance
        or float(np.max(requested)) > float(source_frequency[-1]) + tolerance
    ):
        raise ValueError("Auto IIR evaluation extends outside the available response data")
    return np.interp(
        np.log10(np.maximum(requested, 1e-9)),
        np.log10(source_frequency),
        source_gain,
    )


def _response_supported_frequency_range(response: SpeakerResponse) -> tuple[float, float]:
    frequency = np.asarray(response.frequency, dtype=float)
    gain = np.asarray(response.gain_db, dtype=float)
    valid = np.isfinite(frequency) & np.isfinite(gain) & (frequency > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError("Auto IIR requires at least two finite response points")
    return float(np.min(frequency[valid])), float(np.max(frequency[valid]))


def _estimated_independent_resolution_hz(response: SpeakerResponse) -> float:
    frequency = np.asarray(response.frequency, dtype=float)
    frequency = np.unique(np.sort(frequency[np.isfinite(frequency) & (frequency > 0.0)]))
    if frequency.size < 4:
        return 0.0
    low_limit = min(float(frequency[-1]), max(200.0, float(frequency[0]) * 64.0))
    low = frequency[frequency <= low_limit]
    if low.size < 4:
        low = frequency[: min(frequency.size, 64)]
    spacing = np.diff(low)
    spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    if spacing.size < 3:
        return 0.0
    median = float(np.median(spacing))
    # Only a nearly uniform axis carries an independent FFT-bin resolution.
    # Log-spaced FRD rows are display samples, not proof of the measurement window.
    if float(np.percentile(np.abs(spacing - median), 90.0)) > median * 0.08:
        return 0.0
    return median


def _default_reliable_f_min_hz(response: SpeakerResponse) -> float:
    source_min, _source_max = _response_supported_frequency_range(response)
    resolution = _estimated_independent_resolution_hz(response)
    if resolution > 0.0 and source_min <= max(5.0, resolution * 2.1):
        # Linear FFT spectra often contain numeric DC-adjacent bins even when
        # the acoustic measurement has no useful information there.
        return max(20.0, source_min)
    return source_min


def _auto_reliable_range(
    speaker: SpeakerResponse,
    target: SpeakerResponse,
    sample_rate: int,
    settings: AutoIIRSettings,
) -> tuple[float, float]:
    speaker_min, speaker_max = _response_supported_frequency_range(speaker)
    target_min, target_max = _response_supported_frequency_range(target)
    reliable_min = (
        float(settings.reliable_f_min_hz)
        if settings.reliable_f_min_hz is not None
        else _default_reliable_f_min_hz(speaker)
    )
    reliable_max = (
        float(settings.reliable_f_max_hz)
        if settings.reliable_f_max_hz is not None
        else speaker_max
    )
    supported_min = max(speaker_min, target_min, reliable_min, 1.0)
    supported_max = min(
        speaker_max,
        target_max,
        reliable_max,
        float(sample_rate) / 2.0 - 1.0,
    )
    if supported_max <= supported_min:
        raise ValueError("Auto IIR has no reliable common Input/Target frequency band")
    return supported_min, supported_max


def _auto_gain_band_crossing_hz(
    frequency_a: float,
    frequency_b: float,
    gain_a: float,
    gain_b: float,
    threshold_db: float,
) -> float:
    if not np.isfinite(gain_a) or not np.isfinite(gain_b) or np.isclose(gain_a, gain_b):
        return float(frequency_b)
    position = float(np.clip((threshold_db - gain_a) / (gain_b - gain_a), 0.0, 1.0))
    return float(
        np.exp(
            np.log(max(float(frequency_a), 1e-9))
            + position
            * (
                np.log(max(float(frequency_b), 1e-9))
                - np.log(max(float(frequency_a), 1e-9))
            )
        )
    )


def _auto_gain_band_range(
    frequency: np.ndarray,
    gain_db: np.ndarray,
    *,
    f_min: float,
    f_max: float,
    limit_db: float = -40.0,
    smoothing_oct: float = 1.0 / 3.0,
) -> tuple[float, float, float]:
    """Return the outer audible band of the current response.

    The first and last 1/3-octave-smoothed points above peak + limit define
    the band. Internal notches do not split it, so one narrow null cannot hide
    otherwise usable correction frequencies.
    """

    axis = np.asarray(frequency, dtype=float)
    gain = np.asarray(gain_db, dtype=float)
    if axis.ndim != 1 or gain.shape != axis.shape or axis.size < 2:
        raise ValueError("Auto IIR Gain band requires matching frequency and Gain arrays")
    requested = (
        np.isfinite(axis)
        & np.isfinite(gain)
        & (axis >= float(f_min) * (1.0 - 1e-10))
        & (axis <= float(f_max) * (1.0 + 1e-10))
    )
    indices = np.flatnonzero(requested)
    if indices.size < 2:
        raise ValueError("Auto IIR Gain band has fewer than two usable points")
    smoothed = fractional_octave_smooth(
        gain,
        axis,
        max(float(smoothing_oct), 1.0 / 48.0),
    )
    reference_db = float(np.max(smoothed[indices]))
    threshold_db = reference_db + min(float(limit_db), 0.0)
    audible = indices[smoothed[indices] >= threshold_db]
    if audible.size == 0:
        raise ValueError("Auto IIR has no Gain band above the configured level limit")
    first = int(audible[0])
    last = int(audible[-1])
    lower = float(f_min)
    upper = float(f_max)
    lower_gain = float(
        np.interp(np.log(float(f_min)), np.log(axis), smoothed)
    )
    upper_gain = float(
        np.interp(np.log(float(f_max)), np.log(axis), smoothed)
    )
    if lower_gain < threshold_db and first > 0:
        lower = _auto_gain_band_crossing_hz(
            axis[first - 1],
            axis[first],
            smoothed[first - 1],
            smoothed[first],
            threshold_db,
        )
    if upper_gain < threshold_db and last < axis.size - 1:
        upper = _auto_gain_band_crossing_hz(
            axis[last],
            axis[last + 1],
            smoothed[last],
            smoothed[last + 1],
            threshold_db,
        )
    return max(float(f_min), lower), min(float(f_max), upper), reference_db


def _adaptive_log_axis(f_min: float, f_max: float, *, points_per_octave: float, minimum: int, maximum: int) -> np.ndarray:
    octave_span = max(float(np.log2(f_max / f_min)), 1e-9)
    points = int(np.clip(np.ceil(octave_span * points_per_octave) + 1, minimum, maximum))
    return np.geomspace(float(f_min), float(f_max), points)


def _auto_frequency_kernel(
    sample_rate: int,
    frequency: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the frequency terms shared by every Auto IIR trial."""

    clipped = np.clip(
        np.asarray(frequency, dtype=float),
        0.0,
        float(sample_rate) / 2.0,
    )
    z1 = np.exp(-2j * np.pi * clipped / float(sample_rate))
    return z1, z1 * z1


def _auto_filter_response(
    filters: list[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
    *,
    frequency_kernel: tuple[np.ndarray, np.ndarray] | None = None,
    initial_response: np.ndarray | None = None,
) -> np.ndarray:
    # Auto IIR evaluates the same small SOS cascade thousands of times at
    # explicitly supplied frequencies. Evaluating the transfer polynomial
    # directly avoids scipy.signal.sosfreqz setup overhead while preserving the
    # public response path used by graphs and exports.
    frequency = np.asarray(frequency, dtype=float)
    sections = [design_iir_sos(item, sample_rate) for item in filters if item.enabled]
    sections = [item for item in sections if item.size]
    z1, z2 = (
        _auto_frequency_kernel(sample_rate, frequency)
        if frequency_kernel is None
        else frequency_kernel
    )
    response = (
        np.ones_like(frequency, dtype=complex)
        if initial_response is None
        else np.asarray(initial_response, dtype=complex).copy()
    )
    for section in sections:
        for b0, b1, b2, a0, a1, a2 in section:
            response *= (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return response


def _auto_filter_gain(
    filters: list[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
    *,
    frequency_kernel: tuple[np.ndarray, np.ndarray] | None = None,
    initial_response: np.ndarray | None = None,
) -> np.ndarray:
    response = _auto_filter_response(
        filters,
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
        initial_response=initial_response,
    )
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))


def _auto_filter_response_from_values(
    filters: list[IIRFilter],
    values: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    *,
    frequency_kernel: tuple[np.ndarray, np.ndarray],
    fixed_q: bool = False,
    initial_response: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate optimizer parameters without rebuilding temporary dataclasses."""

    z1, z2 = frequency_kernel
    response = (
        np.ones_like(frequency, dtype=complex)
        if initial_response is None
        else np.asarray(initial_response, dtype=complex).copy()
    )
    stride = 2 if fixed_q else 3
    for index, item in enumerate(filters):
        if not item.enabled:
            continue
        offset = index * stride
        fc = float(np.exp(values[offset]))
        q = float(item.q) if fixed_q else float(np.exp(values[offset + 1]))
        gain_db = float(values[offset + 1] if fixed_q else values[offset + 2])
        b0, b1, b2, _a0, a1, a2 = _rbj_section(
            item.kind,
            fc,
            q,
            gain_db,
            sample_rate,
        )
        response *= (b0 + b1 * z1 + b2 * z2) / (1.0 + a1 * z1 + a2 * z2)
    return response


def _auto_iir_filter_headroom_db(
    filters: Iterable[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
    *,
    frequency_kernel: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Return the maximum positive gain of one combined IIR cascade."""
    enabled = [item for item in filters if item.enabled]
    if not enabled:
        return 0.0
    gain = _auto_filter_gain(
        enabled,
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
    )
    return max(float(np.max(gain)), 0.0)


def _auto_iir_headroom_is_safe(
    filters: Iterable[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
) -> bool:
    return bool(
        _auto_iir_filter_headroom_db(filters, sample_rate, frequency)
        <= float(settings.filter_headroom_limit_db) + 1e-6
    )


def _enforce_auto_iir_filter_headroom(
    generated: list[IIRFilter],
    fixed: Iterable[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    frequency_kernel: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[IIRFilter]:
    """Scale only generated boosts until the combined cascade is within +12 dB.

    Cut gain is intentionally unrestricted. If fixed/manual filters alone
    exceed the limit, generated boosts are removed but fixed filters are never
    rewritten by Auto IIR.
    """
    result = list(generated)
    fixed_filters = [item for item in fixed if item.enabled]
    limit = max(float(settings.filter_headroom_limit_db), 0.0)
    if _auto_iir_filter_headroom_db(
        [*fixed_filters, *result],
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
    ) <= limit + 1e-6:
        return result
    positive = [index for index, item in enumerate(result) if item.gain_db > 0.0]
    if not positive:
        return result

    def scaled(scale: float) -> list[IIRFilter]:
        return [
            replace(item, gain_db=float(item.gain_db) * scale)
            if index in positive
            else item
            for index, item in enumerate(result)
        ]

    zeroed = scaled(0.0)
    if _auto_iir_filter_headroom_db(
        [*fixed_filters, *zeroed],
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
    ) > limit + 1e-6:
        return [item for item in zeroed if abs(float(item.gain_db)) >= 0.01]

    low = 0.0
    high = 1.0
    for _ in range(48):
        middle = 0.5 * (low + high)
        trial = scaled(middle)
        if _auto_iir_filter_headroom_db(
            [*fixed_filters, *trial],
            sample_rate,
            frequency,
            frequency_kernel=frequency_kernel,
        ) <= limit:
            low = middle
        else:
            high = middle
    limited = scaled(low)
    return [item for item in limited if abs(float(item.gain_db)) >= 0.01]


def _auto_iir_gain_search_bounds(item: IIRFilter, settings: AutoIIRSettings) -> tuple[float, float]:
    """Return sign-preserving numerical bounds without product Gain limits."""
    numeric = float(AUTO_IIR_NUMERIC_GAIN_BOUND_DB)
    near_zero = max(float(settings.min_correction_db) * 0.10, 0.01)
    if item.gain_db < 0.0:
        # A PEQ cut is initialized by the positive-peak balance logic. Do not
        # deepen it during the numerical refit, otherwise one audible hill can
        # split into multiple cuts plus compensating boosts. This is not an
        # absolute Max Cut; the data-derived initial cut may be arbitrarily
        # deeper than the removed UI limit.
        lower = max(-numeric, float(item.gain_db)) if item.kind == "peq" else -numeric
        return lower, -near_zero
    return near_zero, numeric


def _estimate_peak_q(frequency: np.ndarray, residual: np.ndarray, peak_index: int) -> float:
    peak = abs(float(residual[peak_index]))
    if peak <= 1e-9:
        return 1.0
    threshold = peak * 0.5
    left = peak_index
    right = peak_index
    while left > 0 and abs(float(residual[left])) >= threshold:
        left -= 1
    while right < residual.size - 1 and abs(float(residual[right])) >= threshold:
        right += 1
    width_oct = max(np.log2(frequency[right] / frequency[left]), 1.0 / 24.0)
    return float(np.clip(1.0 / width_oct, 0.2, 12.0))


def _auto_sensitivity(frequency: np.ndarray, settings: AutoIIRSettings) -> np.ndarray:
    width = max(float(settings.sensitivity_width_oct), 0.1)
    distance_oct = np.log2(np.maximum(frequency, 1e-9) / max(float(settings.sensitivity_center_hz), 1.0))
    return np.exp(-0.5 * np.square(distance_oct / width))


def _auto_frequency_weights(
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    perceptual_level_db: np.ndarray | None = None,
) -> np.ndarray:
    # This is intentionally mild. It is not A-weighting and must not suppress
    # the low/high bands that are part of the requested correction range.
    weights = 1.0 + max(float(settings.sensitivity_weight), 0.0) * _auto_sensitivity(frequency, settings)
    if perceptual_level_db is not None:
        level = np.asarray(perceptual_level_db, dtype=float)
        if level.shape != np.asarray(frequency).shape:
            raise ValueError("Auto IIR perceptual level must match the frequency axis")
        weights = weights * relative_audibility_weights(
            level,
            threshold_db=float(settings.audibility_threshold_db),
            transition_db=float(settings.audibility_transition_db),
            minimum_weight=float(settings.audibility_min_weight),
            reference_percentile=float(settings.audibility_reference_percentile),
        )
    return weights


def _gaussian_smooth(values: np.ndarray, sigma_points: float) -> np.ndarray:
    sigma = max(float(sigma_points), 0.25)
    radius = max(2, int(np.ceil(3.0 * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * np.square(axis / sigma))
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(values, dtype=float), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _auto_smoothed_error(error: np.ndarray, frequency: np.ndarray, smoothing_oct: float) -> np.ndarray:
    octave_span = max(float(np.log2(frequency[-1] / frequency[0])), 1e-9)
    points_per_octave = max((frequency.size - 1) / octave_span, 1.0)
    sigma_points = max(float(smoothing_oct) * points_per_octave / 2.355, 0.25)
    return _gaussian_smooth(error, sigma_points)


def _auto_weighted_rms(error: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(error)) / max(float(np.sum(weights)), 1e-12)))


def _auto_model_score(
    error: np.ndarray,
    filters: list[IIRFilter],
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    weights: np.ndarray | None = None,
) -> float:
    if weights is None:
        weights = _auto_frequency_weights(frequency, settings)
    rms = _auto_weighted_rms(error, weights)
    mean_error = float(np.sum(weights * error) / max(float(np.sum(weights)), 1e-12))
    smooth_error = _auto_smoothed_error(error, frequency, 1.0 / 12.0)
    curvature = np.diff(smooth_error, n=2)
    sensitivity = _auto_sensitivity(frequency, settings)[1:-1]
    smoothness = (
        float(np.sqrt(np.mean(np.square(curvature) * (1.0 + sensitivity))))
        if curvature.size
        else 0.0
    )
    boost_cost = sum(max(float(item.gain_db), 0.0) for item in filters)
    cut_cost = sum(max(-float(item.gain_db), 0.0) for item in filters)
    high_q_cost = sum(max(np.log2(max(float(item.q), 1e-9) / 4.0), 0.0) for item in filters)
    octave_span = max(float(np.log2(frequency[-1] / frequency[0])), 1e-9)
    area_mean = _absolute_error_area(error, frequency, weights) / octave_span
    peak = weighted_peak_error(error, weights)
    return float(
        rms
        + 0.35 * abs(mean_error)
        + float(settings.objective_area_weight) * area_mean
        + float(settings.objective_peak_weight) * peak
        + float(settings.smoothness_weight) * smoothness
        + float(settings.selection_regularization_scale)
        * (
            float(settings.boost_penalty) * boost_cost
            + float(settings.cut_penalty) * cut_cost
            + float(settings.high_q_penalty) * high_q_cost
        )
    )


def _auto_error(
    speaker_gain: np.ndarray,
    target_gain: np.ndarray,
    manual: list[IIRFilter],
    generated: list[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
) -> np.ndarray:
    return target_gain - speaker_gain - _auto_filter_gain(manual + generated, sample_rate, frequency)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    midpoint = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), midpoint, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _absolute_error_area(
    error: np.ndarray,
    frequency: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted absolute error integrated over octaves (dB-octave)."""
    return float(np.trapezoid(np.abs(error) * weights, x=np.log2(frequency)))


def _masked_absolute_error_area(
    error: np.ndarray,
    frequency: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Integrate selected contiguous bands without bridging excluded gaps."""
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size < 2:
        return 0.0
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    total = 0.0
    for part in np.split(indices, split_points):
        if part.size < 2:
            continue
        total += _absolute_error_area(error[part], frequency[part], weights[part])
    return float(total)


def _masked_weighted_rms(error: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return 0.0
    return _auto_weighted_rms(np.asarray(error)[selected], np.asarray(weights)[selected])


def _guard_is_safe(
    before_error: np.ndarray,
    after_error: np.ndarray,
    frequency: np.ndarray,
    weights: np.ndarray,
    guard_mask: np.ndarray,
    settings: AutoIIRSettings,
) -> bool:
    guard = np.asarray(guard_mask, dtype=bool)
    if np.count_nonzero(guard) < 2:
        return True
    before_area = _masked_absolute_error_area(before_error, frequency, weights, guard)
    after_area = _masked_absolute_error_area(after_error, frequency, weights, guard)
    area_tolerance = max(
        float(settings.guard_area_tolerance_db_oct),
        before_area * float(settings.guard_area_tolerance_ratio),
    )
    before_rms = _masked_weighted_rms(before_error, weights, guard)
    after_rms = _masked_weighted_rms(after_error, weights, guard)
    before_p95 = float(np.percentile(np.abs(np.asarray(before_error)[guard]), 95.0))
    after_p95 = float(np.percentile(np.abs(np.asarray(after_error)[guard]), 95.0))
    return bool(
        after_area <= before_area + area_tolerance
        and after_rms <= before_rms + float(settings.guard_rms_tolerance_db)
        and after_p95 <= before_p95 + float(settings.guard_p95_tolerance_db)
    )


def _peak_cut_balance_is_safe(
    before: PositivePeakMetrics,
    after: PositivePeakMetrics,
    before_under_peak_db: float,
    after_under_peak_db: float,
    settings: AutoIIRSettings,
) -> bool:
    """Allow useful Peak Cut without letting it create a new audible dip.

    A fixed under-Target area allowance prevents almost every meaningful cut
    when the source already contains a deep cancellation dip.  Tie the area
    allowance to the positive-peak area actually removed, while retaining the
    strict 1/3-octave maximum-dip guard.
    """

    before_peak_area = float(before.area_db_oct)
    after_peak_area = float(after.area_db_oct)
    peak_area_improvement = max(before_peak_area - after_peak_area, 0.0)
    allowed_under_area = (
        float(settings.under_target_area_tolerance_db_oct)
        + float(settings.positive_peak_under_target_tradeoff) * peak_area_improvement
    )
    return bool(
        float(after.under_target_area_db_oct)
        <= float(before.under_target_area_db_oct) + allowed_under_area
        and float(after_under_peak_db)
        <= float(before_under_peak_db) + float(settings.under_target_peak_tolerance_db)
    )


def _enforce_final_guard(
    filters: list[IIRFilter],
    base_error: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    weights: np.ndarray,
    core_mask: np.ndarray,
    guard_mask: np.ndarray,
    settings: AutoIIRSettings,
) -> list[IIRFilter]:
    result = list(filters)
    while result:
        current_error = _auto_residual_from_base(base_error, result, sample_rate, frequency)
        if _guard_is_safe(base_error, current_error, frequency, weights, guard_mask, settings):
            return result
        alternatives: list[tuple[float, list[IIRFilter]]] = []
        for index in range(len(result)):
            reduced = result[:index] + result[index + 1 :]
            reduced_error = _auto_residual_from_base(base_error, reduced, sample_rate, frequency)
            core_area = _masked_absolute_error_area(
                reduced_error,
                frequency,
                weights,
                core_mask,
            )
            alternatives.append((core_area, reduced))
        result = min(alternatives, key=lambda value: value[0])[1]
    return []


def _auto_candidate_shortlist(
    candidates: list[_AutoIIRCandidate],
    settings: AutoIIRSettings,
) -> list[_AutoIIRCandidate]:
    if not candidates:
        return []
    minimum = max(1, int(settings.candidate_shortlist_size))
    maximum = max(minimum, int(settings.candidate_shortlist_max))
    selected = list(candidates[:minimum])
    if len(candidates) <= minimum:
        return selected
    reference = max(abs(float(candidates[minimum - 1].priority)), 1e-12)
    for candidate in candidates[minimum:maximum]:
        difference = abs(float(candidate.priority) - float(candidates[minimum - 1].priority))
        if difference / reference > max(float(settings.candidate_shortlist_tie_ratio), 0.0):
            break
        selected.append(candidate)
    return selected


def _best_area_shelf_candidate(
    kind: str,
    error: np.ndarray,
    frequency: np.ndarray,
    sample_rate: int,
    settings: AutoIIRSettings,
    *,
    weights: np.ndarray | None = None,
) -> _AutoIIRCandidate | None:
    """Choose the shelf that removes the most residual area across the full band."""
    if weights is None:
        weights = _auto_frequency_weights(frequency, settings)
    before_area = _absolute_error_area(error, frequency, weights)
    octave_span = max(float(np.log2(frequency[-1] / frequency[0])), 1e-9)
    q_max = max(min(float(settings.max_shelf_q), IIR_SHELF_Q_MAX), 0.2)
    q_values = np.unique(np.minimum(np.asarray([0.35, 0.5, 0.7071, q_max]), q_max))
    # The two shelf families may overlap around the middle, but a low shelf
    # must not migrate to the top edge (or vice versa) and impersonate a broad
    # PEQ. Keep each knee in its logical side while still allowing a generous
    # central overlap for real-world tilted responses.
    log_min = float(np.log(frequency[4]))
    log_max = float(np.log(frequency[-5]))
    if kind == "low_shelf":
        fc_min = float(np.exp(log_min))
        fc_max = float(np.exp(log_min + 0.65 * (log_max - log_min)))
    else:
        fc_min = float(np.exp(log_min + 0.35 * (log_max - log_min)))
        fc_max = float(np.exp(log_max))
    fc_values = np.geomspace(fc_min, fc_max, 24)
    best: _AutoIIRCandidate | None = None
    best_improvement = 0.0
    for fc in fc_values:
        for q in q_values:
            unit = IIRFilter(kind=kind, fc=float(fc), q=float(q), gain_db=1.0, origin="auto")
            unit_gain = _auto_filter_gain([unit], sample_rate, frequency)
            valid = np.abs(unit_gain) > 1e-4
            if not np.any(valid):
                continue
            # For a response that is linear in dB gain, the weighted median of
            # error/shape minimizes integrated absolute error. RBJ shelves are
            # extremely close to linear; the actual Biquad is evaluated below.
            ratios = error[valid] / unit_gain[valid]
            ratio_weights = weights[valid] * np.abs(unit_gain[valid])
            gain = _weighted_median(ratios, ratio_weights)
            gain = float(
                np.clip(
                    gain,
                    -AUTO_IIR_NUMERIC_GAIN_BOUND_DB,
                    AUTO_IIR_NUMERIC_GAIN_BOUND_DB,
                )
            )
            if abs(gain) < float(settings.min_correction_db):
                continue
            item = replace(unit, gain_db=gain)
            after_error = error - _auto_filter_gain([item], sample_rate, frequency)
            improvement = before_area - _absolute_error_area(after_error, frequency, weights)
            if improvement <= best_improvement:
                continue
            normalized_improvement = improvement / octave_span
            priority = normalized_improvement * (1.12 if gain < 0.0 else 0.78)
            best = _AutoIIRCandidate(
                item=item,
                prominence_db=float(normalized_improvement),
                priority=float(priority),
            )
            best_improvement = improvement
    return best


def _auto_filters_from_values(filters: list[IIRFilter], values: np.ndarray) -> list[IIRFilter]:
    result = list(filters)
    for index, item in enumerate(filters):
        offset = index * 3
        result[index] = replace(
            item,
            fc=float(np.exp(values[offset])),
            q=float(np.exp(values[offset + 1])),
            gain_db=float(values[offset + 2]),
        )
    return result


def _auto_q_limit(item: IIRFilter, settings: AutoIIRSettings) -> float:
    if item.kind == "peq":
        limit = min(float(settings.max_peq_q), IIR_PEQ_Q_MAX)
        if item.gain_db > 0.0:
            limit = min(limit, float(settings.broad_boost_max_q))
        resolution = float(settings.independent_resolution_hz or 0.0)
        if resolution > 0.0:
            limit = min(limit, float(item.fc) / (3.0 * resolution))
        return max(limit, 0.2)
    if item.kind in {"low_shelf", "high_shelf"}:
        return max(min(float(settings.max_shelf_q), IIR_SHELF_Q_MAX), 0.2)
    return 12.0


def _auto_optimizer_needs_multistart(
    result: object,
    *,
    max_nfev: int,
    initial_cost: float,
) -> bool:
    """Return whether a single-start solve needs the guarded Q alternatives."""
    values = np.asarray(getattr(result, "x", []), dtype=float)
    residual = np.asarray(getattr(result, "fun", []), dtype=float)
    fallback_cost = float(np.sum(np.sqrt(1.0 + np.square(residual)) - 1.0))
    result_cost = float(getattr(result, "cost", fallback_cost))
    return bool(
        not bool(getattr(result, "success", False))
        or int(getattr(result, "status", 0) or 0) <= 0
        or int(getattr(result, "nfev", max_nfev) or max_nfev) >= int(max_nfev)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(residual))
        or not np.isfinite(result_cost)
        or result_cost > initial_cost + 1e-9
    )


def _optimize_auto_filter_set(
    filters: list[IIRFilter],
    manual: list[IIRFilter],
    speaker_gain: np.ndarray,
    target_gain: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    max_nfev: int,
    multi_start: bool = False,
    fc_limit_oct: float | None = None,
    fc_bounds: tuple[float, float] | None = None,
    fit_range: tuple[float, float] | None = None,
    frequency_weights: np.ndarray | None = None,
) -> list[IIRFilter]:
    if not filters:
        return []
    initial: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    if fc_bounds is None:
        f_min = float(frequency[0])
        f_max = float(frequency[-1])
    else:
        f_min = max(float(fc_bounds[0]), float(frequency[0]))
        f_max = min(float(fc_bounds[1]), float(frequency[-1]))
    if f_max <= f_min:
        raise ValueError("Auto IIR filter frequency bounds are empty")
    for item in filters:
        q_limit = _auto_q_limit(item, settings)
        initial_q = float(np.clip(item.q, 0.2, q_limit))
        gain_lower, gain_upper = _auto_iir_gain_search_bounds(item, settings)
        initial_gain = float(np.clip(item.gain_db, gain_lower, gain_upper))
        initial.extend([np.log(item.fc), np.log(initial_q), initial_gain])
        if fc_limit_oct is None:
            item_f_min = f_min
            item_f_max = f_max
        else:
            ratio = 2.0 ** max(float(fc_limit_oct), 0.0)
            item_f_min = max(f_min, float(item.fc) / ratio)
            item_f_max = min(f_max, float(item.fc) * ratio)
        lower.extend([np.log(item_f_min), np.log(0.2), gain_lower])
        upper.extend([np.log(item_f_max), np.log(q_limit), gain_upper])
    weights = (
        _auto_frequency_weights(frequency, settings)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=float).copy()
    )
    if weights.shape != np.asarray(frequency).shape:
        raise ValueError("Auto IIR frequency weights must match the frequency axis")
    if fit_range is not None:
        fit_mask = (frequency >= float(fit_range[0])) & (frequency <= float(fit_range[1]))
        weights = weights * fit_mask.astype(float)
        if np.count_nonzero(fit_mask) < 2:
            raise ValueError("Auto IIR correction band has too few evaluation points")
    sensitivity = _auto_sensitivity(frequency, settings)
    frequency_kernel = _auto_frequency_kernel(sample_rate, frequency)
    fixed_response = _auto_filter_response(
        manual,
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
    )
    unfiltered_error = target_gain - speaker_gain

    def objective(values: np.ndarray) -> np.ndarray:
        combined_response = _auto_filter_response_from_values(
            filters,
            values,
            sample_rate,
            frequency,
            frequency_kernel=frequency_kernel,
            initial_response=fixed_response,
        )
        combined_gain = 20.0 * np.log10(
            np.maximum(np.abs(combined_response), 1e-12)
        )
        error = unfiltered_error - combined_gain
        point_error = np.sqrt(weights) * error
        mean_error = float(np.sum(weights * error) / max(float(np.sum(weights)), 1e-12))
        smooth_error = _auto_smoothed_error(error, frequency, 1.0 / 12.0)
        curvature = np.diff(smooth_error, n=2)
        smooth_residual = (
            np.sqrt(max(float(settings.smoothness_weight), 0.0))
            * np.sqrt(1.0 + sensitivity[1:-1])
            * curvature
        )
        regularization: list[float] = []
        for index, _item in enumerate(filters):
            gain = float(values[index * 3 + 2])
            q = float(np.exp(values[index * 3 + 1]))
            gain_weight = settings.boost_penalty if gain > 0.0 else settings.cut_penalty
            regularization.append(np.sqrt(max(gain_weight, 0.0)) * gain)
            regularization.append(
                np.sqrt(max(float(settings.high_q_penalty), 0.0))
                * max(np.log2(max(q, 1e-9) / 4.0), 0.0)
            )
        headroom_excess = max(
            max(float(np.max(combined_gain)), 0.0)
            - float(settings.filter_headroom_limit_db),
            0.0,
        )
        return np.concatenate(
            [
                point_error,
                np.asarray([0.5 * np.sqrt(error.size) * mean_error]),
                smooth_residual,
                np.asarray(regularization, dtype=float),
                np.asarray([4.0 * np.sqrt(error.size) * headroom_excess]),
            ]
        )

    initial_array = np.asarray(initial, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    initial_residual = objective(initial_array)
    initial_cost = float(np.sum(np.sqrt(1.0 + np.square(initial_residual)) - 1.0))

    def solve(start: np.ndarray) -> tuple[object, list[IIRFilter], float]:
        result = least_squares(
            objective,
            start,
            bounds=(lower_array, upper_array),
            max_nfev=max_nfev,
            loss="soft_l1",
            f_scale=1.0,
            x_scale="jac",
        )
        trial = _enforce_auto_iir_filter_headroom(
            _auto_filters_from_values(filters, result.x),
            manual,
            sample_rate,
            frequency,
            settings,
            frequency_kernel=frequency_kernel,
        )
        error = unfiltered_error - _auto_filter_gain(
            trial,
            sample_rate,
            frequency,
            frequency_kernel=frequency_kernel,
            initial_response=fixed_response,
        )
        score = _auto_model_score(error, trial, frequency, settings, weights=weights)
        return result, trial, score

    first_result, best_filters, best_score = solve(initial_array)
    if multi_start and _auto_optimizer_needs_multistart(
        first_result,
        max_nfev=max_nfev,
        initial_cost=initial_cost,
    ):
        for q_scale in (0.72, 1.38):
            alternate = initial_array.copy()
            alternate[1::3] = np.clip(
                alternate[1::3] + np.log(q_scale),
                lower_array[1::3],
                upper_array[1::3],
            )
            _result, trial, score = solve(alternate)
            if score < best_score:
                best_score = score
                best_filters = trial
    return best_filters


def _optimize_auto_filter_set_fixed_q(
    filters: list[IIRFilter],
    manual: list[IIRFilter],
    speaker_gain: np.ndarray,
    target_gain: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    max_nfev: int,
    fc_limit_oct: float | None = None,
    fc_bounds: tuple[float, float] | None = None,
    fit_range: tuple[float, float] | None = None,
    frequency_weights: np.ndarray | None = None,
) -> list[IIRFilter]:
    """Refit Fc/Gain after Auto IIR Q values have been fixed to 0.1."""
    if not filters:
        return []
    base_filters = [
        replace(
            item,
            q=float(
                np.clip(
                    quantize_iir_q(item.kind, item.q),
                    0.2,
                    _auto_q_limit(item, settings),
                )
            ),
        )
        for item in filters
    ]
    if fc_bounds is None:
        f_min = float(frequency[0])
        f_max = float(frequency[-1])
    else:
        f_min = max(float(fc_bounds[0]), float(frequency[0]))
        f_max = min(float(fc_bounds[1]), float(frequency[-1]))
    if f_max <= f_min:
        raise ValueError("Auto IIR filter frequency bounds are empty")

    initial: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for item in base_filters:
        gain_lower, gain_upper = _auto_iir_gain_search_bounds(item, settings)
        initial_gain = float(np.clip(item.gain_db, gain_lower, gain_upper))
        initial.extend([np.log(item.fc), initial_gain])
        if fc_limit_oct is None:
            item_f_min = f_min
            item_f_max = f_max
        else:
            ratio = 2.0 ** max(float(fc_limit_oct), 0.0)
            item_f_min = max(f_min, float(item.fc) / ratio)
            item_f_max = min(f_max, float(item.fc) * ratio)
        lower.extend([np.log(item_f_min), gain_lower])
        upper.extend([np.log(item_f_max), gain_upper])

    weights = (
        _auto_frequency_weights(frequency, settings)
        if frequency_weights is None
        else np.asarray(frequency_weights, dtype=float).copy()
    )
    if weights.shape != np.asarray(frequency).shape:
        raise ValueError("Auto IIR frequency weights must match the frequency axis")
    if fit_range is not None:
        fit_mask = (frequency >= float(fit_range[0])) & (frequency <= float(fit_range[1]))
        weights = weights * fit_mask.astype(float)
        if np.count_nonzero(fit_mask) < 2:
            raise ValueError("Auto IIR correction band has too few evaluation points")
    sensitivity = _auto_sensitivity(frequency, settings)
    frequency_kernel = _auto_frequency_kernel(sample_rate, frequency)
    fixed_response = _auto_filter_response(
        manual,
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
    )
    unfiltered_error = target_gain - speaker_gain

    def filters_from_values(values: np.ndarray) -> list[IIRFilter]:
        return [
            replace(
                item,
                fc=float(np.exp(values[index * 2])),
                gain_db=float(values[index * 2 + 1]),
            )
            for index, item in enumerate(base_filters)
        ]

    def objective(values: np.ndarray) -> np.ndarray:
        combined_response = _auto_filter_response_from_values(
            base_filters,
            values,
            sample_rate,
            frequency,
            frequency_kernel=frequency_kernel,
            fixed_q=True,
            initial_response=fixed_response,
        )
        combined_gain = 20.0 * np.log10(
            np.maximum(np.abs(combined_response), 1e-12)
        )
        error = unfiltered_error - combined_gain
        point_error = np.sqrt(weights) * error
        mean_error = float(np.sum(weights * error) / max(float(np.sum(weights)), 1e-12))
        smooth_error = _auto_smoothed_error(error, frequency, 1.0 / 12.0)
        curvature = np.diff(smooth_error, n=2)
        smooth_residual = (
            np.sqrt(max(float(settings.smoothness_weight), 0.0))
            * np.sqrt(1.0 + sensitivity[1:-1])
            * curvature
        )
        regularization = []
        for index, _item in enumerate(base_filters):
            gain = float(values[index * 2 + 1])
            gain_weight = settings.boost_penalty if gain > 0.0 else settings.cut_penalty
            regularization.append(np.sqrt(max(gain_weight, 0.0)) * gain)
        headroom_excess = max(
            max(float(np.max(combined_gain)), 0.0)
            - float(settings.filter_headroom_limit_db),
            0.0,
        )
        return np.concatenate(
            [
                point_error,
                np.asarray([0.5 * np.sqrt(error.size) * mean_error]),
                smooth_residual,
                np.asarray(regularization, dtype=float),
                np.asarray([4.0 * np.sqrt(error.size) * headroom_excess]),
            ]
        )

    result = least_squares(
        objective,
        np.asarray(initial, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        max_nfev=max_nfev,
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
    )
    fitted = _enforce_auto_iir_filter_headroom(
        filters_from_values(result.x),
        manual,
        sample_rate,
        frequency,
        settings,
        frequency_kernel=frequency_kernel,
    )
    if not np.all(np.isfinite(np.asarray(result.x, dtype=float))):
        return base_filters
    base_filters = _enforce_auto_iir_filter_headroom(
        base_filters,
        manual,
        sample_rate,
        frequency,
        settings,
        frequency_kernel=frequency_kernel,
    )
    base_error = unfiltered_error - _auto_filter_gain(
        base_filters,
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
        initial_response=fixed_response,
    )
    fitted_error = unfiltered_error - _auto_filter_gain(
        fitted,
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
        initial_response=fixed_response,
    )
    base_score = _auto_model_score(base_error, base_filters, frequency, settings, weights=weights)
    fitted_score = _auto_model_score(fitted_error, fitted, frequency, settings, weights=weights)
    return fitted if fitted_score <= base_score + 1e-9 else base_filters


def _auto_total_filter_budget(settings: AutoIIRSettings) -> int:
    if settings.max_peq is not None:
        # Old callers expressed only the PEQ allowance. Preserve those API
        # calls while all UI/state paths use the new total-filter contract.
        return max(0, int(settings.max_peq)) + int(settings.include_low_shelf) + int(
            settings.include_high_shelf
        )
    return max(0, int(settings.max_filters))


def _auto_peq_is_enabled(settings: AutoIIRSettings) -> bool:
    return bool(settings.include_peq) and (settings.max_peq is None or int(settings.max_peq) > 0)


def _masked_shelf_trend(
    error: np.ndarray,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
) -> np.ndarray:
    """Return a broad trend with narrow peaks bridged before shelf fitting."""
    trend = _auto_smoothed_error(error, frequency, float(settings.shelf_smoothing_oct))
    detail = _auto_smoothed_error(error, frequency, float(settings.peq_smoothing_oct)) - trend
    threshold = max(float(settings.min_correction_db) * 0.35, 0.20)
    peak_indices, _properties = signal.find_peaks(
        np.abs(detail),
        prominence=threshold,
        distance=max(2, int(frequency.size / 80)),
    )
    if peak_indices.size == 0:
        return trend
    log_frequency = np.log2(frequency)
    mask = np.zeros(frequency.size, dtype=bool)
    half_width_oct = max(float(settings.peq_smoothing_oct) * 3.0, 1.0 / 4.0)
    for index in peak_indices:
        mask |= np.abs(log_frequency - log_frequency[int(index)]) <= half_width_oct
    valid = ~mask
    if np.count_nonzero(valid) < 4:
        return trend
    bridged = trend.copy()
    bridged[mask] = np.interp(log_frequency[mask], log_frequency[valid], trend[valid])
    return bridged


def _auto_peq_candidates(
    error: np.ndarray,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    existing: list[IIRFilter],
    *,
    audibility_weights: np.ndarray | None = None,
    perceptual_level_db: np.ndarray | None = None,
) -> list[_AutoIIRCandidate]:
    """Extract one-filter-per-hill candidates from a 1/12-octave residual."""
    min_correction = max(float(settings.min_correction_db), 0.0)
    octave_span = max(float(np.log2(frequency[-1] / frequency[0])), 1e-9)
    points_per_octave = max((frequency.size - 1) / octave_span, 1.0)
    peak_distance = max(2, int(points_per_octave / 8.0))
    prominence_floor = max(min_correction * 0.35, 0.05)
    sensitivity = _auto_sensitivity(frequency, settings)
    audible = (
        np.ones_like(frequency, dtype=float)
        if audibility_weights is None
        else np.clip(np.asarray(audibility_weights, dtype=float), 0.0, None)
    )
    if audible.shape != frequency.shape:
        raise ValueError("Auto IIR audibility weights must match the frequency axis")
    peak_excess = np.zeros_like(frequency, dtype=float)
    if perceptual_level_db is not None:
        perceptual = np.asarray(perceptual_level_db, dtype=float)
        if perceptual.shape != frequency.shape:
            raise ValueError("Auto IIR perceptual level must match the frequency axis")
        peak_curve = _auto_smoothed_error(
            perceptual,
            frequency,
            float(settings.wavelet_peak_smoothing_oct),
        )
        context_curve = _auto_smoothed_error(
            perceptual,
            frequency,
            float(settings.wavelet_peak_context_oct),
        )
        peak_excess = np.maximum(peak_curve - context_curve, 0.0)
    log_frequency = np.log2(frequency)
    third_error = _auto_smoothed_error(error, frequency, 1.0 / 3.0)
    one_error = _auto_smoothed_error(error, frequency, 1.0)
    # Error is Target - response, so a positive response protrusion is
    # ``one_error - third_error``. Dips deliberately receive no peak bonus.
    response_peak_prominence = np.maximum(one_error - third_error, 0.0)
    candidates: list[_AutoIIRCandidate] = []
    for is_cut, signed_error in ((True, -error), (False, error)):
        indices, properties = signal.find_peaks(
            signed_error,
            prominence=prominence_floor,
            distance=peak_distance,
        )
        prominence_by_index = {
            int(index): float(value)
            for index, value in zip(indices, properties["prominences"])
        }
        extreme_index = int(np.argmin(error) if is_cut else np.argmax(error))
        indices = np.unique(np.append(indices, extreme_index))
        for index_value in indices:
            index = int(index_value)
            gain = float(error[index])
            if (is_cut and gain >= -min_correction) or (not is_cut and gain <= min_correction):
                continue
            gain = float(
                np.clip(
                    gain,
                    -AUTO_IIR_NUMERIC_GAIN_BOUND_DB,
                    AUTO_IIR_NUMERIC_GAIN_BOUND_DB,
                )
            )
            q = min(
                _estimate_peak_q(frequency, error, index),
                max(float(settings.max_peq_q), 0.2),
            )
            response_peak = float(response_peak_prominence[index])
            wavelet_peak = float(peak_excess[index])
            wavelet_agrees = response_peak >= max(
                float(settings.wavelet_response_agreement_min_db),
                float(settings.wavelet_peak_min_prominence_db) * 0.5,
            )
            confirmed_wavelet_peak = wavelet_peak if wavelet_agrees else 0.0
            audible_peak = max(response_peak, confirmed_wavelet_peak)
            if is_cut and response_peak >= float(settings.wavelet_peak_min_prominence_db):
                q = min(q, max(float(settings.wavelet_peak_max_q), 0.2))
                full_cut = abs(gain)
                response_above_target = max(float(-third_error[index]), 0.0)
                balanced_cut = max(
                    audible_peak - float(settings.positive_peak_residual_db),
                    response_above_target * float(settings.positive_peak_level_weight),
                    min_correction,
                )
                gain = -min(
                    full_cut * float(settings.positive_peak_max_reduction_ratio),
                    balanced_cut,
                )
            elif not is_cut:
                # Only broad, measurement-robust dips may be boosted. Narrow
                # cancellation notches remain visible but are not chased.
                q = min(q, max(float(settings.broad_boost_max_q), 0.2))
            item = IIRFilter(
                kind="peq",
                fc=float(frequency[index]),
                q=q,
                gain_db=gain,
                origin="auto",
            )
            if any(
                other.kind == "peq"
                and np.sign(other.gain_db) == np.sign(gain)
                and abs(np.log2(other.fc / item.fc)) < 1.0 / 10.0
                for other in existing
            ):
                continue
            prominence = float(prominence_by_index.get(index, abs(gain) * 0.35))
            half_width_oct = max(1.0 / max(2.0 * q, 1e-9), 1.0 / 6.0)
            local = np.abs(log_frequency - log_frequency[index]) <= half_width_oct
            local_area = _masked_absolute_error_area(
                error,
                frequency,
                audible,
                local,
            )
            local_area_density = local_area / max(2.0 * half_width_oct, 1e-9)
            positive_peak_area = _masked_absolute_error_area(
                response_peak_prominence,
                frequency,
                audible,
                local,
            )
            wavelet_factor = 1.0
            if is_cut and confirmed_wavelet_peak > 0.0:
                reference = max(float(settings.wavelet_peak_min_prominence_db), 1e-9)
                wavelet_factor = min(
                    1.0
                    + float(settings.wavelet_peak_priority_weight)
                    * confirmed_wavelet_peak
                    / reference,
                    float(settings.wavelet_peak_priority_max_factor),
                )
            positive_peak_priority = (
                0.45 * response_peak_prominence[index] + 0.25 * positive_peak_area
                if is_cut
                else 0.0
            )
            priority = (
                (
                    0.35 * prominence
                    + 0.20 * abs(gain)
                    + 0.25 * local_area_density
                    + positive_peak_priority
                )
                * (1.18 if gain < 0.0 else 0.82)
                * (1.0 + 0.08 * sensitivity[index])
                * audible[index]
                * wavelet_factor
            )
            candidates.append(_AutoIIRCandidate(item, prominence, float(priority)))
    candidates.sort(key=lambda value: value.priority, reverse=True)
    return candidates[:18]


def _cap_wavelet_peak_cut_q(
    filters: list[IIRFilter],
    frequency: np.ndarray,
    perceptual_level_db: np.ndarray,
    settings: AutoIIRSettings,
    *,
    response_error_db: np.ndarray | None = None,
) -> list[IIRFilter]:
    """Keep confirmed audible response-peak cuts broad enough to be robust."""

    peak_curve = _auto_smoothed_error(
        np.asarray(perceptual_level_db, dtype=float),
        frequency,
        float(settings.wavelet_peak_smoothing_oct),
    )
    context_curve = _auto_smoothed_error(
        np.asarray(perceptual_level_db, dtype=float),
        frequency,
        float(settings.wavelet_peak_context_oct),
    )
    excess = np.maximum(peak_curve - context_curve, 0.0)
    response_prominence = np.full_like(frequency, np.inf, dtype=float)
    if response_error_db is not None:
        response_error = np.asarray(response_error_db, dtype=float)
        if response_error.shape != np.asarray(frequency).shape:
            raise ValueError("Auto IIR response error must match the frequency axis")
        response_third = _auto_smoothed_error(response_error, frequency, 1.0 / 3.0)
        response_one = _auto_smoothed_error(response_error, frequency, 1.0)
        response_prominence = np.maximum(response_one - response_third, 0.0)
    limit = max(float(settings.wavelet_peak_max_q), 0.2)
    result: list[IIRFilter] = []
    for item in filters:
        if item.kind != "peq" or item.gain_db >= 0.0:
            result.append(item)
            continue
        index = int(np.argmin(np.abs(np.log2(frequency / max(float(item.fc), 1e-9)))))
        if (
            excess[index] >= float(settings.wavelet_peak_min_prominence_db)
            and response_prominence[index]
            >= max(
                float(settings.wavelet_response_agreement_min_db),
                float(settings.wavelet_peak_min_prominence_db) * 0.5,
            )
        ):
            result.append(replace(item, q=min(float(item.q), limit)))
        else:
            result.append(item)
    return result


def _optimize_auto_from_base_error(
    filters: list[IIRFilter],
    fixed_filters: list[IIRFilter],
    base_error: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    max_nfev: int,
    multi_start: bool = False,
    fc_limit_oct: float | None = None,
    fc_bounds: tuple[float, float] | None = None,
    fit_range: tuple[float, float] | None = None,
    frequency_weights: np.ndarray | None = None,
) -> list[IIRFilter]:
    return _optimize_auto_filter_set(
        filters,
        fixed_filters,
        -np.asarray(base_error, dtype=float),
        np.zeros_like(base_error, dtype=float),
        sample_rate,
        frequency,
        settings,
        max_nfev=max_nfev,
        multi_start=multi_start,
        fc_limit_oct=fc_limit_oct,
        fc_bounds=fc_bounds,
        fit_range=fit_range,
        frequency_weights=frequency_weights,
    )


def _auto_residual_from_base(
    base_error: np.ndarray,
    filters: list[IIRFilter],
    sample_rate: int,
    frequency: np.ndarray,
) -> np.ndarray:
    return np.asarray(base_error, dtype=float) - _auto_filter_gain(filters, sample_rate, frequency)


def _auto_center_references(
    filters: list[IIRFilter],
    base_error: np.ndarray,
    frequency: np.ndarray,
    weights: np.ndarray,
    settings: AutoIIRSettings,
    perceptual_level_db: np.ndarray | None,
) -> dict[int, float]:
    """Return one sign-preserving 1/3-octave error centroid per PEQ."""

    third_error = fractional_octave_smooth(base_error, frequency, 1.0 / 3.0)
    log_frequency = np.log2(frequency)
    wavelet_emphasis = np.ones_like(frequency, dtype=float)
    if perceptual_level_db is not None:
        perceptual = np.asarray(perceptual_level_db, dtype=float)
        peak_curve = fractional_octave_smooth(perceptual, frequency, 1.0 / 3.0)
        context_curve = fractional_octave_smooth(perceptual, frequency, 1.0)
        prominence = np.maximum(peak_curve - context_curve, 0.0)
        reference = max(float(np.percentile(prominence, 95.0)), 0.75)
        wavelet_emphasis += 0.5 * np.clip(prominence / reference, 0.0, 1.0)
    result: dict[int, float] = {}
    half_width = max(float(settings.peq_fc_limit_oct), 1.0 / 3.0)
    for index, item in enumerate(filters):
        if item.kind != "peq":
            continue
        local = np.abs(log_frequency - np.log2(max(float(item.fc), 1e-9))) <= half_width
        same_sign = np.sign(third_error) == np.sign(float(item.gain_db))
        selected = local & same_sign & (np.abs(third_error) >= float(settings.min_correction_db) * 0.25)
        if np.count_nonzero(selected) < 2:
            result[index] = float(np.log2(item.fc))
            continue
        centroid_weight = (
            np.abs(third_error[selected])
            * np.maximum(np.asarray(weights)[selected], 0.0)
            * wavelet_emphasis[selected]
        )
        denominator = max(float(np.sum(centroid_weight)), 1e-12)
        result[index] = float(np.sum(log_frequency[selected] * centroid_weight) / denominator)
    return result


def _auto_center_shift_oct(
    filters: list[IIRFilter],
    references: dict[int, float],
) -> float:
    distances = [
        abs(float(np.log2(filters[index].fc)) - center)
        for index, center in references.items()
        if index < len(filters) and filters[index].kind == "peq"
    ]
    return float(np.mean(distances)) if distances else 0.0


def _auto_wavelet_peak_cut_indices(
    filters: list[IIRFilter],
    frequency: np.ndarray,
    perceptual_level_db: np.ndarray | None,
    settings: AutoIIRSettings,
    *,
    response_error_db: np.ndarray | None = None,
) -> set[int]:
    if perceptual_level_db is None:
        return set()
    peak_curve = fractional_octave_smooth(perceptual_level_db, frequency, 1.0 / 3.0)
    context_curve = fractional_octave_smooth(perceptual_level_db, frequency, 1.0)
    excess = np.maximum(peak_curve - context_curve, 0.0)
    response_prominence = np.full_like(frequency, np.inf, dtype=float)
    if response_error_db is not None:
        response_error = np.asarray(response_error_db, dtype=float)
        if response_error.shape != np.asarray(frequency).shape:
            raise ValueError("Auto IIR response error must match the frequency axis")
        response_third = fractional_octave_smooth(response_error, frequency, 1.0 / 3.0)
        response_one = fractional_octave_smooth(response_error, frequency, 1.0)
        response_prominence = np.maximum(response_one - response_third, 0.0)
    result: set[int] = set()
    for index, item in enumerate(filters):
        if item.kind != "peq" or item.gain_db >= 0.0:
            continue
        nearest = int(np.argmin(np.abs(np.log2(frequency / max(float(item.fc), 1e-9)))))
        if (
            excess[nearest] >= float(settings.wavelet_peak_min_prominence_db)
            and response_prominence[nearest]
            >= max(
                float(settings.wavelet_response_agreement_min_db),
                float(settings.wavelet_peak_min_prominence_db) * 0.5,
            )
        ):
            result.add(index)
    return result


def _auto_sobol_search_parameters(
    filters: list[IIRFilter],
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    fc_limit_oct: float,
    fc_bounds: tuple[float, float],
    wavelet_limited_indices: set[int] | None = None,
) -> tuple[tuple[SearchParameter, ...], dict[str, float]]:
    f_min = max(float(fc_bounds[0]), float(frequency[0]))
    f_max = min(float(fc_bounds[1]), float(frequency[-1]))
    ratio = 2.0 ** max(float(fc_limit_oct), 0.0)
    parameters: list[SearchParameter] = []
    initial: dict[str, float] = {}
    available_cut_offset = sum(max(-float(item.gain_db), 0.0) for item in filters)
    for index, item in enumerate(filters):
        fc_low = max(f_min, float(item.fc) / ratio)
        fc_high = min(f_max, float(item.fc) * ratio)
        q_low = 0.2
        q_high = _auto_q_limit(item, settings)
        if wavelet_limited_indices is not None and index in wavelet_limited_indices:
            q_high = min(q_high, max(float(settings.wavelet_peak_max_q), 0.2))
        gain_low, gain_high = _auto_iir_gain_search_bounds(item, settings)
        if item.gain_db >= 0.0:
            # Keep Sobol's finite box concentrated around combinations that
            # can satisfy the aggregate headroom constraint. Existing cuts
            # may create room for a boost above +12 dB without raising the
            # combined cascade above +12 dB.
            gain_high = min(
                gain_high,
                float(settings.filter_headroom_limit_db) + available_cut_offset,
            )
        prefix = f"filter_{index}"
        parameters.extend(
            [
                SearchParameter(f"{prefix}_fc", fc_low, fc_high, "log"),
                SearchParameter(f"{prefix}_q", q_low, q_high, "log"),
                SearchParameter(f"{prefix}_gain", gain_low, gain_high),
            ]
        )
        initial[f"{prefix}_fc"] = float(np.clip(item.fc, fc_low, fc_high))
        initial[f"{prefix}_q"] = float(np.clip(item.q, q_low, q_high))
        initial[f"{prefix}_gain"] = float(np.clip(item.gain_db, gain_low, gain_high))
    return tuple(parameters), initial


def _auto_filters_from_search_params(
    filters: list[IIRFilter],
    values: dict[str, float],
) -> list[IIRFilter]:
    return [
        replace(
            item,
            fc=float(values[f"filter_{index}_fc"]),
            q=float(values[f"filter_{index}_q"]),
            gain_db=float(values[f"filter_{index}_gain"]),
        )
        for index, item in enumerate(filters)
    ]


def _global_optimize_auto_from_base_error(
    filters: list[IIRFilter],
    base_error: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    settings: AutoIIRSettings,
    *,
    core_weights: np.ndarray,
    guard_weights: np.ndarray,
    core_mask: np.ndarray,
    guard_mask: np.ndarray,
    perceptual_level_db: np.ndarray | None,
    fc_limit_oct: float,
    fc_bounds: tuple[float, float],
    headroom_fixed_filters: Iterable[IIRFilter] = (),
) -> tuple[list[tuple[str, list[IIRFilter]]], SobolOptunaResult]:
    """Globally refine a fixed Auto IIR topology, then locally polish its leaders."""

    fixed_headroom = [item for item in headroom_fixed_filters if item.enabled]
    filters = _enforce_auto_iir_filter_headroom(
        filters,
        fixed_headroom,
        sample_rate,
        frequency,
        settings,
    )
    parameters, initial = _auto_sobol_search_parameters(
        filters,
        frequency,
        settings,
        fc_limit_oct=fc_limit_oct,
        fc_bounds=fc_bounds,
        wavelet_limited_indices=_auto_wavelet_peak_cut_indices(
            filters,
            frequency,
            perceptual_level_db,
            settings,
            response_error_db=base_error,
        ),
    )
    reference = auto_iir_objective_reference(base_error, frequency, core_weights)
    centers = _auto_center_references(
        filters,
        base_error,
        frequency,
        core_weights,
        settings,
        perceptual_level_db,
    )
    objective_weights = AutoIIRObjectiveWeights(
        one_octave_abs=float(settings.objective_one_octave_abs_weight),
        third_octave_abs=float(settings.objective_third_octave_abs_weight),
        perceptual_peak=float(settings.objective_perceptual_peak_weight),
        positive_peak_area=float(settings.objective_positive_peak_area_weight),
        center_shift=float(settings.objective_center_shift_weight),
    )
    base_peak_metrics = positive_peak_metrics(base_error, frequency, core_weights)
    base_guard_area = _masked_absolute_error_area(
        base_error,
        frequency,
        guard_weights,
        guard_mask,
    )
    base_guard_rms = _masked_weighted_rms(base_error, guard_weights, guard_mask)
    base_guard_p95 = (
        float(np.percentile(np.abs(base_error[guard_mask]), 95.0))
        if np.count_nonzero(guard_mask)
        else 0.0
    )
    frequency_kernel = _auto_frequency_kernel(sample_rate, frequency)
    fixed_headroom_response = _auto_filter_response(
        fixed_headroom,
        sample_rate,
        frequency,
        frequency_kernel=frequency_kernel,
    )
    area_tolerance = max(
        float(settings.guard_area_tolerance_db_oct),
        base_guard_area * float(settings.guard_area_tolerance_ratio),
    )
    def objective(values: dict[str, float]) -> OptimizationEvaluation:
        trial_filters = _auto_filters_from_search_params(filters, values)
        trial_response = _auto_filter_response(
            trial_filters,
            sample_rate,
            frequency,
            frequency_kernel=frequency_kernel,
        )
        trial_gain = 20.0 * np.log10(np.maximum(np.abs(trial_response), 1e-12))
        error = np.asarray(base_error, dtype=float) - trial_gain
        components = evaluate_auto_iir_objective(
            error,
            frequency,
            core_weights,
            reference=reference,
            center_shift_oct=_auto_center_shift_oct(trial_filters, centers),
            objective_weights=objective_weights,
        )
        boost_cost = sum(max(float(item.gain_db), 0.0) for item in trial_filters)
        cut_cost = sum(max(-float(item.gain_db), 0.0) for item in trial_filters)
        high_q_cost = sum(
            max(np.log2(max(float(item.q), 1e-9) / 4.0), 0.0)
            for item in trial_filters
        )
        regularization = float(settings.selection_regularization_scale) * (
            float(settings.boost_penalty) * boost_cost
            + float(settings.cut_penalty) * cut_cost
            + float(settings.high_q_penalty) * high_q_cost
        )
        guard_area = _masked_absolute_error_area(error, frequency, guard_weights, guard_mask)
        guard_rms = _masked_weighted_rms(error, guard_weights, guard_mask)
        guard_p95 = (
            float(np.percentile(np.abs(error[guard_mask]), 95.0))
            if np.count_nonzero(guard_mask)
            else 0.0
        )
        peak_area_improvement = max(
            base_peak_metrics.area_db_oct - components.positive_peak_area_db_oct,
            0.0,
        )
        allowed_under_target_area = (
            base_peak_metrics.under_target_area_db_oct
            + float(settings.under_target_area_tolerance_db_oct)
            + float(settings.positive_peak_under_target_tradeoff) * peak_area_improvement
        )
        filter_headroom = max(
            float(
                np.max(
                    20.0
                    * np.log10(
                        np.maximum(
                            np.abs(fixed_headroom_response * trial_response),
                            1e-12,
                        )
                    )
                )
            ),
            0.0,
        )
        violations = (
            guard_area - (base_guard_area + area_tolerance),
            guard_rms - (base_guard_rms + float(settings.guard_rms_tolerance_db)),
            guard_p95 - (base_guard_p95 + float(settings.guard_p95_tolerance_db)),
            components.under_target_area_db_oct - allowed_under_target_area,
            filter_headroom - float(settings.filter_headroom_limit_db),
        )
        return OptimizationEvaluation(
            score=float(components.score + regularization),
            components={
                "one_octave_abs_db": components.one_octave_abs_db,
                "third_octave_abs_db": components.third_octave_abs_db,
                "contour_abs_db": components.contour_abs_db,
                "perceptual_peak_db": components.perceptual_peak_db,
                "positive_peak_area_db_oct": components.positive_peak_area_db_oct,
                "under_target_area_db_oct": components.under_target_area_db_oct,
                "center_shift_oct": components.center_shift_oct,
                "regularization": regularization,
                "filter_headroom_db": filter_headroom,
            },
            constraint_violations=violations,
        )

    search = optimize_sobol_optuna(
        parameters,
        objective,
        settings=SobolOptunaSettings(
            sobol_trials=int(settings.optimizer_sobol_trials),
            optuna_trials=int(settings.optimizer_optuna_trials),
            seed=int(settings.optimizer_seed),
            timeout_seconds=settings.optimizer_timeout_seconds,
            top_k=int(settings.optimizer_top_k),
        ),
        initial_params=(initial,),
    )
    polished: list[tuple[str, list[IIRFilter]]] = []
    for trial in search.top_trials:
        candidate = _auto_filters_from_search_params(filters, trial.params)
        candidate = _optimize_auto_from_base_error(
            candidate,
            [],
            base_error,
            sample_rate,
            frequency,
            settings,
            max_nfev=100,
            multi_start=False,
            fc_limit_oct=fc_limit_oct,
            fc_bounds=fc_bounds,
            fit_range=fc_bounds,
            frequency_weights=core_weights,
        )
        candidate = _enforce_auto_iir_filter_headroom(
            candidate,
            fixed_headroom,
            sample_rate,
            frequency,
            settings,
            frequency_kernel=frequency_kernel,
        )
        if perceptual_level_db is not None:
            candidate = _cap_wavelet_peak_cut_q(
                candidate,
                frequency,
                perceptual_level_db,
                settings,
                response_error_db=base_error,
            )
        candidate_evaluation = objective(
            {
                f"filter_{index}_{field}": value
                for index, item in enumerate(candidate)
                for field, value in (
                    ("fc", item.fc),
                    ("q", item.q),
                    ("gain", item.gain_db),
                )
            }
        )
        if candidate_evaluation.feasible:
            polished.append((trial.source, candidate))
    if not polished:
        return [
            (search.best.source, _auto_filters_from_search_params(filters, search.best.params))
        ], search
    return polished, search


def _auto_remove_close_opposite_pairs(
    filters: list[IIRFilter],
    base_error: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    weights: np.ndarray,
    settings: AutoIIRSettings,
) -> list[IIRFilter]:
    result = list(filters)
    while True:
        current_area = _absolute_error_area(
            _auto_residual_from_base(base_error, result, sample_rate, frequency),
            frequency,
            weights,
        )
        removable_pairs: list[tuple[float, list[IIRFilter]]] = []
        for left in range(len(result)):
            for right in range(left + 1, len(result)):
                first = result[left]
                second = result[right]
                if first.kind != "peq" or second.kind != "peq":
                    continue
                distance_oct = abs(np.log2(first.fc / second.fc))
                opposite = np.sign(first.gain_db) != np.sign(second.gain_db)
                same_hill = not opposite and distance_oct < 1.0 / 10.0
                close_opposite = opposite and distance_oct < float(
                    settings.close_opposite_spacing_oct
                )
                if not same_hill and not close_opposite:
                    continue
                alternatives: list[tuple[float, list[IIRFilter]]] = []
                for index in (left, right):
                    reduced = result[:index] + result[index + 1 :]
                    area = _absolute_error_area(
                        _auto_residual_from_base(
                            base_error,
                            reduced,
                            sample_rate,
                            frequency,
                        ),
                        frequency,
                        weights,
                    )
                    alternatives.append((area, reduced))
                best_area, best_reduced = min(alternatives, key=lambda value: value[0])
                if same_hill or best_area - current_area <= float(
                    settings.close_opposite_strong_area_db_oct
                ):
                    removable_pairs.append((best_area, best_reduced))
        if not removable_pairs:
            return result
        result = min(removable_pairs, key=lambda value: value[0])[1]


def _auto_prune_by_error_area(
    filters: list[IIRFilter],
    base_error: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    weights: np.ndarray,
    settings: AutoIIRSettings,
) -> list[IIRFilter]:
    result = list(filters)
    reference = auto_iir_objective_reference(base_error, frequency, weights)
    objective_weights = AutoIIRObjectiveWeights(
        one_octave_abs=float(settings.objective_one_octave_abs_weight),
        third_octave_abs=float(settings.objective_third_octave_abs_weight),
        perceptual_peak=float(settings.objective_perceptual_peak_weight),
        positive_peak_area=float(settings.objective_positive_peak_area_weight),
        center_shift=0.0,
    )
    while len(result) > 1:
        current_error = _auto_residual_from_base(base_error, result, sample_rate, frequency)
        current_area = _absolute_error_area(current_error, frequency, weights)
        current_rms = _auto_weighted_rms(current_error, weights)
        current_p95 = float(np.percentile(np.abs(current_error), 95.0))
        current_objective = evaluate_auto_iir_objective(
            current_error,
            frequency,
            weights,
            reference=reference,
            objective_weights=objective_weights,
        )
        current_under_peak = float(
            np.max(
                np.maximum(
                    fractional_octave_smooth(current_error, frequency, 1.0 / 3.0),
                    0.0,
                )
            )
        )
        removable: list[tuple[float, list[IIRFilter]]] = []
        for index in range(len(result)):
            reduced = result[:index] + result[index + 1 :]
            reduced = _optimize_auto_from_base_error(
                reduced,
                [],
                base_error,
                sample_rate,
                frequency,
                settings,
                max_nfev=60,
                multi_start=False,
                fc_limit_oct=float(settings.peq_fc_limit_oct),
                fc_bounds=(float(frequency[0]), float(frequency[-1])),
                fit_range=(float(frequency[0]), float(frequency[-1])),
                frequency_weights=weights,
            )
            reduced_error = _auto_residual_from_base(base_error, reduced, sample_rate, frequency)
            area_loss = _absolute_error_area(reduced_error, frequency, weights) - current_area
            rms_loss = _auto_weighted_rms(reduced_error, weights) - current_rms
            p95_loss = float(np.percentile(np.abs(reduced_error), 95.0)) - current_p95
            reduced_objective = evaluate_auto_iir_objective(
                reduced_error,
                frequency,
                weights,
                reference=reference,
                objective_weights=objective_weights,
            )
            objective_loss = reduced_objective.score - current_objective.score
            under_peak_increase = (
                float(
                    np.max(
                        np.maximum(
                            fractional_octave_smooth(
                                reduced_error,
                                frequency,
                                1.0 / 3.0,
                            ),
                            0.0,
                        )
                    )
                )
                - current_under_peak
            )
            allowed_area_loss = max(
                float(settings.peq_min_area_improvement_db_oct),
                current_area * 0.003,
            )
            if (
                area_loss <= allowed_area_loss
                and rms_loss <= 0.03
                and p95_loss <= 0.10
                and objective_loss <= float(settings.prune_objective_tolerance)
                and reduced_objective.perceptual_peak_db
                <= current_objective.perceptual_peak_db + 0.10
                and reduced_objective.positive_peak_area_db_oct
                <= current_objective.positive_peak_area_db_oct + 0.04
                and reduced_objective.under_target_area_db_oct
                <= current_objective.under_target_area_db_oct
                + float(settings.under_target_area_tolerance_db_oct)
                and under_peak_increase <= float(settings.under_target_peak_tolerance_db)
            ):
                removable.append((objective_loss, reduced))
        if not removable:
            break
        result = min(removable, key=lambda value: value[0])[1]
    return result


def _auto_common_center_references(
    filters: list[IIRFilter],
    base_error: np.ndarray,
    frequency: np.ndarray,
    weights: np.ndarray,
    settings: AutoIIRSettings,
    perceptual_level_db: np.ndarray | None,
) -> tuple[tuple[str, int, float], ...]:
    centers = _auto_center_references(
        filters,
        base_error,
        frequency,
        weights,
        settings,
        perceptual_level_db,
    )
    return tuple(
        (item.kind, int(np.sign(item.gain_db)), centers[index])
        for index, item in enumerate(filters)
        if index in centers
    )


def _auto_center_shift_from_common_references(
    filters: list[IIRFilter],
    references: tuple[tuple[str, int, float], ...],
) -> float:
    remaining = list(references)
    distances: list[float] = []
    for item in filters:
        if item.kind != "peq":
            continue
        log_fc = float(np.log2(max(item.fc, 1e-9)))
        sign = int(np.sign(item.gain_db))
        matches = [
            (abs(log_fc - center), index)
            for index, (kind, reference_sign, center) in enumerate(remaining)
            if kind == item.kind and reference_sign == sign
        ]
        if not matches:
            continue
        distance, index = min(matches)
        distances.append(float(distance))
        remaining.pop(index)
    return float(np.mean(distances)) if distances else 0.0


def _finalize_auto_iir_candidate(
    filters: list[IIRFilter],
    *,
    manual: list[IIRFilter],
    before_error: np.ndarray,
    peq_base_error: np.ndarray,
    speaker_gain: np.ndarray,
    target_gain: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    evaluation_weights: np.ndarray,
    core_weights: np.ndarray,
    core_mask: np.ndarray,
    guard_mask: np.ndarray,
    settings: AutoIIRSettings,
    perceptual_level_db: np.ndarray | None,
    fc_bounds: tuple[float, float],
) -> list[IIRFilter]:
    result = _auto_remove_close_opposite_pairs(
        list(filters),
        peq_base_error,
        sample_rate,
        frequency,
        core_weights,
        settings,
    )
    result = _auto_prune_by_error_area(
        result,
        peq_base_error,
        sample_rate,
        frequency,
        core_weights,
        settings,
    )
    if perceptual_level_db is not None:
        result = _cap_wavelet_peak_cut_q(
            result,
            frequency,
            perceptual_level_db,
            settings,
            response_error_db=peq_base_error,
        )
    result = _optimize_auto_filter_set_fixed_q(
        result,
        manual,
        speaker_gain,
        target_gain,
        sample_rate,
        frequency,
        settings,
        max_nfev=160,
        fc_limit_oct=float(settings.peq_fc_limit_oct),
        fc_bounds=fc_bounds,
        fit_range=fc_bounds,
        frequency_weights=evaluation_weights,
    )
    result = _enforce_final_guard(
        result,
        before_error,
        sample_rate,
        frequency,
        evaluation_weights,
        core_mask,
        guard_mask,
        settings,
    )
    return _enforce_auto_iir_filter_headroom(
        result,
        manual,
        sample_rate,
        frequency,
        settings,
    )


def _evaluate_final_auto_iir_candidate(
    source: str,
    filters: list[IIRFilter],
    *,
    manual: list[IIRFilter],
    speaker_gain: np.ndarray,
    target_gain: np.ndarray,
    sample_rate: int,
    frequency: np.ndarray,
    evaluation_weights: np.ndarray,
    core_weights: np.ndarray,
    core_mask: np.ndarray,
    before_smooth: np.ndarray,
    smoothing_oct: float,
    objective_reference: AutoIIRObjectiveReference,
    objective_weights: AutoIIRObjectiveWeights,
    center_references: tuple[tuple[str, int, float], ...],
) -> _AutoIIRFinalCandidate:
    error = _auto_error(
        speaker_gain,
        target_gain,
        manual,
        filters,
        sample_rate,
        frequency,
    )
    smooth = _auto_smoothed_error(error, frequency, float(smoothing_oct))
    center_shift = _auto_center_shift_from_common_references(filters, center_references)
    components = evaluate_auto_iir_objective(
        error,
        frequency,
        core_weights,
        reference=objective_reference,
        center_shift_oct=center_shift,
        objective_weights=objective_weights,
    )
    area = _masked_absolute_error_area(smooth, frequency, evaluation_weights, core_mask)
    rms = _masked_weighted_rms(smooth, evaluation_weights, core_mask)
    p95 = float(np.percentile(np.abs(smooth[core_mask]), 95.0))
    under_target_peak = float(
        np.max(
            np.maximum(
                fractional_octave_smooth(
                    error[core_mask],
                    frequency[core_mask],
                    1.0 / 3.0,
                ),
                0.0,
            )
        )
    )
    before_area = _masked_absolute_error_area(
        before_smooth,
        frequency,
        evaluation_weights,
        core_mask,
    )
    before_rms = _masked_weighted_rms(before_smooth, evaluation_weights, core_mask)
    before_p95 = float(np.percentile(np.abs(before_smooth[core_mask]), 95.0))
    final_score = (
        components.score
        + 0.35 * area / max(before_area, 0.25)
        + 0.15 * rms / max(before_rms, 0.25)
        + 0.10 * p95 / max(before_p95, 0.25)
    )
    return _AutoIIRFinalCandidate(
        source=source,
        filters=list(filters),
        score=float(final_score),
        one_octave_abs_db=components.one_octave_abs_db,
        third_octave_abs_db=components.third_octave_abs_db,
        contour_abs_db=components.contour_abs_db,
        perceptual_peak_db=components.perceptual_peak_db,
        positive_peak_area_db_oct=components.positive_peak_area_db_oct,
        under_target_area_db_oct=components.under_target_area_db_oct,
        under_target_peak_db=under_target_peak,
        center_shift_oct=components.center_shift_oct,
        error_area_db_oct=area,
        weighted_rms_db=rms,
        p95_abs_db=p95,
    )


def _select_final_auto_iir_candidate(
    candidates: list[_AutoIIRFinalCandidate],
    settings: AutoIIRSettings,
) -> _AutoIIRFinalCandidate:
    if not candidates:
        raise ValueError("Auto IIR final selection requires at least one candidate")
    legacy = candidates[0]
    eligible = [legacy]
    for candidate in candidates[1:]:
        peak_area_improvement = max(
            legacy.positive_peak_area_db_oct - candidate.positive_peak_area_db_oct,
            0.0,
        )
        allowed_under_area = (
            legacy.under_target_area_db_oct
            + float(settings.under_target_area_tolerance_db_oct)
            + float(settings.positive_peak_under_target_tradeoff) * peak_area_improvement
        )
        if (
            candidate.error_area_db_oct
            <= legacy.error_area_db_oct + max(0.01, legacy.error_area_db_oct * 0.001)
            and candidate.weighted_rms_db <= legacy.weighted_rms_db + 0.01
            and candidate.p95_abs_db <= legacy.p95_abs_db + 0.10
            and candidate.perceptual_peak_db <= legacy.perceptual_peak_db + 0.10
            and candidate.positive_peak_area_db_oct
            <= legacy.positive_peak_area_db_oct + 0.04
            and candidate.under_target_area_db_oct <= allowed_under_area
            and candidate.under_target_peak_db
            <= legacy.under_target_peak_db
            + float(settings.under_target_peak_tolerance_db)
        ):
            eligible.append(candidate)
    return min(
        eligible,
        key=lambda item: (
            item.score,
            item.error_area_db_oct,
            item.weighted_rms_db,
            item.perceptual_peak_db,
            item.positive_peak_area_db_oct,
            len(item.filters),
        ),
    )


def _optimize_auto_iir_impl(
    speaker: SpeakerResponse,
    target: SpeakerResponse,
    sample_rate: int,
    *,
    existing_filters: Iterable[IIRFilter] = (),
    settings: AutoIIRSettings = AutoIIRSettings(),
    perceptual_profile: AutoIIRPerceptualProfile | None = None,
) -> AutoIIRResult:
    """Design one fixed Shelf topology and fill the remaining budget with PEQs."""
    started_at = perf_counter()
    requested_f_min = max(float(settings.f_min), 1.0)
    requested_f_max = min(float(settings.f_max), float(sample_rate) / 2.0 - 1.0)
    if requested_f_max <= requested_f_min:
        raise ValueError("Auto IIR end frequency must be above start frequency")
    reliable_f_min, reliable_f_max = _auto_reliable_range(
        speaker,
        target,
        sample_rate,
        settings,
    )
    f_min = max(requested_f_min, reliable_f_min)
    f_max = min(requested_f_max, reliable_f_max)
    if f_max <= f_min:
        raise ValueError("Auto IIR Start/End does not overlap the reliable Input/Target band")
    manual = [
        replace(item, q=clamp_iir_q(item.kind, item.q), origin="manual")
        for item in existing_filters
        if item.enabled and item.origin != "auto"
    ]
    if settings.target_mask_enabled:
        target_mask_frequency = _adaptive_log_axis(
            reliable_f_min,
            reliable_f_max,
            points_per_octave=48.0,
            minimum=384,
            maximum=768,
        )
        target_mask_gain = _response_gain_on_log_axis(target, target_mask_frequency)
        target_mask_f_min, target_mask_f_max, _target_mask_reference_db = (
            _auto_gain_band_range(
                target_mask_frequency,
                target_mask_gain,
                f_min=f_min,
                f_max=f_max,
                limit_db=float(settings.target_mask_limit_db),
                smoothing_oct=float(settings.target_mask_smoothing_oct),
            )
        )
        f_min = max(f_min, target_mask_f_min)
        f_max = min(f_max, target_mask_f_max)
        if f_max <= f_min:
            raise ValueError(
                "Auto IIR has no correction band above the -40 dB Target mask"
            )
    independent_resolution = float(
        settings.independent_resolution_hz
        if settings.independent_resolution_hz is not None
        else _estimated_independent_resolution_hz(speaker)
    )
    runtime_settings = replace(
        settings,
        f_min=f_min,
        f_max=f_max,
        independent_resolution_hz=independent_resolution if independent_resolution > 0.0 else None,
    )
    correction_frequency = _adaptive_log_axis(
        f_min,
        f_max,
        points_per_octave=36.0,
        minimum=256,
        maximum=512,
    )
    evaluation_f_min, evaluation_f_max = reliable_f_min, reliable_f_max
    evaluation_frequency = _adaptive_log_axis(
        evaluation_f_min,
        evaluation_f_max,
        points_per_octave=float(settings.evaluation_points_per_octave),
        minimum=max(int(settings.evaluation_minimum_points), 2),
        maximum=max(
            int(settings.evaluation_maximum_points),
            int(settings.evaluation_minimum_points),
            2,
        ),
    )
    speaker_gain = _response_gain_on_log_axis(speaker, evaluation_frequency)
    target_gain = _response_gain_on_log_axis(target, evaluation_frequency)
    if perceptual_profile is None:
        evaluation_perceptual_level = _auto_smoothed_error(
            speaker_gain,
            evaluation_frequency,
            float(runtime_settings.wavelet_peak_smoothing_oct),
        )
        perceptual_profile_source = "response_1_6_oct_proxy"
    else:
        evaluation_perceptual_level = perceptual_level_on_axis(
            perceptual_profile,
            evaluation_frequency,
        )
        perceptual_profile_source = str(perceptual_profile.source_type)
    correction_perceptual_level = np.interp(
        np.log2(correction_frequency),
        np.log2(evaluation_frequency),
        evaluation_perceptual_level,
    )
    settings = runtime_settings
    evaluation_weights = _auto_frequency_weights(
        evaluation_frequency,
        settings,
        perceptual_level_db=evaluation_perceptual_level,
    )
    correction_weights = _auto_frequency_weights(
        correction_frequency,
        settings,
        perceptual_level_db=correction_perceptual_level,
    )
    correction_audibility = relative_audibility_weights(
        correction_perceptual_level,
        threshold_db=float(settings.audibility_threshold_db),
        transition_db=float(settings.audibility_transition_db),
        minimum_weight=float(settings.audibility_min_weight),
        reference_percentile=float(settings.audibility_reference_percentile),
    )
    core_mask = (
        (evaluation_frequency >= f_min * (1.0 - 1e-10))
        & (evaluation_frequency <= f_max * (1.0 + 1e-10))
    )
    guard_mask = ~core_mask
    core_weights = evaluation_weights * core_mask.astype(float)
    before_error = _auto_error(
        speaker_gain,
        target_gain,
        manual,
        [],
        sample_rate,
        evaluation_frequency,
    )
    shelf_base_error = _masked_shelf_trend(before_error, evaluation_frequency, settings)
    peq_base_error = _auto_smoothed_error(
        before_error,
        evaluation_frequency,
        float(settings.peq_smoothing_oct),
    )
    correction_shelf_base_error = np.interp(
        np.log2(correction_frequency),
        np.log2(evaluation_frequency),
        shelf_base_error,
    )
    correction_peq_base_error = np.interp(
        np.log2(correction_frequency),
        np.log2(evaluation_frequency),
        peq_base_error,
    )
    total_budget = _auto_total_filter_budget(settings)
    shelves: list[IIRFilter] = []
    peqs: list[IIRFilter] = []
    stop_reason = "No useful candidate above threshold"
    global_search_result: SobolOptunaResult | None = None
    optimizer_mode = str(settings.optimizer_mode or "legacy")
    selected_final_candidate: _AutoIIRFinalCandidate | None = None
    optimizer_final_candidate_count = 0
    optimizer_selected_source = "none"

    # Stage 1: fit broad trends only. Narrow local peaks are bridged before a
    # shelf is evaluated, so a shelf cannot win merely by leaning into one hill.
    unused_shelf_kinds = []
    if settings.include_low_shelf:
        unused_shelf_kinds.append("low_shelf")
    if settings.include_high_shelf:
        unused_shelf_kinds.append("high_shelf")
    while unused_shelf_kinds and len(shelves) < total_budget:
        correction_error = _auto_residual_from_base(
            correction_shelf_base_error,
            shelves,
            sample_rate,
            correction_frequency,
        )
        evaluation_error = _auto_residual_from_base(
            shelf_base_error,
            shelves,
            sample_rate,
            evaluation_frequency,
        )
        before_area = _masked_absolute_error_area(
            evaluation_error,
            evaluation_frequency,
            evaluation_weights,
            core_mask,
        )
        correction_area = _absolute_error_area(
            correction_error,
            correction_frequency,
            correction_weights,
        )
        evaluated: list[tuple[float, list[IIRFilter], str]] = []
        for kind in unused_shelf_kinds:
            candidate = _best_area_shelf_candidate(
                kind,
                correction_error,
                correction_frequency,
                sample_rate,
                settings,
                weights=correction_weights,
            )
            if candidate is None:
                continue
            trial = _optimize_auto_from_base_error(
                [*shelves, candidate.item],
                [],
                shelf_base_error,
                sample_rate,
                evaluation_frequency,
                settings,
                max_nfev=100,
                multi_start=True,
                fc_limit_oct=1.0,
                fc_bounds=(f_min, f_max),
                fit_range=(f_min, f_max),
                frequency_weights=evaluation_weights,
            )
            trial_error = _auto_residual_from_base(
                shelf_base_error,
                trial,
                sample_rate,
                evaluation_frequency,
            )
            improvement = before_area - _masked_absolute_error_area(
                trial_error,
                evaluation_frequency,
                evaluation_weights,
                core_mask,
            )
            if not _guard_is_safe(
                evaluation_error,
                trial_error,
                evaluation_frequency,
                evaluation_weights,
                guard_mask,
                settings,
            ):
                continue
            evaluated.append((improvement, trial, kind))
        if not evaluated:
            break
        improvement, trial, kind = max(evaluated, key=lambda value: value[0])
        threshold = max(
            float(settings.shelf_min_area_improvement_db_oct),
            correction_area * 0.0075,
        )
        if improvement < threshold:
            break
        shelves = trial
        unused_shelf_kinds.remove(kind)
        stop_reason = "Shelf trend stage completed"

    # Stage 2: fit the 1/12-octave residual. Every candidate represents one
    # visible hill, and Shelf + PEQ together may never exceed the total budget.
    while _auto_peq_is_enabled(settings) and len(shelves) + len(peqs) < total_budget:
        current_filters = [*shelves, *peqs]
        correction_error = _auto_residual_from_base(
            correction_peq_base_error,
            current_filters,
            sample_rate,
            correction_frequency,
        )
        evaluation_error = _auto_residual_from_base(
            peq_base_error,
            current_filters,
            sample_rate,
            evaluation_frequency,
        )
        current_area = _masked_absolute_error_area(
            evaluation_error,
            evaluation_frequency,
            evaluation_weights,
            core_mask,
        )
        correction_area = _absolute_error_area(
            correction_error,
            correction_frequency,
            correction_weights,
        )
        current_rms = _masked_weighted_rms(evaluation_error, evaluation_weights, core_mask)
        current_p95 = float(np.percentile(np.abs(evaluation_error[core_mask]), 95.0))
        current_peak_metrics = positive_peak_metrics(
            evaluation_error[core_mask],
            evaluation_frequency[core_mask],
            evaluation_weights[core_mask],
        )
        current_under_peak = float(
            np.max(
                np.maximum(
                    fractional_octave_smooth(
                        evaluation_error[core_mask],
                        evaluation_frequency[core_mask],
                        1.0 / 3.0,
                    ),
                    0.0,
                )
            )
        )
        candidates = _auto_peq_candidates(
            correction_error,
            correction_frequency,
            settings,
            peqs,
            audibility_weights=correction_audibility,
            perceptual_level_db=(
                correction_perceptual_level if perceptual_profile is not None else None
            ),
        )
        if not candidates:
            stop_reason = "No 1/12-octave hill above threshold"
            break
        evaluated_peqs: list[tuple[float, float, list[IIRFilter]]] = []
        for candidate in _auto_candidate_shortlist(candidates, settings):
            # Keep already accepted PEQs fixed while evaluating the next hill.
            # Jointly moving their Q/Fc here can create a new shoulder dip and
            # incorrectly reject an otherwise useful second peak. The complete
            # set is still jointly optimized after topology selection.
            optimized_new = _optimize_auto_from_base_error(
                [candidate.item],
                [*shelves, *peqs],
                peq_base_error,
                sample_rate,
                evaluation_frequency,
                settings,
                max_nfev=100,
                multi_start=len(peqs) == 0,
                fc_limit_oct=float(settings.peq_fc_limit_oct),
                fc_bounds=(f_min, f_max),
                fit_range=(f_min, f_max),
                frequency_weights=evaluation_weights,
            )
            trial_peqs = [*peqs, *optimized_new]
            if perceptual_profile is not None:
                trial_peqs = _cap_wavelet_peak_cut_q(
                    trial_peqs,
                    evaluation_frequency,
                    evaluation_perceptual_level,
                    settings,
                    response_error_db=peq_base_error,
                )
            if len(trial_peqs) != len(peqs) + 1:
                continue
            if trial_peqs[-1].gain_db < 0.0:
                # Reduce a new cut until its 1/3-oct shoulders stay within the
                # 0.25 dB dip tolerance and the added under-Target area stays
                # within 0.05 dB-oct. This is evaluated against the already
                # accepted filters, not against an idealized isolated PEQ.
                balanced_trial: list[IIRFilter] | None = None
                for scale in np.linspace(1.0, 0.1, 10):
                    scaled = [
                        *trial_peqs[:-1],
                        replace(
                            trial_peqs[-1],
                            gain_db=float(trial_peqs[-1].gain_db) * float(scale),
                        ),
                    ]
                    scaled_error = _auto_residual_from_base(
                        peq_base_error,
                        [*shelves, *scaled],
                        sample_rate,
                        evaluation_frequency,
                    )
                    scaled_metrics = positive_peak_metrics(
                        scaled_error[core_mask],
                        evaluation_frequency[core_mask],
                        evaluation_weights[core_mask],
                    )
                    scaled_under_peak = float(
                        np.max(
                            np.maximum(
                                fractional_octave_smooth(
                                    scaled_error[core_mask],
                                    evaluation_frequency[core_mask],
                                    1.0 / 3.0,
                                ),
                                0.0,
                            )
                        )
                    )
                    if _peak_cut_balance_is_safe(
                        current_peak_metrics,
                        scaled_metrics,
                        current_under_peak,
                        scaled_under_peak,
                        settings,
                    ):
                        balanced_trial = scaled
                        break
                if balanced_trial is None:
                    continue
                trial_peqs = balanced_trial
            trial_filters = [*shelves, *trial_peqs]
            trial_error = _auto_residual_from_base(
                peq_base_error,
                trial_filters,
                sample_rate,
                evaluation_frequency,
            )
            trial_area = _masked_absolute_error_area(
                trial_error,
                evaluation_frequency,
                evaluation_weights,
                core_mask,
            )
            area_improvement = current_area - trial_area
            rms_improvement = current_rms - _masked_weighted_rms(
                trial_error,
                evaluation_weights,
                core_mask,
            )
            p95_improvement = current_p95 - float(np.percentile(np.abs(trial_error[core_mask]), 95.0))
            trial_peak_metrics = positive_peak_metrics(
                trial_error[core_mask],
                evaluation_frequency[core_mask],
                evaluation_weights[core_mask],
            )
            peak_improvement = (
                current_peak_metrics.height_db - trial_peak_metrics.height_db
            )
            peak_area_improvement = (
                current_peak_metrics.area_db_oct - trial_peak_metrics.area_db_oct
            )
            trial_under_peak = float(
                np.max(
                    np.maximum(
                        fractional_octave_smooth(
                            trial_error[core_mask],
                            evaluation_frequency[core_mask],
                            1.0 / 3.0,
                        ),
                        0.0,
                    )
                )
            )
            if not _guard_is_safe(
                evaluation_error,
                trial_error,
                evaluation_frequency,
                evaluation_weights,
                guard_mask,
                settings,
            ):
                continue
            new_item = trial_peqs[-1]
            close_opposite = any(
                np.sign(item.gain_db) != np.sign(new_item.gain_db)
                and abs(np.log2(item.fc / new_item.fc))
                < float(settings.close_opposite_spacing_oct)
                for item in peqs
            )
            if close_opposite and area_improvement < float(
                settings.close_opposite_strong_area_db_oct
            ):
                continue
            threshold = max(
                float(settings.peq_min_area_improvement_db_oct),
                correction_area * 0.004,
            )
            peak_worsening_tolerance = (
                0.10
                if new_item.gain_db < 0.0
                else float(settings.wavelet_peak_min_prominence_db)
            )
            if (
                area_improvement < threshold
                or rms_improvement < -0.01
                or p95_improvement < -0.10
                or peak_improvement < -peak_worsening_tolerance
                or not _peak_cut_balance_is_safe(
                    current_peak_metrics,
                    trial_peak_metrics,
                    current_under_peak,
                    trial_under_peak,
                    settings,
                )
            ):
                continue
            cut_factor = 1.35 if new_item.gain_db < 0.0 else 0.75
            quality = (
                cut_factor * float(settings.selection_area_weight) * area_improvement
                + float(settings.selection_rms_weight) * rms_improvement
                + float(settings.selection_peak_weight)
                * (0.65 * peak_improvement + 0.35 * peak_area_improvement)
                + 0.15 * p95_improvement
                + 0.25 * float(candidate.priority)
            )
            evaluated_peqs.append((quality, trial_area, trial_peqs))
        if not evaluated_peqs:
            stop_reason = "Remaining hills did not improve area, RMS, and peak score"
            break
        _quality, _area, peqs = max(evaluated_peqs, key=lambda value: value[0])
        stop_reason = "Total useful filter budget reached"

    generated = [*shelves, *peqs]
    # Keep the staged/legacy route as an explicit final candidate. Global
    # leaders are polished, Q-quantized, guarded, and scored with the same
    # metrics shown in the UI before any one of them can replace it.
    if generated:
        current_error = _auto_residual_from_base(
            peq_base_error,
            generated,
            sample_rate,
            evaluation_frequency,
        )
        current_area = _masked_absolute_error_area(
            current_error,
            evaluation_frequency,
            evaluation_weights,
            core_mask,
        )
        legacy_joint = _optimize_auto_from_base_error(
            generated,
            [],
            peq_base_error,
            sample_rate,
            evaluation_frequency,
            settings,
            max_nfev=160,
            multi_start=True,
            fc_limit_oct=float(settings.peq_fc_limit_oct),
            fc_bounds=(f_min, f_max),
            fit_range=(f_min, f_max),
            frequency_weights=evaluation_weights,
        )
        legacy_error = _auto_residual_from_base(
            peq_base_error,
            legacy_joint,
            sample_rate,
            evaluation_frequency,
        )
        legacy_area = _masked_absolute_error_area(
            legacy_error,
            evaluation_frequency,
            evaluation_weights,
            core_mask,
        )
        current_peaks = positive_peak_metrics(
            current_error[core_mask],
            evaluation_frequency[core_mask],
            evaluation_weights[core_mask],
        )
        legacy_peaks = positive_peak_metrics(
            legacy_error[core_mask],
            evaluation_frequency[core_mask],
            evaluation_weights[core_mask],
        )
        current_under_peak = float(
            np.max(
                np.maximum(
                    fractional_octave_smooth(
                        current_error[core_mask],
                        evaluation_frequency[core_mask],
                        1.0 / 3.0,
                    ),
                    0.0,
                )
            )
        )
        legacy_under_peak = float(
            np.max(
                np.maximum(
                    fractional_octave_smooth(
                        legacy_error[core_mask],
                        evaluation_frequency[core_mask],
                        1.0 / 3.0,
                    ),
                    0.0,
                )
            )
        )
        legacy_seed = list(generated)
        if (
            legacy_area < current_area - 1e-6
            and legacy_peaks.height_db <= current_peaks.height_db + 0.10
            and legacy_peaks.area_db_oct <= current_peaks.area_db_oct + 0.04
            and _peak_cut_balance_is_safe(
                current_peaks,
                legacy_peaks,
                current_under_peak,
                legacy_under_peak,
                settings,
            )
            and _guard_is_safe(
                current_error,
                legacy_error,
                evaluation_frequency,
                evaluation_weights,
                guard_mask,
                settings,
            )
        ):
            legacy_seed = legacy_joint

        global_candidates: list[tuple[str, list[IIRFilter]]] = []
        if optimizer_mode == "sobol_optuna":
            try:
                global_candidates, global_search_result = _global_optimize_auto_from_base_error(
                    legacy_joint,
                    peq_base_error,
                    sample_rate,
                    evaluation_frequency,
                    settings,
                    core_weights=core_weights,
                    guard_weights=evaluation_weights,
                    core_mask=core_mask,
                    guard_mask=guard_mask,
                    perceptual_level_db=(
                        evaluation_perceptual_level
                        if perceptual_profile is not None
                        else None
                    ),
                    fc_limit_oct=float(settings.peq_fc_limit_oct),
                    fc_bounds=(f_min, f_max),
                    headroom_fixed_filters=manual,
                )
            except (FloatingPointError, ValueError):
                optimizer_mode = "sobol_optuna_fallback"

        before_smooth_for_selection = _auto_smoothed_error(
            before_error,
            evaluation_frequency,
            float(settings.peq_smoothing_oct),
        )
        objective_reference_for_selection = auto_iir_objective_reference(
            before_error,
            evaluation_frequency,
            core_weights,
        )
        objective_weights_for_selection = AutoIIRObjectiveWeights(
            one_octave_abs=float(settings.objective_one_octave_abs_weight),
            third_octave_abs=float(settings.objective_third_octave_abs_weight),
            perceptual_peak=float(settings.objective_perceptual_peak_weight),
            positive_peak_area=float(settings.objective_positive_peak_area_weight),
            center_shift=float(settings.objective_center_shift_weight),
        )
        center_references = _auto_common_center_references(
            legacy_joint,
            peq_base_error,
            evaluation_frequency,
            core_weights,
            settings,
            evaluation_perceptual_level if perceptual_profile is not None else None,
        )
        final_candidates: list[_AutoIIRFinalCandidate] = []
        seen: set[tuple[tuple[object, ...], ...]] = set()
        for source, seed_filters in [("legacy", legacy_seed), *global_candidates]:
            finalized = _finalize_auto_iir_candidate(
                seed_filters,
                manual=manual,
                before_error=before_error,
                peq_base_error=peq_base_error,
                speaker_gain=speaker_gain,
                target_gain=target_gain,
                sample_rate=sample_rate,
                frequency=evaluation_frequency,
                evaluation_weights=evaluation_weights,
                core_weights=core_weights,
                core_mask=core_mask,
                guard_mask=guard_mask,
                settings=settings,
                perceptual_level_db=(
                    evaluation_perceptual_level if perceptual_profile is not None else None
                ),
                fc_bounds=(f_min, f_max),
            )
            signature = tuple(
                (
                    item.kind,
                    round(float(item.fc), 6),
                    round(float(item.q), 6),
                    round(float(item.gain_db), 6),
                )
                for item in finalized
            )
            if signature in seen:
                continue
            seen.add(signature)
            final_candidates.append(
                _evaluate_final_auto_iir_candidate(
                    source,
                    finalized,
                    manual=manual,
                    speaker_gain=speaker_gain,
                    target_gain=target_gain,
                    sample_rate=sample_rate,
                    frequency=evaluation_frequency,
                    evaluation_weights=evaluation_weights,
                    core_weights=core_weights,
                    core_mask=core_mask,
                    before_smooth=before_smooth_for_selection,
                    smoothing_oct=float(settings.peq_smoothing_oct),
                    objective_reference=objective_reference_for_selection,
                    objective_weights=objective_weights_for_selection,
                    center_references=center_references,
                )
            )
        selected_final_candidate = _select_final_auto_iir_candidate(final_candidates, settings)
        generated = selected_final_candidate.filters
        optimizer_final_candidate_count = len(final_candidates)
        optimizer_selected_source = selected_final_candidate.source

    combined_filter_gain = _auto_filter_gain(
        [*manual, *generated],
        sample_rate,
        evaluation_frequency,
    )
    after_error = target_gain - speaker_gain - combined_filter_gain
    before_smooth = _auto_smoothed_error(
        before_error,
        evaluation_frequency,
        float(settings.peq_smoothing_oct),
    )
    after_smooth = _auto_smoothed_error(
        after_error,
        evaluation_frequency,
        float(settings.peq_smoothing_oct),
    )
    filter_headroom_db = max(float(np.max(combined_filter_gain)), 0.0)
    objective_reference = auto_iir_objective_reference(
        before_error,
        evaluation_frequency,
        core_weights,
    )
    optimizer_center_shift_oct = (
        0.0
        if selected_final_candidate is None
        else float(selected_final_candidate.center_shift_oct)
    )
    after_objective = evaluate_auto_iir_objective(
        after_error,
        evaluation_frequency,
        core_weights,
        reference=objective_reference,
        center_shift_oct=optimizer_center_shift_oct,
        objective_weights=AutoIIRObjectiveWeights(
            one_octave_abs=float(settings.objective_one_octave_abs_weight),
            third_octave_abs=float(settings.objective_third_octave_abs_weight),
            perceptual_peak=float(settings.objective_perceptual_peak_weight),
            positive_peak_area=float(settings.objective_positive_peak_area_weight),
            center_shift=float(settings.objective_center_shift_weight),
        ),
    )
    outside_mask = guard_mask
    shelf_count = sum(item.kind in {"low_shelf", "high_shelf"} for item in generated)
    peq_count = sum(item.kind == "peq" for item in generated)
    diagnostics = AutoIIRDiagnostics(
        requested_f_min_hz=requested_f_min,
        requested_f_max_hz=requested_f_max,
        correction_f_min_hz=f_min,
        correction_f_max_hz=f_max,
        reliable_f_min_hz=reliable_f_min,
        reliable_f_max_hz=reliable_f_max,
        evaluation_f_min_hz=evaluation_f_min,
        evaluation_f_max_hz=evaluation_f_max,
        independent_resolution_hz=max(independent_resolution, 0.0),
        before_weighted_rms_db=_masked_weighted_rms(before_smooth, evaluation_weights, core_mask),
        after_weighted_rms_db=_masked_weighted_rms(after_smooth, evaluation_weights, core_mask),
        before_raw_weighted_rms_db=_masked_weighted_rms(before_error, evaluation_weights, core_mask),
        after_raw_weighted_rms_db=_masked_weighted_rms(after_error, evaluation_weights, core_mask),
        before_error_area_db_oct=_masked_absolute_error_area(
            before_smooth,
            evaluation_frequency,
            evaluation_weights,
            core_mask,
        ),
        after_error_area_db_oct=_masked_absolute_error_area(
            after_smooth,
            evaluation_frequency,
            evaluation_weights,
            core_mask,
        ),
        before_outside_error_area_db_oct=_masked_absolute_error_area(
            before_smooth,
            evaluation_frequency,
            evaluation_weights,
            outside_mask,
        ),
        after_outside_error_area_db_oct=_masked_absolute_error_area(
            after_smooth,
            evaluation_frequency,
            evaluation_weights,
            outside_mask,
        ),
        before_p95_abs_db=float(np.percentile(np.abs(before_smooth[core_mask]), 95.0)),
        after_p95_abs_db=float(np.percentile(np.abs(after_smooth[core_mask]), 95.0)),
        before_p99_abs_db=float(np.percentile(np.abs(before_smooth[core_mask]), 99.0)),
        after_p99_abs_db=float(np.percentile(np.abs(after_smooth[core_mask]), 99.0)),
        before_max_abs_db=float(np.max(np.abs(before_error))),
        after_max_abs_db=float(np.max(np.abs(after_error))),
        before_one_octave_abs_db=objective_reference.one_octave_abs_db,
        after_one_octave_abs_db=after_objective.one_octave_abs_db,
        before_third_octave_abs_db=objective_reference.third_octave_abs_db,
        after_third_octave_abs_db=after_objective.third_octave_abs_db,
        before_contour_abs_db=objective_reference.contour_abs_db,
        after_contour_abs_db=after_objective.contour_abs_db,
        before_perceptual_peak_db=objective_reference.perceptual_peak_db,
        after_perceptual_peak_db=after_objective.perceptual_peak_db,
        before_positive_peak_area_db_oct=objective_reference.positive_peak_area_db_oct,
        after_positive_peak_area_db_oct=after_objective.positive_peak_area_db_oct,
        before_under_target_area_db_oct=objective_reference.under_target_area_db_oct,
        after_under_target_area_db_oct=after_objective.under_target_area_db_oct,
        before_under_target_peak_db=float(
            np.max(
                np.maximum(
                    fractional_octave_smooth(
                        before_error[core_mask],
                        evaluation_frequency[core_mask],
                        1.0 / 3.0,
                    ),
                    0.0,
                )
            )
        ),
        after_under_target_peak_db=float(
            np.max(
                np.maximum(
                    fractional_octave_smooth(
                        after_error[core_mask],
                        evaluation_frequency[core_mask],
                        1.0 / 3.0,
                    ),
                    0.0,
                )
            )
        ),
        audibility_threshold_db=float(settings.audibility_threshold_db),
        audibility_min_weight=float(settings.audibility_min_weight),
        perceptual_profile_source=perceptual_profile_source,
        max_peq_q=min(float(settings.max_peq_q), float(IIR_PEQ_Q_MAX)),
        wavelet_peak_smoothing_oct=float(settings.wavelet_peak_smoothing_oct),
        optimizer_mode=optimizer_mode,
        optimizer_seed=int(settings.optimizer_seed),
        optimizer_sobol_trials=(
            0 if global_search_result is None else global_search_result.sobol_trials
        ),
        optimizer_optuna_trials=(
            0 if global_search_result is None else global_search_result.optuna_trials
        ),
        optimizer_completed_trials=(
            0 if global_search_result is None else global_search_result.completed_trials
        ),
        optimizer_feasible_trials=(
            0 if global_search_result is None else global_search_result.feasible_trials
        ),
        optimizer_best_score=(
            0.0
            if selected_final_candidate is None
            else float(selected_final_candidate.score)
        ),
        optimizer_center_shift_oct=optimizer_center_shift_oct,
        optimizer_final_candidate_count=optimizer_final_candidate_count,
        optimizer_selected_source=optimizer_selected_source,
        filter_headroom_db=filter_headroom_db,
        filter_headroom_limit_db=float(settings.filter_headroom_limit_db),
        filter_headroom_ok=bool(
            filter_headroom_db <= float(settings.filter_headroom_limit_db) + 1e-6
        ),
        total_filter_budget=total_budget,
        generated_filter_count=len(generated),
        peq_count=peq_count,
        shelf_count=shelf_count,
        low_shelf_allowed=bool(settings.include_low_shelf),
        high_shelf_allowed=bool(settings.include_high_shelf),
        low_shelf_selected=any(item.kind == "low_shelf" for item in generated),
        high_shelf_selected=any(item.kind == "high_shelf" for item in generated),
        shelf_topology_candidate_count=1,
        optimization_seconds=perf_counter() - started_at,
        stop_reason=stop_reason,
    )
    return AutoIIRResult(filters=generated, diagnostics=diagnostics)


def _auto_iir_shelf_topology_flags(settings: AutoIIRSettings) -> list[tuple[bool, bool]]:
    """Return every Shelf topology allowed by the UI switches, baseline first."""
    variants = [(False, False)]
    if settings.include_low_shelf:
        variants.append((True, False))
    if settings.include_high_shelf:
        variants.append((False, True))
    if settings.include_low_shelf and settings.include_high_shelf:
        variants.append((True, True))
    return variants


def _auto_iir_diagnostics_selection_score(diagnostics: AutoIIRDiagnostics) -> float:
    """Compare topology results with normalized versions of the visible metrics."""

    def ratio(after: float, before: float, floor: float) -> float:
        return float(after) / max(float(before), float(floor))

    score = (
        0.24
        * ratio(
            diagnostics.after_error_area_db_oct,
            diagnostics.before_error_area_db_oct,
            0.25,
        )
        + 0.18
        * ratio(
            diagnostics.after_weighted_rms_db,
            diagnostics.before_weighted_rms_db,
            0.10,
        )
        + 0.08
        * ratio(diagnostics.after_p95_abs_db, diagnostics.before_p95_abs_db, 0.10)
        + 0.12
        * ratio(
            diagnostics.after_one_octave_abs_db,
            diagnostics.before_one_octave_abs_db,
            0.10,
        )
        + 0.14
        * ratio(
            diagnostics.after_third_octave_abs_db,
            diagnostics.before_third_octave_abs_db,
            0.10,
        )
        + 0.14
        * ratio(
            diagnostics.after_perceptual_peak_db,
            diagnostics.before_perceptual_peak_db,
            0.10,
        )
        + 0.10
        * ratio(
            diagnostics.after_positive_peak_area_db_oct,
            diagnostics.before_positive_peak_area_db_oct,
            0.05,
        )
    )
    under_area_increase = max(
        diagnostics.after_under_target_area_db_oct
        - diagnostics.before_under_target_area_db_oct,
        0.0,
    )
    under_peak_increase = max(
        diagnostics.after_under_target_peak_db - diagnostics.before_under_target_peak_db,
        0.0,
    )
    return float(
        score
        + 0.10
        * under_area_increase
        / max(diagnostics.before_under_target_area_db_oct, 0.25)
        + 0.10
        * under_peak_increase
        / max(diagnostics.before_under_target_peak_db, 0.25)
    )


def _auto_iir_shelf_result_is_safe(
    candidate: AutoIIRResult,
    baseline: AutoIIRResult,
    settings: AutoIIRSettings,
) -> bool:
    """Prevent an allowed Shelf topology from degrading the no-Shelf result."""
    trial = candidate.diagnostics
    reference = baseline.diagnostics
    peak_area_improvement = max(
        reference.after_positive_peak_area_db_oct
        - trial.after_positive_peak_area_db_oct,
        0.0,
    )
    allowed_under_area = (
        reference.after_under_target_area_db_oct
        + float(settings.under_target_area_tolerance_db_oct)
        + float(settings.positive_peak_under_target_tradeoff) * peak_area_improvement
    )
    return bool(
        trial.after_error_area_db_oct
        <= reference.after_error_area_db_oct
        + max(0.01, reference.after_error_area_db_oct * 0.001)
        and trial.after_weighted_rms_db <= reference.after_weighted_rms_db + 0.01
        and trial.after_p95_abs_db <= reference.after_p95_abs_db + 0.10
        and trial.after_perceptual_peak_db
        <= reference.after_perceptual_peak_db + 0.10
        and trial.after_positive_peak_area_db_oct
        <= reference.after_positive_peak_area_db_oct + 0.04
        and trial.after_under_target_area_db_oct <= allowed_under_area
        and trial.after_under_target_peak_db
        <= reference.after_under_target_peak_db
        + float(settings.under_target_peak_tolerance_db)
    )


def _select_auto_iir_shelf_result(
    baseline: AutoIIRResult,
    candidates: Iterable[AutoIIRResult],
    settings: AutoIIRSettings,
) -> AutoIIRResult:
    """Keep the no-Shelf result unless a Shelf result is safe and scores better."""
    eligible = [baseline]
    for candidate in candidates:
        if _auto_iir_shelf_result_is_safe(candidate, baseline, settings):
            eligible.append(candidate)
    return min(
        eligible,
        key=lambda result: (
            _auto_iir_diagnostics_selection_score(result.diagnostics),
            result.diagnostics.after_error_area_db_oct,
            result.diagnostics.after_weighted_rms_db,
            result.diagnostics.after_perceptual_peak_db,
            result.diagnostics.after_positive_peak_area_db_oct,
            len(result.filters),
        ),
    )


def _optimize_auto_iir_at_resolution(
    speaker: SpeakerResponse,
    target: SpeakerResponse,
    sample_rate: int,
    *,
    existing_filters: Iterable[IIRFilter] = (),
    settings: AutoIIRSettings = AutoIIRSettings(),
    perceptual_profile: AutoIIRPerceptualProfile | None = None,
) -> AutoIIRResult:
    """Design Auto IIR at the evaluation resolution stored in settings."""
    started_at = perf_counter()
    existing = tuple(existing_filters)
    variants = _auto_iir_shelf_topology_flags(settings)

    if len(variants) == 1:
        return _optimize_auto_iir_impl(
            speaker,
            target,
            sample_rate,
            existing_filters=existing,
            settings=settings,
            perceptual_profile=perceptual_profile,
        )

    # Topology previews are intentionally Legacy-only. They cheaply determine
    # which optional Shelf layout is worth a full Sobol/Optuna run.
    preview_results: dict[tuple[bool, bool], AutoIIRResult] = {}
    preview_settings = replace(settings, optimizer_mode="legacy")
    for low_shelf, high_shelf in variants:
        preview_results[(low_shelf, high_shelf)] = _optimize_auto_iir_impl(
            speaker,
            target,
            sample_rate,
            existing_filters=existing,
            settings=replace(
                preview_settings,
                include_low_shelf=low_shelf,
                include_high_shelf=high_shelf,
            ),
            perceptual_profile=perceptual_profile,
        )

    preview_baseline = preview_results[(False, False)]
    preview_shelf = _select_auto_iir_shelf_result(
        preview_baseline,
        (result for flags, result in preview_results.items() if flags != (False, False)),
        settings,
    )
    selected_preview_flags = (
        preview_shelf.diagnostics.low_shelf_allowed,
        preview_shelf.diagnostics.high_shelf_allowed,
    )

    # Always optimize the no-Shelf baseline with the requested optimizer. If a
    # Shelf topology survived the preview guards, optimize it too and compare
    # the actual final results. Thus switching Shelf ON cannot make the result
    # worse merely because a Shelf consumed a PEQ slot.
    if str(settings.optimizer_mode or "legacy") == "legacy":
        final_baseline = preview_baseline
    else:
        final_baseline = _optimize_auto_iir_impl(
            speaker,
            target,
            sample_rate,
            existing_filters=existing,
            settings=replace(
                settings,
                include_low_shelf=False,
                include_high_shelf=False,
            ),
            perceptual_profile=perceptual_profile,
        )
    final_candidates: list[AutoIIRResult] = []
    if selected_preview_flags != (False, False):
        if str(settings.optimizer_mode or "legacy") == "legacy":
            final_candidates.append(preview_shelf)
        else:
            final_candidates.append(
                _optimize_auto_iir_impl(
                    speaker,
                    target,
                    sample_rate,
                    existing_filters=existing,
                    settings=replace(
                        settings,
                        include_low_shelf=selected_preview_flags[0],
                        include_high_shelf=selected_preview_flags[1],
                    ),
                    perceptual_profile=perceptual_profile,
                )
            )
    selected = _select_auto_iir_shelf_result(final_baseline, final_candidates, settings)
    diagnostics = replace(
        selected.diagnostics,
        total_filter_budget=_auto_total_filter_budget(settings),
        low_shelf_allowed=bool(settings.include_low_shelf),
        high_shelf_allowed=bool(settings.include_high_shelf),
        low_shelf_selected=any(item.kind == "low_shelf" for item in selected.filters),
        high_shelf_selected=any(item.kind == "high_shelf" for item in selected.filters),
        shelf_topology_candidate_count=len(variants),
        optimization_seconds=perf_counter() - started_at,
    )
    return replace(selected, diagnostics=diagnostics)


def _reevaluate_auto_iir_result(
    result: AutoIIRResult,
    speaker: SpeakerResponse,
    target: SpeakerResponse,
    sample_rate: int,
    *,
    existing_filters: Iterable[IIRFilter],
    settings: AutoIIRSettings,
    perceptual_profile: AutoIIRPerceptualProfile | None,
    policy: AutoIIRResolutionPolicy,
) -> AutoIIRResult:
    """Recompute visible diagnostics once on the full validation axis."""

    diagnostics = result.diagnostics
    frequency = _adaptive_log_axis(
        diagnostics.evaluation_f_min_hz,
        diagnostics.evaluation_f_max_hz,
        points_per_octave=float(policy.validation_points_per_octave),
        minimum=int(policy.validation_minimum_points),
        maximum=int(policy.validation_maximum_points),
    )
    runtime_settings = replace(
        settings,
        f_min=float(diagnostics.correction_f_min_hz),
        f_max=float(diagnostics.correction_f_max_hz),
        independent_resolution_hz=(
            float(diagnostics.independent_resolution_hz)
            if diagnostics.independent_resolution_hz > 0.0
            else None
        ),
    )
    speaker_gain = _response_gain_on_log_axis(speaker, frequency)
    target_gain = _response_gain_on_log_axis(target, frequency)
    if perceptual_profile is None:
        perceptual_level = _auto_smoothed_error(
            speaker_gain,
            frequency,
            float(runtime_settings.wavelet_peak_smoothing_oct),
        )
    else:
        perceptual_level = perceptual_level_on_axis(perceptual_profile, frequency)
    weights = _auto_frequency_weights(
        frequency,
        runtime_settings,
        perceptual_level_db=perceptual_level,
    )
    core_mask = (
        (frequency >= float(diagnostics.correction_f_min_hz) * (1.0 - 1e-10))
        & (frequency <= float(diagnostics.correction_f_max_hz) * (1.0 + 1e-10))
    )
    guard_mask = ~core_mask
    core_weights = weights * core_mask.astype(float)
    manual = [
        replace(item, q=clamp_iir_q(item.kind, item.q), origin="manual")
        for item in existing_filters
        if item.enabled and item.origin != "auto"
    ]
    before_error = _auto_error(
        speaker_gain,
        target_gain,
        manual,
        [],
        sample_rate,
        frequency,
    )
    combined_gain = _auto_filter_gain(
        [*manual, *result.filters],
        sample_rate,
        frequency,
    )
    after_error = target_gain - speaker_gain - combined_gain
    before_smooth = _auto_smoothed_error(
        before_error,
        frequency,
        float(runtime_settings.peq_smoothing_oct),
    )
    after_smooth = _auto_smoothed_error(
        after_error,
        frequency,
        float(runtime_settings.peq_smoothing_oct),
    )
    objective_reference = auto_iir_objective_reference(
        before_error,
        frequency,
        core_weights,
    )
    after_objective = evaluate_auto_iir_objective(
        after_error,
        frequency,
        core_weights,
        reference=objective_reference,
        center_shift_oct=float(diagnostics.optimizer_center_shift_oct),
        objective_weights=AutoIIRObjectiveWeights(
            one_octave_abs=float(runtime_settings.objective_one_octave_abs_weight),
            third_octave_abs=float(runtime_settings.objective_third_octave_abs_weight),
            perceptual_peak=float(runtime_settings.objective_perceptual_peak_weight),
            positive_peak_area=float(
                runtime_settings.objective_positive_peak_area_weight
            ),
            center_shift=float(runtime_settings.objective_center_shift_weight),
        ),
    )
    headroom_db = max(float(np.max(combined_gain)), 0.0)
    shelf_count = sum(
        item.kind in {"low_shelf", "high_shelf"} for item in result.filters
    )
    peq_count = sum(item.kind == "peq" for item in result.filters)
    before_under_peak = float(
        np.max(
            np.maximum(
                fractional_octave_smooth(
                    before_error[core_mask], frequency[core_mask], 1.0 / 3.0
                ),
                0.0,
            )
        )
    )
    after_under_peak = float(
        np.max(
            np.maximum(
                fractional_octave_smooth(
                    after_error[core_mask], frequency[core_mask], 1.0 / 3.0
                ),
                0.0,
            )
        )
    )
    updated = replace(
        diagnostics,
        before_weighted_rms_db=_masked_weighted_rms(
            before_smooth, weights, core_mask
        ),
        after_weighted_rms_db=_masked_weighted_rms(
            after_smooth, weights, core_mask
        ),
        before_raw_weighted_rms_db=_masked_weighted_rms(
            before_error, weights, core_mask
        ),
        after_raw_weighted_rms_db=_masked_weighted_rms(
            after_error, weights, core_mask
        ),
        before_error_area_db_oct=_masked_absolute_error_area(
            before_smooth, frequency, weights, core_mask
        ),
        after_error_area_db_oct=_masked_absolute_error_area(
            after_smooth, frequency, weights, core_mask
        ),
        before_outside_error_area_db_oct=_masked_absolute_error_area(
            before_smooth, frequency, weights, guard_mask
        ),
        after_outside_error_area_db_oct=_masked_absolute_error_area(
            after_smooth, frequency, weights, guard_mask
        ),
        before_p95_abs_db=float(np.percentile(np.abs(before_smooth[core_mask]), 95.0)),
        after_p95_abs_db=float(np.percentile(np.abs(after_smooth[core_mask]), 95.0)),
        before_p99_abs_db=float(np.percentile(np.abs(before_smooth[core_mask]), 99.0)),
        after_p99_abs_db=float(np.percentile(np.abs(after_smooth[core_mask]), 99.0)),
        before_max_abs_db=float(np.max(np.abs(before_error))),
        after_max_abs_db=float(np.max(np.abs(after_error))),
        before_one_octave_abs_db=objective_reference.one_octave_abs_db,
        after_one_octave_abs_db=after_objective.one_octave_abs_db,
        before_third_octave_abs_db=objective_reference.third_octave_abs_db,
        after_third_octave_abs_db=after_objective.third_octave_abs_db,
        before_contour_abs_db=objective_reference.contour_abs_db,
        after_contour_abs_db=after_objective.contour_abs_db,
        before_perceptual_peak_db=objective_reference.perceptual_peak_db,
        after_perceptual_peak_db=after_objective.perceptual_peak_db,
        before_positive_peak_area_db_oct=objective_reference.positive_peak_area_db_oct,
        after_positive_peak_area_db_oct=after_objective.positive_peak_area_db_oct,
        before_under_target_area_db_oct=objective_reference.under_target_area_db_oct,
        after_under_target_area_db_oct=after_objective.under_target_area_db_oct,
        before_under_target_peak_db=before_under_peak,
        after_under_target_peak_db=after_under_peak,
        filter_headroom_db=headroom_db,
        filter_headroom_ok=bool(
            headroom_db <= float(runtime_settings.filter_headroom_limit_db) + 1e-6
        ),
        generated_filter_count=len(result.filters),
        peq_count=peq_count,
        shelf_count=shelf_count,
        low_shelf_selected=any(
            item.kind == "low_shelf" for item in result.filters
        ),
        high_shelf_selected=any(
            item.kind == "high_shelf" for item in result.filters
        ),
    )
    return replace(result, diagnostics=updated)


def _auto_iir_resolution_metrics(result: AutoIIRResult) -> AutoIIRResolutionMetrics:
    diagnostics = result.diagnostics
    return AutoIIRResolutionMetrics(
        twelfth_rms_db=float(diagnostics.after_weighted_rms_db),
        third_abs_db=float(diagnostics.after_third_octave_abs_db),
        perceptual_peak_db=float(diagnostics.after_perceptual_peak_db),
        positive_peak_area_db_oct=float(
            diagnostics.after_positive_peak_area_db_oct
        ),
        headroom_db=float(diagnostics.filter_headroom_db),
        headroom_ok=bool(diagnostics.filter_headroom_ok),
        filter_count=len(result.filters),
        shelf_kinds=tuple(
            sorted(
                item.kind
                for item in result.filters
                if item.kind in {"low_shelf", "high_shelf"}
            )
        ),
    )


def _quantize_auto_iir_candidate(
    filters: list[IIRFilter],
    manual: list[IIRFilter],
    sample_rate: int,
    f_min: float,
    f_max: float,
    evaluation_f_min: float,
    evaluation_f_max: float,
    settings: AutoIIRSettings,
    policy: AutoIIRResolutionPolicy,
) -> list[IIRFilter]:
    """Quantize final values and reduce only generated boosts for headroom."""

    result = quantize_auto_iir_filters(
        filters,
        f_min=f_min,
        f_max=f_max,
        policy=policy,
    )
    frequency = _adaptive_log_axis(
        evaluation_f_min,
        evaluation_f_max,
        points_per_octave=float(policy.validation_points_per_octave),
        minimum=int(policy.validation_minimum_points),
        maximum=int(policy.validation_maximum_points),
    )
    gain_step = max(float(policy.gain_step_db), 1e-9)
    for _ in range(2_500):
        if _auto_iir_headroom_is_safe(
            [*manual, *result], sample_rate, frequency, settings
        ):
            break
        positive = {
            index for index, item in enumerate(result) if float(item.gain_db) > 0.0
        }
        if not positive:
            break
        result = [
            replace(item, gain_db=max(float(item.gain_db) - gain_step, 0.0))
            if index in positive
            else item
            for index, item in enumerate(result)
        ]
    return result


def optimize_auto_iir(
    speaker: SpeakerResponse,
    target: SpeakerResponse,
    sample_rate: int,
    *,
    existing_filters: Iterable[IIRFilter] = (),
    settings: AutoIIRSettings = AutoIIRSettings(),
    perceptual_profile: AutoIIRPerceptualProfile | None = None,
) -> AutoIIRResult:
    """Design Auto IIR with the validated 28/48 multi-resolution policy."""

    existing = tuple(existing_filters)
    if not settings.adaptive_resolution_enabled:
        return _optimize_auto_iir_at_resolution(
            speaker,
            target,
            sample_rate,
            existing_filters=existing,
            settings=settings,
            perceptual_profile=perceptual_profile,
        )

    started_at = perf_counter()
    policy = AutoIIRResolutionPolicy()
    candidate_settings = replace(
        settings,
        adaptive_resolution_enabled=False,
        evaluation_points_per_octave=float(policy.candidate_points_per_octave),
        evaluation_minimum_points=int(policy.candidate_minimum_points),
        evaluation_maximum_points=int(policy.candidate_maximum_points),
    )
    candidate = _optimize_auto_iir_at_resolution(
        speaker,
        target,
        sample_rate,
        existing_filters=existing,
        settings=candidate_settings,
        perceptual_profile=perceptual_profile,
    )
    candidate_seconds = perf_counter() - started_at
    reference = _reevaluate_auto_iir_result(
        candidate,
        speaker,
        target,
        sample_rate,
        existing_filters=existing,
        settings=settings,
        perceptual_profile=perceptual_profile,
        policy=policy,
    )
    manual = [
        replace(item, q=clamp_iir_q(item.kind, item.q), origin="manual")
        for item in existing
        if item.enabled and item.origin != "auto"
    ]
    quantized_filters = _quantize_auto_iir_candidate(
        candidate.filters,
        manual,
        sample_rate,
        float(candidate.diagnostics.correction_f_min_hz),
        float(candidate.diagnostics.correction_f_max_hz),
        float(candidate.diagnostics.evaluation_f_min_hz),
        float(candidate.diagnostics.evaluation_f_max_hz),
        settings,
        policy,
    )
    quantized = _reevaluate_auto_iir_result(
        replace(candidate, filters=quantized_filters),
        speaker,
        target,
        sample_rate,
        existing_filters=existing,
        settings=settings,
        perceptual_profile=perceptual_profile,
        policy=policy,
    )
    validation = validate_auto_iir_resolution_candidate(
        _auto_iir_resolution_metrics(quantized),
        _auto_iir_resolution_metrics(reference),
        policy=policy,
    )
    if validation.passed:
        diagnostics = replace(
            quantized.diagnostics,
            optimization_seconds=perf_counter() - started_at,
            resolution_mode="adaptive_28_48",
            resolution_candidate_points_per_octave=float(
                policy.candidate_points_per_octave
            ),
            resolution_validation_points_per_octave=float(
                policy.validation_points_per_octave
            ),
            resolution_guard_passed=True,
            resolution_fallback_used=False,
            resolution_guard_reasons=(),
            resolution_candidate_seconds=candidate_seconds,
        )
        return replace(quantized, diagnostics=diagnostics)

    full_settings = replace(
        settings,
        adaptive_resolution_enabled=False,
        evaluation_points_per_octave=float(policy.validation_points_per_octave),
        evaluation_minimum_points=int(policy.validation_minimum_points),
        evaluation_maximum_points=int(policy.validation_maximum_points),
    )
    fallback = _optimize_auto_iir_at_resolution(
        speaker,
        target,
        sample_rate,
        existing_filters=existing,
        settings=full_settings,
        perceptual_profile=perceptual_profile,
    )
    diagnostics = replace(
        fallback.diagnostics,
        optimization_seconds=perf_counter() - started_at,
        resolution_mode="full_48_fallback",
        resolution_candidate_points_per_octave=float(
            policy.candidate_points_per_octave
        ),
        resolution_validation_points_per_octave=float(
            policy.validation_points_per_octave
        ),
        resolution_guard_passed=False,
        resolution_fallback_used=True,
        resolution_guard_reasons=tuple(validation.reasons),
        resolution_candidate_seconds=candidate_seconds,
    )
    return replace(fallback, diagnostics=diagnostics)


def auto_iir_filters(
    speaker: SpeakerResponse,
    target: SpeakerResponse,
    sample_rate: int,
    *,
    existing_filters: Iterable[IIRFilter] = (),
    settings: AutoIIRSettings = AutoIIRSettings(),
    perceptual_profile: AutoIIRPerceptualProfile | None = None,
) -> list[IIRFilter]:
    """Compatibility wrapper returning the optimized editable filter list."""
    return optimize_auto_iir(
        speaker,
        target,
        sample_rate,
        existing_filters=existing_filters,
        settings=settings,
        perceptual_profile=perceptual_profile,
    ).filters


def biquad_rows(filters: Iterable[IIRFilter], sample_rate: int, *, extra_sos=()) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    section_index = 0
    for filter_index, item in enumerate(filters, start=1):
        if not item.enabled:
            continue
        for local_index, section in enumerate(design_iir_sos(item, sample_rate), start=1):
            section_index += 1
            rows.append(
                {
                    "section": section_index,
                    "filter": filter_index,
                    "filter_type": item.kind,
                    "filter_section": local_index,
                    "b0": float(section[0]),
                    "b1": float(section[1]),
                    "b2": float(section[2]),
                    "a0": 1.0,
                    "a1": float(section[4]),
                    "a2": float(section[5]),
                }
            )
    extra = np.asarray(extra_sos, dtype=float)
    if extra.size:
        if extra.ndim != 2 or extra.shape[1] != 6 or not np.all(np.isfinite(extra)) or np.any(extra[:, 3] != 1):
            raise ValueError("Output alignment SOS must be finite normalized six-coefficient rows")
        for local_index, section in enumerate(extra, start=1):
            rows.append(dict(section=len(rows)+1, filter=0, filter_type="output_alignment",
                             filter_section=local_index,
                             **dict(zip(("b0", "b1", "b2", "a0", "a1", "a2"), map(float, section)))))
    return rows


def export_biquad_csv(filters: Iterable[IIRFilter], sample_rate: int, *, extra_sos=()) -> str:
    output = io.StringIO()
    columns = ("section", "filter", "filter_type", "filter_section", "b0", "b1", "b2", "a0", "a1", "a2")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in biquad_rows(filters, sample_rate, extra_sos=extra_sos):
        writer.writerow(row)
    return output.getvalue()


def export_biquad_json(filters: Iterable[IIRFilter], sample_rate: int, *, extra_sos=()) -> str:
    payload = {
        "sample_rate": int(sample_rate),
        "transfer_function": "(b0+b1*z^-1+b2*z^-2)/(1+a1*z^-1+a2*z^-2)",
        "sections": biquad_rows(filters, sample_rate, extra_sos=extra_sos),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_iir_parameter_document(filters: Iterable[IIRFilter], sample_rate: int, *, extra_sos=()) -> str:
    """Return a vendor-neutral, human-readable IIR parameter handoff."""
    items = list(filters)
    lines = [
        "# PhaseEQ Generic IIR Parameters",
        "",
        f"Sample rate: {int(sample_rate)} Hz",
        "Application order: top to bottom; disabled rows are bypassed.",
        "",
        "| Apply order | Type | Frequency [Hz] | Gain [dB] | Q | Order | Enabled | Family | Origin | All-pass preset |",
        "|---:|---|---:|---:|---:|---:|:---:|---|---|---|",
    ]
    for apply_order, item in enumerate(items, start=1):
        lines.append(
            "| "
            + " | ".join((
                str(apply_order),
                str(item.kind),
                f"{float(item.fc):.10g}",
                f"{float(item.gain_db):.10g}",
                f"{float(item.q):.10g}",
                str(int(item.order)),
                "Yes" if item.enabled else "No",
                str(item.family),
                str(item.origin),
                str(item.allpass_preset),
            ))
            + " |"
        )
    if not items:
        lines.append("| - | - | - | - | - | - | - | - | - | - |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Type values: `peq`, `high_shelf`, `low_shelf`, `allpass`, `high_pass`, `low_pass`.",
        "- Enter the filters into the destination DSP in Apply order.",
        "- A destination DSP may use different Shelf-Q, crossover-family, or All-pass conventions. Verify the realized response after entry.",
        "- The table above contains design parameters, not Biquad coefficients.",
        "",
    ])
    if np.asarray(extra_sos).size:
        lines.extend(["", "## Output alignment Biquads", "",
                      "Apply these coefficients after the parameter filters. Polarity is included once.",
                      "```json", export_biquad_json([], sample_rate, extra_sos=extra_sos), "```"])
    return "\n".join(lines)


def export_minidsp_biquads(filters: Iterable[IIRFilter], sample_rate: int, *, extra_sos=()) -> str:
    lines: list[str] = []
    for row in biquad_rows(filters, sample_rate, extra_sos=extra_sos):
        lines.extend(
            [
                f"biquad{row['section']},",
                f"b0={float(row['b0']):.16g},",
                f"b1={float(row['b1']):.16g},",
                f"b2={float(row['b2']):.16g},",
                f"a1={-float(row['a1']):.16g},",
                f"a2={-float(row['a2']):.16g}",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def export_sigmastudio_biquads(filters: Iterable[IIRFilter], sample_rate: int, *, extra_sos=()) -> str:
    lines = [
        "# SigmaStudio General 2nd-Order / IIR Coefficient UI",
        "# Order per section: B0, B1, B2, A1, A2",
        "# A values use H(z)=B(z)/(1+A1*z^-1+A2*z^-2); this is not a raw DSP RAM image.",
    ]
    for row in biquad_rows(filters, sample_rate, extra_sos=extra_sos):
        lines.append(f"# section {row['section']} / {row['filter_type']}")
        lines.extend(
            f"{float(row[name]):.16g}"
            for name in ("b0", "b1", "b2", "a1", "a2")
        )
    return "\n".join(lines) + "\n"


def export_camilladsp_biquads(filters: Iterable[IIRFilter], sample_rate: int, *, extra_sos=()) -> str:
    """Return CamillaDSP filter definitions using normalized DiffEq sections."""
    lines = ["# PhaseEQ CamillaDSP IIR filter snippet", "filters:"]
    names: list[str] = []
    for row in biquad_rows(filters, sample_rate, extra_sos=extra_sos):
        name = f"phaseeq_iir_{int(row['section'])}"
        names.append(name)
        lines.extend(
            [
                f"  {name}:",
                "    type: DiffEq",
                "    parameters:",
                f"      a: [1.0, {float(row['a1']):.16g}, {float(row['a2']):.16g}]",
                f"      b: [{float(row['b0']):.16g}, {float(row['b1']):.16g}, {float(row['b2']):.16g}]",
            ]
        )
    lines.extend(["# Apply in this order:", f"# names: [{', '.join(names)}]"])
    return "\n".join(lines) + "\n"
