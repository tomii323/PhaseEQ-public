from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class MeasurementState(StrEnum):
    IDLE = "IDLE"
    ARMING = "ARMING"
    MEASURING_NOISE = "MEASURING_NOISE"
    PREFLIGHT = "PREFLIGHT"
    SEARCHING = "SEARCHING"
    CAPTURING_SHOT = "CAPTURING_SHOT"
    ANALYZING_SHOT = "ANALYZING_SHOT"
    STOPPING = "STOPPING"
    STOPPED_WITH_RESULTS = "STOPPED_WITH_RESULTS"
    ERROR_RECOVERABLE = "ERROR_RECOVERABLE"
    ERROR_FATAL = "ERROR_FATAL"


class ShotStatus(StrEnum):
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    CLIPPED = "clipped"
    DETECTION_FAILED = "detection_failed"


class AverageMode(StrEnum):
    RUNNING = "running"
    LAST_N = "last_n"
    SELECTED = "selected"


@dataclass(frozen=True)
class MeasurementSettings:
    sample_rate: int = 48_000
    channels: int = 2
    input_mode: str = "dual_gain"
    reference_mode: str = "ir_peak"
    sync_polarity: str = "auto"
    start_frequency_hz: float = 2.0
    end_frequency_hz: float = 22_050.0
    sweep_duration_s: float = 0.480
    shot_period_s: float = 0.683
    search_margin_before_s: float = 0.100
    search_margin_after_s: float = 0.100
    correlation_threshold: float = 0.35
    clip_threshold: float = 0.995
    gate_start_ms: float = -2.0
    gate_end_ms: float = 5.0
    merge_crossover_hz: float = 0.0
    fft_size: int = 65_536  # Minimum FFT size; never a limit on retained deconvolution IR.
    wavelet_update_mode: str = "every_shot"
    wavelet_every_n: int = 3
    include_warnings_in_average: bool = True
    autosave_enabled: bool = True
    raw_audio_storage: str = "accepted_shots"
    quality_gate_mode: str = "standard"
    clock_correction_mode: str = "auto"
    input_device: int | str | None = None
    reference_source_name: str = ""
    reference_source_sample_rate: int = 0
    reference_peak_dbfs: float = 0.0
    reference_repeat_count: int = 0
    reference_loop_gap_variable: bool = False
    reference_source_channel: int = 1
    shot_capture_duration_s: float = 0.0
    reference_has_timing_marker: bool = False
    timing_marker_to_ess_s: float = 0.0
    timing_reference_capture_preroll_s: float = 0.020
    timing_reference_kind: str = "sequence_marker"
    timing_reference_session_id: str = ""
    reference_tweeter_channel_id: str = ""
    reference_measurement_id: str = ""
    reference_arrival_sample: float = float("nan")
    timing_reference_role: str = "target_speaker"
    reference_output_channel: str = ""
    measurement_output_channel: str = ""
    reference_routing_verified: bool = False
    external_playback_setup_confirmed: bool = False
    reference_protection_confirmed: bool = False
    alignment_processing_bypassed_confirmed: bool = False
    external_playback_setup_note: str = ""
    measurement_distance_m: float = 1.0
    level_calibration_mode: str = "uncalibrated"
    mic_sensitivity_dbfs_per_pa: float = 0.0
    mic_sensitivity_mv_pa: float = 0.0
    interface_full_scale_vrms: float = 0.0
    input_gain_db: float = 0.0
    response_gain_reference_db: float | None = None
    input_gain_db_channels: tuple[float, ...] = ()
    input_gain_max_db_channels: tuple[float, ...] = ()
    input_gain_device_name: str = ""
    input_analog_gain_db: float | None = None
    require_stable_input_gain: bool = False
    minidsp_sens_factor_db: float = 0.0
    minidsp_analog_gain_db: float | None = None
    minidsp_gain_adjustment_db: float = 0.0
    level_reference_spl_db: float = 94.0
    level_reference_dbfs: float = 0.0
    level_calibration_note: str = ""
    microphone_profile_id: str = ""
    microphone_unit_id: str = ""
    microphone_serial_number: str = ""
    microphone_nominal_bit_depth: int = 0
    input_channel: int = 1
    mic_calibration_angle_deg: int = 0
    mic_calibration_id: str = ""
    mic_calibration_frequency_hz: tuple[float, ...] = ()
    mic_calibration_gain_db: tuple[float, ...] = ()
    mic_calibration_extrapolation: str = "hold_edge"
    mic_phase_calibration_id: str = ""
    mic_phase_calibration_frequency_hz: tuple[float, ...] = ()
    mic_phase_calibration_deg: tuple[float, ...] = ()
    noise_preflight_enabled: bool = False
    ambient_noise_duration_s: float = 5.0
    target_snr_db: float = 25.0
    minimum_snr_db: float = 18.0
    minimum_passing_band_fraction: float = 0.80
    preflight_mask_relative_db: float = -25.0
    preflight_mask_min_bands: int = 12
    preflight_mask_min_octaves: float = 1.0
    preflight_mask_edge_trim_bands: int = 2
    preflight_mask_max_gap_bands: int = 2
    target_headroom_db: float = 10.0
    minimum_headroom_db: float = 6.0
    automatic_usb_gain: bool = True
    guided_measurement: bool = False
    playback_target_spl_db: float = 75.0
    playback_warning_spl_db: float = 80.0
    playback_red_warning_spl_db: float = 85.0
    maximum_speaker_increase_db: float = 3.0
    low_snr_retry_limit: int = 3

    def __post_init__(self) -> None:
        if self.sample_rate not in {48_000, 96_000, 192_000}:
            raise ValueError("sample_rate must be 48000, 96000, or 192000 Hz")
        if self.channels not in {1, 2}:
            raise ValueError("continuous ESS measurement requires one or two input channels")
        if self.input_channel < 1:
            raise ValueError("input_channel must be 1 or greater")
        if self.input_mode not in {"mono", "dual_gain", "known_wav_dual_gain", "known_wav_mono"}:
            raise ValueError("unsupported measurement input_mode")
        if self.reference_mode not in {"ir_peak", "timing_marker"}:
            raise ValueError("unsupported measurement reference_mode")
        if self.sync_polarity not in {"auto", "positive", "negative"}:
            raise ValueError("unsupported synchronization polarity")
        if self.input_mode == "known_wav_dual_gain" and self.channels != 2:
            raise ValueError("ESS Library reference with dual-gain recording requires two input channels")
        if self.input_mode == "known_wav_mono" and self.channels != 1:
            raise ValueError("ESS Library reference with mono recording requires one input channel")
        if self.sweep_duration_s <= 0 or self.shot_period_s <= self.sweep_duration_s:
            raise ValueError("shot period must be longer than sweep duration")
        if self.reference_repeat_count < 0:
            raise ValueError("reference repeat count cannot be negative")
        if self.reference_source_channel not in {1, 2}:
            raise ValueError("reference source channel must be 1 or 2")
        if self.shot_capture_duration_s < 0:
            raise ValueError("shot capture duration cannot be negative")
        if self.shot_capture_duration_s and self.shot_capture_duration_s < self.sweep_duration_s:
            raise ValueError("shot capture duration must cover the ESS sweep")
        if self.timing_marker_to_ess_s < 0:
            raise ValueError("timing marker offset cannot be negative")
        if not 0.0 <= self.timing_reference_capture_preroll_s <= 0.050:
            raise ValueError("timing reference capture preroll must be between 0 and 50 ms")
        if self.timing_reference_kind not in {
            "unknown", "sequence_marker", "phaseeq_tweeter_reference",
        }:
            raise ValueError("unsupported timing reference kind")
        if self.timing_reference_kind == "phaseeq_tweeter_reference" and (
            not self.timing_reference_session_id.strip()
            or not self.reference_tweeter_channel_id.strip()
        ):
            raise ValueError("Tweeter timing requires a session and reference Channel")
        if self.timing_reference_role not in {"reference_tweeter", "target_speaker"}:
            raise ValueError("unsupported timing reference role")
        if self.reference_output_channel not in {"", "left", "right"}:
            raise ValueError("reference output channel must be left or right")
        if self.measurement_output_channel not in {"", "left", "right"}:
            raise ValueError("measurement output channel must be left or right")
        if self.reference_routing_verified and (
            not self.reference_output_channel
            or not self.measurement_output_channel
            or self.reference_output_channel == self.measurement_output_channel
        ):
            raise ValueError("verified Tweeter routing requires separate left and right outputs")
        if self.gate_end_ms <= self.gate_start_ms:
            raise ValueError("gate end must be after gate start")
        if self.end_frequency_hz > self.sample_rate / 2:
            raise ValueError("ESS end frequency exceeds Nyquist")
        if self.mic_calibration_extrapolation not in {"hold_edge", "no_correction", "manual_extension"}:
            raise ValueError("unsupported microphone calibration extrapolation")
        if self.clock_correction_mode not in {"auto", "force", "off"}:
            raise ValueError("unsupported clock correction mode")
        if self.quality_gate_mode not in {"strict", "standard", "lenient"}:
            raise ValueError("unsupported quality gate mode")
        if self.raw_audio_storage not in {"accepted_shots", "session_wav", "ring_only"}:
            raise ValueError("unsupported raw audio storage mode")
        if self.measurement_distance_m <= 0:
            raise ValueError("measurement distance must be positive")
        if self.level_calibration_mode not in {
            "uncalibrated",
            "digital_sensitivity",
            "estimated_digital_sensitivity",
            "minidsp_sens_factor",
            "analog_sensitivity",
            "acoustic_calibrator",
            "spl_meter_comparison",
        }:
            raise ValueError("unsupported level calibration mode")
        if self.ambient_noise_duration_s < 1.0:
            raise ValueError("ambient noise duration must be at least one second")
        if self.target_snr_db < self.minimum_snr_db:
            raise ValueError("target S/N must not be below minimum S/N")
        if not 0.0 < self.minimum_passing_band_fraction <= 1.0:
            raise ValueError("minimum passing-band fraction must be in (0, 1]")
        if self.preflight_mask_relative_db >= 0.0:
            raise ValueError("preflight mask relative level must be negative")
        if self.preflight_mask_min_bands < 1:
            raise ValueError("preflight mask must contain at least one band")
        if self.preflight_mask_min_octaves <= 0.0:
            raise ValueError("preflight mask minimum width must be positive")
        if self.preflight_mask_edge_trim_bands < 0 or self.preflight_mask_max_gap_bands < 0:
            raise ValueError("preflight mask trim and gap sizes cannot be negative")
        if self.target_headroom_db < self.minimum_headroom_db:
            raise ValueError("target headroom must not be below minimum headroom")
        if not (
            self.playback_target_spl_db
            <= self.playback_warning_spl_db
            <= self.playback_red_warning_spl_db
        ):
            raise ValueError("playback SPL levels must be ordered target <= warning <= red warning")
        if self.maximum_speaker_increase_db <= 0.0:
            raise ValueError("maximum speaker increase must be positive")
        if self.low_snr_retry_limit < 1:
            raise ValueError("low S/N retry limit must be at least one")
        if self.response_gain_reference_db is not None and not np.isfinite(self.response_gain_reference_db):
            raise ValueError("response gain reference must be finite")

    @property
    def tweeter_reference_setup_ready(self) -> bool:
        if self.timing_reference_kind != "phaseeq_tweeter_reference":
            return True
        return bool(
            self.reference_mode == "timing_marker"
            and self.reference_has_timing_marker
            and self.reference_routing_verified
            and self.reference_output_channel in {"left", "right"}
            and self.measurement_output_channel in {"left", "right"}
            and self.reference_output_channel != self.measurement_output_channel
            and self.external_playback_setup_confirmed
            and self.reference_protection_confirmed
            and self.alignment_processing_bypassed_confirmed
        )


@dataclass(frozen=True)
class ShotQuality:
    status: ShotStatus
    snr_db: float
    left_peak: float
    right_peak: float
    left_rms: float
    right_rms: float
    gain_difference_db: float
    lr_correlation: float
    delay_difference_samples: float
    polarity_inverted: bool
    left_clipped: bool
    right_clipped: bool
    selected_channel: str
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShotResult:
    shot_index: int
    status: ShotStatus
    detected_start_sample: int
    detected_end_sample: int
    correlation: float
    detection_confidence: float
    timing_error_ms: float
    detected_duration_s: float
    detected_period_s: float | None
    raw_stereo_audio: np.ndarray | None
    combined_raw_ir: np.ndarray | None
    gated_ir: np.ndarray | None
    frequency_hz: np.ndarray | None
    ungated_complex_response: np.ndarray | None
    gated_complex_response: np.ndarray | None
    merged_complex_response: np.ndarray | None
    wavelet_map: Any | None
    quality: ShotQuality
    settings_snapshot: MeasurementSettings
    included_in_average: bool = True
    sweep_index: int = 0
    timing_segment: int = 0
    timing_index: int = -1
    loop_boundary_before: bool = False
    reference_mode: str = "ir_peak"
    reference_plane: str = "peak_centered"
    absolute_timing_valid: bool = False
    marker_detected: bool = False
    marker_verified: bool = False
    center_method: str = "absolute_peak"
    center_polarity: str = "auto"
    center_sample: float = float("nan")
    center_confidence: float = 0.0
    center_tracking_state: str = "initial"
    center_search_start_sample: int = 0
    center_search_end_sample: int = 0
    input_gain_normalization_db: float = 0.0
    raw_peak_sample: float = float("nan")
    centered_ir_similarity: float = float("nan")
    gain_valid: bool = True
    relative_phase_valid: bool = True
    timing_valid: bool = False
    quality_tier: str = "basic"
    timing_ungated_complex_response: np.ndarray | None = None
    timing_gated_complex_response: np.ndarray | None = None
    timing_merged_complex_response: np.ndarray | None = None
    merge_valid: bool = True
    effective_merge_crossover_hz: float = float("nan")

    @property
    def merged_gain_db(self) -> np.ndarray | None:
        return _gain(self.merged_complex_response)

    @property
    def merged_phase_deg(self) -> np.ndarray | None:
        if self.merged_complex_response is None:
            return None
        return np.rad2deg(np.unwrap(np.angle(self.merged_complex_response)))


@dataclass(frozen=True)
class HighPrecisionResult:
    """A derived, phase-preserving result made from retained raw shots."""

    frequency_hz: np.ndarray
    ungated_complex_response: np.ndarray
    gated_complex_response: np.ndarray
    merged_complex_response: np.ndarray
    used_shot_indices: tuple[int, ...]
    reference_shot_index: int
    alignment_samples: tuple[float, ...]
    coherence: np.ndarray
    repeatability_db: np.ndarray
    confidence: np.ndarray
    clock_drift_ppm: float = 0.0
    clock_residual_samples: float = float("nan")
    clock_corrected: bool = False
    algorithm: str = "subsample_aligned_quality_weighted_robust_complex_average"
    integrated_uncalibrated_ir: np.ndarray | None = None
    arrival_sample: float = float("nan")
    observation_samples: int = 0
    minimum_center_confidence: float = 0.0
    summary_median_coherence: float | None = None
    summary_repeatability_p90_db: float | None = None
    reference_mode: str = "ir_peak"
    timing_valid_shot_count: int = 0
    absolute_timing_valid: bool = False

    @property
    def shot_count(self) -> int:
        return len(self.used_shot_indices)

    @property
    def median_coherence(self) -> float:
        if self.summary_median_coherence is not None:
            return self.summary_median_coherence
        finite = self.coherence[np.isfinite(self.coherence)]
        return float(np.median(finite)) if finite.size else 0.0

    @property
    def repeatability_p90_db(self) -> float:
        if self.summary_repeatability_p90_db is not None:
            return self.summary_repeatability_p90_db
        finite = self.repeatability_db[np.isfinite(self.repeatability_db)]
        return float(np.percentile(finite, 90)) if finite.size else float("inf")


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    state: MeasurementState
    settings: MeasurementSettings
    revision: int = 0
    shots: tuple[ShotResult, ...] = ()
    started_at: str = ""
    stopped_at: str = ""
    error_message: str = ""
    device_info: dict[str, Any] = field(default_factory=dict)
    autosave_directory: Path | None = None
    standard_result: HighPrecisionResult | None = None
    denoised_result: HighPrecisionResult | None = None
    saved_original: Any | None = None

    @property
    def valid_shots(self) -> tuple[ShotResult, ...]:
        allowed = {ShotStatus.VALID}
        if self.settings.include_warnings_in_average:
            allowed.add(ShotStatus.WARNING)
        return tuple(shot for shot in self.shots if shot.included_in_average and shot.status in allowed)

    @property
    def latest(self) -> ShotResult | None:
        return self.shots[-1] if self.shots else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def readonly(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    value = np.ascontiguousarray(array)
    value.flags.writeable = False
    return value


def _gain(response: np.ndarray | None) -> np.ndarray | None:
    if response is None:
        return None
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-15))
