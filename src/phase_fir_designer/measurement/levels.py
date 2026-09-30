from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import MeasurementSettings
from .sounddevice_compat import import_sounddevice


PA_REFERENCE_SPL_DB = 94.0
SINE_PEAK_TO_RMS_DB = 20.0 * np.log10(1.0 / np.sqrt(2.0))


@dataclass(frozen=True)
class InputLevelReading:
    rms_dbfs: float
    peak_dbfs: float
    selected_channel: int
    channel_rms_dbfs: tuple[float, ...]


@dataclass(frozen=True)
class AcousticCalibrationEstimate:
    """ARTA-style acoustic-calibrator estimate for the current input chain."""

    reference_spl_db: float
    measured_rms_dbfs: float
    peak_dbfs: float
    selected_channel: int
    preamp_gain_db: float
    chain_sensitivity_dbfs_per_pa: float
    unity_gain_sensitivity_dbfs_per_pa: float
    calibration_offset_db: float


def estimate_acoustic_calibration(
    reading: InputLevelReading,
    *,
    reference_spl_db: float,
    preamp_gain_db: float,
) -> AcousticCalibrationEstimate:
    """Estimate microphone/input sensitivity from a steady acoustic calibrator."""

    values = (
        float(reading.rms_dbfs),
        float(reading.peak_dbfs),
        float(reference_spl_db),
        float(preamp_gain_db),
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("acoustic calibration values must be finite")
    if reading.rms_dbfs >= 0.0 or reading.peak_dbfs >= 0.0:
        raise ValueError("acoustic calibration recording must remain below 0 dBFS")
    if not 40.0 <= reference_spl_db <= 150.0:
        raise ValueError("acoustic calibrator level must be between 40 and 150 dB SPL")
    pressure_above_one_pa_db = float(reference_spl_db) - PA_REFERENCE_SPL_DB
    chain_sensitivity = float(reading.rms_dbfs) - pressure_above_one_pa_db
    return AcousticCalibrationEstimate(
        reference_spl_db=float(reference_spl_db),
        measured_rms_dbfs=float(reading.rms_dbfs),
        peak_dbfs=float(reading.peak_dbfs),
        selected_channel=int(reading.selected_channel),
        preamp_gain_db=float(preamp_gain_db),
        chain_sensitivity_dbfs_per_pa=chain_sensitivity,
        unity_gain_sensitivity_dbfs_per_pa=chain_sensitivity - float(preamp_gain_db),
        calibration_offset_db=float(reference_spl_db) - float(reading.rms_dbfs),
    )


def level_calibration_offset_db(settings: MeasurementSettings) -> float | None:
    """Return the dB offset from app-convention dBFS RMS to dB SPL."""

    mode = settings.level_calibration_mode
    if mode in {"digital_sensitivity", "estimated_digital_sensitivity"}:
        if not np.isfinite(settings.mic_sensitivity_dbfs_per_pa) or settings.mic_sensitivity_dbfs_per_pa >= 0:
            return None
        return PA_REFERENCE_SPL_DB - float(settings.mic_sensitivity_dbfs_per_pa)
    if mode == "minidsp_sens_factor":
        if not np.isfinite(settings.minidsp_sens_factor_db):
            return None
        if not np.isfinite(settings.minidsp_gain_adjustment_db) or settings.minidsp_gain_adjustment_db < -0.1:
            return None
        return (
            PA_REFERENCE_SPL_DB
            - float(settings.minidsp_sens_factor_db)
            + float(settings.minidsp_gain_adjustment_db)
        )
    if mode == "analog_sensitivity":
        sensitivity_v_pa = float(settings.mic_sensitivity_mv_pa) / 1000.0
        full_scale_vrms = float(settings.interface_full_scale_vrms)
        if sensitivity_v_pa <= 0 or full_scale_vrms <= 0:
            return None
        sensitivity_dbfs_per_pa = (
            20.0 * np.log10(sensitivity_v_pa / full_scale_vrms)
            + float(settings.input_gain_db)
        )
        return PA_REFERENCE_SPL_DB - sensitivity_dbfs_per_pa
    if mode in {"acoustic_calibrator", "spl_meter_comparison"}:
        if (
            not np.isfinite(settings.level_reference_spl_db)
            or not np.isfinite(settings.level_reference_dbfs)
            or settings.level_reference_dbfs >= 0
        ):
            return None
        return float(settings.level_reference_spl_db) - float(settings.level_reference_dbfs)
    return None


def level_calibration_kind(settings: MeasurementSettings) -> str:
    if level_calibration_offset_db(settings) is None:
        return "uncalibrated"
    if settings.level_calibration_mode == "estimated_digital_sensitivity":
        return "estimated"
    return "calibrated"


def response_level_dbfs(response: np.ndarray, settings: MeasurementSettings) -> np.ndarray:
    """Convert ESS transfer magnitude to recorded sine-equivalent dBFS RMS."""

    transfer_db = 20.0 * np.log10(np.maximum(np.abs(np.asarray(response)), 1e-15))
    return transfer_db + float(settings.reference_peak_dbfs) + float(SINE_PEAK_TO_RMS_DB)


def response_level_spl(response: np.ndarray, settings: MeasurementSettings) -> np.ndarray | None:
    offset = level_calibration_offset_db(settings)
    if offset is None:
        return None
    return response_level_dbfs(response, settings) + offset


def capture_input_level(settings: MeasurementSettings, duration_s: float = 1.0) -> InputLevelReading:
    """Capture a short steady calibration signal and return app-convention dBFS RMS."""

    try:
        sd = import_sounddevice()
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"python-sounddevice is unavailable: {exc}") from exc

    stream_channels = settings.input_channel if settings.channels == 1 else settings.channels
    frames = max(1, int(round(float(duration_s) * settings.sample_rate)))
    try:
        recording = sd.rec(
            frames,
            samplerate=settings.sample_rate,
            channels=stream_channels,
            dtype="float32",
            device=settings.input_device,
            blocking=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Calibration input could not be recorded: {exc}") from exc

    audio = np.asarray(recording, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] != frames:
        raise RuntimeError("Calibration input returned incomplete audio")
    if settings.channels == 1:
        audio = audio[:, settings.input_channel - 1 : settings.input_channel]
    else:
        audio = audio[:, : settings.channels]
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise RuntimeError("Calibration input contains no usable samples")

    rms = np.sqrt(np.mean(audio * audio, axis=0))
    peak = np.max(np.abs(audio), axis=0)
    selected = int(np.argmax(rms))
    if peak[selected] >= settings.clip_threshold:
        raise RuntimeError("Calibration signal clipped on the high-gain channel; reduce input gain")
    if rms[selected] < 1e-7:
        raise RuntimeError("Calibration signal is too low")
    channel_rms_dbfs = tuple(float(20.0 * np.log10(max(value, 1e-15))) for value in rms)
    return InputLevelReading(
        rms_dbfs=channel_rms_dbfs[selected],
        peak_dbfs=float(20.0 * np.log10(max(peak[selected], 1e-15))),
        selected_channel=selected + 1,
        channel_rms_dbfs=channel_rms_dbfs,
    )
