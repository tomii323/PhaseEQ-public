from __future__ import annotations

from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import numpy as np
from scipy.io import wavfile

from .levels import response_level_dbfs, response_level_spl

from .models import (
    MeasurementSettings,
    MeasurementState,
    HighPrecisionResult,
    SessionSnapshot,
    ShotQuality,
    ShotResult,
    ShotStatus,
)


ARRAY_FIELDS = (
    "raw_stereo_audio",
    "combined_raw_ir",
    "gated_ir",
    "frequency_hz",
    "ungated_complex_response",
    "gated_complex_response",
    "merged_complex_response",
    "timing_ungated_complex_response",
    "timing_gated_complex_response",
    "timing_merged_complex_response",
)

_ACTIVE_SESSION_STATES = {
    "ARMING", "MEASURING_NOISE", "PREFLIGHT", "SEARCHING",
    "CAPTURING_SHOT", "ANALYZING_SHOT", "STOPPING",
}
_STARTUP_CLEANED_ROOTS: set[Path] = set()


def cleanup_unrestorable_measurement_sessions_once(
    root: Path,
    *,
    active_grace_seconds: float = 12 * 60 * 60,
) -> tuple[int, int]:
    """Delete autosaves that the current UI cannot restore directly.

    A recently active manifest may belong to another running PhaseEQ instance,
    so it is protected. The operation runs once per process/root.
    """
    resolved_root = root.resolve()
    if resolved_root in _STARTUP_CLEANED_ROOTS:
        return 0, 0
    _STARTUP_CLEANED_ROOTS.add(resolved_root)
    if not root.is_dir():
        return 0, 0
    removed_count = 0
    released_bytes = 0
    now = time.time()
    for directory in tuple(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        manifest_path = directory / "session_manifest.json"
        if not manifest_path.exists():
            candidates = tuple(directory.rglob("measurement.json"))
            manifest_path = candidates[0] if candidates else manifest_path
        state = ""
        try:
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get('retain_until_original_committed') is True:
                    continue
                state = str(manifest.get("state", "")).upper()
        except (OSError, ValueError):
            state = ""
        newest_mtime = max((path.stat().st_mtime for path in directory.rglob("*") if path.is_file()), default=directory.stat().st_mtime)
        if state in _ACTIVE_SESSION_STATES and now - newest_mtime < active_grace_seconds:
            continue
        released_bytes += sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory)
        removed_count += 1
    return removed_count, int(released_bytes)


def autosave_shot(directory: Path, snapshot: SessionSnapshot, shot: ShotResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_shot(directory / f"shot_{shot.shot_index:04d}.npz", shot)
    _write_manifest(directory / "session_manifest.json", snapshot)


def autosave_manifest(directory: Path, snapshot: SessionSnapshot) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_manifest(directory / "session_manifest.json", snapshot)


def autosave_standard(directory: Path, result: HighPrecisionResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_standard(directory / "standard_result.npz", result)


def autosave_denoised(directory: Path, result: HighPrecisionResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_standard(directory / "denoised_result.npz", result)


def load_autosaved_shot(path: Path, settings: MeasurementSettings) -> ShotResult:
    return _read_shot(path, settings)


def export_measurement_session(snapshot: SessionSnapshot, destination: Path) -> Path:
    """Portable original only; acquisition Raw and individual shots stay temporary."""
    from .original import original_from_snapshot, MeasurementOriginal

    original = original_from_snapshot(snapshot)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("測定パッケージには空の保存先を指定してください。")
    data = original.ir_bytes()
    MeasurementOriginal.from_bytes(data, original.recipe, original.calibration, original.storage_metadata)
    (destination / "integrated_uncalibrated_ir.npz").write_bytes(data)
    (destination / "calibration.json").write_text(json.dumps(original.calibration, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"schema_version": 5, "session_id": snapshot.session_id,
                "recipe": original.recipe, "original_metadata": original.storage_metadata}
    (destination / "measurement.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def load_measurement_session(directory: Path) -> SessionSnapshot:
    manifest_path = directory / "measurement.json"
    if not manifest_path.exists():
        manifest_path = directory / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") == 5:
        from .original import MeasurementOriginal, shot_from_original, result_from_original
        original = MeasurementOriginal.from_bytes(
            (directory / "integrated_uncalibrated_ir.npz").read_bytes(), manifest["recipe"],
            json.loads((directory / "calibration.json").read_text(encoding="utf-8")), manifest["original_metadata"],
        )
        return SessionSnapshot(
            session_id=manifest["session_id"], state=MeasurementState.STOPPED_WITH_RESULTS,
            settings=original.settings, shots=(shot_from_original(original),),
            standard_result=result_from_original(original), autosave_directory=None, saved_original=original,
        )
    settings_data = dict(manifest["settings"])
    for field_name in (
        "input_gain_db_channels",
        "input_gain_max_db_channels",
        "mic_calibration_frequency_hz",
        "mic_calibration_gain_db",
        "mic_phase_calibration_frequency_hz",
        "mic_phase_calibration_deg",
    ):
        if field_name in settings_data:
            settings_data[field_name] = tuple(settings_data[field_name])
    settings = MeasurementSettings(**settings_data)
    shot_dir = directory / "shots" if (directory / "shots").is_dir() else directory
    shots = tuple(_read_shot(path, settings) for path in sorted(shot_dir.glob("shot_*.npz")))
    standard_path = directory / "standard_result.npz"
    standard = _read_standard(standard_path) if standard_path.exists() else None
    denoised_path = directory / "denoised_result.npz"
    denoised = _read_standard(denoised_path) if denoised_path.exists() else None
    return SessionSnapshot(
        session_id=str(manifest["session_id"]),
        state=MeasurementState(manifest.get("state", MeasurementState.STOPPED_WITH_RESULTS)),
        settings=settings,
        revision=max(0, int(manifest.get("revision", len(shots)))),
        shots=shots,
        started_at=str(manifest.get("started_at", "")),
        stopped_at=str(manifest.get("stopped_at", "")),
        error_message=str(manifest.get("error_message", "")),
        device_info=dict(manifest.get("device_info", {})),
        autosave_directory=directory,
        standard_result=standard,
        denoised_result=denoised,
    )


def cleanup_measurement_autosave(snapshot: SessionSnapshot) -> int:
    """Remove a completed session's recoverable autosave after formal DB save.

    The manifest identity check prevents a stale or malformed snapshot from
    deleting an unrelated directory.  Callers must only invoke this after the
    durable measurement record has been committed successfully.
    """
    directory = snapshot.autosave_directory
    if directory is None or not directory.exists():
        return 0
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("measurement autosave path is not a safe directory")
    if snapshot.state not in {
        MeasurementState.STOPPED_WITH_RESULTS,
        MeasurementState.ERROR_RECOVERABLE,
        MeasurementState.ERROR_FATAL,
    }:
        raise ValueError("active measurement autosave cannot be removed")

    manifest_path = directory / "session_manifest.json"
    if not manifest_path.exists():
        manifest_path = directory / "measurement.json"
    if not manifest_path.is_file():
        raise ValueError("measurement autosave manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("session_id", "")) != str(snapshot.session_id):
        raise ValueError("measurement autosave identity does not match")

    released_bytes = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    shutil.rmtree(directory)
    return int(released_bytes)


def session_zip_bytes(snapshot: SessionSnapshot) -> bytes:
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory() as temp:
        root = export_measurement_session(snapshot, Path(temp) / "measurement_session")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            from ..output_terms import write_output_terms
            write_output_terms(archive)
            for path in root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(root.parent))
        return output.getvalue()


def _write_manifest(path: Path, snapshot: SessionSnapshot) -> None:
    payload = {
        "schema_version": 4,
        "retain_until_original_committed": True,
        "session_id": snapshot.session_id,
        "state": str(snapshot.state),
        "revision": max(0, int(snapshot.revision)),
        "settings": _jsonable(asdict(snapshot.settings)),
        "shot_count": len(snapshot.shots),
        "started_at": snapshot.started_at,
        "stopped_at": snapshot.stopped_at,
        "error_message": snapshot.error_message,
        "device_info": _jsonable(snapshot.device_info),
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_shot(path: Path, shot: ShotResult) -> None:
    arrays = {name: value for name in ARRAY_FIELDS if (value := getattr(shot, name)) is not None}
    metadata = {
        "shot_index": shot.shot_index,
        "status": str(shot.status),
        "detected_start_sample": shot.detected_start_sample,
        "detected_end_sample": shot.detected_end_sample,
        "correlation": shot.correlation,
        "detection_confidence": shot.detection_confidence,
        "timing_error_ms": shot.timing_error_ms,
        "detected_duration_s": shot.detected_duration_s,
        "detected_period_s": shot.detected_period_s,
        "included_in_average": shot.included_in_average,
        "sweep_index": shot.sweep_index or shot.shot_index,
        "timing_segment": shot.timing_segment,
        "timing_index": shot.timing_index,
        "loop_boundary_before": shot.loop_boundary_before,
        "reference_mode": shot.reference_mode,
        "reference_plane": shot.reference_plane,
        "absolute_timing_valid": shot.absolute_timing_valid,
        "marker_detected": shot.marker_detected,
        "marker_verified": shot.marker_verified,
        "center_method": shot.center_method,
        "center_polarity": shot.center_polarity,
        "center_sample": shot.center_sample,
        "center_confidence": shot.center_confidence,
        "center_tracking_state": shot.center_tracking_state,
        "center_search_start_sample": shot.center_search_start_sample,
        "center_search_end_sample": shot.center_search_end_sample,
        "input_gain_normalization_db": shot.input_gain_normalization_db,
        "shot_input_gain_db": shot.settings_snapshot.input_gain_db,
        "shot_input_gain_db_channels": shot.settings_snapshot.input_gain_db_channels,
        "shot_input_gain_max_db_channels": shot.settings_snapshot.input_gain_max_db_channels,
        "shot_input_analog_gain_db": shot.settings_snapshot.input_analog_gain_db,
        "shot_minidsp_gain_adjustment_db": shot.settings_snapshot.minidsp_gain_adjustment_db,
        "shot_response_gain_reference_db": shot.settings_snapshot.response_gain_reference_db,
        "raw_peak_sample": shot.raw_peak_sample,
        "centered_ir_similarity": shot.centered_ir_similarity,
        "gain_valid": shot.gain_valid,
        "relative_phase_valid": shot.relative_phase_valid,
        "timing_valid": shot.timing_valid,
        "quality_tier": shot.quality_tier,
        "merge_valid": shot.merge_valid,
        "effective_merge_crossover_hz": shot.effective_merge_crossover_hz,
        "quality": _jsonable(asdict(shot.quality)),
    }
    if shot.wavelet_map is not None:
        wavelet = shot.wavelet_map
        for name in (
            "time_ms", "frequency_hz", "level_db", "peak_time_ms", "centroid_time_ms",
            "spread_ms", "confidence", "reflection_index",
        ):
            value = getattr(wavelet, name, None)
            if value is not None:
                arrays[f"wavelet__{name}"] = value
        metadata["wavelet"] = {
            name: _jsonable(getattr(wavelet, name))
            for name in (
                "label", "reference_db", "source_type", "reconstruction_confidence",
                "phase_available", "smoothing_detected", "estimated_smoothing_oct",
                "smoothing_likelihood", "source_warning", "alignment_center_sample",
            )
        }
    np.savez_compressed(path, metadata_json=json.dumps(metadata, ensure_ascii=False), **arrays)


def _write_standard(path: Path, result: HighPrecisionResult) -> None:
    metadata = {
        "used_shot_indices": result.used_shot_indices,
        "reference_shot_index": result.reference_shot_index,
        "alignment_samples": result.alignment_samples,
        "clock_drift_ppm": result.clock_drift_ppm,
        "clock_residual_samples": result.clock_residual_samples,
        "clock_corrected": result.clock_corrected,
        "algorithm": result.algorithm,
        "reference_mode": result.reference_mode,
        "timing_valid_shot_count": result.timing_valid_shot_count,
        "absolute_timing_valid": result.absolute_timing_valid,
        "arrival_sample": result.arrival_sample,
        "observation_samples": result.observation_samples,
        "minimum_center_confidence": result.minimum_center_confidence,
    }
    np.savez_compressed(
        path,
        metadata_json=json.dumps(metadata, ensure_ascii=False),
        frequency_hz=result.frequency_hz,
        ungated_complex_response=result.ungated_complex_response,
        gated_complex_response=result.gated_complex_response,
        merged_complex_response=result.merged_complex_response,
        coherence=result.coherence,
        repeatability_db=result.repeatability_db,
        confidence=result.confidence,
        **({"integrated_uncalibrated_ir": result.integrated_uncalibrated_ir} if result.integrated_uncalibrated_ir is not None else {}),
    )


def _read_standard(path: Path) -> HighPrecisionResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        return HighPrecisionResult(
            frequency_hz=np.array(data["frequency_hz"]),
            ungated_complex_response=np.array(data["ungated_complex_response"]),
            gated_complex_response=np.array(data["gated_complex_response"]),
            merged_complex_response=np.array(data["merged_complex_response"]),
            coherence=np.array(data["coherence"]),
            repeatability_db=np.array(data["repeatability_db"]),
            confidence=np.array(data["confidence"]),
            used_shot_indices=tuple(int(value) for value in metadata["used_shot_indices"]),
            reference_shot_index=int(metadata["reference_shot_index"]),
            alignment_samples=tuple(float(value) for value in metadata["alignment_samples"]),
            clock_drift_ppm=float(metadata.get("clock_drift_ppm", 0.0)),
            clock_residual_samples=float(metadata.get("clock_residual_samples", float("nan"))),
            clock_corrected=bool(metadata.get("clock_corrected", False)),
            algorithm=str(metadata.get("algorithm", "subsample_aligned_quality_weighted_robust_complex_average")),
            reference_mode=str(metadata.get("reference_mode", "ir_peak")),
            timing_valid_shot_count=int(metadata.get("timing_valid_shot_count", 0)),
            absolute_timing_valid=bool(metadata.get("absolute_timing_valid", False)),
            integrated_uncalibrated_ir=(np.array(data['integrated_uncalibrated_ir']) if 'integrated_uncalibrated_ir' in data else None),
            arrival_sample=float(metadata.get('arrival_sample', float('nan'))),
            observation_samples=int(metadata.get('observation_samples', 0)),
            minimum_center_confidence=float(metadata.get('minimum_center_confidence', 0.0)),
        )


def _read_shot(path: Path, settings: MeasurementSettings) -> ShotResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        quality_data = metadata["quality"]
        quality_data["status"] = ShotStatus(quality_data["status"])
        quality_data["messages"] = tuple(quality_data.get("messages", ()))
        quality = ShotQuality(**quality_data)
        arrays: dict[str, np.ndarray | None] = {}
        for name in ARRAY_FIELDS:
            arrays[name] = np.array(data[name]) if name in data else None
        wavelet_map = None
        if "wavelet" in metadata:
            from wavelet_analysis import WaveletMap

            wavelet_arrays = {}
            for name in (
                "time_ms", "frequency_hz", "level_db", "peak_time_ms", "centroid_time_ms",
                "spread_ms", "confidence", "reflection_index",
            ):
                key = f"wavelet__{name}"
                wavelet_arrays[name] = np.array(data[key]) if key in data else None
            required = {name: wavelet_arrays.pop(name) for name in ("time_ms", "frequency_hz", "level_db")}
            wavelet_map = WaveletMap(**required, **wavelet_arrays, **metadata["wavelet"])
    shot_settings = replace(
        settings,
        input_gain_db=float(metadata.get("shot_input_gain_db", settings.input_gain_db)),
        input_gain_db_channels=tuple(metadata.get("shot_input_gain_db_channels", settings.input_gain_db_channels)),
        input_gain_max_db_channels=tuple(metadata.get("shot_input_gain_max_db_channels", settings.input_gain_max_db_channels)),
        input_analog_gain_db=metadata.get("shot_input_analog_gain_db", settings.input_analog_gain_db),
        minidsp_gain_adjustment_db=float(metadata.get("shot_minidsp_gain_adjustment_db", settings.minidsp_gain_adjustment_db)),
        response_gain_reference_db=metadata.get("shot_response_gain_reference_db", settings.response_gain_reference_db),
    )
    return ShotResult(
        shot_index=int(metadata["shot_index"]),
        status=ShotStatus(metadata["status"]),
        detected_start_sample=int(metadata["detected_start_sample"]),
        detected_end_sample=int(metadata["detected_end_sample"]),
        correlation=float(metadata["correlation"]),
        detection_confidence=float(metadata["detection_confidence"]),
        timing_error_ms=float(metadata["timing_error_ms"]),
        detected_duration_s=float(metadata["detected_duration_s"]),
        detected_period_s=metadata.get("detected_period_s"),
        wavelet_map=wavelet_map,
        quality=quality,
        settings_snapshot=shot_settings,
        included_in_average=bool(metadata.get("included_in_average", True)),
        sweep_index=int(metadata.get("sweep_index", metadata["shot_index"])),
        timing_segment=int(metadata.get("timing_segment", 0)),
        timing_index=int(metadata.get("timing_index", -1)),
        loop_boundary_before=bool(metadata.get("loop_boundary_before", False)),
        reference_mode=str(metadata.get("reference_mode", settings.reference_mode)),
        reference_plane=str(metadata.get("reference_plane", "peak_centered")),
        absolute_timing_valid=bool(metadata.get("absolute_timing_valid", False)),
        marker_detected=bool(metadata.get("marker_detected", False)),
        marker_verified=bool(metadata.get("marker_verified", False)),
        center_method=str(metadata.get("center_method", "absolute_peak")),
        center_polarity=str(metadata.get("center_polarity", "auto")),
        center_sample=float(metadata.get("center_sample", float("nan"))),
        center_confidence=float(metadata.get("center_confidence", 0.0)),
        center_tracking_state=str(metadata.get("center_tracking_state", "initial")),
        center_search_start_sample=int(metadata.get("center_search_start_sample", 0)),
        center_search_end_sample=int(metadata.get("center_search_end_sample", 0)),
        input_gain_normalization_db=float(metadata.get("input_gain_normalization_db", 0.0)),
        raw_peak_sample=float(metadata.get("raw_peak_sample", float("nan"))),
        centered_ir_similarity=float(metadata.get("centered_ir_similarity", float("nan"))),
        gain_valid=bool(metadata.get("gain_valid", True)),
        relative_phase_valid=bool(metadata.get("relative_phase_valid", True)),
        timing_valid=bool(metadata.get("timing_valid", False)),
        quality_tier=str(metadata.get("quality_tier", "basic")),
        merge_valid=bool(metadata.get("merge_valid", True)),
        effective_merge_crossover_hz=float(metadata.get("effective_merge_crossover_hz", float("nan"))),
        **arrays,
    )


def _write_result_files(directory: Path, shot: ShotResult, prefix: str) -> None:
    sample_rate = shot.settings_snapshot.sample_rate
    if shot.combined_raw_ir is not None:
        wavfile.write(directory / f"{prefix}combined_raw_ir.wav", sample_rate, np.asarray(shot.combined_raw_ir, dtype=np.float32))
    if shot.gated_ir is not None:
        wavfile.write(directory / f"{prefix}gated_ir.wav", sample_rate, np.asarray(shot.gated_ir, dtype=np.float32))
    if shot.wavelet_map is not None:
        wavelet = shot.wavelet_map
        arrays = {
            "time_ms": wavelet.time_ms,
            "frequency_hz": wavelet.frequency_hz,
            "level_db": wavelet.level_db,
        }
        for name in ("peak_time_ms", "centroid_time_ms", "spread_ms", "confidence", "reflection_index"):
            value = getattr(wavelet, name, None)
            if value is not None:
                arrays[name] = value
        np.savez_compressed(directory / f"{prefix}wavelet.npz", **arrays)
    if shot.frequency_hz is not None:
        for name, response in (
            ("ungated", shot.ungated_complex_response),
            ("gated", shot.gated_complex_response),
            ("merged", shot.merged_complex_response),
        ):
            if response is not None:
                _write_frd(directory / f"{prefix}response_{name}.frd", shot.frequency_hz, response)
                _write_level_frd(
                    directory / f"{prefix}level_{name}_dbfs.frd",
                    shot.frequency_hz,
                    response_level_dbfs(response, shot.settings_snapshot),
                    "Level(dBFS RMS)",
                )
                spl = response_level_spl(response, shot.settings_snapshot)
                if spl is not None:
                    _write_level_frd(
                        directory / f"{prefix}level_{name}_spl.frd",
                        shot.frequency_hz,
                        spl,
                        "Level(dB SPL)",
                    )
    quality = _jsonable(asdict(shot.quality))
    quality["input_levels_dbfs"] = {
        "left_peak": _dbfs(shot.quality.left_peak),
        "right_peak": _dbfs(shot.quality.right_peak),
        "left_rms": _dbfs(shot.quality.left_rms),
        "right_rms": _dbfs(shot.quality.right_rms),
    }
    (directory / f"{prefix}quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_frd(path: Path, frequency: np.ndarray, response: np.ndarray) -> None:
    gain = 20 * np.log10(np.maximum(np.abs(response), 1e-15))
    phase = np.rad2deg(np.unwrap(np.angle(response)))
    rows = (f"{f:.9g}\t{g:.9g}\t{p:.9g}" for f, g, p in zip(frequency, gain, phase))
    path.write_text("* Frequency(Hz)\tGain(dB)\tPhase(deg)\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _write_metric_frd(path: Path, frequency: np.ndarray, values: np.ndarray, label: str) -> None:
    rows = (
        f"{float(f):.9g}\t{float(value):.9g}"
        for f, value in zip(frequency, values)
        if np.isfinite(f) and np.isfinite(value)
    )
    path.write_text(f"* Frequency(Hz)\t{label}\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _write_level_frd(path: Path, frequency: np.ndarray, values: np.ndarray, label: str) -> None:
    rows = (
        f"{float(f):.9g}\t{float(value):.9g}"
        for f, value in zip(frequency, values)
        if np.isfinite(f) and np.isfinite(value)
    )
    path.write_text(f"* Frequency(Hz)\t{label}\n" + "\n".join(rows) + "\n", encoding="utf-8")


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-15)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
