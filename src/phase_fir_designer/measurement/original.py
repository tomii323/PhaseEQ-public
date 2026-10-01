"""One immutable, uncalibrated integrated IR and reproducible derived outputs.

DSP contract v1: align and gain-normalize raw IRs, integrate, gate, then
subtract microphone deviations. The IR is circular, with its center at sample
zero; the separately retained arrival is never inferred again after calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
import zipfile

import numpy as np
from scipy.io import wavfile

from .analysis import high_precision_result, reanalyze_gate
from .levels import level_calibration_offset_db, response_level_dbfs, response_level_spl
from .models import HighPrecisionResult, MeasurementSettings, SessionSnapshot, ShotQuality, ShotResult, ShotStatus


CALIBRATION_FIELDS = tuple(
    name for name in MeasurementSettings.__dataclass_fields__
    if name.startswith(('mic_', 'microphone_', 'minidsp_', 'level_'))
)


@dataclass(frozen=True)
class MeasurementOriginal:
    impulse: np.ndarray
    recipe: dict
    calibration: dict
    metadata: dict

    @property
    def sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.impulse, dtype='<f4').tobytes()).hexdigest()

    @property
    def storage_metadata(self) -> dict:
        return {**self.metadata,
                'calibration_sha256': hashlib.sha256(json.dumps(self.calibration, sort_keys=True).encode()).hexdigest(),
                'recipe_sha256': hashlib.sha256(json.dumps(self.recipe, sort_keys=True).encode()).hexdigest()}

    @property
    def settings(self) -> MeasurementSettings:
        values = {**self.recipe, **self.calibration['settings']}
        for key, field in MeasurementSettings.__dataclass_fields__.items():
            if isinstance(field.default, tuple) and key in values:
                values[key] = tuple(values[key])
        return MeasurementSettings(**values)

    def ir_bytes(self) -> bytes:
        output = io.BytesIO()
        np.savez_compressed(output, integrated_uncalibrated_ir=np.asarray(self.impulse, dtype='<f4'))
        return output.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, recipe: dict, calibration: dict, metadata: dict) -> MeasurementOriginal:
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            if set(arrays.files) != {'integrated_uncalibrated_ir'}:
                raise ValueError('測定原本の配列構成が不正です。')
            impulse = np.array(arrays['integrated_uncalibrated_ir'], dtype=np.float32)
        if impulse.ndim != 1 or impulse.size < 2 or not np.all(np.isfinite(impulse)):
            raise ValueError('測定原本IRが不正です。')
        result = cls(impulse, recipe, calibration, metadata)
        if result.sha256 != metadata.get('original_sha256'):
            raise ValueError('測定原本IRのhashが一致しません。')
        for name in ('calibration_sha256', 'recipe_sha256'):
            if name in metadata and metadata[name] != result.storage_metadata[name]:
                raise ValueError('測定原本の校正・条件hashが一致しません。')
        if calibration.get('curve_semantics') != 'microphone_deviation':
            raise ValueError('校正データはマイク偏差である必要があります。')
        return result


def original_from_snapshot(snapshot: SessionSnapshot) -> MeasurementOriginal:
    if snapshot.saved_original is not None:
        return snapshot.saved_original
    result = snapshot.standard_result
    if result is None or result.integrated_uncalibrated_ir is None:
        result = high_precision_result(snapshot.valid_shots)
    if result is None or result.integrated_uncalibrated_ir is None:
        raise ValueError('統合未校正IRを確定できません。')
    values = asdict(snapshot.settings)
    calibration = {'curve_semantics': 'microphone_deviation', 'operation': 'subtract',
                   'settings': {key: values.pop(key) for key in CALIBRATION_FIELDS}, 'sources': []}
    metadata = {
        'schema_version': 1, 'processing_version': 'integrated-ir-gate-deviation-v1',
        'session_id': snapshot.session_id, 'sample_rate_hz': snapshot.settings.sample_rate,
        'time_origin': 'circular_peak_centered', 'center_sample': 0.0,
        'arrival_sample': result.arrival_sample, 'observation_samples': result.observation_samples,
        'observation_kind': 'captured_audio_samples_not_ir_decay_support',
        'retained_samples': len(result.integrated_uncalibrated_ir),
        'input_gain_normalization': 'included_before_integration', 'normalization': 'none',
        'used_shot_indices': list(result.used_shot_indices),
        'reference_shot_index': result.reference_shot_index,
        'alignment_samples': list(result.alignment_samples),
        'clock_drift_ppm': result.clock_drift_ppm, 'clock_corrected': result.clock_corrected,
        'clock_residual_samples': result.clock_residual_samples,
        'minimum_center_confidence': result.minimum_center_confidence,
        'absolute_timing_valid': result.absolute_timing_valid,
        'timing_valid_shot_count': result.timing_valid_shot_count,
        'median_coherence': result.median_coherence, 'repeatability_p90_db': result.repeatability_p90_db,
    }
    original = MeasurementOriginal(result.integrated_uncalibrated_ir, values, calibration, metadata)
    metadata['original_sha256'] = original.sha256
    return original


def shot_from_original(original: MeasurementOriginal) -> ShotResult:
    settings = original.settings
    # This is a runtime view of the aggregate, not a retained acquisition shot.
    quality = ShotQuality(ShotStatus.VALID, 0., 0., 0., 0., 0., 0., 1., 0., False, False, False, 'integrated')
    shot = ShotResult(
        shot_index=0, status=ShotStatus.VALID, detected_start_sample=0,
        detected_end_sample=int(original.metadata['observation_samples']),
        correlation=1., detection_confidence=1., timing_error_ms=0.,
        detected_duration_s=float(original.metadata['observation_samples']) / settings.sample_rate,
        detected_period_s=None, raw_stereo_audio=None, combined_raw_ir=original.impulse,
        gated_ir=None, frequency_hz=None, ungated_complex_response=None,
        gated_complex_response=None, merged_complex_response=None, wavelet_map=None,
        quality=quality, settings_snapshot=settings, center_sample=0.,
        center_search_start_sample=-1, center_search_end_sample=len(original.impulse),
        center_tracking_state='integrated_original', reference_mode=settings.reference_mode,
        timing_valid=bool(original.metadata['absolute_timing_valid']),
        absolute_timing_valid=bool(original.metadata['absolute_timing_valid']),
    )
    return reanalyze_gate(shot, settings)


def result_from_original(original: MeasurementOriginal) -> HighPrecisionResult:
    shot = shot_from_original(original)
    n = len(shot.frequency_hz)
    meta = original.metadata
    return HighPrecisionResult(
        frequency_hz=shot.frequency_hz, ungated_complex_response=shot.ungated_complex_response,
        gated_complex_response=shot.gated_complex_response, merged_complex_response=shot.merged_complex_response,
        used_shot_indices=tuple(meta['used_shot_indices']), reference_shot_index=int(meta['reference_shot_index']),
        alignment_samples=tuple(meta['alignment_samples']), coherence=np.full(n, np.nan),
        repeatability_db=np.full(n, np.nan), confidence=np.full(n, np.nan),
        summary_median_coherence=meta['median_coherence'], summary_repeatability_p90_db=meta['repeatability_p90_db'],
        minimum_center_confidence=meta['minimum_center_confidence'],
        integrated_uncalibrated_ir=original.impulse, arrival_sample=float(meta['arrival_sample']),
        observation_samples=int(meta['observation_samples']), clock_drift_ppm=float(meta['clock_drift_ppm']),
        clock_residual_samples=float(meta['clock_residual_samples']),
        clock_corrected=bool(meta['clock_corrected']), reference_mode=original.settings.reference_mode,
        absolute_timing_valid=bool(meta['absolute_timing_valid']), timing_valid_shot_count=int(meta['timing_valid_shot_count']),
    )


def calibrated_output(original: MeasurementOriginal, response_kind: str = 'merged') -> tuple[np.ndarray, np.ndarray, dict]:
    if response_kind not in {'ungated', 'gated', 'merged'}:
        raise ValueError('Unknown response kind')
    shot = shot_from_original(original)
    response = np.asarray(getattr(shot, f'{response_kind}_complex_response'), dtype=np.complex128)
    settings = original.settings
    spl = response_level_spl(response, settings)
    gain = response_level_dbfs(response, settings) if spl is None else spl
    scaled = 10 ** (gain / 20) * np.exp(1j * np.angle(response))
    # A real IR has real DC/Nyquist bins. Use the same representable response
    # for FRD and WAV rather than silently discarding imaginary endpoint bins.
    scaled[[0, -1]] = scaled[[0, -1]].real
    lower = float(settings.start_frequency_hz)
    upper = min(float(settings.end_frequency_hz), settings.sample_rate / 2)
    if settings.mic_calibration_frequency_hz:
        lower = max(lower, min(settings.mic_calibration_frequency_hz))
        upper = min(upper, max(settings.mic_calibration_frequency_hz))
    meta = {**original.storage_metadata, 'response_kind': response_kind, 'gate_before_calibration': True,
            'calibration_application': {
                'schema_version': 1, 'curve_semantics': 'microphone_deviation', 'operation': 'subtract',
                'magnitude_applied': bool(settings.mic_calibration_frequency_hz and settings.mic_calibration_gain_db),
                'phase_applied': bool(settings.mic_phase_calibration_frequency_hz and settings.mic_phase_calibration_deg),
                'sensitivity_applied': level_calibration_offset_db(settings) is not None,
            },
            'gain_unit': 'dBFS RMS' if spl is None else 'dB SPL',
            'wav_sample_unit': 'relative_to_20uPa' if spl is not None else 'sine_equivalent_full_scale_rms',
            'reference_peak_dbfs': settings.reference_peak_dbfs,
            'level_offset_db': level_calibration_offset_db(settings),
            'valid_frequency_range_hz': [lower, upper], 'calibration': original.calibration,
            'gate_start_ms': settings.gate_start_ms, 'gate_end_ms': settings.gate_end_ms}
    return np.asarray(shot.frequency_hz), scaled, meta


def calibrated_export_zip(original: MeasurementOriginal, response_kind: str = 'merged') -> bytes:
    frequency, response, meta = calibrated_output(original, response_kind)
    impulse = np.fft.irfft(response, n=len(original.impulse)).astype(np.float32)
    wav = io.BytesIO()
    wavfile.write(wav, original.settings.sample_rate, impulse)
    valid = (frequency >= meta['valid_frequency_range_hz'][0]) & (frequency <= meta['valid_frequency_range_hz'][1])
    rows = np.column_stack((frequency[valid], 20 * np.log10(np.maximum(np.abs(response[valid]), 1e-15)),
                            np.rad2deg(np.unwrap(np.angle(response)))[valid]))
    frd = io.StringIO()
    np.savetxt(frd, rows, fmt='%.12g', header=f"Frequency(Hz) Gain({meta['gain_unit']}) Phase(deg); calibrated, {response_kind}", comments='* ')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        from ..output_terms import write_output_terms
        write_output_terms(archive)
        archive.writestr('calibrated_ir.wav', wav.getvalue())
        archive.writestr('calibrated_response.frd', frd.getvalue())
        archive.writestr('calibrated_output.json', json.dumps(_external_metadata(meta), ensure_ascii=False, indent=2, allow_nan=False))
    return output.getvalue()


def _external_metadata(value):
    """Unavailable timings/levels are JSON null, never nonstandard NaN tokens."""
    if isinstance(value, dict):
        return {key: _external_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_external_metadata(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
