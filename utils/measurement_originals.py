"""Library boundary for original IR persistence, recalibration and export."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from phase_fir_designer.measurement.original import MeasurementOriginal, calibrated_output


def payload_for_original(original: MeasurementOriginal, previous: dict | None = None) -> dict:
    payload = deepcopy(previous or {})
    frequency, response, metadata = calibrated_output(original)
    payload.update(frequency=frequency.tolist(),
                   gain_db=(20 * np.log10(np.maximum(np.abs(response), 1e-15))).tolist(),
                   phase_deg=np.rad2deg(np.unwrap(np.angle(response))).tolist())
    session = payload.setdefault('measurement_session', {})
    if isinstance(original.metadata.get('timing_provenance'), dict):
        payload.setdefault('timing_provenance', deepcopy(original.metadata['timing_provenance']))
        session.setdefault('timing_provenance', deepcopy(payload['timing_provenance']))
    # No waveform is embedded in response JSON. The database transaction stores
    # the sole original BLOB separately from its calibration and response cache.
    session.pop('combined_raw_ir', None)
    session.pop('timed_response', None)
    session['original'] = {**original.storage_metadata, 'storage': 'measurement_ir_originals'}
    session['calibration_application'] = metadata['calibration_application']
    session['calibration_application']['output_finalized'] = True
    session['level'] = {**session.get('level', {}), 'gain_unit': metadata['gain_unit']}
    session['valid_frequency_range_hz'] = metadata['valid_frequency_range_hz']
    payload['auto_iir_metadata'] = {
        'valid_f_min_hz': metadata['valid_frequency_range_hz'][0],
        'valid_f_max_hz': metadata['valid_frequency_range_hz'][1],
        'source': 'integrated_measurement_original',
    }
    session['mic_calibration_id'] = original.settings.mic_calibration_id
    session['ir_sample_rate_hz'] = original.settings.sample_rate
    session['center_sample'] = 0.0
    session['derivation'] = {key: original.metadata[key] for key in
                             ('used_shot_indices', 'reference_shot_index', 'alignment_samples', 'processing_version')}
    return payload


def load_measurement_original(path: Path, record_id: str) -> MeasurementOriginal:
    from utils.speaker_db import _connect
    with _connect(path) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='measurement_ir_originals'").fetchone():
            raise ValueError('この測定には統合未校正IRがありません。旧データは変更しません。')
        row = conn.execute('SELECT * FROM measurement_ir_originals WHERE measurement_id=?', (record_id,)).fetchone()
    if row is None:
        raise ValueError('測定原本がありません。元の測定DBまたは測定パッケージを復元してください。')
    return MeasurementOriginal.from_bytes(bytes(row['ir_npz']), json.loads(row['recipe_json']),
                                         json.loads(row['calibration_json']), json.loads(row['metadata_json']))


def attach_calibration_sources(original: MeasurementOriginal, path: Path) -> MeasurementOriginal:
    from utils.speaker_db import get_measurement
    calibration = deepcopy(original.calibration)
    previous_sources = {item['id']: item for item in calibration.get('sources', []) if 'id' in item}
    sources = []
    for record_id in (original.settings.mic_calibration_id, original.settings.mic_phase_calibration_id):
        if not record_id or any(source['id'] == record_id for source in sources):
            continue
        record = get_measurement(path, record_id)
        if record is not None:
            sources.append({'id': record.id, 'name': record.name, 'source_name': record.source_name,
                            'raw_text': record.raw_text, 'response_payload': record.response_payload})
        elif record_id in previous_sources:
            sources.append(previous_sources[record_id])
    calibration['sources'] = sources
    return replace(original, calibration=calibration)


def recalibrate_measurement(path: Path, record_id: str, calibration_id: str):
    from utils.speaker_db import get_measurement, save_measurement
    from utils.settings_io import speaker_response_from_payload
    original = load_measurement_original(path, record_id)
    record = get_measurement(path, record_id)
    calibration_record = get_measurement(path, calibration_id)
    if record is None or calibration_record is None or calibration_record.source_type != 'mic_calibration_raw':
        raise ValueError('測定またはマイク校正データがありません。')
    curve = speaker_response_from_payload(calibration_record.response_payload)
    if curve is None:
        raise ValueError('マイク校正曲線を読み込めません。')
    calibration = deepcopy(original.calibration)
    calibration['settings'].update(mic_calibration_id=calibration_id,
        mic_calibration_frequency_hz=list(curve.frequency), mic_calibration_gain_db=list(curve.gain_db))
    changed = attach_calibration_sources(replace(original, calibration=calibration), path)
    # Calibration changes are not a new arrival measurement. Bind the existing
    # timing provenance to the regenerated response while retaining its times.
    payload = payload_for_original(changed, record.response_payload)
    for provenance in (payload.get('timing_provenance'), payload['measurement_session'].get('timing_provenance')):
        if isinstance(provenance, dict):
            # P0 binds its processed response hash at the publishing boundary.
            provenance['response_hash'] = ''
    return save_measurement(path, record=replace(record, response_payload=payload), original=changed)
