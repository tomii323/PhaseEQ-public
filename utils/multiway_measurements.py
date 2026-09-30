"""Manual, workspace-scoped measurement handoff; never infer a channel."""
from __future__ import annotations

from utils.exchange_io import locked_exchange
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid
from copy import deepcopy

from utils.composite_exchange import _atomic_json, read_multiway_workspace
from utils.measurement_payloads import (
    speaker_input_from_measurement_payload, timing_provenance_from_measurement_payload,
    speaker_calibration_state_from_measurement_payload,
)
from utils.settings_io import speaker_response_payload


def workspace_key(workspace):
    identity = [workspace['system_id'], workspace['sample_rate_hz'], sorted(workspace['channel_ids'])]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def mapping_path(root, workspace):
    return Path(root) / 'measurement_inputs' / (workspace_key(workspace) + '.json')


def read_measurement_inputs(root, workspace):
    path = mapping_path(root, workspace)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('測定入力の形式が不正です。')
    if payload.get('workspace_key') != workspace_key(workspace) or not isinstance(payload.get('channels'), dict):
        raise ValueError('測定データの受け渡し情報を読み込めません。')
    channels = payload['channels']
    for channel, item in channels.items():
        if not isinstance(channel, str) or not isinstance(item, dict):
            raise ValueError('測定入力のチャンネル項目が不正です。')
        for key in ('request_id', 'measurement_id', 'name'):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f'測定入力の{key}がありません。再送信してください。')
        if not all(c.isalnum() or c in '_-' for c in item['request_id']):
            raise ValueError('測定入力のrequest_idが不正です。')
        if not isinstance(item.get('response'), dict) or not isinstance(item.get('metadata', {}), dict):
            raise ValueError('測定入力の応答またはメタデータが不正です。')
        if 'measurement_payload' in item and not isinstance(item['measurement_payload'], dict):
            raise ValueError('測定入力の保存データが不正です。')
    return channels


@locked_exchange
def set_measurement_input(root, workspace, channel_id, record):
    current = read_multiway_workspace(root)
    if current is None or workspace_key(current) != workspace_key(workspace):
        raise ValueError('マルチウェイの構成が変わりました。画面を更新して送信先を選び直してください。')
    if channel_id not in current['channel_ids']:
        raise ValueError('送信先チャンネルが現在のマルチウェイにありません。')
    response = speaker_input_from_measurement_payload(record.response_payload)
    if record.source_type != 'speaker_input_raw' or response is None:
        raise ValueError('スピーカーの測定データを選んでください。')
    channels = read_measurement_inputs(root, current)
    channels[channel_id] = dict(
        request_id=uuid.uuid4().hex, measurement_id=record.id, name=record.name,
        project_name=str(current.get('system_name') or ''),
        updated_at=datetime.now(timezone.utc).isoformat(),
        response=speaker_response_payload(response),
        measurement_payload=deepcopy(record.response_payload),
        timing_provenance=timing_provenance_from_measurement_payload(record.response_payload),
        calibration_state=speaker_calibration_state_from_measurement_payload(record.response_payload),
        metadata={key: getattr(record, key) for key in (
            'brand', 'model', 'measurement_date', 'distance', 'angle', 'microphone',
            'location', 'channel', 'position_name', 'system_state', 'note')},
    )
    _atomic_json(mapping_path(root, current), dict(workspace_key=workspace_key(current), channels=channels))
    return channels[channel_id]


def assignment_measurement_input(root, assignment):
    """Resolve the roster for this Assignment, never the last opened project."""
    if assignment is None:
        return None
    workspace = read_multiway_workspace(root, assignment=assignment)
    if workspace is None:
        return None
    return read_measurement_inputs(root, workspace).get(assignment.channel_id)


def measurement_input_record(item, db_path):
    from utils.speaker_db import SpeakerMeasurementRecord, get_measurement
    payload = item.get('measurement_payload')
    if not isinstance(payload, dict):
        # Earlier handoffs only carried the main response. Recover the bundle
        # from the DB only while it still matches the sent snapshot.
        saved = get_measurement(db_path, item['measurement_id'])
        response = speaker_input_from_measurement_payload(saved.response_payload) if saved else None
        if (response is not None and speaker_response_payload(response) == item['response']
                and speaker_calibration_state_from_measurement_payload(saved.response_payload) == item.get('calibration_state', 'raw')):
            return saved
        payload = dict(item['response'], timing_provenance=item.get('timing_provenance'),
                       input_calibration_state=item.get('calibration_state'),
                       external_mic_calibration_applied=item.get('calibration_state') in {'measurement_calibrated', 'legacy_calibrated'})
    try:
        return SpeakerMeasurementRecord(id=item['measurement_id'], name=item['name'],
            response_payload=deepcopy(payload), **item.get('metadata', {}))
    except TypeError as exc:
        raise ValueError('測定入力のメタデータ形式が不正です。') from exc


def measurement_asset(item):
    from utils.settings_io import speaker_response_from_payload
    response = speaker_response_from_payload(item['response'])
    if response is None:
        raise ValueError('測定応答がありません。')
    phases = response.phase_deg if response.phase_deg is not None else [0.0] * len(response.frequency)
    text = '\n'.join(f'{f:.17g} {g:.17g} {p:.17g}' for f, g, p in zip(response.frequency, response.gain_db, phases))
    return dict(filename='measurement.frd', data=text.encode(), name=item['name'])


def apply_measurement_inputs(root, workspace, rows):
    """Set only explicitly addressed rows; newer PhaseEQ returns may supersede them."""
    inputs = read_measurement_inputs(root, workspace)
    applied = []
    for row in rows:
        item = inputs.get(str(row.get('channel_id', '')))
        if item is None:
            continue
        row['speaker_library_response'] = measurement_asset(item)
        row['measurement_input'] = item
        # A new explicit selection supersedes the old returned speaker response.
        # Subsequent PhaseEQ processing remains the authoritative processed input.
        from composite_engine.adapter.control import read_phaseeq_assignment_status
        assignment_id = str(row.get('phaseeq_assignment_id', ''))
        status = read_phaseeq_assignment_status(root, assignment_id=assignment_id) if assignment_id else None
        published = (status.published_at or status.updated_at) if status else ''
        if not published or datetime.fromisoformat(published.replace('Z', '+00:00')) <= datetime.fromisoformat(item['updated_at']):
            row.pop('phaseeq_speaker_response', None)
            row['timing_provenance'] = item.get('timing_provenance')
        applied.append(dict(channel_id=row['channel_id'], name=row.get('name', ''), measurement=item['name']))
        ack_path = Path(root) / 'measurement_inputs' / 'received' / (item['request_id'] + '.json')
        if not ack_path.exists():
            _atomic_json(ack_path, dict(channel_id=row['channel_id'], measurement_id=item['measurement_id']))
    return applied
