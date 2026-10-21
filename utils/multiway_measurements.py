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
    _validate_channels(channels)
    return channels


def _validate_channels(channels):
    if not isinstance(channels, dict):
        raise ValueError('測定入力のチャンネル一覧が不正です。')
    for channel, item in channels.items():
        if not isinstance(channel, str) or not isinstance(item, dict):
            raise ValueError('測定入力のチャンネル項目が不正です。')
        for key in ('request_id', 'name'):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f'測定入力の{key}がありません。再送信してください。')
        if not all(c.isalnum() or c in '_-' for c in item['request_id']):
            raise ValueError('測定入力のrequest_idが不正です。')
        if input_is_empty(item):
            if item.get('response') is not None or item.get('measurement_id'):
                raise ValueError('解除済み入力に測定応答が含まれています。')
            continue
        if (not isinstance(item.get('measurement_id'), str) or not item['measurement_id']
                or not isinstance(item.get('response'), dict) or not isinstance(item.get('metadata', {}), dict)):
            raise ValueError('測定入力の応答またはメタデータが不正です。')
        if 'measurement_payload' in item and not isinstance(item['measurement_payload'], dict):
            raise ValueError('測定入力の保存データが不正です。')


@locked_exchange
def restore_measurement_inputs(root, workspace, channels, *, known_channel_ids=None):
    """Restore a complete saved snapshot; missing legacy fields never call this."""
    _validate_channels(channels)
    known = workspace['channel_ids'] if known_channel_ids is None else known_channel_ids
    if set(channels) - set(known):
        raise ValueError('保存した測定入力とチャンネル構成が一致しません。')
    _atomic_json(mapping_path(root, workspace),
                 dict(workspace_key=workspace_key(workspace), channels=deepcopy(channels)))


@locked_exchange
def measurement_input_snapshot(root, workspace):
    """Read once; acknowledge exactly this content, never a later file stat."""
    channels = read_measurement_inputs(root, workspace)
    signature = hashlib.sha256(json.dumps(channels, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
    return (workspace_key(workspace), signature), channels


def measurement_update_required(applied, pending, incoming):
    """One rerun per unconsumed snapshot; initialization is consumed in render."""
    return applied is not None and incoming != applied and incoming != pending


def input_is_empty(item):
    """Absence is legacy/unconfigured; an explicit empty item masks all fallbacks."""
    return isinstance(item, dict) and item.get('state') == 'empty'


def _current_input_channels(root, workspace, channel_id):
    current = read_multiway_workspace(root)
    if current is None or workspace_key(current) != workspace_key(workspace):
        raise ValueError('マルチウェイの構成が変わりました。画面を更新して送信先を選び直してください。')
    if channel_id not in current['channel_ids']:
        raise ValueError('送信先チャンネルが現在のマルチウェイにありません。')
    return read_measurement_inputs(root, current)


@locked_exchange
def clear_measurement_input(root, workspace, channel_id, *, expected_request_id):
    """Commit a confirmed generation, retaining filters and the library original."""
    channels = _current_input_channels(root, workspace, channel_id)
    if channels.get(channel_id, {}).get('request_id', '') != expected_request_id:
        raise ValueError('割り当てが更新されています。内容を確認して解除をやり直してください。')
    item = dict(state='empty', request_id=uuid.uuid4().hex,
                measurement_id='', name='未設定（Unity）', response=None,
                updated_at=datetime.now(timezone.utc).isoformat())
    channels[channel_id] = item
    _atomic_json(mapping_path(root, workspace), dict(workspace_key=workspace_key(workspace), channels=channels))
    return item


def measurement_result_matches(root, assignment, result, *, allow_retained_filters=False):
    """One receipt contract for FIR ON/OFF and cached results, including legacy."""
    item = assignment_measurement_input(root, assignment)
    if item is None:
        return not result.get('measurement_input_request_id')
    request = result.get('measurement_input_request_id')
    if not input_is_empty(item) and request is not None and request == item['request_id']:
        return True
    from composite_engine.adapter.control import read_phaseeq_assignment_status
    status = read_phaseeq_assignment_status(root, assignment_id=assignment.assignment_id)
    published = (status.published_at or status.updated_at) if status else ''
    # Already published filters survive release/reassignment. The source resolver
    # and apply step mask their Speaker snapshot; a late stale publication fails.
    if allow_retained_filters and published and datetime.fromisoformat(published.replace('Z', '+00:00')) <= datetime.fromisoformat(item['updated_at']):
        return True
    if input_is_empty(item):
        return False
    if request is not None:
        return request == item['request_id']
    if item.get('state') == 'assigned':
        return False
    # Old publishers have no generation. Accept only before a later explicit change.
    return bool(published and datetime.fromisoformat(published.replace('Z', '+00:00'))
                >= datetime.fromisoformat(item['updated_at']))


@locked_exchange
def set_measurement_input(root, workspace, channel_id, record):
    channels = _current_input_channels(root, workspace, channel_id)
    response = speaker_input_from_measurement_payload(record.response_payload)
    if record.source_type != 'speaker_input_raw' or response is None:
        raise ValueError('スピーカーの測定データを選んでください。')
    channels[channel_id] = _measurement_input_item(read_multiway_workspace(root), record)
    _atomic_json(mapping_path(root, workspace), dict(workspace_key=workspace_key(workspace), channels=channels))
    return channels[channel_id]


def _measurement_input_item(current, record):
    response = speaker_input_from_measurement_payload(record.response_payload)
    from phase_fir_designer.ideal_input import normalize_input_definition, ideal_speaker_response
    definition = normalize_input_definition((record.response_payload or {}).get('input_definition'))
    if definition is not None:
        response = ideal_speaker_response(int(current['sample_rate_hz']), 16384)

    if response is None:
        raise ValueError("スピーカー入力応答がありません。")
    return dict(
        state='assigned',
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


def prepare_initial_measurement_inputs(root, workspace, *, previous_workspace, eligible_ids, record):
    """Stage only new Channel defaults before publishing the new roster."""
    retained = ({key: item for key, item in read_measurement_inputs(root, previous_workspace).items()
                 if key in workspace['channel_ids']} if previous_workspace else {})
    current = read_measurement_inputs(root, workspace)
    channels = {**retained, **current}
    previous_ids = set(previous_workspace.get('channel_ids', [])) if previous_workspace else set()
    from phase_fir_designer.ideal_input import plan_input_selection
    changed = channels != current
    for channel in workspace['channel_ids']:
        selection = plan_input_selection(saved_state_exists=channel in previous_ids or channel in channels)
        if channel in eligible_ids and selection.kind == 'ideal':
            channels[channel] = _measurement_input_item(workspace, record)
            changed = True
    if changed:
        _atomic_json(mapping_path(root, workspace), dict(workspace_key=workspace_key(workspace), channels=channels))


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
    from phase_fir_designer.ideal_input import normalize_input_definition
    definition = normalize_input_definition((item.get('measurement_payload') or {}).get('input_definition'))
    return dict(filename='measurement.frd', data=text.encode(), name=item['name'],
                generated_ideal=definition is not None)


def apply_measurement_inputs(root, workspace, rows, *, inputs=None):
    """Set only explicitly addressed rows; newer PhaseEQ returns may supersede them."""
    inputs = read_measurement_inputs(root, workspace) if inputs is None else inputs
    applied = []
    prepared = []
    for row in rows:
        item = inputs.get(str(row.get('channel_id', '')))
        if item is None:
            continue
        asset = None if input_is_empty(item) else measurement_asset(item)
        # A new explicit selection supersedes the old returned speaker response.
        # Subsequent PhaseEQ processing remains the authoritative processed input.
        from composite_engine.adapter.control import read_phaseeq_assignment_status
        assignment_id = str(row.get('phaseeq_assignment_id', ''))
        status = read_phaseeq_assignment_status(root, assignment_id=assignment_id) if assignment_id else None
        published = (status.published_at or status.updated_at) if status else ''
        supersedes = (not published or datetime.fromisoformat(published.replace('Z', '+00:00'))
                      <= datetime.fromisoformat(item['updated_at']))
        result = getattr(status, 'result', None)
        if isinstance(result, dict) and result.get('measurement_input_request_id') is not None:
            supersedes = result['measurement_input_request_id'] != item['request_id']
        prepared.append((row, item, asset, supersedes))
    # Validate the whole read snapshot before changing any Channel response.
    for row, item, asset, supersedes in prepared:
        row['measurement_input'] = item
        if asset is None:
            for key in ('speaker_library_response', 'phaseeq_speaker_response', 'timing_provenance'):
                row.pop(key, None)
        else:
            row['speaker_library_response'] = asset
        if asset is not None and supersedes:
            row.pop('phaseeq_speaker_response', None)
            row['timing_provenance'] = item.get('timing_provenance')
        applied.append(dict(channel_id=row['channel_id'], name=row.get('name', ''), measurement=item['name']))
        ack_path = Path(root) / 'measurement_inputs' / 'received' / (item['request_id'] + '.json')
        if not ack_path.exists():
            _atomic_json(ack_path, dict(channel_id=row['channel_id'], measurement_id=item['measurement_id']))
    return applied
