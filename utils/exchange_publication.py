"""Immutable result assets projected from the committed status record."""
import hashlib
import json
from pathlib import Path


def working_session_from_result(root, assignment_id, result):
    path = Path(str(result.get('working_session', ''))).resolve()
    path.relative_to((Path(root) / 'assignments' / assignment_id / 'results').resolve())
    if hashlib.sha256(path.read_bytes()).hexdigest() != result.get('working_session_hash'):
        raise ValueError('公開Working Sessionのhashが一致しません。')
    return path


def validated_fir_asset(root, sample_rate, channel):
    directory = Path(root) / str(sample_rate)
    name = channel.get('wav') or channel.get('frd')
    if not isinstance(name, str) or not name:
        raise ValueError('公開FIRの参照先がありません。')
    path = (directory / name).resolve()
    path.relative_to(directory.resolve())
    data = path.read_bytes()
    if channel.get('content_hash') and hashlib.sha256(data).hexdigest() != channel['content_hash']:
        raise ValueError('公開FIRのhashが一致しません。')
    return data


def _published(root, sample_rate):
    from utils.exchange_io import read_status_payload
    latest = {}
    for path in (Path(root) / 'assignments').glob('*/status.json'):
        try:
            payload = read_status_payload(path)
        except (OSError, ValueError):
            continue
        result = payload.get('result')
        if (payload.get('state') not in ('Ready', 'Editing') or not isinstance(result, dict)
                or result.get('asset_schema') != 1 or result.get('sample_rate_hz') != sample_rate):
            continue
        key = payload.get('channel_id')
        if not isinstance(key, str) or not key:
            continue
        order = (str(payload.get('published_at') or ''), str(payload.get('assignment_id') or ''))
        if key not in latest or order > latest[key][0]:
            latest[key] = (order, result)
    return [(key, item[1]) for key, item in latest.items()]


def has_published_exchange(root, sample_rate):
    try:
        return bool(_projected_payload(root, sample_rate)['channels'])
    except (OSError, ValueError, TypeError, KeyError):
        # The package loader reports the concrete error in the UI.
        return True


def _projected_payload(root, sample_rate):
    directory = Path(root) / str(sample_rate)
    manifest = directory / 'manifest.json'
    payload = json.loads(manifest.read_text()) if manifest.is_file() else dict(
        format_version=2, sample_rate_hz=sample_rate, channels=[])
    channels = list(payload['channels'])
    for channel_id, result in _published(root, sample_rate):
        channel = result.get('exchange_channel')
        if result.get('tap_count') == 0:
            channels = [row for row in channels if row.get('channel_id') != channel_id
                        and row.get('name') != result.get('channel_name')]
            continue
        if not isinstance(channel, dict):
            continue
        channels = [row for row in channels if row.get('channel_id') != channel.get('channel_id')
                    and row.get('name') != channel.get('name')]
        channels.append(channel)
    payload['channels'] = channels
    payload['composite_groups'] = list(dict.fromkeys(row['group'] for row in channels))
    return payload


def published_exchange_package(root, sample_rate):
    from composite_engine.adapter import CompositePackage
    payload = _projected_payload(root, sample_rate)
    assets = {}
    for row in payload['channels']:
        name = row.get('wav') or row.get('frd')
        assets[name] = validated_fir_asset(root, sample_rate, row)
    return CompositePackage.from_uploads(json.dumps(payload).encode(), assets)


def publication_identity(result):
    """Ignore ZIP container timestamps, never ignore saved configuration bytes."""
    import io
    import zipfile
    value = dict(result)
    if value.get('asset_schema') == 1:
        data = Path(value['working_session']).read_bytes()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            contents = [(name, hashlib.sha256(archive.read(name)).hexdigest())
                        for name in sorted(archive.namelist())]
        value['working_session'] = contents
        value.pop('working_session_hash', None)
        value.pop('manifest', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_compatible_writers(root):
    import time
    for path in (Path(root) / 'browser_presence').glob('*/*.json'):
        try:
            payload = json.loads(path.read_text())
            if 0 <= time.time() - float(payload.get('updated', 0)) < 15 and payload.get('exchange_protocol') != 1:
                raise ValueError('旧版の連携画面が開いています。PhaseEQとマルチウェイを再起動してから送信してください。')
        except (OSError, json.JSONDecodeError):
            continue
