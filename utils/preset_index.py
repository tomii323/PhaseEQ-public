"""Read-only byte index for legacy JSON catalogs, without decoding responses."""
from copy import deepcopy
from functools import lru_cache
import json
import mmap
from pathlib import Path
import re

from utils.preset_store import _preset_id

_TOKEN = re.compile(rb'"(?:[^"\\]|\\.)*"|[{}\[\]]')
_SPACE = re.compile(rb'\s*')
_SCALAR = re.compile(rb'[^,}\]\s]+')
HEAVY_FIELDS = frozenset({'target_response_raw', 'materialized_response'})


def file_revision(path):
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return ()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _start(data, position):
    return _SPACE.match(data, position).end()


def _end(data, position):
    first = data[position:position + 1]
    if first == b'"':
        match = _TOKEN.match(data, position)
        if not match:
            raise ValueError('Invalid preset JSON string')
        return match.end()
    if first not in (b'{', b'['):
        match = _SCALAR.match(data, position)
        if not match:
            raise ValueError('Invalid preset JSON value')
        return match.end()
    stack = []
    for match in _TOKEN.finditer(data, position):
        token = match.group()
        if token in (b'{', b'['):
            stack.append(token)
        elif token in (b'}', b']'):
            if not stack or stack.pop() != (b'{' if token == b'}' else b'['):
                raise ValueError('Invalid preset JSON nesting')
            if not stack:
                return match.end()
    raise ValueError('Unterminated preset JSON value')


def _members(data, position):
    if data[position:position + 1] != b'{':
        raise ValueError('Preset JSON must be an object')
    position = _start(data, position + 1)
    while data[position:position + 1] != b'}':
        end = _end(data, position)
        key = json.loads(data[position:end])
        position = _start(data, end)
        if data[position:position + 1] != b':':
            raise ValueError('Invalid preset JSON member')
        start = _start(data, position + 1)
        end = _end(data, start)
        yield key, start, end
        position = _start(data, end)
        if data[position:position + 1] == b'}':
            break
        if data[position:position + 1] != b',':
            raise ValueError('Invalid preset JSON separator')
        position = _start(data, position + 1)


@lru_cache(maxsize=4)
def _index(path, revision):
    if not revision:
        return ()
    records = []
    with Path(path).open('rb') as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        top = {key: (start, end) for key, start, end in _members(data, _start(data, 0))}
        if 'kind' not in top or json.loads(data[slice(*top['kind'])]) != 'target_response':
            raise ValueError('Preset file kind must be target_response')
        if 'presets' not in top:
            return ()
        position = top['presets'][0]
        if data[position:position + 1] != b'[':
            raise ValueError('Preset list is invalid')
        position = _start(data, position + 1)
        while data[position:position + 1] != b']':
            end = _end(data, position)
            if data[position:position + 1] == b'{':
                metadata = {}
                for key, start, stop in _members(data, position):
                    if key in HEAVY_FIELDS:
                        # Preserve presence/type for legacy category inference.
                        metadata[key] = {} if data[start:start + 1] == b'{' else None
                    else:
                        metadata[key] = json.loads(data[start:stop])
                metadata['id'] = _preset_id(metadata.get('id'), str(metadata.get('name', 'preset')))
                metadata['name'] = str(metadata.get('name', 'Unnamed preset')).strip() or 'Unnamed preset'
                records.append((metadata, position, end))
            position = _start(data, end)
            if data[position:position + 1] == b']':
                break
            if data[position:position + 1] != b',':
                raise ValueError('Invalid preset list separator')
            position = _start(data, position + 1)
    return tuple(records)


def preset_summaries(path):
    return [deepcopy(item) for item, _, _ in _index(Path(path), file_revision(path))]


def preset_record(path, record_id):
    path = Path(path)
    for _ in range(2):
        revision = file_revision(path)
        spans = [(start, end) for item, start, end in _index(path, revision) if item['id'] == record_id]
        if not spans:
            return []
        with path.open('rb') as handle:
            records = []
            for start, end in spans:
                handle.seek(start)
                records.append(json.loads(handle.read(end - start)))
        if file_revision(path) == revision:
            return records
    raise ValueError('Target一覧が更新されました。再度選択してください。')
