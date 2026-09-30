"""Bounded process locks and atomic replacement for local Exchange writers."""
from contextlib import contextmanager
from functools import wraps
import json
import os
from pathlib import Path
import tempfile
import threading
import time

_local = threading.local()


@contextmanager
def exchange_lock(root, *, timeout=1.0):
    # Never unlink this inode: another process may already be waiting on it.
    root = Path(root).resolve()
    token = str(root)
    held = getattr(_local, 'held', set())
    if token in held:
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    from utils.runtime_instances import _lock_file, _unlock_file
    with (root / '.exchange.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'1')
            handle.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                _lock_file(handle, blocking=False)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError('連携データを更新中です。少し待って再試行してください。') from exc
                time.sleep(0.02)
        _local.held = held | {token}
        try:
            yield
        finally:
            _local.held = held
            _unlock_file(handle)


def locked_exchange(function):
    import inspect
    root_parameter = next(iter(inspect.signature(function).parameters))
    @wraps(function)
    def wrapped(*args, **kwargs):
        root = args[0] if args else kwargs[root_parameter]
        with exchange_lock(root):
            return function(*args, **kwargs)
    return wrapped


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    if shutil.disk_usage(path.parent).free < len(data) + 1024 * 1024:
        raise OSError('連携データを保存する空き容量が不足しています。前回結果を保持します。')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, payload):
    atomic_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))


def read_status_payload(path):
    """Both UI and adapter readers use the same legacy fallback contract."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('連携statusの形式が不正です。')
    if payload.get('publication_schema') == 1:
        if payload.get('result') is not None and not isinstance(payload['result'], dict):
            raise ValueError('公開結果の形式が不正です。')
        return payload
    if not isinstance(payload.get('result'), dict):
        workspace_path = path.parent / 'workspace.json'
        if workspace_path.is_file():
            workspace = json.loads(workspace_path.read_text(encoding='utf-8'))
            payload['result'] = workspace.get('last_result')
            payload['published_at'] = payload.get('published_at') or workspace.get('last_result_published_at', '')
    return payload
