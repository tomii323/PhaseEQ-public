"""One-shot Target-menu navigation for an already connected PhaseEQ session."""
from utils.exchange_io import atomic_bytes
import json
import time
import uuid
from pathlib import Path
from utils.browser_presence import browser_session_is_open


def request_target_menu(root):
    root = Path(root)
    sessions = [p for p in (root / 'browser_presence' / 'phaseeq').glob('*.json')
                if browser_session_is_open(root, 'phaseeq', p.stem)]
    if not sessions:
        return False
    session = max(sessions, key=lambda p: p.stat().st_mtime)
    directory = root / 'navigation_requests'
    directory.mkdir(parents=True, exist_ok=True)
    payload = {'id':uuid.uuid4().hex, 'session_id':session.stem, 'created':time.time(), 'page':'Target'}
    temporary = directory / (payload['id'] + '.tmp')
    try:
        temporary.write_text(json.dumps(payload))
        temporary.replace(directory / f'{session.stem}.json')
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def target_menu_request(root, session_id):
    if not session_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in session_id):
        return None
    path = Path(root) / 'navigation_requests' / f'{session_id}.json'
    try:
        payload = json.loads(path.read_text())
        # Inactive browser tabs may suspend Streamlit's fragment timer. Keep
        # the request for this exact session until it can receive it.
        if payload.get('page') == 'Target' and payload.get('session_id') == session_id:
            return str(payload['id'])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def acknowledge_target_menu(root, session_id, request_id):
    if target_menu_request(root, session_id) != request_id:
        return
    directory = Path(root) / 'navigation_requests'
    path = directory / f'{request_id}.ack'
    try:
        if path.read_text() == session_id:
            return
    except FileNotFoundError:
        pass
    atomic_bytes(path, session_id.encode())


def target_menu_acknowledged(root, request):
    try:
        return (Path(root) / 'navigation_requests' / f"{request['id']}.ack").read_text() == request['session_id']
    except OSError:
        return False
