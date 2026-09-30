"""Short lived leases written only by connected Streamlit browser sessions."""
import json
import time
import tempfile
from pathlib import Path


def publish_presence(root: Path, app: str, session_id: str) -> None:
    directory = Path(root) / "browser_presence" / app
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session_id}.json"
    # Foreground status updates and the heartbeat can publish the same session
    # concurrently. Each writer owns its temporary file until atomic replace.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=f"{session_id}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps({"updated": time.time(), "exchange_protocol": 1}))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def browser_session_is_open(root: Path, app: str, session_id: str, *, now: float | None = None) -> bool:
    if not session_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in session_id):
        return False
    timestamp = time.time() if now is None else now
    try:
        path = Path(root) / "browser_presence" / app / f"{session_id}.json"
        age = timestamp - float(json.loads(path.read_text())["updated"])
        return 0 <= age < 15
    except (OSError, ValueError, TypeError, KeyError):
        return False


def browser_is_open(root: Path, app: str, *, now: float | None = None) -> bool:
    return any(browser_session_is_open(root, app, path.stem, now=now)
               for path in (Path(root) / "browser_presence" / app).glob("*.json"))


# A live browser can be busy rendering for longer than its lease. Track the
# WebSocket session independently of Streamlit's script/fragment execution.
import logging
import threading

_presence_threads: dict[tuple[str, str, str], threading.Thread] = {}
_presence_lock = threading.Lock()


def start_browser_presence(root, app, session_id, is_connected, *, interval=2.0):
    token = (str(root), app, session_id)
    with _presence_lock:
        existing = _presence_threads.get(token)
        if existing is not None and existing.is_alive():
            return existing

        def maintain():
            try:
                while is_connected():
                    publish_presence(root, app, session_id)
                    time.sleep(interval)
            except (OSError, RuntimeError):
                logging.getLogger(__name__).exception("Browser presence tracking stopped")
            finally:
                with _presence_lock:
                    _presence_threads.pop(token, None)

        thread = threading.Thread(target=maintain, name=f"{app}-browser-presence", daemon=True)
        _presence_threads[token] = thread
        thread.start()
        return thread
