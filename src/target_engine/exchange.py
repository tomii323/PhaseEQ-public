from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid

from .definition import canonical_target_definition


TARGET_EDIT_SESSION_FORMAT_VERSION = 1


@dataclass(frozen=True)
class TargetEditSession:
    session_id: str
    group: str
    return_url: str
    state: str
    source_definition: dict[str, Any]
    current_definition: dict[str, Any]
    revision: int
    updated_at: str


def create_target_edit_session(
    root: Path,
    *,
    group: str,
    return_url: str,
    definition: dict[str, Any],
    session_id: str | None = None,
) -> TargetEditSession:
    session_id = _safe_id(session_id or str(uuid.uuid4()))
    if not session_id:
        raise ValueError("Target edit session ID is invalid")
    canonical = canonical_target_definition(definition)
    session = TargetEditSession(
        session_id=session_id,
        group=str(group).strip() or "Main",
        return_url=str(return_url).strip(),
        state="Editing",
        source_definition=canonical,
        current_definition=canonical,
        revision=0,
        updated_at=_utc_now(),
    )
    _write_session(root, session)
    return session


def read_target_edit_session(root: Path, session_id: str) -> TargetEditSession | None:
    normalized = _safe_id(session_id)
    if not normalized:
        return None
    path = Path(root) / "target_edit_sessions" / normalized / "session.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != TARGET_EDIT_SESSION_FORMAT_VERSION:
        raise ValueError("Unsupported Target edit session format")
    source = payload.get("source_definition")
    current = payload.get("current_definition")
    if not isinstance(source, dict) or not isinstance(current, dict):
        raise ValueError("Target edit session definition is invalid")
    return TargetEditSession(
        session_id=normalized,
        group=str(payload.get("group", "Main")).strip() or "Main",
        return_url=str(payload.get("return_url", "")).strip(),
        state=str(payload.get("state", "Editing")),
        source_definition=canonical_target_definition(source),
        current_definition=canonical_target_definition(current),
        revision=max(0, int(payload.get("revision", 0))),
        updated_at=str(payload.get("updated_at", "")),
    )


def save_target_edit_result(
    root: Path,
    session_id: str,
    definition: dict[str, Any],
) -> TargetEditSession:
    current = read_target_edit_session(root, session_id)
    if current is None:
        raise ValueError("Target edit session was not found")
    updated = TargetEditSession(
        session_id=current.session_id,
        group=current.group,
        return_url=current.return_url,
        state="Ready",
        source_definition=current.source_definition,
        current_definition=canonical_target_definition(definition),
        revision=current.revision + 1,
        updated_at=_utc_now(),
    )
    _write_session(root, updated)
    return updated


def _write_session(root: Path, session: TargetEditSession) -> None:
    directory = Path(root) / "target_edit_sessions" / session.session_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "session.json"
    temporary = path.with_name(".session.json.tmp")
    payload = {
        "format_version": TARGET_EDIT_SESSION_FORMAT_VERSION,
        "session_id": session.session_id,
        "group": session.group,
        "return_url": session.return_url,
        "state": session.state,
        "source_definition": session.source_definition,
        "current_definition": session.current_definition,
        "revision": session.revision,
        "updated_at": session.updated_at,
    }
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_id(value: str) -> str:
    normalized = str(value).strip()
    return normalized if normalized and all(
        character.isalnum() or character in "_-" for character in normalized
    ) else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
