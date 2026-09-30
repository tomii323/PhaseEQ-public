from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from utils.exchange_io import atomic_json as _safe_atomic_json, atomic_bytes, locked_exchange

import numpy as np
from scipy.io import wavfile


@dataclass(frozen=True)
class CompositeExchangeState:
    directory: Path
    manifest_path: Path
    channel_names: tuple[str, ...]
    group_names: tuple[str, ...]


@dataclass(frozen=True)
class CompositeAssignment:
    sample_rate_hz: int
    tap_count: int
    channel_name: str
    way: str
    group: str
    assignment_id: str = ""
    channel_id: str = ""
    return_url: str = ""
    revision: int = 0
    multiway_mode: str = ""
    highpass_hz: float = 0.0
    lowpass_hz: float = 0.0
    linear_fir_filters: tuple[dict[str, object], ...] = ()
    iir_crossover: dict[str, object] | None = None
    band_split_recipe: dict[str, object] | None = None
    target_definition: dict[str, object] | None = None
    target_application: str = "legacy_apply"
    format_version: int = 2
    system_id: str = ""
    management_no: str = ""
    system_revision_number: int | None = None
    system_content_hash: str = ""
    target_id: str = ""
    target_revision_number: int | None = None
    target_content_hash: str = ""
    assignment_signature: str = ""

    @property
    def fir_enabled(self) -> bool:
        return int(self.tap_count) > 0


@dataclass(frozen=True)
class CompositeAssignmentStatus:
    assignment_id: str
    channel_id: str
    state: str
    revision: int
    updated_at: str
    message: str = ""
    result: dict[str, object] | None = None
    published_at: str = ""


@dataclass(frozen=True)
class PhaseEQSessionStatus:
    session_id: str
    mode: str
    updated_at: str
    assignment_id: str = ""
    channel_id: str = ""
    channel_name: str = ""
    group: str = ""
    target_edit_session_id: str = ""
    active_page: str = ""


def write_phaseeq_session_status(
    root: Path,
    *,
    session_id: str,
    mode: str,
    assignment_id: str = "",
    channel_id: str = "",
    channel_name: str = "",
    group: str = "",
    target_edit_session_id: str = "",
    active_page: str = "",
) -> PhaseEQSessionStatus:
    normalized_session_id = _safe_id(session_id)
    if not normalized_session_id:
        raise ValueError("invalid PhaseEQ session id")
    normalized_mode = str(mode).strip()
    if normalized_mode not in {"Standalone", "TargetEdit", "Assignment"}:
        raise ValueError("invalid PhaseEQ session mode")
    status = PhaseEQSessionStatus(
        session_id=normalized_session_id,
        mode=normalized_mode,
        updated_at=_utc_now(),
        assignment_id=_safe_id(assignment_id),
        channel_id=_safe_id(channel_id),
        channel_name=str(channel_name).strip(),
        group=str(group).strip(),
        target_edit_session_id=_safe_id(target_edit_session_id),
        active_page=str(active_page),
    )
    _atomic_json(
        Path(root) / "phaseeq_sessions" / f"{normalized_session_id}.json",
        status.__dict__,
    )
    return status


def list_active_phaseeq_sessions(
    root: Path, *, max_age_seconds: float = 30.0,
) -> tuple[PhaseEQSessionStatus, ...]:
    """Return recently heartbeating PhaseEQ tabs without deleting stale records."""
    sessions_root = Path(root) / "phaseeq_sessions"
    if not sessions_root.is_dir():
        return ()
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(float(max_age_seconds), 0.0))
    sessions: list[PhaseEQSessionStatus] = []
    for path in sessions_root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            updated_at = datetime.fromisoformat(str(payload.get("updated_at", "")))
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            from utils.browser_presence import browser_session_is_open
            if updated_at < cutoff and not browser_session_is_open(
                root, "phaseeq", str(payload.get("session_id", "")),
            ):
                continue
            status = PhaseEQSessionStatus(
                session_id=_safe_id(str(payload.get("session_id", ""))),
                mode=str(payload.get("mode", "")),
                updated_at=updated_at.isoformat(),
                assignment_id=_safe_id(str(payload.get("assignment_id", ""))),
                channel_id=_safe_id(str(payload.get("channel_id", ""))),
                channel_name=str(payload.get("channel_name", "")).strip(),
                group=str(payload.get("group", "")).strip(),
                target_edit_session_id=_safe_id(str(payload.get("target_edit_session_id", ""))),
                active_page=str(payload.get("active_page", "")),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if status.session_id and status.mode in {"Standalone", "TargetEdit", "Assignment"}:
            sessions.append(status)
    return tuple(sorted(sessions, key=lambda item: (item.mode, item.session_id)))


def phaseeq_session_status_signature(root: Path, *, max_age_seconds: float = 30.0) -> str:
    # Heartbeat timestamps change every two seconds.  They are deliberately
    # excluded so a healthy tab does not look like a new user-visible event.
    records = [{
        "session_id": status.session_id,
        "mode": status.mode,
        "assignment_id": status.assignment_id,
        "channel_id": status.channel_id,
        "channel_name": status.channel_name,
        "group": status.group,
        "target_edit_session_id": status.target_edit_session_id,
    } for status in list_active_phaseeq_sessions(
        root, max_age_seconds=max_age_seconds,
    )]
    serialized = json.dumps(records, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def current_composite_return_url(current_base_url: str, stored_return_url: str) -> str:
    """Use the live launcher URL while preserving only the assignment query."""
    base = str(current_base_url).strip().rstrip("/")
    if not base:
        raise ValueError("current Composite Engine URL is required")
    query = urlsplit(str(stored_return_url).strip()).query
    return f"{base}?{query}" if query else base


def read_composite_assignment(
    root: Path,
    assignment_id: str | None = None,
    *, require_current: bool = False,
) -> CompositeAssignment | None:
    normalized_id = _safe_id(assignment_id or "")
    path = (
        Path(root) / "assignments" / normalized_id / "assignment.json"
        if normalized_id else Path(root) / "phaseeq_assignment.json"
    )
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    format_version = int(payload.get("format_version", 2))
    if format_version not in {2, 3}:
        raise ValueError("Unsupported Composite assignment format")
    system_payload = payload.get("system") if isinstance(payload.get("system"), dict) else {}
    target_payload = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    target_definition = (
        target_payload.get("definition_snapshot")
        if isinstance(target_payload.get("definition_snapshot"), dict)
        else payload.get("target_definition")
    )
    assignment = CompositeAssignment(
        sample_rate_hz=int(payload.get("sample_rate_hz", 0)),
        tap_count=int(payload.get("tap_count", 0)),
        channel_name=str(payload.get("channel_name", "")).strip(),
        way=str(payload.get("way", "")).strip(),
        group=str(payload.get("group", "")).strip(),
        assignment_id=str(payload.get("assignment_id", "")).strip(),
        channel_id=str(payload.get("channel_id", "")).strip(),
        return_url=str(payload.get("return_url", "")).strip(),
        revision=int(payload.get("assignment_state_revision", payload.get("revision", 0))),
        multiway_mode=str(payload.get("multiway_mode", "")).strip(),
        highpass_hz=float(payload.get("highpass_hz", 0.0)),
        band_split_recipe=payload.get("band_split_recipe"),
        lowpass_hz=float(payload.get("lowpass_hz", 0.0)),
        linear_fir_filters=tuple(
            dict(item) for item in payload.get("linear_fir_filters", [])
            if isinstance(item, dict)
        ),
        iir_crossover=(
            dict(payload["iir_crossover"])
            if isinstance(payload.get("iir_crossover"), dict) else None
        ),
        target_definition=dict(target_definition) if isinstance(target_definition, dict) else None,
        target_application=str(
            target_payload.get("application", payload.get("target_application", "legacy_apply"))
        ),
        format_version=format_version,
        system_id=str(system_payload.get("system_id", payload.get("system_id", ""))),
        management_no=str(system_payload.get("management_no", payload.get("management_no", ""))),
        system_revision_number=(
            int(system_payload.get("revision_number", payload.get("system_revision_number")))
            if system_payload.get("revision_number", payload.get("system_revision_number")) is not None
            else None
        ),
        system_content_hash=str(system_payload.get("content_hash", payload.get("system_content_hash", ""))),
        target_id=str(target_payload.get("target_id", payload.get("target_id", ""))),
        target_revision_number=(
            int(target_payload.get("revision_number", payload.get("target_revision_number")))
            if target_payload.get("revision_number", payload.get("target_revision_number")) is not None
            else None
        ),
        target_content_hash=str(target_payload.get("content_hash", payload.get("target_content_hash", ""))),
        assignment_signature=str(payload.get("assignment_signature", "")),
    )
    if assignment.tap_count < 0 or not assignment.channel_name or not assignment.way or not assignment.group:
        raise ValueError("Composite assignment is incomplete")
    if normalized_id and assignment.assignment_id != normalized_id:
        raise ValueError("Composite assignment ID mismatch")
    if require_current:
        pointer = Path(root) / "channels" / f"{_safe_id(assignment.channel_id)}.json"
        if pointer.is_file():
            current = json.loads(pointer.read_text(encoding="utf-8"))
            if str(current.get("assignment_id", "")) != assignment.assignment_id:
                raise ValueError("Composite Assignmentが古くなっています。最新のWayを選択してください。")
        workspace = read_multiway_workspace(root, assignment=assignment)
        channel_ids = workspace["channel_ids"] if workspace is not None else None
        if channel_ids is not None and assignment.channel_id not in channel_ids:
            raise ValueError("Composite AssignmentのChannelが現在のマルチウェイ構成にありません。")
    return assignment


def read_assignment_status(root: Path, assignment_id: str) -> CompositeAssignmentStatus | None:
    normalized_id = _safe_id(assignment_id)
    if not normalized_id:
        return None
    path = Path(root) / "assignments" / normalized_id / "status.json"
    if not path.is_file():
        return None
    from utils.exchange_io import read_status_payload
    payload = read_status_payload(path)
    result = payload.get('result')
    return CompositeAssignmentStatus(
        assignment_id=str(payload.get("assignment_id", "")),
        channel_id=str(payload.get("channel_id", "")),
        state=str(payload.get("state", "Assigned")),
        revision=int(payload.get("revision", 0)),
        updated_at=str(payload.get("updated_at", "")),
        message=str(payload.get("message", "")),
        result=dict(result) if isinstance(result, dict) else None,
        published_at=str(payload.get("published_at", "")),
    )


def latest_unclaimed_assignment_id(root: Path) -> str | None:
    """Return the newest valid Assignment that PhaseEQ has not opened yet."""
    assignments_root = Path(root) / "assignments"
    if not assignments_root.is_dir():
        return None
    candidates: list[tuple[str, float, str]] = []
    for status_path in assignments_root.glob("*/status.json"):
        assignment_id = _safe_id(status_path.parent.name)
        if not assignment_id:
            continue
        try:
            status = read_assignment_status(root, assignment_id)
            assignment = read_composite_assignment(root, assignment_id)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if status is None or assignment is None or status.state != "Assigned":
            continue
        try:
            modified = status_path.stat().st_mtime
        except OSError:
            modified = 0.0
        candidates.append((status.updated_at, modified, assignment_id))
    return max(candidates)[2] if candidates else None


def assignment_status_signature(
    root: Path,
    *,
    channel_ids: tuple[str, ...] = (),
    states: tuple[str, ...] = ("Ready", "Editing"),
) -> str:
    """Build a stable polling token without reading result assets."""
    allowed_channels = {_safe_id(value) for value in channel_ids if _safe_id(value)}
    allowed_states = {str(value) for value in states}
    assignments_root = Path(root) / "assignments"
    records: list[tuple[str, str, str, int, str]] = []
    if assignments_root.is_dir():
        for status_path in assignments_root.glob("*/status.json"):
            assignment_id = _safe_id(status_path.parent.name)
            if not assignment_id:
                continue
            try:
                status = read_assignment_status(root, assignment_id)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                status is None
                or status.state not in allowed_states
                or (allowed_channels and _safe_id(status.channel_id) not in allowed_channels)
            ):
                continue
            records.append((
                status.assignment_id, status.channel_id, status.state,
                status.revision, status.published_at,
            ))
    serialized = json.dumps(sorted(records), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def list_latest_composite_assignments(
    root: Path, *, system_id: str = "",
) -> tuple[CompositeAssignment, ...]:
    """List each channel's latest Assignment, optionally within one System."""
    channels_root = Path(root) / "channels"
    assignments: list[CompositeAssignment] = []
    if not channels_root.is_dir():
        return ()
    for pointer_path in channels_root.glob("*.json"):
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            assignment = read_composite_assignment(
                root, str(pointer.get("assignment_id", "")),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if assignment is None or (system_id and assignment.system_id != system_id):
            continue
        assignments.append(assignment)
    return tuple(sorted(
        assignments,
        key=lambda item: (item.group, item.way, item.channel_name, item.assignment_id),
    ))


def _multiway_workspace_path(root: Path, system_id: str, sample_rate_hz: int) -> Path:
    scope = hashlib.sha256(f"{system_id}:{int(sample_rate_hz)}".encode()).hexdigest()
    return Path(root) / "multiway_workspaces" / f"{scope}.json"


@locked_exchange
def write_multiway_workspace(
    root: Path, *, system_id: str, sample_rate_hz: int, channel_ids: list[str],
    channels: list[dict[str, str]] | None = None,
    system_name: str = "",
    previous_channel_ids: list[str] | None = None,
) -> None:
    """Publish editable Channels independently of the returned-FIR manifest."""
    payload = dict(format="phaseeq-multiway-workspace", format_version=1,
                   system_id=str(system_id), sample_rate_hz=int(sample_rate_hz),
                   channel_ids=list(dict.fromkeys(str(value) for value in channel_ids)))
    if channels is not None:
        payload["channels"] = channels
    if system_name:
        payload["system_name"] = str(system_name)
    # Removing returned/external rows from an old roster must not strand the
    # measurements already assigned to its remaining editable channels.
    scoped_path = _multiway_workspace_path(root, system_id, sample_rate_hz)
    if previous_channel_ids is not None and scoped_path.is_file():
        previous = json.loads(scoped_path.read_text(encoding="utf-8"))
        if (previous.get("system_id") == payload["system_id"]
                and previous.get("sample_rate_hz") == payload["sample_rate_hz"]
                and set(previous.get("channel_ids", [])) == set(previous_channel_ids)
                and set(payload["channel_ids"]) < set(previous_channel_ids)):
            from utils.multiway_measurements import read_measurement_inputs, mapping_path, workspace_key
            retained = {key: value for key, value in read_measurement_inputs(root, previous).items()
                        if key in payload["channel_ids"]}
            if retained:
                retained.update(read_measurement_inputs(root, payload))
                _atomic_json(mapping_path(root, payload), {
                    "workspace_key": workspace_key(payload), "channels": retained,
                })
    for path in (
        _multiway_workspace_path(root, system_id, sample_rate_hz),
        Path(root) / "multiway_workspace.json",
    ):
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) == payload:
            continue
        _atomic_json(path, payload)


def read_multiway_workspace(
    root: Path, *, assignment: CompositeAssignment | None = None,
) -> dict[str, object] | None:
    path = (Path(root) / "multiway_workspace.json" if assignment is None else
            _multiway_workspace_path(root, assignment.system_id, assignment.sample_rate_hz))
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict)
            or payload.get("format") != "phaseeq-multiway-workspace"
            or payload.get("format_version") != 1
            or not isinstance(payload.get("system_id"), str)
            or not isinstance(payload.get("sample_rate_hz"), int)
            or isinstance(payload.get("sample_rate_hz"), bool)
            or payload["sample_rate_hz"] <= 0
            or not isinstance(payload.get("channel_ids"), list)
            or not all(isinstance(value, str) and value for value in payload["channel_ids"])):
        raise ValueError("マルチウェイのChannel構成を読み込めません。Multiwayを再表示してください。")
    if assignment is not None and (
        payload["system_id"] != assignment.system_id or payload["sample_rate_hz"] != assignment.sample_rate_hz
    ):
        raise ValueError("マルチウェイのChannel構成とAssignmentのSystem・Fsが一致しません。")
    return payload


def read_current_composite_assignment(root: Path, assignment_id: str) -> CompositeAssignment | None:
    """Resolve old editor URLs only within the same live System and Channel."""
    previous = read_composite_assignment(root, assignment_id)
    if previous is None:
        return None
    latest = next((item for item in list_switchable_composite_assignments(root, assignment=previous)
                   if item.channel_id == previous.channel_id and item.system_id == previous.system_id), None)
    return read_composite_assignment(root, latest.assignment_id if latest is not None else assignment_id,
                                     require_current=True)


def list_switchable_composite_assignments(
    root: Path, *, assignment: CompositeAssignment | None = None,
) -> tuple[CompositeAssignment, ...]:
    workspace = read_multiway_workspace(root, assignment=assignment)
    reference = assignment or (read_composite_assignment(root) if workspace is None else None)
    if workspace is None and reference is None:
        return ()
    system_id = workspace["system_id"] if workspace is not None else reference.system_id
    sample_rate = workspace["sample_rate_hz"] if workspace is not None else reference.sample_rate_hz
    candidates = []
    for item in list_latest_composite_assignments(root, system_id=system_id):
        if item.system_id != system_id or item.sample_rate_hz != sample_rate:
            continue
        if workspace is not None and item.channel_id not in workspace["channel_ids"]:
            continue
        candidates.append(item)
    if workspace is None:
        # Legacy workspaces have no roster. Keep the newest logical Channel,
        # never infer membership from the partial returned-FIR manifest.
        candidates.sort(key=lambda item: (
            Path(root) / "assignments" / item.assignment_id / "assignment.json"
        ).stat().st_mtime_ns)
        candidates = list({(item.group, item.way, item.channel_name): item for item in candidates}.values())
    return tuple(sorted(candidates, key=lambda item: (item.group, item.way, item.channel_name)))


def current_manifest_channel_ids(
    root: Path, assignment: CompositeAssignment,
) -> tuple[str, ...] | None:
    """Return the channel set owned by the Assignment's current Multiway manifest.

    ``None`` means that no matching manifest can be resolved. An empty tuple is
    a valid manifest with no identifiable channels.
    """
    query = parse_qs(urlsplit(str(assignment.return_url)).query)
    exchange_values = query.get("exchange", [])
    exchange_key = str(exchange_values[0]).strip() if exchange_values else ""
    if not exchange_key:
        exchange_key = str(int(assignment.sample_rate_hz))
    manifest_path = Path(root) / exchange_key / "manifest.json"
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    channels = payload.get("channels", [])
    if not isinstance(channels, list):
        raise ValueError("Composite manifest channels must be a list")
    channel_ids = {
        _safe_id(str(item.get("channel_id", "")))
        for item in channels if isinstance(item, dict)
    }
    return tuple(sorted(channel_id for channel_id in channel_ids if channel_id))


@locked_exchange
def mark_assignment_status(
    root: Path,
    assignment_id: str,
    *,
    state: str,
    message: str = "",
    result: dict[str, object] | None = None,
    operation_id: str | None = None,
) -> CompositeAssignmentStatus:
    normalized_id = _safe_id(assignment_id)
    if not normalized_id:
        raise ValueError("invalid assignment id")
    assignment = read_composite_assignment(root, normalized_id)
    if assignment is None:
        raise ValueError("Composite assignment does not exist")
    directory = Path(root) / "assignments" / normalized_id
    previous = read_assignment_status(root, normalized_id)
    status_path = directory / 'status.json'
    previous_payload = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if operation_id and previous_payload.get('operation_id') == operation_id:
        from utils.exchange_publication import publication_identity
        if previous is None or publication_identity(previous.result) != publication_identity(result):
            raise ValueError('同じ送信IDの内容が変更されています。新しい送信として実行してください。')
        return previous
    revision = previous.revision if previous is not None else assignment.revision
    if previous is None or previous.state != state or previous.message != message or result is not None:
        revision += 1
    updated_at = _utc_now()
    payload: dict[str, object] = {
        "assignment_id": normalized_id,
        "channel_id": assignment.channel_id,
        "state": str(state),
        "revision": revision,
        "assignment_state_revision": revision,
        "assignment_signature": assignment.assignment_signature,
        "updated_at": updated_at,
        "message": str(message),
    }
    effective_result = result if result is not None else (
        previous.result if previous is not None else None
    )
    published_at = (
        updated_at if result is not None
        else previous.published_at if previous is not None else ""
    )
    if effective_result is not None:
        payload["result"] = effective_result
        if published_at:
            payload["published_at"] = published_at
    workspace_path = directory / "workspace.json"
    workspace = json.loads(workspace_path.read_text(encoding="utf-8")) if workspace_path.is_file() else {}
    workspace.update({
        "assignment_id": normalized_id,
        "channel_id": assignment.channel_id,
        "revision": revision,
        "assignment_state_revision": revision,
        "assignment_signature": assignment.assignment_signature,
        "state": str(state),
        "updated_at": updated_at,
    })
    if result is not None:
        workspace["last_result"] = result
        workspace["last_result_published_at"] = published_at
    if operation_id:
        from utils.exchange_publication import require_compatible_writers
        require_compatible_writers(root)
        if result and result.get('asset_schema') == 1:
            from utils.exchange_publication import working_session_from_result
            from composite_engine.assignment_snapshot import validated_speaker_asset
            read_composite_assignment(root, normalized_id, require_current=True)
            if result.get('sample_rate_hz') != assignment.sample_rate_hz or result.get('tap_count') != assignment.tap_count:
                raise ValueError('公開結果のFs／tapsとAssignmentが一致しません。')
            working_session_from_result(root, normalized_id, result)
            validated_speaker_asset(result, root, revision)
        payload['publication_schema'] = 1
        payload['operation_id'] = operation_id
        # Immutable workspace preparation precedes the sole publication point.
        workspace_data = json.dumps(workspace, ensure_ascii=False, sort_keys=True).encode()
        generation = hashlib.sha256(workspace_data).hexdigest()
        immutable_workspace = directory / 'results' / 'generations' / generation / 'workspace.json'
        atomic_bytes(immutable_workspace, workspace_data)
        payload['workspace_snapshot'] = str(immutable_workspace)
        _atomic_json(directory / 'status.json', payload)
        # Legacy mirror is non-authoritative; failure cannot undo publication.
        try:
            _atomic_json(workspace_path, workspace)
        except OSError:
            import logging
            logging.getLogger(__name__).exception('Published result; legacy workspace mirror could not be updated')
    else:
        if previous_payload.get('publication_schema') == 1:
            for key in ('publication_schema', 'operation_id', 'workspace_snapshot'):
                payload[key] = previous_payload[key]
        _atomic_json(workspace_path, workspace)
        _atomic_json(directory / 'status.json', payload)
    return CompositeAssignmentStatus(
        assignment_id=normalized_id,
        channel_id=assignment.channel_id,
        state=str(state),
        revision=revision,
        updated_at=updated_at,
        message=str(message),
        result=dict(effective_result) if isinstance(effective_result, dict) else None,
        published_at=published_at,
    )


def assignment_working_session_path(root: Path, assignment_id: str) -> Path | None:
    normalized_id = _safe_id(assignment_id)
    if not normalized_id:
        return None
    status = read_assignment_status(root, normalized_id)
    result = status.result if status is not None else None
    if isinstance(result, dict) and result.get('asset_schema') == 1:
        from utils.exchange_publication import working_session_from_result
        return working_session_from_result(root, normalized_id, result)
    path = Path(root) / "assignments" / normalized_id / "results" / "working_session.zip"
    return path if path.is_file() else None


def latest_published_assignment_working_session(
    root: Path,
    *,
    channel_id: str,
    exclude_assignment_id: str = "",
    system_id: str | None = None,
) -> tuple[str, Path] | None:
    """Find the newest published Working Session for one stable Composite channel."""
    normalized_channel_id = _safe_id(channel_id)
    excluded = _safe_id(exclude_assignment_id)
    if not normalized_channel_id:
        return None
    assignments_root = Path(root) / "assignments"
    if not assignments_root.is_dir():
        return None
    candidates: list[tuple[float, str, Path]] = []
    for status_path in assignments_root.glob("*/status.json"):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        assignment_id = _safe_id(str(payload.get("assignment_id", "")))
        if not assignment_id or assignment_id == excluded:
            continue
        try:
            status = read_assignment_status(root, assignment_id)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            status is None
            or _safe_id(status.channel_id) != normalized_channel_id
            or status.state not in {"Ready", "Editing"}
            or not isinstance(status.result, dict)
        ):
            continue
        working_session = assignment_working_session_path(root, assignment_id)
        if working_session is None:
            continue
        try:
            assignment = read_composite_assignment(root, assignment_id)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if assignment is None or assignment.channel_id != normalized_channel_id:
            continue
        if system_id is not None and assignment.system_id != system_id:
            continue
        try:
            published_order = datetime.fromisoformat(status.published_at).timestamp()
        except (TypeError, ValueError):
            try:
                published_order = working_session.stat().st_mtime
            except OSError:
                try:
                    published_order = datetime.fromisoformat(status.updated_at).timestamp()
                except (TypeError, ValueError):
                    published_order = 0.0
        candidates.append((
            published_order,
            assignment_id,
            working_session,
        ))
    if not candidates:
        return None
    _published_order, assignment_id, working_session = max(
        candidates, key=lambda item: (item[0], item[1]),
    )
    return assignment_id, working_session


def latest_ready_assignment_working_session(
    root: Path,
    *,
    channel_id: str,
    exclude_assignment_id: str = "",
) -> tuple[str, Path] | None:
    """Compatibility alias for the published-session resolver."""
    return latest_published_assignment_working_session(
        root,
        channel_id=channel_id,
        exclude_assignment_id=exclude_assignment_id,
    )


def write_assignment_working_session(root: Path, assignment_id: str, data: bytes, *, immutable: bool = False) -> Path:
    normalized_id = _safe_id(assignment_id)
    if not normalized_id:
        raise ValueError("invalid assignment id")
    if read_composite_assignment(root, normalized_id) is None:
        raise ValueError("Composite assignment does not exist")
    path = Path(root) / "assignments" / normalized_id / "results" / "working_session.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable:
        path = path.parent / 'generations' / hashlib.sha256(data).hexdigest() / path.name
    atomic_bytes(path, bytes(data))
    return path


def write_assignment_speaker_response(
    root: Path,
    assignment_id: str,
    *,
    frequency_hz: list[float] | tuple[float, ...] | np.ndarray,
    gain_db: list[float] | tuple[float, ...] | np.ndarray,
    phase_deg: list[float] | tuple[float, ...] | np.ndarray | None,
    immutable: bool = False,
) -> tuple[Path, str]:
    """Atomically publish the final Response Processing output as an FRD asset."""
    normalized_id = _safe_id(assignment_id)
    if not normalized_id or read_composite_assignment(root, normalized_id) is None:
        raise ValueError("Composite assignment does not exist")
    frequency = np.asarray(frequency_hz, dtype=float)
    gain = np.asarray(gain_db, dtype=float)
    phase = np.zeros_like(frequency) if phase_deg is None else np.asarray(phase_deg, dtype=float)
    if (
        frequency.ndim != 1 or frequency.size < 2 or gain.shape != frequency.shape
        or phase.shape != frequency.shape or not np.isfinite(frequency).all()
        or not np.isfinite(gain).all() or not np.isfinite(phase).all()
        or np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0)
    ):
        raise ValueError("Speaker response must be finite, positive and strictly ordered")
    lines = [
        "# PhaseEQ Response Processing output",
        "# Frequency(Hz) Magnitude(dB)" + (" Phase(deg)" if phase_deg is not None else ""),
    ]
    lines.extend(
        f"{freq:.12g} {level:.12g}" + (f" {angle:.12g}" if phase_deg is not None else "")
        for freq, level, angle in zip(frequency, gain, phase, strict=True)
    )
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path = Path(root) / "assignments" / normalized_id / "results" / "speaker_response.frd"
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable:
        path = path.parent / 'generations' / hashlib.sha256(data).hexdigest() / path.name
    atomic_bytes(path, data)
    return path, hashlib.sha256(data).hexdigest()


def exchange_directory(root: Path, sample_rate_hz: int) -> Path:
    return Path(root) / str(int(sample_rate_hz))


@locked_exchange
def upsert_exchange_channel(
    root: Path,
    *,
    sample_rate_hz: int,
    channel_name: str,
    way: str,
    group: str,
    coefficients: np.ndarray,
    gain_db: float = 0.0,
    polarity: int = 1,
    delay_samples: float = 0.0,
    assignment_id: str = "",
    channel_id: str = "",
    deferred: bool = False,
) -> CompositeExchangeState:
    """Write one PhaseEQ FIR into a Manifest-only exchange boundary."""
    name = channel_name.strip()
    way_name = way.strip()
    group_name = group.strip()
    if not name or not way_name or not group_name:
        raise ValueError("Channel name, Way and Composite Group are required")
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("FIR must be a finite non-empty mono vector")
    if polarity not in (-1, 1):
        raise ValueError("polarity must be +1 or -1")
    if not np.isfinite([gain_db, delay_samples]).all():
        raise ValueError("gain and delay must be finite")

    directory = exchange_directory(root, sample_rate_hz)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    payload = _read_manifest(manifest_path, sample_rate_hz)
    channels = list(payload.get("channels", []))
    previous = next((item for item in channels if str(item.get("name", "")) == name), None)
    asset_name = str(previous.get("wav")) if isinstance(previous, dict) and previous.get("wav") else f"channels/{_asset_stem(name)}.wav"
    asset_path = directory / asset_name
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    import io
    buffer = io.BytesIO()
    wavfile.write(buffer, int(sample_rate_hz), values.astype(np.float32))
    asset_data = buffer.getvalue()
    if deferred:
        asset_name = f"channels/{hashlib.sha256(asset_data).hexdigest()}.wav"
        asset_path = directory / asset_name
    atomic_bytes(asset_path, asset_data)
    channel = {
        "name": name,
        "way": way_name,
        "group": group_name,
        "gain": float(gain_db),
        "polarity": int(polarity),
        "delay": float(delay_samples),
        "wav": asset_name,
        "frd": None,
        "tap_count": int(values.size),
        "center_position": (int(values.size) - 1) / 2.0,
        "time_reference": "tap_center",
        "delay_application": "after_center_alignment",
        "assignment_id": str(assignment_id).strip(),
        "channel_id": str(channel_id).strip(),
    }
    existing_index = next(
        (index for index, item in enumerate(channels) if str(item.get("name", "")) == name),
        None,
    )
    if existing_index is None:
        channels.append(channel)
    else:
        channels[existing_index] = channel
    payload = {
        "format_version": 2,
        "sample_rate_hz": int(sample_rate_hz),
        "composite_groups": list(dict.fromkeys(str(item["group"]) for item in channels)),
        "channels": channels,
    }
    if deferred:
        channel['content_hash'] = hashlib.sha256(asset_data).hexdigest()
        payload['channels'] = [channel]
        payload['composite_groups'] = [group_name]
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        manifest_path = directory / f'.prepared-{fingerprint}.json'
    _atomic_json(manifest_path, payload)
    return CompositeExchangeState(
        directory=directory,
        manifest_path=manifest_path,
        channel_names=tuple(str(item["name"]) for item in channels),
        group_names=tuple(payload["composite_groups"]),
    )


def _read_manifest(path: Path, sample_rate_hz: int) -> dict[str, object]:
    if not path.exists():
        return {"format_version": 2, "sample_rate_hz": int(sample_rate_hz), "channels": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("sample_rate_hz", 0)) != int(sample_rate_hz):
        raise ValueError("Exchange Manifest sample rate mismatch")
    return payload


def _asset_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return stem or "channel"


def _safe_id(value: str) -> str:
    normalized = str(value).strip()
    if normalized and all(character.isalnum() or character in "_-" for character in normalized):
        return normalized
    return ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    _safe_atomic_json(path, payload)
