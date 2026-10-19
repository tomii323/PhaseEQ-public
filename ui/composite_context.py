"""Explicit editor switch requests, independent of a changing option list."""
from copy import deepcopy
from pathlib import Path


def standalone_settings_path(settings_path):
    return Path(settings_path).with_name("settings.standalone.json")


def startup_settings_path(settings_path):
    dedicated = standalone_settings_path(settings_path)
    return dedicated if dedicated.exists() else Path(settings_path)


def load_standalone_settings(settings_path):
    from utils.settings_io import load_startup_settings

    path = standalone_settings_path(settings_path)
    if not path.exists():
        return None
    loaded = load_startup_settings(path)
    if loaded.status == "unreadable_preserved":
        raise ValueError(f"Standalone設定を復元できません。元ファイルを保持しています: {loaded.error}")
    return loaded.payload


def save_standalone_settings(state, settings_path, payload):
    """Persist only Standalone; retain optimistic multi-session protection."""
    import json
    from utils.storage_paths import file_revision, write_text_if_revision
    from utils.settings_io import payload_revision

    if state.get("_active_context_assignment") is not None or state.get("_settings_autosave_blocked"):
        return True
    path = standalone_settings_path(settings_path)
    standalone = {key: value for key, value in payload.items() if not key.startswith("eq_link")}
    standalone["revision"] = payload_revision(standalone)
    text = json.dumps(standalone, ensure_ascii=False, indent=2, sort_keys=True)
    revision = file_revision(path)
    expected = state.get("_standalone_settings_file_revision", revision)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        written, revision = write_text_if_revision(path, text, expected_revision=expected)
        if not written:
            return False
    state["_standalone_settings_file_revision"] = revision
    return True


def context_owner(assignment):
    """Stable ownership, independent of display names, Way labels and taps."""
    if assignment is None:
        return ("standalone",)
    return ("channel", str(assignment.system_id), str(assignment.channel_id),
            int(assignment.sample_rate_hz))


def source_context_payload(state, assignment, pending=None):
    """Materialize only the snapshot captured for the editor being left."""
    pending = pending if isinstance(pending, dict) else {}
    payload = pending.get("payload", state.get("_composite_context_payload"))
    owner = pending.get("source_owner", state.get("_composite_context_payload_owner"))
    if payload is not None and owner is not None and tuple(owner) != context_owner(assignment):
        raise ValueError("切替元と編集データの所有者が一致しません。編集内容を保持しました。")
    return payload() if callable(payload) else deepcopy(payload)


def validate_draft_owner(owners, identifier, assignment):
    owner = owners.get(identifier)
    if owner is not None and tuple(owner) != context_owner(assignment):
        raise ValueError("復帰先とDraftの所有者が一致しません。編集内容を保持しました。")


def active_assignment_crossover(state):
    """Only a connected Assignment can own the Studio crossover stage."""
    if state.get('_active_context_assignment') is None:
        return None
    payload = state.get('_composite_iir_crossover')
    return payload if isinstance(payload, dict) else None


def detach_standalone_crossover(state):
    """Retain orphaned Studio data for recovery, never as an active filter."""
    if state.get('_active_context_assignment') is not None:
        return False
    payload = state.pop('_composite_iir_crossover', None)
    if payload is None:
        return False
    detached = deepcopy(state.get('_detached_channel_band_split') or {})
    detached['iir'] = deepcopy(payload)
    state['_detached_channel_band_split'] = detached
    return True


def queue_channel_switch(state):
    state["_pending_composite_context"] = {
        "assignment_id": str(state.get("composite_phaseeq_channel_switcher", "") or ""),
        "payload": state.get("_composite_context_payload"),
        "source_owner": state.get("_composite_context_payload_owner"),
    }


def requested_context_id(query_id, current_id, pending):
    # An incoming Multiway URL wins over an older queued widget action.
    if query_id and query_id != current_id:
        return query_id
    return str(pending["assignment_id"]) if isinstance(pending, dict) else query_id


def same_channel_context(source, destination):
    return (source is not None and destination is not None
            and (source.system_id, source.channel_id) == (destination.system_id, destination.channel_id))
