from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import os
import pickle
from pathlib import Path
from typing import Any

from utils.preset_store import read_preset_file
from utils.storage_paths import application_storage_paths


TARGET_PRESET_SCHEMA_VERSION = 2
TARGET_PRESET_SCOPES = ("phaseeq_channel", "composite_group")
TARGET_PRESET_CATEGORIES = (
    "Reference",
    "House curve",
    "Band-limited reference",
    "Imported response",
    "Custom",
)


def shared_target_preset_paths(app_root: Path) -> tuple[Path, Path]:
    """Return the single Target preset store used by both applications."""
    override = str(os.environ.get("PHASEEQ_DATA_DIR", "")).strip()
    paths = application_storage_paths(
        app_root,
        data_dir=Path(override).expanduser() if override else None,
    )
    return (
        paths["bundled_presets"] / "target_response_presets.json",
        paths["user_presets"] / "target_response_presets.json",
    )


def normalize_target_preset(record: dict[str, Any]) -> dict[str, Any]:
    """Upgrade one legacy or current record to the shared Target schema."""
    from response_completion.settings import migrate_extension_choices
    item = migrate_extension_choices(deepcopy(record))
    item["preset_schema_version"] = TARGET_PRESET_SCHEMA_VERSION
    item["category"] = str(item.get("category") or _infer_category(item))
    scopes = item.get("compatible_scopes", TARGET_PRESET_SCOPES)
    if not isinstance(scopes, (list, tuple)):
        scopes = TARGET_PRESET_SCOPES
    item["compatible_scopes"] = list(dict.fromkeys(
        scope for scope in (str(value).strip() for value in scopes) if scope
    )) or list(TARGET_PRESET_SCOPES)
    item["target_edit"] = deepcopy(item.get("target_edit")) if isinstance(
        item.get("target_edit"), dict,
    ) else {}
    item["target_url"] = str(item.get("target_url", ""))
    item["target_source_name"] = str(item.get("target_source_name", ""))
    item["source_content_mode"] = (
        "Gain only" if str(item.get("source_content_mode", "Gain + Phase")) == "Gain only"
        else "Gain + Phase"
    )
    return item


def shared_target_preset_catalog(
    app_root: Path, *, scope: str | None = None, summaries: bool = False,
) -> list[dict[str, Any]]:
    """Load the common Built-in/User Target repository with schema upgrades.

    A User record with the same ID complements and overrides Built-in fields.
    This preserves one logical preset instead of creating app-specific copies.
    """
    if summaries:
        return _target_summaries(app_root, scope=scope)
    bundled, user = shared_target_preset_paths(app_root)
    revision = tuple(_preset_file_revision(path) for path in (bundled, user))
    # Deserialize only bytes produced here, never a pickle supplied by a file.
    # Each caller receives its own mutable records without Python-level deep
    # copies of every sample. Retain only the latest repository snapshot.
    catalog = pickle.loads(_catalog_snapshot(bundled, user, revision))
    if scope is None:
        return catalog
    normalized_scope = str(scope).strip()
    return [item for item in catalog
            if normalized_scope in item.get("compatible_scopes", TARGET_PRESET_SCOPES)]


def _preset_file_revision(path: Path) -> tuple:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return ()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=1)
def _catalog_snapshot(bundled: Path, user: Path, revision: tuple) -> bytes:
    raw_catalog = [
        *({**item, "source": "Built-in"} for item in read_preset_file(
            bundled, kind="target_response",
        )),
        *({**item, "source": "User"} for item in read_preset_file(
            user, kind="target_response",
        )),
    ]
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for raw in raw_catalog:
        preset_id = str(raw.get("id", "")).strip()
        base_id = preset_id
        if base_id not in merged:
            merged[base_id] = normalize_target_preset({**raw, "id": base_id})
            merged[base_id]["sources"] = [str(raw.get("source", "User"))]
            order.append(base_id)
            continue
        previous = merged[base_id]
        sources = [*previous.get("sources", []), str(raw.get("source", "User"))]
        combined = {**previous, **raw, "id": base_id}
        combined["sources"] = list(dict.fromkeys(sources))
        combined["source"] = str(raw.get("source", previous.get("source", "User")))
        merged[base_id] = normalize_target_preset(combined)
    catalog = [merged[preset_id] for preset_id in order]
    return pickle.dumps(catalog, protocol=5)


def _infer_category(item: dict[str, Any]) -> str:
    preset_id = str(item.get("id", "")).lower()
    edit = item.get("target_edit") if isinstance(item.get("target_edit"), dict) else {}
    if "house" in preset_id or edit.get("gain_tilt"):
        return "House curve"
    if edit.get("gain_linear") or any(
        str(value.get("kind", "")) in {"high_pass", "low_pass"}
        for value in edit.get("iir_filters", []) if isinstance(value, dict)
    ):
        return "Band-limited reference"
    if isinstance(item.get("target_response_raw"), dict):
        return "Imported response"
    return "Reference"


def _merge_target_records(tiers):
    merged = {}
    for source, records in tiers:
        for raw in records:
            from utils.preset_store import _preset_id
            key = _preset_id(raw.get("id"), str(raw.get("name", "preset")))
            previous = merged.get(key, {})
            sources = list(dict.fromkeys([*previous.get("sources", []), source]))
            merged[key] = normalize_target_preset({**previous, **raw, "id": key,
                                                   "source": source, "sources": sources})
    return list(merged.values())


def _target_summaries(app_root, *, scope=None):
    from utils.preset_index import preset_summaries
    bundled, user = shared_target_preset_paths(app_root)
    rows = _merge_target_records([(source, preset_summaries(path))
                                 for source, path in (("Built-in", bundled), ("User", user))])
    return [{**row, "_summary": True} for row in rows
            if scope is None or str(scope).strip() in row["compatible_scopes"]]


def get_shared_target_preset(app_root, record_id):
    """Decode only the selected record, preserving User/Built-in overlay rules."""
    from utils.preset_index import preset_record
    bundled, user = shared_target_preset_paths(app_root)
    rows = _merge_target_records([(source, preset_record(path, str(record_id)))
                                 for source, path in (("Built-in", bundled), ("User", user))])
    return rows[0] if rows else None


def materialize_target_preset(app_root, definition):
    if not definition or not definition.get("_summary"):
        return definition
    loaded = get_shared_target_preset(app_root, definition["id"])
    if loaded is None:
        raise ValueError("選択したTargetが削除されました。一覧を更新してください。")
    return loaded
