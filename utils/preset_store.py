from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any


def _preset_id(value: Any, name: str = "preset") -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or name).lower()).strip("-")
    return text or f"preset-{uuid.uuid4().hex[:10]}"


def read_preset_file(path: Path, *, kind: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ValueError(f"Preset file kind must be {kind!r}: {path}")
    presets = payload.get("presets", [])
    if not isinstance(presets, list):
        raise ValueError(f"Preset list is invalid: {path}")
    normalized: list[dict[str, Any]] = []
    for raw in presets:
        if not isinstance(raw, dict):
            continue
        item = deepcopy(raw)
        item["id"] = _preset_id(item.get("id"), str(item.get("name", "preset")))
        item["name"] = str(item.get("name", "Unnamed preset")).strip() or "Unnamed preset"
        normalized.append(item)
    return normalized


def load_preset_catalog(
    bundled_path: Path,
    user_path: Path,
    *,
    kind: str,
) -> list[dict[str, Any]]:
    bundled = read_preset_file(bundled_path, kind=kind)
    user = read_preset_file(user_path, kind=kind)
    catalog: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, records in (("Built-in", bundled), ("User", user)):
        for record in records:
            item = deepcopy(record)
            item["source"] = source
            if item["id"] in seen:
                item["id"] = f"{item['id']}-{source.lower()}"
            seen.add(item["id"])
            catalog.append(item)
    return catalog


def write_preset_file(path: Path, *, kind: str, presets: list[dict[str, Any]]) -> None:
    """Atomically write one preset tier."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean: list[dict[str, Any]] = []
    for raw in presets:
        if raw.get("_summary"):
            raise ValueError("Targetの概要だけでは保存できません。主データを読み込んでください。")
        item = deepcopy(raw)
        item.pop("source", None)
        item["id"] = _preset_id(item.get("id"), str(item.get("name", "preset")))
        clean.append(item)
    payload = {
        "schema_version": 2 if kind == "target_response" else 1,
        "kind": kind,
        "presets": clean,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_user_presets(path: Path, *, kind: str, presets: list[dict[str, Any]]) -> None:
    write_preset_file(path, kind=kind, presets=presets)


def transfer_preset(
    preset_id: str,
    *,
    source_path: Path,
    destination_path: Path,
    kind: str,
) -> dict[str, Any]:
    """Move a preset between Built-in and User tiers with rollback protection."""
    source_records = read_preset_file(source_path, kind=kind)
    destination_records = read_preset_file(destination_path, kind=kind)
    selected = next((item for item in source_records if str(item.get("id")) == str(preset_id)), None)
    if selected is None:
        raise ValueError(f"Preset not found in source tier: {preset_id}")
    if any(str(item.get("id")) == str(preset_id) for item in destination_records):
        raise ValueError(f"Preset ID already exists in destination tier: {preset_id}")

    destination_existed = destination_path.exists()
    destination_original = destination_path.read_bytes() if destination_existed else b""
    write_preset_file(destination_path, kind=kind, presets=[*destination_records, selected])
    try:
        write_preset_file(
            source_path,
            kind=kind,
            presets=[item for item in source_records if str(item.get("id")) != str(preset_id)],
        )
    except Exception:
        if destination_existed:
            destination_path.write_bytes(destination_original)
        else:
            destination_path.unlink(missing_ok=True)
        raise
    return deepcopy(selected)


def user_presets_from_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [deepcopy(item) for item in catalog if item.get("source") == "User"]


def new_preset_id(name: str) -> str:
    return f"{_preset_id('', name)}-{uuid.uuid4().hex[:8]}"
