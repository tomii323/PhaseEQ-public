"""Split persistence for Speaker Packages, Target Packages, and Designs.

The public ``DesignRecord.payload`` is a runtime view composed from the three
stored responsibilities.  It keeps the existing editor and DSP synthesis APIs
working while the database itself no longer stores a monolithic legacy Project
payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal, Mapping
import uuid

from phase_fir_designer import SpeakerResponse

from utils.response_assets import (
    load_response_asset,
    response_asset_ref_payload,
    save_response_asset,
)
from utils.settings_io import speaker_response_from_payload, speaker_response_payload
from utils.sqlite_repository import (
    canonical_json,
    connect_sqlite,
    content_sha256,
    ensure_app_meta,
    utc_now,
)


DESIGN_LIBRARY_SCHEMA_VERSION = 3
DesignSortKey = Literal[
    "updated_at",
    "last_opened_at",
    "name",
    "speaker_type",
    "side",
    "position",
    "sample_rate",
]

_SPEAKER_UI_PREFIXES = (
    "speaker_",
    "driver_band_",
    "current_speaker_",
    "current_near_field_",
    "current_port_",
)
_SPEAKER_UI_KEYS = {"apply_mic_cal", "current_mic_calibration"}
_TARGET_UI_PREFIXES = ("target_",)


@dataclass(frozen=True)
class DesignRecord:
    id: str
    name: str
    speaker_type: str = ""
    side: str = ""
    position: str = ""
    tags: str = ""
    note: str = ""
    speaker_package_id: str = ""
    target_package_id: str = ""
    sample_rate: int | None = None
    taps: int | None = None
    analysis_fft_size: int | None = None
    payload: dict[str, Any] | None = None
    content_signature: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_opened_at: str = ""
    archived: bool = False


@dataclass(frozen=True)
class SpeakerPackageRecord:
    id: str
    name: str
    payload: dict[str, Any]
    runtime_payload: dict[str, Any]
    content_signature: str
    created_at: str
    updated_at: str
    archived: bool = False
    speaker_type: str = ""
    side: str = ""
    position: str = ""
    tags: str = ""
    note: str = ""
    input_profile_signature: str = ""


@dataclass(frozen=True)
class InputProfileRevision:
    id: str
    speaker_package_id: str
    content_signature: str
    summary: str
    created_at: str


def speaker_package_payload_from_runtime(
    response_asset_path: Path,
    *,
    payload: Mapping[str, Any],
    name: str,
    speaker_type: str = "",
    side: str = "",
    position: str = "",
    tags: str = "",
    note: str = "",
    existing_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize the INPUT-only Speaker Package payload.

    DSP conditions and every Target/EQ field are deliberately ignored.  The
    returned payload is also the canonical content used by Input Profile
    revision signatures.
    """

    normalized = dict(payload)
    config = dict(normalized.get("config", {}))
    io_payload = dict(normalized.get("io", {}))
    ui_payload = dict(normalized.get("ui", {}))
    response_refs = io_payload.get("response_refs", {})
    if not isinstance(response_refs, Mapping):
        response_refs = {}
    previous = dict(existing_payload or {})
    previous_refs = previous.get("response_refs", {})
    if not isinstance(previous_refs, Mapping):
        previous_refs = {}

    raw_speaker = _response(io_payload.get("speaker_response_raw")) or _response_from_ref(
        response_asset_path, response_refs.get("speaker_response_raw")
    )
    processed_speaker = _response(config.get("speaker_response")) or raw_speaker
    mic_calibration = _response(io_payload.get("mic_cal_response_raw")) or _response_from_ref(
        response_asset_path, response_refs.get("mic_cal_response_raw")
    )
    speaker_refs = {
        "speaker_response_raw": _save_asset(
            response_asset_path,
            raw_speaker,
            kind="speaker_input_raw",
            current_ref=previous_refs.get("speaker_response_raw"),
        ),
        "speaker_response_processed": _save_asset(
            response_asset_path,
            processed_speaker,
            kind="speaker_input_processed",
            current_ref=previous_refs.get("speaker_response_processed"),
        ),
        "mic_cal_response_raw": _save_asset(
            response_asset_path,
            mic_calibration,
            kind="mic_calibration_raw",
            current_ref=previous_refs.get("mic_cal_response_raw"),
        ),
    }
    return {
        "schema_version": 2,
        "name": str(name).strip() or "Untitled Speaker Package",
        "speaker_type": str(speaker_type).strip(),
        "side": str(side).strip(),
        "position": str(position).strip(),
        "tags": str(tags).strip(),
        "note": str(note).strip(),
        "source_name": str(io_payload.get("speaker_source_name", "")),
        "mic_cal_source_name": str(io_payload.get("mic_cal_source_name", "")),
        "response_refs": speaker_refs,
        "input_ui": {
            key: value for key, value in ui_payload.items() if _is_speaker_ui_key(str(key))
        },
        "processing": dict(normalized.get("processing", {})).get("input", {}),
    }


def input_profile_signature_from_runtime(
    payload: Mapping[str, Any],
    *,
    name: str,
    speaker_type: str = "",
    side: str = "",
    position: str = "",
    tags: str = "",
    note: str = "",
) -> str:
    """Hash the user-visible INPUT result without DSP/Target conditions."""

    normalized = dict(payload)
    config = dict(normalized.get("config", {}))
    io_payload = dict(normalized.get("io", {}))
    ui_payload = dict(normalized.get("ui", {}))
    return content_sha256({
        "schema_version": 2,
        "name": str(name).strip() or "Untitled Speaker Package",
        "speaker_type": str(speaker_type).strip(),
        "side": str(side).strip(),
        "position": str(position).strip(),
        "tags": str(tags).strip(),
        "note": str(note).strip(),
        "source_name": str(io_payload.get("speaker_source_name", "")),
        "mic_cal_source_name": str(io_payload.get("mic_cal_source_name", "")),
        "speaker_response_raw": speaker_response_payload(
            _response(io_payload.get("speaker_response_raw"))
        ),
        "speaker_response_processed": speaker_response_payload(
            _response(config.get("speaker_response"))
        ),
        "mic_cal_response_raw": speaker_response_payload(
            _response(io_payload.get("mic_cal_response_raw"))
        ),
        "input_ui": {
            key: value for key, value in ui_payload.items() if _is_speaker_ui_key(str(key))
        },
        "processing": dict(normalized.get("processing", {})).get("input", {}),
    })


def save_input_profile_revision(
    path: Path,
    *,
    speaker_package_id: str,
    payload: Mapping[str, Any],
    summary: str = "Input Profile saved",
) -> None:
    """Persist one deduplicated INPUT-only revision."""

    ensure_design_library_db(path)
    with connect_sqlite(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _record_input_profile_revision(
            connection,
            str(speaker_package_id),
            payload,
            summary=str(summary),
        )


def save_speaker_package(
    path: Path,
    response_asset_path: Path,
    *,
    payload: Mapping[str, Any],
    package_id: str | None = None,
    name: str,
    speaker_type: str = "",
    side: str = "",
    position: str = "",
    tags: str = "",
    note: str = "",
    revision_summary: str = "Input Profile saved",
) -> SpeakerPackageRecord:
    """Save only Speaker Package and its Input Profile revision."""

    ensure_design_library_db(path)
    clean_id = str(package_id or uuid.uuid4())
    clean_name = str(name).strip() or "Untitled Speaker Package"
    existing = _package_payload(path, "speaker_packages", clean_id)
    package = speaker_package_payload_from_runtime(
        response_asset_path,
        payload=payload,
        name=clean_name,
        speaker_type=speaker_type,
        side=side,
        position=position,
        tags=tags,
        note=note,
        existing_payload=existing,
    )
    with connect_sqlite(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        row = connection.execute(
            "SELECT created_at FROM speaker_packages WHERE id=?", (clean_id,)
        ).fetchone()
        now = utc_now(connection)
        created_at = str(row[0]) if row else now
        _upsert_package(
            connection,
            "speaker_packages",
            clean_id,
            clean_name,
            package,
            created_at,
            now,
        )
        _record_input_profile_revision(
            connection,
            clean_id,
            package,
            summary=revision_summary,
        )
    record = get_speaker_package(path, response_asset_path, clean_id)
    if record is None:
        raise RuntimeError("Speaker Package save failed.")
    return record


def duplicate_speaker_package(
    path: Path,
    response_asset_path: Path,
    source: SpeakerPackageRecord,
    *,
    name: str | None = None,
) -> SpeakerPackageRecord:
    if source.payload.get("_summary"):
        raise ValueError("概要から複製できません。選択した主データを読み込んでください。")
    runtime = source.runtime_payload
    metadata = source.payload
    return save_speaker_package(
        path,
        response_asset_path,
        payload=runtime,
        name=str(name or f"{source.name} copy"),
        speaker_type=str(metadata.get("speaker_type", "")),
        side=str(metadata.get("side", "")),
        position=str(metadata.get("position", "")),
        tags=str(metadata.get("tags", "")),
        note=str(metadata.get("note", "")),
        revision_summary="Speaker Package duplicated",
    )


def ensure_design_library_db(path: Path) -> None:
    with connect_sqlite(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        ensure_app_meta(
            connection,
            schema_version=DESIGN_LIBRARY_SCHEMA_VERSION,
            db_kind="phaseeq_design_library",
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS speaker_packages (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_signature TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS input_profile_revisions (
                id TEXT PRIMARY KEY,
                speaker_package_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_signature TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                UNIQUE(speaker_package_id, content_signature),
                FOREIGN KEY(speaker_package_id) REFERENCES speaker_packages(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS target_packages (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_signature TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS designs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                speaker_type TEXT NOT NULL DEFAULT '',
                side TEXT NOT NULL DEFAULT '',
                position TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '',
                speaker_package_id TEXT NOT NULL,
                target_package_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                ui_json TEXT NOT NULL,
                ui_profile_json TEXT NOT NULL,
                processing_json TEXT NOT NULL,
                source_meta_json TEXT NOT NULL,
                content_signature TEXT NOT NULL,
                sample_rate INTEGER,
                taps INTEGER,
                analysis_fft_size INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_opened_at TEXT NOT NULL DEFAULT '',
                archived INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(speaker_package_id) REFERENCES speaker_packages(id),
                FOREIGN KEY(target_package_id) REFERENCES target_packages(id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_designs_updated ON designs(archived, updated_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_designs_name ON designs(name COLLATE NOCASE)"
        )


def save_design(
    path: Path,
    response_asset_path: Path,
    *,
    payload: Mapping[str, Any],
    design_id: str | None = None,
    name: str,
    speaker_type: str = "",
    side: str = "",
    position: str = "",
    tags: str = "",
    note: str = "",
) -> DesignRecord:
    ensure_design_library_db(path)
    clean_id = str(design_id or uuid.uuid4())
    clean_name = str(name).strip() or "Untitled Design"
    normalized = dict(payload)
    config = dict(normalized.get("config", {}))
    io_payload = dict(normalized.get("io", {}))
    ui_payload = dict(normalized.get("ui", {}))
    response_refs = io_payload.get("response_refs", {})
    if not isinstance(response_refs, Mapping):
        response_refs = {}

    existing = _design_row(path, clean_id)
    existing_speaker = _package_payload(path, "speaker_packages", str(existing["speaker_package_id"])) if existing else {}
    existing_target = _package_payload(path, "target_packages", str(existing["target_package_id"])) if existing else {}

    raw_speaker = _response(io_payload.get("speaker_response_raw")) or _response_from_ref(
        response_asset_path, response_refs.get("speaker_response_raw")
    )
    processed_speaker = _response(config.get("speaker_response")) or raw_speaker
    raw_target = _response(io_payload.get("target_response_raw")) or _response_from_ref(
        response_asset_path, response_refs.get("target_response_raw")
    )
    effective_target = _response(config.get("target_response")) or raw_target
    mic_calibration = _response(io_payload.get("mic_cal_response_raw")) or _response_from_ref(
        response_asset_path, response_refs.get("mic_cal_response_raw")
    )

    speaker_refs = {
        "speaker_response_raw": _save_asset(
            response_asset_path,
            raw_speaker,
            kind="speaker_input_raw",
            current_ref=existing_speaker.get("response_refs", {}).get("speaker_response_raw"),
        ),
        "speaker_response_processed": _save_asset(
            response_asset_path,
            processed_speaker,
            kind="speaker_input_processed",
            current_ref=existing_speaker.get("response_refs", {}).get("speaker_response_processed"),
        ),
        "mic_cal_response_raw": _save_asset(
            response_asset_path,
            mic_calibration,
            kind="mic_calibration_raw",
            current_ref=existing_speaker.get("response_refs", {}).get("mic_cal_response_raw"),
        ),
    }
    target_refs = {
        "target_response_raw": _save_asset(
            response_asset_path,
            raw_target,
            kind="target_response_raw",
            current_ref=existing_target.get("response_refs", {}).get("target_response_raw"),
        ),
        "target_response_effective": _save_asset(
            response_asset_path,
            effective_target,
            kind="target_response_effective",
            current_ref=existing_target.get("response_refs", {}).get("target_response_effective"),
        ),
    }

    speaker_package_id = str(existing["speaker_package_id"]) if existing else str(uuid.uuid4())
    target_package_id = str(existing["target_package_id"]) if existing else str(uuid.uuid4())
    speaker_package = {
        "schema_version": 1,
        "name": clean_name,
        "speaker_type": str(speaker_type).strip(),
        "side": str(side).strip(),
        "position": str(position).strip(),
        "tags": str(tags).strip(),
        "note": str(note).strip(),
        "source_name": str(io_payload.get("speaker_source_name", "")),
        "mic_cal_source_name": str(io_payload.get("mic_cal_source_name", "")),
        "response_refs": speaker_refs,
        "input_ui": {
            key: value for key, value in ui_payload.items() if _is_speaker_ui_key(str(key))
        },
        "processing": dict(normalized.get("processing", {})).get("input", {}),
    }
    target_package = {
        "schema_version": 1,
        "name": clean_name,
        "source_name": str(io_payload.get("target_source_name", "")),
        "response_refs": target_refs,
        "target_ui": {
            key: value for key, value in ui_payload.items() if _is_target_ui_key(str(key))
        },
        "processing": dict(normalized.get("processing", {})).get("target", {}),
        "raw_available": raw_target is not None,
        "effective_available": effective_target is not None,
    }
    design_ui = {
        key: value
        for key, value in ui_payload.items()
        if not _is_speaker_ui_key(str(key)) and not _is_target_ui_key(str(key))
    }
    stored_config = dict(config)
    stored_config.pop("speaker_response", None)
    stored_config.pop("target_response", None)
    source_meta = {
        "app": normalized.get("app", "PhaseEQ"),
        "app_version": normalized.get("app_version", ""),
        "author": normalized.get("author", ""),
        "schema_version": int(normalized.get("schema_version", 1) or 1),
        "snapshot_name": normalized.get("snapshot_name"),
    }
    processing = dict(normalized.get("processing", {}))
    content_signature = content_sha256(
        {
            "speaker_package": speaker_package,
            "target_package": target_package,
            "config": stored_config,
            "ui": design_ui,
            "ui_profile": normalized.get("ui_profile", {}),
            "processing": processing,
        }
    )
    sample_rate, taps, analysis_fft_size = _payload_summary(config)

    with connect_sqlite(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        now = utc_now(connection)
        created_at = str(existing["created_at"]) if existing else now
        _upsert_package(
            connection,
            "speaker_packages",
            speaker_package_id,
            clean_name,
            speaker_package,
            created_at,
            now,
        )
        _record_input_profile_revision(
            connection,
            speaker_package_id,
            speaker_package,
            summary="Speaker/Input profile saved",
        )
        _upsert_package(
            connection,
            "target_packages",
            target_package_id,
            clean_name,
            target_package,
            created_at,
            now,
        )
        connection.execute(
            """
            INSERT INTO designs(
                id, name, speaker_type, side, position, tags, note,
                speaker_package_id, target_package_id, config_json, ui_json,
                ui_profile_json, processing_json, source_meta_json,
                content_signature, sample_rate, taps, analysis_fft_size,
                created_at, updated_at, last_opened_at, archived
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                speaker_type=excluded.speaker_type,
                side=excluded.side,
                position=excluded.position,
                tags=excluded.tags,
                note=excluded.note,
                speaker_package_id=excluded.speaker_package_id,
                target_package_id=excluded.target_package_id,
                config_json=excluded.config_json,
                ui_json=excluded.ui_json,
                ui_profile_json=excluded.ui_profile_json,
                processing_json=excluded.processing_json,
                source_meta_json=excluded.source_meta_json,
                content_signature=excluded.content_signature,
                sample_rate=excluded.sample_rate,
                taps=excluded.taps,
                analysis_fft_size=excluded.analysis_fft_size,
                updated_at=excluded.updated_at,
                archived=0
            """,
            (
                clean_id,
                clean_name,
                str(speaker_type).strip(),
                str(side).strip(),
                str(position).strip(),
                str(tags).strip(),
                str(note).strip(),
                speaker_package_id,
                target_package_id,
                canonical_json(stored_config),
                canonical_json(design_ui),
                canonical_json(normalized.get("ui_profile", {})),
                canonical_json(processing),
                canonical_json(source_meta),
                content_signature,
                sample_rate,
                taps,
                analysis_fft_size,
                created_at,
                now,
                now,
            ),
        )
    record = get_design(path, response_asset_path, clean_id)
    if record is None:
        raise RuntimeError("Design save failed.")
    return record


def get_design(path: Path, response_asset_path: Path, design_id: str) -> DesignRecord | None:
    ensure_design_library_db(path)
    with connect_sqlite(path) as connection:
        row = connection.execute("SELECT * FROM designs WHERE id = ?", (str(design_id),)).fetchone()
    return _record_from_row(path, response_asset_path, row) if row else None


def list_designs(
    path: Path,
    response_asset_path: Path,
    *,
    query: str = "",
    include_archived: bool = False,
    sort_by: DesignSortKey = "updated_at",
    descending: bool = True,
    limit: int = 1000,
    offset: int = 0,
) -> list[DesignRecord]:
    ensure_design_library_db(path)
    valid_sort = {
        "updated_at", "last_opened_at", "name", "speaker_type", "side",
        "position", "sample_rate",
    }
    sort_key = sort_by if sort_by in valid_sort else "updated_at"
    clauses: list[str] = []
    params: list[Any] = []
    if not include_archived:
        clauses.append("archived = 0")
    if str(query).strip():
        like = f"%{str(query).strip()}%"
        clauses.append(
            "(name LIKE ? COLLATE NOCASE OR speaker_type LIKE ? COLLATE NOCASE OR "
            "side LIKE ? COLLATE NOCASE OR position LIKE ? COLLATE NOCASE OR "
            "tags LIKE ? COLLATE NOCASE OR note LIKE ? COLLATE NOCASE)"
        )
        params.extend([like] * 6)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "DESC" if descending else "ASC"
    params.extend([max(1, int(limit)), max(0, int(offset))])
    with connect_sqlite(path) as connection:
        rows = connection.execute(
            f"""
            SELECT * FROM designs
            {where_sql}
            ORDER BY {sort_key} {order}, name COLLATE NOCASE ASC, id ASC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
    return [_record_from_row(path, response_asset_path, row) for row in rows]


def mark_design_opened(path: Path, design_id: str) -> None:
    ensure_design_library_db(path)
    with connect_sqlite(path) as connection:
        connection.execute(
            "UPDATE designs SET last_opened_at = ? WHERE id = ?",
            (utc_now(connection), str(design_id)),
        )


def archive_design(path: Path, design_id: str) -> None:
    ensure_design_library_db(path)
    with connect_sqlite(path) as connection:
        connection.execute(
            "UPDATE designs SET archived = 1, updated_at = ? WHERE id = ?",
            (utc_now(connection), str(design_id)),
        )


def duplicate_design(
    path: Path,
    response_asset_path: Path,
    source: DesignRecord,
    *,
    name: str | None = None,
) -> DesignRecord:
    if not isinstance(source.payload, dict):
        raise ValueError("Source Design has no runtime payload.")
    return save_design(
        path,
        response_asset_path,
        payload=source.payload,
        name=str(name or f"{source.name} copy"),
        speaker_type=source.speaker_type,
        side=source.side,
        position=source.position,
        tags=source.tags,
        note=source.note,
    )


def list_speaker_packages(
    path: Path,
    response_asset_path: Path,
    *,
    include_archived: bool = False,
) -> list[SpeakerPackageRecord]:
    ensure_design_library_db(path)
    where = "" if include_archived else "WHERE archived=0"
    with connect_sqlite(path) as connection:
        rows = connection.execute(
            f"SELECT * FROM speaker_packages {where} ORDER BY updated_at DESC,name COLLATE NOCASE"
        ).fetchall()
    return [_speaker_package_record(response_asset_path, row) for row in rows]


def get_speaker_package(
    path: Path,
    response_asset_path: Path,
    package_id: str,
) -> SpeakerPackageRecord | None:
    ensure_design_library_db(path)
    with connect_sqlite(path) as connection:
        row = connection.execute(
            "SELECT * FROM speaker_packages WHERE id=?", (str(package_id),)
        ).fetchone()
    return _speaker_package_record(response_asset_path, row) if row else None


def archive_speaker_package(path: Path, package_id: str) -> None:
    ensure_design_library_db(path)
    with connect_sqlite(path) as connection:
        connection.execute(
            "UPDATE speaker_packages SET archived=1,updated_at=? WHERE id=?",
            (utc_now(connection), str(package_id)),
        )


def list_input_profile_revisions(path: Path, package_id: str) -> list[InputProfileRevision]:
    ensure_design_library_db(path)
    with connect_sqlite(path) as connection:
        rows = connection.execute(
            "SELECT id,speaker_package_id,content_signature,summary,created_at "
            "FROM input_profile_revisions WHERE speaker_package_id=? ORDER BY created_at DESC",
            (str(package_id),),
        ).fetchall()
    return [InputProfileRevision(str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])) for row in rows]


def _design_row(path: Path, design_id: str) -> sqlite3.Row | None:
    with connect_sqlite(path) as connection:
        return connection.execute("SELECT * FROM designs WHERE id = ?", (str(design_id),)).fetchone()


def _package_payload(path: Path, table: str, package_id: str) -> dict[str, Any]:
    if table not in {"speaker_packages", "target_packages"}:
        raise ValueError(table)
    with connect_sqlite(path) as connection:
        row = connection.execute(
            f"SELECT payload_json FROM {table} WHERE id = ?", (str(package_id),)
        ).fetchone()
    return json.loads(str(row["payload_json"])) if row else {}


def _upsert_package(
    connection: sqlite3.Connection,
    table: str,
    package_id: str,
    name: str,
    payload: Mapping[str, Any],
    created_at: str,
    updated_at: str,
) -> None:
    if table not in {"speaker_packages", "target_packages"}:
        raise ValueError(table)
    payload_json = canonical_json(dict(payload))
    connection.execute(
        f"""
        INSERT INTO {table}(id, name, payload_json, content_signature, created_at, updated_at, archived)
        VALUES (?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            payload_json=excluded.payload_json,
            content_signature=excluded.content_signature,
            updated_at=excluded.updated_at,
            archived=0
        """,
        (package_id, name, payload_json, content_sha256(payload_json), created_at, updated_at),
    )


def _record_input_profile_revision(
    connection: sqlite3.Connection,
    speaker_package_id: str,
    payload: Mapping[str, Any],
    *,
    summary: str,
) -> None:
    normalized_payload = dict(payload)
    normalized_payload.pop("design_condition", None)
    payload_json = canonical_json(normalized_payload)
    signature = content_sha256(payload_json)
    latest = connection.execute(
        """SELECT payload_json FROM input_profile_revisions
        WHERE speaker_package_id=? ORDER BY created_at DESC LIMIT 1""",
        (str(speaker_package_id),),
    ).fetchone()
    if latest is not None:
        try:
            latest_payload = json.loads(str(latest[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            latest_payload = None
        if isinstance(latest_payload, dict):
            latest_payload.pop("design_condition", None)
            if content_sha256(canonical_json(latest_payload)) == signature:
                return
    connection.execute(
        """INSERT OR IGNORE INTO input_profile_revisions(
            id,speaker_package_id,payload_json,content_signature,summary,created_at
        ) VALUES(?,?,?,?,?,?)""",
        (str(uuid.uuid4()), str(speaker_package_id), payload_json, signature,
         str(summary), utc_now(connection)),
    )


def _speaker_package_record(response_asset_path: Path, row: sqlite3.Row) -> SpeakerPackageRecord:
    package = json.loads(str(row["payload_json"]))
    refs = dict(package.get("response_refs", {}))
    raw = _response_from_ref(response_asset_path, refs.get("speaker_response_raw"))
    processed = _response_from_ref(response_asset_path, refs.get("speaker_response_processed")) or raw
    mic = _response_from_ref(response_asset_path, refs.get("mic_cal_response_raw"))
    # Old packages can still contain ``design_condition``.  It is deliberately
    # ignored: Speaker Package ends at Input Profile, while the DSP Device owns
    # sample rate/taps and PhaseEQ owns the common analysis FFT policy.
    config = {
        "sample_rate": 48_000,
        "taps": 4097,
        "analysis_fft_size": 16_384,
        "speaker_response": speaker_response_payload(processed),
        "target_response": None,
        "iir_filters": [],
        "linear_fir_filters": [],
    }
    runtime = {
        "app": "PhaseEQ",
        "schema_version": 3,
        "snapshot_name": str(row["name"]),
        "io": {
            "speaker_source_name": package.get("source_name", ""),
            "mic_cal_source_name": package.get("mic_cal_source_name", ""),
            "speaker_response_raw": speaker_response_payload(raw),
            "target_response_raw": None,
            "mic_cal_response_raw": speaker_response_payload(mic),
        },
        "ui": {**dict(package.get("input_ui", {})), "project_name": str(row["name"])},
        "ui_profile": {},
        "config": config,
        "processing": {"input": dict(package.get("processing", {}))},
    }
    runtime["revision"] = str(row["content_signature"])
    runtime_signature = input_profile_signature_from_runtime(
        runtime,
        name=str(row["name"]),
        speaker_type=str(package.get("speaker_type", "")),
        side=str(package.get("side", "")),
        position=str(package.get("position", "")),
        tags=str(package.get("tags", "")),
        note=str(package.get("note", "")),
    )
    return SpeakerPackageRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        payload=package,
        runtime_payload=runtime,
        content_signature=str(row["content_signature"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        archived=bool(row["archived"]),
        speaker_type=str(package.get("speaker_type", "")),
        side=str(package.get("side", "")),
        position=str(package.get("position", "")),
        tags=str(package.get("tags", "")),
        note=str(package.get("note", "")),
        input_profile_signature=runtime_signature,
    )


def _record_from_row(path: Path, response_asset_path: Path, row: sqlite3.Row) -> DesignRecord:
    speaker_package = _package_payload(path, "speaker_packages", str(row["speaker_package_id"]))
    target_package = _package_payload(path, "target_packages", str(row["target_package_id"]))
    config = json.loads(str(row["config_json"]))
    design_ui = json.loads(str(row["ui_json"]))
    ui_profile = json.loads(str(row["ui_profile_json"]))
    processing = json.loads(str(row["processing_json"]))
    source_meta = json.loads(str(row["source_meta_json"]))
    payload = _compose_runtime_payload(
        response_asset_path,
        speaker_package=speaker_package,
        target_package=target_package,
        config=config,
        design_ui=design_ui,
        ui_profile=ui_profile,
        processing=processing,
        source_meta=source_meta,
        name=str(row["name"]),
    )
    return DesignRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        speaker_type=str(row["speaker_type"]),
        side=str(row["side"]),
        position=str(row["position"]),
        tags=str(row["tags"]),
        note=str(row["note"]),
        speaker_package_id=str(row["speaker_package_id"]),
        target_package_id=str(row["target_package_id"]),
        sample_rate=_optional_int(row["sample_rate"]),
        taps=_optional_int(row["taps"]),
        analysis_fft_size=_optional_int(row["analysis_fft_size"]),
        payload=payload,
        content_signature=str(row["content_signature"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_opened_at=str(row["last_opened_at"]),
        archived=bool(row["archived"]),
    )


def _compose_runtime_payload(
    response_asset_path: Path,
    *,
    speaker_package: Mapping[str, Any],
    target_package: Mapping[str, Any],
    config: Mapping[str, Any],
    design_ui: Mapping[str, Any],
    ui_profile: Mapping[str, Any],
    processing: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    speaker_refs = dict(speaker_package.get("response_refs", {}))
    target_refs = dict(target_package.get("response_refs", {}))
    raw_speaker = _response_from_ref(response_asset_path, speaker_refs.get("speaker_response_raw"))
    processed_speaker = _response_from_ref(response_asset_path, speaker_refs.get("speaker_response_processed"))
    raw_target = _response_from_ref(response_asset_path, target_refs.get("target_response_raw"))
    effective_target = _response_from_ref(response_asset_path, target_refs.get("target_response_effective"))
    mic_calibration = _response_from_ref(response_asset_path, speaker_refs.get("mic_cal_response_raw"))
    combined_config = dict(config)
    combined_config["speaker_response"] = speaker_response_payload(processed_speaker or raw_speaker)
    combined_config["target_response"] = speaker_response_payload(effective_target or raw_target)
    combined_ui = {
        **dict(speaker_package.get("input_ui", {})),
        **dict(target_package.get("target_ui", {})),
        **dict(design_ui),
    }
    combined_ui["project_name"] = name
    response_refs = {
        "speaker_response_raw": speaker_refs.get("speaker_response_raw"),
        "target_response_raw": target_refs.get("target_response_raw"),
        "mic_cal_response_raw": speaker_refs.get("mic_cal_response_raw"),
    }
    payload = {
        "app": source_meta.get("app", "PhaseEQ"),
        "app_version": source_meta.get("app_version", ""),
        "author": source_meta.get("author", ""),
        "schema_version": max(3, int(source_meta.get("schema_version", 1) or 1)),
        "snapshot_name": name,
        "io": {
            "speaker_source_name": speaker_package.get("source_name", ""),
            "target_source_name": target_package.get("source_name", ""),
            "mic_cal_source_name": speaker_package.get("mic_cal_source_name", ""),
            # Runtime composition is intentionally self-contained. The stored
            # packages still hold only response references; embedding happens
            # after load so existing editors need no legacy DB or resolver.
            "speaker_response_raw": speaker_response_payload(raw_speaker),
            "target_response_raw": speaker_response_payload(raw_target),
            "mic_cal_response_raw": speaker_response_payload(mic_calibration),
            "response_refs": response_refs,
        },
        "ui": combined_ui,
        "ui_profile": dict(ui_profile),
        "config": combined_config,
        "processing": dict(processing),
    }
    payload["revision"] = content_sha256(payload)
    return payload


def _response(value: Any) -> SpeakerResponse | None:
    return speaker_response_from_payload(value)


def _response_from_ref(response_asset_path: Path, ref: Any) -> SpeakerResponse | None:
    if not isinstance(ref, Mapping) or not str(ref.get("id", "")).strip():
        return None
    return load_response_asset(response_asset_path, dict(ref))


def _save_asset(
    path: Path,
    response: SpeakerResponse | None,
    *,
    kind: str,
    current_ref: Any = None,
) -> dict[str, Any] | None:
    if response is None:
        return None
    valid_current = dict(current_ref) if isinstance(current_ref, Mapping) else None
    saved = save_response_asset(path, response, kind=kind, current_ref=valid_current)
    return response_asset_ref_payload(saved)


def _is_speaker_ui_key(key: str) -> bool:
    return key in _SPEAKER_UI_KEYS or key.startswith(_SPEAKER_UI_PREFIXES)


def _is_target_ui_key(key: str) -> bool:
    return key.startswith(_TARGET_UI_PREFIXES)


def _payload_summary(config: Mapping[str, Any]) -> tuple[int | None, int | None, int | None]:
    return (
        _optional_int(config.get("sample_rate")),
        _optional_int(config.get("taps")),
        _optional_int(config.get("analysis_fft_size")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def list_speaker_package_summaries(path, response_asset_path=None, *, include_archived=False):
    """List metadata without resolving any response asset or runtime payload."""
    ensure_design_library_db(path)
    names = ('speaker_type', 'side', 'position', 'tags', 'note', 'source_name', 'mic_cal_source_name')
    projected = ', '.join(f"json_extract(payload_json, '$.{name}') AS {name}" for name in names)
    where = '' if include_archived else 'WHERE archived=0'
    with connect_sqlite(path) as connection:
        rows = connection.execute(f"SELECT id, name, content_signature, created_at, updated_at, archived, {projected} FROM speaker_packages {where} ORDER BY updated_at DESC,name COLLATE NOCASE").fetchall()
    return [SpeakerPackageRecord(
        id=str(row['id']), name=str(row['name']),
        payload={'_summary': True, 'source_name': str(row['source_name'] or ''),
                 'mic_cal_source_name': str(row['mic_cal_source_name'] or '')},
        runtime_payload={}, content_signature=str(row['content_signature']),
        created_at=str(row['created_at']), updated_at=str(row['updated_at']), archived=bool(row['archived']),
        **{name: str(row[name] or '') for name in names[:5]}) for row in rows]
