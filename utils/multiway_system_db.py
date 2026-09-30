"""Independent Multiway System Library.

The repository owns system identity and references only. Speaker/Input assets,
PhaseEQ designs, Assignments, and Working Session ZIPs remain in their existing
repositories and are referenced by stable IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable
import uuid

from utils.sqlite_repository import (
    backup_sqlite,
    canonical_json,
    connect_sqlite,
    content_sha256,
    ensure_app_meta,
    utc_now,
)


MULTIWAY_SYSTEM_SCHEMA_VERSION = 3
MULTIWAY_SYSTEM_DB_KIND = "phaseeq_multiway_system_library"
VALID_MODES = frozenset({
    "Fullrange", "Fullrange+SUB",
    "2Way", "2Way+SUB", "3Way", "3Way+SUB", "4Way", "4Way+SUB",
})
VALID_WAYS = frozenset({"High", "Mid", "Low", "SUB", "Fullrange"})


@dataclass(frozen=True)
class MultiwayChannelReference:
    channel_id: str
    name: str
    way: str
    group: str
    band: str = ""
    enabled: bool = True
    gain_db: float = 0.0
    polarity: int = 1
    delay_samples: float = 0.0
    auto_alignment_delay_samples: float = 0.0
    auto_alignment_allpass: tuple[dict[str, Any], ...] = ()
    speaker_package_id: str = ""
    latest_assignment_id: str = ""
    target_preset_id: str = ""
    sort_order: int = 0
    dc_gain_normalize: bool = False

    def validate(self) -> None:
        if not self.channel_id.strip():
            raise ValueError("Multiway Channel ID is required")
        if not self.name.strip() or not self.group.strip():
            raise ValueError("Multiway Channel name and Group are required")
        if self.way not in VALID_WAYS:
            raise ValueError(f"Unsupported Multiway Way: {self.way}")
        if int(self.polarity) not in {-1, 1}:
            raise ValueError("Multiway Channel polarity must be -1 or 1")
        if int(self.sort_order) < 0:
            raise ValueError("Multiway Channel sort order must not be negative")


@dataclass(frozen=True)
class MultiwaySystem:
    id: str
    name: str
    mode: str
    sample_rate_hz: int
    settings: dict[str, Any]
    channels: tuple[MultiwayChannelReference, ...] = ()
    selected_group: str = "Main"
    note: str = ""
    revision: int = 1

    def validate(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("Multiway System ID and name are required")
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported Multiway mode: {self.mode}")
        if int(self.sample_rate_hz) <= 0:
            raise ValueError("Multiway System sample rate must be positive")
        if not isinstance(self.settings, dict):
            raise ValueError("Multiway System settings must be an object")
        channel_ids: set[str] = set()
        identities: set[tuple[str, str, str]] = set()
        for channel in self.channels:
            channel.validate()
            if channel.channel_id in channel_ids:
                raise ValueError(f"Duplicate Multiway Channel ID: {channel.channel_id}")
            identity = (channel.name, channel.way, channel.group)
            if identity in identities:
                raise ValueError(f"Duplicate Multiway Channel identity: {identity}")
            channel_ids.add(channel.channel_id)
            identities.add(identity)
        if int(self.revision) < 1:
            raise ValueError("Multiway System revision must be at least 1")


@dataclass(frozen=True)
class MultiwaySystemRecord:
    system: MultiwaySystem
    archived: bool
    created_at: str
    updated_at: str
    management_no: str = ""
    content_hash: str = ""


@dataclass(frozen=True)
class MultiwaySystemRevisionRecord:
    system_id: str
    revision_number: int
    content_hash: str
    snapshot: dict[str, Any]
    restored_from_revision_number: int | None
    operation_id: str
    created_at: str


@dataclass(frozen=True)
class MultiwaySystemDraftRecord:
    draft_id: str
    session_id: str
    system_id: str
    base_revision_number: int | None
    payload: dict[str, Any]
    content_hash: str
    lease_expires_at: str
    state: str
    updated_at: str


class MultiwaySystemRevisionConflict(RuntimeError):
    pass


def new_multiway_system(
    name: str,
    *,
    mode: str,
    sample_rate_hz: int,
    settings: dict[str, Any],
    channels: Iterable[MultiwayChannelReference] = (),
    selected_group: str = "Main",
    note: str = "",
) -> MultiwaySystem:
    system = MultiwaySystem(
        id=str(uuid.uuid4()), name=str(name).strip(), mode=str(mode),
        sample_rate_hz=int(sample_rate_hz),
        settings=json.loads(canonical_json(settings)),
        channels=tuple(channels), selected_group=str(selected_group).strip() or "Main",
        note=str(note).strip(), revision=1,
    )
    system.validate()
    return system


def ensure_multiway_system_db(path: Path) -> None:
    path = Path(path)
    if path.is_file():
        try:
            with connect_sqlite(path, read_only=True) as probe:
                meta = probe.execute(
                    "SELECT value FROM app_meta WHERE key='schema_version'"
                ).fetchone()
        except Exception:
            meta = None
        existing_schema = int(meta[0]) if meta is not None else 0
        if 0 < existing_schema < MULTIWAY_SYSTEM_SCHEMA_VERSION:
            backup = path.with_name(
                f"{path.stem}.schema-v{existing_schema}-backup{path.suffix}"
            )
            if not backup.exists():
                backup_sqlite(path, backup)
    with connect_sqlite(path) as connection:
        ensure_app_meta(
            connection,
            schema_version=MULTIWAY_SYSTEM_SCHEMA_VERSION,
            db_kind=MULTIWAY_SYSTEM_DB_KIND,
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS multiway_systems (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            mode TEXT NOT NULL,
            sample_rate_hz INTEGER NOT NULL,
            settings_json TEXT NOT NULL,
            selected_group TEXT NOT NULL,
            note TEXT NOT NULL,
            revision INTEGER NOT NULL,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS multiway_system_channels (
            system_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            name TEXT NOT NULL,
            way TEXT NOT NULL,
            group_name TEXT NOT NULL,
            band TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            gain_db REAL NOT NULL,
            polarity INTEGER NOT NULL,
            delay_samples REAL NOT NULL,
            auto_alignment_delay_samples REAL NOT NULL DEFAULT 0.0,
            auto_alignment_allpass_json TEXT NOT NULL DEFAULT '[]',
            speaker_package_id TEXT NOT NULL,
            latest_assignment_id TEXT NOT NULL,
            target_preset_id TEXT NOT NULL DEFAULT '',
            sort_order INTEGER NOT NULL,
            PRIMARY KEY(system_id, channel_id),
            UNIQUE(system_id, name, way, group_name),
            FOREIGN KEY(system_id) REFERENCES multiway_systems(id) ON DELETE CASCADE)"""
        )
        channel_columns = {
            str(row[1]) for row in connection.execute(
                "PRAGMA table_info(multiway_system_channels)"
            ).fetchall()
        }
        if "band" not in channel_columns:
            connection.execute(
                "ALTER TABLE multiway_system_channels "
                "ADD COLUMN band TEXT NOT NULL DEFAULT ''"
            )
        for column_name, definition in (
            ("gain_db", "REAL NOT NULL DEFAULT 0.0"),
            ("polarity", "INTEGER NOT NULL DEFAULT 1"),
            ("dc_gain_normalize", "INTEGER NOT NULL DEFAULT 0"),
            ("delay_samples", "REAL NOT NULL DEFAULT 0.0"),
            ("auto_alignment_delay_samples", "REAL NOT NULL DEFAULT 0.0"),
            ("auto_alignment_allpass_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("target_preset_id", "TEXT NOT NULL DEFAULT ''"),
        ):
            if column_name not in channel_columns:
                connection.execute(
                    f"ALTER TABLE multiway_system_channels ADD COLUMN {column_name} {definition}"
                )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_multiway_systems_active "
            "ON multiway_systems(archived, updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_multiway_channels_identity "
            "ON multiway_system_channels(system_id, group_name, sort_order)"
        )
        system_columns = {
            str(row[1]) for row in connection.execute(
                "PRAGMA table_info(multiway_systems)"
            ).fetchall()
        }
        for column_name, definition in (
            ("management_no", "TEXT NOT NULL DEFAULT ''"),
            ("management_no_normalized", "TEXT NOT NULL DEFAULT ''"),
            ("content_hash", "TEXT NOT NULL DEFAULT ''"),
        ):
            if column_name not in system_columns:
                connection.execute(
                    f"ALTER TABLE multiway_systems ADD COLUMN {column_name} {definition}"
                )
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_multiway_management_no "
            "ON multiway_systems(management_no_normalized) "
            "WHERE management_no_normalized<>''"
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS multiway_system_revisions (
            system_id TEXT NOT NULL,
            revision_number INTEGER NOT NULL,
            base_revision_number INTEGER,
            restored_from_revision_number INTEGER,
            content_hash TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            operation_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            PRIMARY KEY(system_id,revision_number),
            FOREIGN KEY(system_id) REFERENCES multiway_systems(id) ON DELETE CASCADE)"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_multiway_revisions_hash "
            "ON multiway_system_revisions(system_id,content_hash)"
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS multiway_system_aliases (
            alias_normalized TEXT PRIMARY KEY,
            alias_display TEXT NOT NULL,
            system_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            retired_at TEXT,
            FOREIGN KEY(system_id) REFERENCES multiway_systems(id) ON DELETE CASCADE)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS change_journal (
            change_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            revision_number INTEGER,
            change_kind TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            operation_id TEXT NOT NULL,
            created_at TEXT NOT NULL)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS audit_log (
            audit_sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_id TEXT NOT NULL,
            action TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            before_revision_number INTEGER,
            after_revision_number INTEGER,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS system_drafts (
            draft_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            system_id TEXT NOT NULL DEFAULT '',
            base_revision_number INTEGER,
            draft_json TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            lease_expires_at TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('editing','recoverable','completed','archived')),
            updated_at TEXT NOT NULL)"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_system_drafts_session "
            "ON system_drafts(session_id,state,updated_at DESC)"
        )
        migration_rows = connection.execute(
            """SELECT s.id,s.name,s.mode,s.sample_rate_hz,s.settings_json,
            s.selected_group,s.note,s.revision,s.management_no,
            s.management_no_normalized FROM multiway_systems s
            LEFT JOIN multiway_system_revisions r
              ON r.system_id=s.id AND r.revision_number=s.revision
            WHERE r.system_id IS NULL"""
        ).fetchall()
        for row in migration_rows:
            channel_rows = connection.execute(
                """SELECT channel_id,name,way,group_name,band,enabled,gain_db,
                polarity,delay_samples,speaker_package_id,latest_assignment_id,
                target_preset_id,sort_order,auto_alignment_delay_samples,
                auto_alignment_allpass_json,dc_gain_normalize FROM multiway_system_channels
                WHERE system_id=? ORDER BY sort_order,group_name,name""",
                (str(row[0]),),
            ).fetchall()
            migrated = MultiwaySystem(
                id=str(row[0]), name=str(row[1]), mode=str(row[2]),
                sample_rate_hz=int(row[3]), settings=json.loads(str(row[4])),
                channels=tuple(
                    MultiwayChannelReference(
                        channel_id=str(item[0]), name=str(item[1]), way=str(item[2]),
                        group=str(item[3]), band=str(item[4]), enabled=bool(item[5]),
                        gain_db=float(item[6]), polarity=int(item[7]),
                        delay_samples=float(item[8]), speaker_package_id=str(item[9]),
                        latest_assignment_id=str(item[10]), target_preset_id=str(item[11]),
                        sort_order=int(item[12]),
                        auto_alignment_delay_samples=float(item[13]),
                        auto_alignment_allpass=tuple(json.loads(str(item[14]))),
                        dc_gain_normalize=bool(item[15]),
                    )
                    for item in channel_rows
                ),
                selected_group=str(row[5]), note=str(row[6]), revision=int(row[7]),
            )
            migrated.validate()
            management_no = str(row[8]).strip()
            management_no_normalized = str(row[9]).strip()
            if not management_no:
                management_no, management_no_normalized = _allocate_management_no(
                    connection, migrated.name,
                )
            snapshot = _system_snapshot(migrated)
            snapshot_json = canonical_json(snapshot)
            digest = content_sha256(snapshot_json)
            operation_id = str(uuid.uuid4())
            now = utc_now(connection)
            connection.execute(
                """UPDATE multiway_systems SET management_no=?,
                management_no_normalized=?,content_hash=? WHERE id=?""",
                (management_no, management_no_normalized, digest, migrated.id),
            )
            connection.execute(
                """INSERT INTO multiway_system_revisions(
                system_id,revision_number,base_revision_number,
                restored_from_revision_number,content_hash,snapshot_json,
                operation_id,created_at) VALUES(?,?,NULL,NULL,?,?,?,?)""",
                (
                    migrated.id, migrated.revision, digest, snapshot_json,
                    operation_id, now,
                ),
            )
            connection.execute(
                """INSERT INTO change_journal(
                entity_type,entity_id,revision_number,change_kind,content_hash,
                operation_id,created_at) VALUES('system',?,?,'system.migrated',?,?,?)""",
                (migrated.id, migrated.revision, digest, operation_id, now),
            )


def check_multiway_system_integrity(path: Path) -> tuple[str, ...]:
    ensure_multiway_system_db(path)
    issues: list[str] = []
    with connect_sqlite(path, read_only=True) as connection:
        meta = connection.execute(
            "SELECT value FROM app_meta WHERE key='schema_version'"
        ).fetchone()
        if meta is None or int(meta[0]) != MULTIWAY_SYSTEM_SCHEMA_VERSION:
            issues.append("schema_version")
        for row in connection.execute("PRAGMA foreign_key_check").fetchall():
            issues.append(f"foreign_key:{row[0]}:{row[1]}")
        missing = connection.execute(
            """SELECT s.id FROM multiway_systems s
            LEFT JOIN multiway_system_revisions r
              ON r.system_id=s.id AND r.revision_number=s.revision
            WHERE r.system_id IS NULL"""
        ).fetchall()
        issues.extend(f"missing_revision:{row[0]}" for row in missing)
    return tuple(issues)


def save_multiway_system_draft(
    path: Path,
    *,
    session_id: str,
    payload: dict[str, Any],
    system_id: str = "",
    base_revision_number: int | None = None,
    draft_id: str | None = None,
    expected_content_hash: str | None = None,
) -> MultiwaySystemDraftRecord:
    if not str(session_id).strip() or not isinstance(payload, dict):
        raise ValueError("Draft session ID and payload are required")
    ensure_multiway_system_db(path)
    draft_id = str(draft_id or uuid.uuid4())
    payload_json = canonical_json(payload)
    digest = content_sha256(payload_json)
    with connect_sqlite(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT content_hash FROM system_drafts WHERE draft_id=?", (draft_id,),
        ).fetchone()
        if expected_content_hash is not None:
            actual = str(existing[0]) if existing is not None else "missing"
            if actual != str(expected_content_hash):
                raise MultiwaySystemRevisionConflict("System Draft changed in another session")
        now = utc_now(connection)
        lease = str(connection.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now','+24 hours')"
        ).fetchone()[0])
        connection.execute(
            """INSERT INTO system_drafts(
            draft_id,session_id,system_id,base_revision_number,draft_json,
            content_hash,lease_expires_at,state,updated_at)
            VALUES(?,?,?,?,?,?,?,'editing',?)
            ON CONFLICT(draft_id) DO UPDATE SET
            draft_json=excluded.draft_json,content_hash=excluded.content_hash,
            lease_expires_at=excluded.lease_expires_at,state='editing',
            updated_at=excluded.updated_at""",
            (
                draft_id, str(session_id), str(system_id), base_revision_number,
                payload_json, digest, lease, now,
            ),
        )
    return get_multiway_system_draft(path, draft_id)


def get_multiway_system_draft(
    path: Path, draft_id: str,
) -> MultiwaySystemDraftRecord:
    ensure_multiway_system_db(path)
    with connect_sqlite(path, read_only=True) as connection:
        row = connection.execute(
            """SELECT draft_id,session_id,system_id,base_revision_number,draft_json,
            content_hash,lease_expires_at,state,updated_at
            FROM system_drafts WHERE draft_id=?""",
            (str(draft_id),),
        ).fetchone()
    if row is None:
        raise ValueError("System Draft does not exist")
    return MultiwaySystemDraftRecord(
        draft_id=str(row[0]), session_id=str(row[1]), system_id=str(row[2]),
        base_revision_number=None if row[3] is None else int(row[3]),
        payload=json.loads(str(row[4])), content_hash=str(row[5]),
        lease_expires_at=str(row[6]), state=str(row[7]), updated_at=str(row[8]),
    )


def recover_multiway_system_drafts(
    path: Path, session_id: str,
) -> tuple[MultiwaySystemDraftRecord, ...]:
    ensure_multiway_system_db(path)
    with connect_sqlite(path) as connection:
        connection.execute(
            """UPDATE system_drafts SET state='recoverable'
            WHERE state='editing' AND lease_expires_at <
              strftime('%Y-%m-%dT%H:%M:%fZ','now')"""
        )
        rows = connection.execute(
            """SELECT draft_id FROM system_drafts
            WHERE session_id=? AND state IN ('editing','recoverable')
            ORDER BY updated_at DESC""",
            (str(session_id),),
        ).fetchall()
    return tuple(get_multiway_system_draft(path, str(row[0])) for row in rows)


def _management_no_normalized(value: str) -> str:
    return "-".join(str(value).strip().upper().replace("_", "-").split())


def _management_no_candidate(name: str) -> str:
    ascii_source = "".join(
        character if character.isascii() and character.isalnum() else "-"
        for character in str(name).upper()
    )
    stem = "-".join(part for part in ascii_source.split("-") if part)[:40]
    return stem or "SYS"


def _allocate_management_no(connection: Any, name: str) -> tuple[str, str]:
    stem = _management_no_candidate(name)
    for sequence in range(1, 1_000_000):
        display = f"{stem}-{sequence:03d}"
        normalized = _management_no_normalized(display)
        exists = connection.execute(
            "SELECT 1 FROM multiway_systems WHERE management_no_normalized=?",
            (normalized,),
        ).fetchone()
        if exists is None:
            return display, normalized
    raise RuntimeError("Multiway System management number space is exhausted")


def _system_snapshot(
    system: MultiwaySystem,
    *,
    target_bindings: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "format": "phaseeq-multiway-system-snapshot",
        "format_version": 1,
        "system_id": system.id,
        "mode": system.mode,
        "sample_rate_hz": int(system.sample_rate_hz),
        "selected_group": system.selected_group,
        "settings": system.settings,
        "channels": [
            {
                "channel_id": item.channel_id,
                "name": item.name,
                "way": item.way,
                "group": item.group,
                "band": item.band,
                "enabled": bool(item.enabled),
                "gain_db": float(item.gain_db),
                "polarity": int(item.polarity),
                "dc_gain_normalize": bool(item.dc_gain_normalize),
                "delay_samples": float(item.delay_samples),
                "auto_alignment_delay_samples": float(item.auto_alignment_delay_samples),
                "auto_alignment_allpass": [dict(section) for section in item.auto_alignment_allpass],
                "speaker_package_id": item.speaker_package_id,
                "latest_assignment_id": item.latest_assignment_id,
                "target_preset_id": item.target_preset_id,
                "sort_order": int(item.sort_order),
            }
            for item in sorted(
                system.channels,
                key=lambda channel: (channel.group, channel.sort_order, channel.channel_id),
            )
        ],
        "target_bindings": sorted(
            (dict(item) for item in target_bindings),
            key=lambda item: (
                str(item.get("scope", "")), str(item.get("group", "")),
                str(item.get("channel_id", "")), str(item.get("way", "")),
            ),
        ),
    }


def _target_binding_snapshots(connection: Any, system_id: str) -> list[dict[str, Any]]:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='system_target_bindings'"
    ).fetchone()
    if exists is None:
        return []
    rows = connection.execute(
        """SELECT scope,group_name,channel_id,way,target_id,binding_mode,
        pinned_revision_number,resolved_revision_number,resolved_content_hash
        FROM system_target_bindings WHERE system_id=?""",
        (str(system_id),),
    ).fetchall()
    return [
        {
            "scope": str(row[0]), "group": str(row[1]),
            "channel_id": str(row[2]), "way": str(row[3]),
            "target_id": str(row[4]), "binding_mode": str(row[5]),
            "pinned_revision_number": None if row[6] is None else int(row[6]),
            "resolved_revision_number": int(row[7]),
            "resolved_content_hash": str(row[8]),
        }
        for row in rows
    ]


def _refresh_latest_target_bindings(connection: Any, system_id: str, now: str) -> None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='system_target_bindings'"
    ).fetchone()
    if exists is None:
        return
    rows = connection.execute(
        """SELECT b.scope,b.group_name,b.channel_id,b.way,t.latest_revision_number,
        r.content_hash FROM system_target_bindings b
        JOIN targets t ON t.target_id=b.target_id
        JOIN target_revisions r ON r.target_id=t.target_id
          AND r.revision_number=t.latest_revision_number
        WHERE b.system_id=? AND b.binding_mode='latest'""",
        (str(system_id),),
    ).fetchall()
    connection.executemany(
        """UPDATE system_target_bindings
        SET resolved_revision_number=?,resolved_content_hash=?,updated_at=?
        WHERE system_id=? AND scope=? AND group_name=? AND channel_id=? AND way=?""",
        [
            (
                int(row[4]), str(row[5]), now, str(system_id), str(row[0]),
                str(row[1]), str(row[2]), str(row[3]),
            )
            for row in rows
        ],
    )


def save_multiway_system(
    path: Path,
    system: MultiwaySystem,
    *,
    expected_revision: int | None = None,
    operation_id: str | None = None,
    restored_from_revision: int | None = None,
    allow_identical_revision: bool = True,
    restored_target_bindings: Iterable[dict[str, Any]] | None = None,
) -> MultiwaySystemRecord:
    system.validate()
    ensure_multiway_system_db(path)
    with connect_sqlite(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        existing = connection.execute(
            "SELECT created_at,archived,revision,management_no,management_no_normalized "
            "FROM multiway_systems WHERE id=?",
            (system.id,),
        ).fetchone()
        if expected_revision is not None:
            actual_revision = int(existing[2]) if existing else 0
            if actual_revision != int(expected_revision):
                raise MultiwaySystemRevisionConflict(
                    f"Multiway System changed from revision {expected_revision} "
                    f"to {actual_revision}"
                )
        now = utc_now(connection)
        created_at = str(existing[0]) if existing else now
        archived = bool(existing[1]) if existing else False
        revision = int(existing[2]) + 1 if existing else int(system.revision)
        saved = replace(system, revision=revision)
        if existing and str(existing[3]).strip():
            management_no = str(existing[3])
            management_no_normalized = str(existing[4])
        else:
            management_no, management_no_normalized = _allocate_management_no(
                connection, saved.name,
            )
        _refresh_latest_target_bindings(connection, saved.id, now)
        binding_snapshots = (
            [dict(item) for item in restored_target_bindings]
            if restored_target_bindings is not None
            else _target_binding_snapshots(connection, saved.id)
        )
        if restored_target_bindings is not None:
            # Restored System revisions always pin the exact Target revisions
            # captured by the historical snapshot.
            binding_snapshots = [
                {
                    **item,
                    "binding_mode": "pinned",
                    "pinned_revision_number": int(item["resolved_revision_number"]),
                }
                for item in binding_snapshots
            ]
        snapshot = _system_snapshot(saved, target_bindings=binding_snapshots)
        snapshot_json = canonical_json(snapshot)
        content_hash = content_sha256(snapshot_json)
        operation_id = str(operation_id or uuid.uuid4())
        prior_operation = connection.execute(
            "SELECT system_id,revision_number FROM multiway_system_revisions "
            "WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if prior_operation is not None:
            if str(prior_operation[0]) != saved.id:
                raise MultiwaySystemRevisionConflict(
                    "Operation ID is already used by another Multiway System"
                )
            connection.rollback()
            existing_record = get_multiway_system(path, saved.id)
            if existing_record is None:
                raise RuntimeError("Idempotent Multiway System result is missing")
            return existing_record
        if existing and not allow_identical_revision:
            previous_hash = str(connection.execute(
                "SELECT content_hash FROM multiway_systems WHERE id=?", (saved.id,),
            ).fetchone()[0])
            if previous_hash == content_hash:
                connection.rollback()
                existing_record = get_multiway_system(path, saved.id)
                if existing_record is None:
                    raise RuntimeError("Unchanged Multiway System could not be reloaded")
                return existing_record
        connection.execute(
            """INSERT INTO multiway_systems(
            id,name,mode,sample_rate_hz,settings_json,selected_group,note,revision,
            archived,created_at,updated_at,management_no,management_no_normalized,content_hash)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,mode=excluded.mode,sample_rate_hz=excluded.sample_rate_hz,
            settings_json=excluded.settings_json,selected_group=excluded.selected_group,
            note=excluded.note,revision=excluded.revision,updated_at=excluded.updated_at,
            management_no=excluded.management_no,
            management_no_normalized=excluded.management_no_normalized,
            content_hash=excluded.content_hash""",
            (
                saved.id, saved.name, saved.mode, saved.sample_rate_hz,
                canonical_json(saved.settings), saved.selected_group, saved.note,
                saved.revision, int(archived), created_at, now,
                management_no, management_no_normalized, content_hash,
            ),
        )
        connection.execute(
            "DELETE FROM multiway_system_channels WHERE system_id=?", (saved.id,),
        )
        connection.executemany(
            """INSERT INTO multiway_system_channels(
            system_id,channel_id,name,way,group_name,band,enabled,gain_db,polarity,
            delay_samples,speaker_package_id,latest_assignment_id,target_preset_id,sort_order,
            auto_alignment_delay_samples,auto_alignment_allpass_json,dc_gain_normalize)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    saved.id, item.channel_id, item.name, item.way, item.group,
                    item.band, int(item.enabled), float(item.gain_db), int(item.polarity),
                    float(item.delay_samples), item.speaker_package_id,
                    item.latest_assignment_id, item.target_preset_id, int(item.sort_order),
                    float(item.auto_alignment_delay_samples),
                    canonical_json(list(item.auto_alignment_allpass)),
                    int(item.dc_gain_normalize),
                )
                for item in saved.channels
            ],
        )
        if restored_target_bindings is not None:
            bindings_table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='system_target_bindings'"
            ).fetchone()
            if bindings_table is not None:
                connection.execute(
                    "DELETE FROM system_target_bindings WHERE system_id=?", (saved.id,),
                )
                connection.executemany(
                    """INSERT INTO system_target_bindings(
                    system_id,scope,group_name,channel_id,way,target_id,binding_mode,
                    pinned_revision_number,resolved_revision_number,
                    resolved_content_hash,updated_at)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    [
                        (
                            saved.id, str(item.get("scope", "system")),
                            str(item.get("group", "")), str(item.get("channel_id", "")),
                            str(item.get("way", "")), str(item.get("target_id", "")),
                            "pinned", int(item.get("resolved_revision_number", 0)),
                            int(item.get("resolved_revision_number", 0)),
                            str(item.get("resolved_content_hash", "")), now,
                        )
                        for item in binding_snapshots
                    ],
                )
        connection.execute(
            """INSERT INTO multiway_system_revisions(
            system_id,revision_number,base_revision_number,restored_from_revision_number,
            content_hash,snapshot_json,operation_id,created_at)
            VALUES(?,?,?,?,?,?,?,?)""",
            (
                saved.id, saved.revision,
                int(existing[2]) if existing else None, restored_from_revision,
                content_hash, snapshot_json, operation_id, now,
            ),
        )
        connection.execute(
            """INSERT INTO change_journal(
            entity_type,entity_id,revision_number,change_kind,content_hash,operation_id,created_at)
            VALUES('system',?,?,?,?,?,?)""",
            (
                saved.id, saved.revision,
                "system.revision_saved" if existing else "system.created",
                content_hash, operation_id, now,
            ),
        )
        connection.execute(
            """INSERT INTO audit_log(
            operation_id,action,entity_type,entity_id,before_revision_number,
            after_revision_number,metadata_json,created_at)
            VALUES(?,?,'system',?,?,?,?,?)""",
            (
                operation_id,
                "system.revision_saved" if existing else "system.created",
                saved.id, int(existing[2]) if existing else None, saved.revision,
                canonical_json({"management_no": management_no}), now,
            ),
        )
    record = get_multiway_system(path, saved.id)
    if record is None:
        raise RuntimeError("Saved Multiway System could not be reloaded")
    return record


def get_multiway_system(path: Path, system_id: str) -> MultiwaySystemRecord | None:
    ensure_multiway_system_db(path)
    with connect_sqlite(path, read_only=True) as connection:
        row = connection.execute(
            """SELECT id,name,mode,sample_rate_hz,settings_json,selected_group,note,
            revision,archived,created_at,updated_at,management_no,content_hash
            FROM multiway_systems WHERE id=?""",
            (str(system_id),),
        ).fetchone()
        if row is None:
            return None
        channel_rows = connection.execute(
            """SELECT channel_id,name,way,group_name,band,enabled,gain_db,polarity,
            delay_samples,speaker_package_id,latest_assignment_id,target_preset_id,sort_order,
            auto_alignment_delay_samples,auto_alignment_allpass_json,dc_gain_normalize
            FROM multiway_system_channels
            WHERE system_id=? ORDER BY sort_order,group_name,name""",
            (str(system_id),),
        ).fetchall()
    channels = tuple(
        MultiwayChannelReference(
            channel_id=str(item[0]), name=str(item[1]), way=str(item[2]),
            group=str(item[3]), band=str(item[4]), enabled=bool(item[5]),
            gain_db=float(item[6]), polarity=int(item[7]), delay_samples=float(item[8]),
            speaker_package_id=str(item[9]), latest_assignment_id=str(item[10]),
            target_preset_id=str(item[11]), sort_order=int(item[12]),
            auto_alignment_delay_samples=float(item[13]),
            auto_alignment_allpass=tuple(json.loads(str(item[14]))),
            dc_gain_normalize=bool(item[15]),
        )
        for item in channel_rows
    )
    system = MultiwaySystem(
        id=str(row[0]), name=str(row[1]), mode=str(row[2]),
        sample_rate_hz=int(row[3]), settings=json.loads(str(row[4])),
        channels=channels, selected_group=str(row[5]), note=str(row[6]),
        revision=int(row[7]),
    )
    system.validate()
    return MultiwaySystemRecord(
        system, bool(row[8]), str(row[9]), str(row[10]), str(row[11]), str(row[12]),
    )


def get_multiway_system_revision(
    path: Path, system_id: str, revision_number: int,
) -> MultiwaySystemRevisionRecord | None:
    ensure_multiway_system_db(path)
    with connect_sqlite(path, read_only=True) as connection:
        row = connection.execute(
            """SELECT system_id,revision_number,content_hash,snapshot_json,
            restored_from_revision_number,operation_id,created_at
            FROM multiway_system_revisions WHERE system_id=? AND revision_number=?""",
            (str(system_id), int(revision_number)),
        ).fetchone()
    if row is None:
        return None
    return MultiwaySystemRevisionRecord(
        system_id=str(row[0]), revision_number=int(row[1]), content_hash=str(row[2]),
        snapshot=json.loads(str(row[3])),
        restored_from_revision_number=None if row[4] is None else int(row[4]),
        operation_id=str(row[5]), created_at=str(row[6]),
    )


def list_multiway_system_revisions(
    path: Path, system_id: str,
) -> tuple[MultiwaySystemRevisionRecord, ...]:
    ensure_multiway_system_db(path)
    with connect_sqlite(path, read_only=True) as connection:
        rows = connection.execute(
            """SELECT revision_number FROM multiway_system_revisions
            WHERE system_id=? ORDER BY revision_number DESC""",
            (str(system_id),),
        ).fetchall()
    return tuple(
        record for row in rows
        if (record := get_multiway_system_revision(path, system_id, int(row[0]))) is not None
    )


def save_multiway_system_revision(
    path: Path,
    system: MultiwaySystem,
    *,
    expected_revision: int,
    operation_id: str | None = None,
) -> MultiwaySystemRecord:
    return save_multiway_system(
        path, system, expected_revision=expected_revision,
        operation_id=operation_id, allow_identical_revision=False,
    )


def restore_multiway_system_revision(
    path: Path,
    system_id: str,
    revision_number: int,
    *,
    expected_revision: int,
    operation_id: str | None = None,
) -> MultiwaySystemRecord:
    current = get_multiway_system(path, system_id)
    source = get_multiway_system_revision(path, system_id, revision_number)
    if current is None or source is None:
        raise ValueError("Multiway System revision does not exist")
    payload = source.snapshot
    channels = tuple(
        MultiwayChannelReference(
            channel_id=str(item.get("channel_id", "")),
            name=str(item.get("name", "")), way=str(item.get("way", "")),
            group=str(item.get("group", "")), band=str(item.get("band", "")),
            enabled=bool(item.get("enabled", True)),
            gain_db=float(item.get("gain_db", 0.0)),
            polarity=int(item.get("polarity", 1)),
            dc_gain_normalize=bool(item.get("dc_gain_normalize", False)),
            delay_samples=float(item.get("delay_samples", 0.0)),
            auto_alignment_delay_samples=float(item.get("auto_alignment_delay_samples", 0.0)),
            auto_alignment_allpass=tuple(
                dict(section) for section in item.get("auto_alignment_allpass", ())
                if isinstance(section, dict)
            ),
            speaker_package_id=str(item.get("speaker_package_id", "")),
            latest_assignment_id=str(item.get("latest_assignment_id", "")),
            target_preset_id=str(item.get("target_preset_id", "")),
            sort_order=int(item.get("sort_order", 0)),
        )
        for item in payload.get("channels", []) if isinstance(item, dict)
    )
    restored = MultiwaySystem(
        id=current.system.id, name=current.system.name,
        mode=str(payload.get("mode", current.system.mode)),
        sample_rate_hz=int(payload.get("sample_rate_hz", current.system.sample_rate_hz)),
        settings=dict(payload.get("settings", {})), channels=channels,
        selected_group=str(payload.get("selected_group", "Main")),
        note=current.system.note, revision=current.system.revision,
    )
    return save_multiway_system(
        path, restored, expected_revision=expected_revision,
        operation_id=operation_id, restored_from_revision=int(revision_number),
        allow_identical_revision=True,
        restored_target_bindings=(
            payload.get("target_bindings", [])
            if isinstance(payload.get("target_bindings", []), list) else []
        ),
    )


def rename_multiway_system(
    path: Path,
    system_id: str,
    *,
    name: str,
    management_no: str,
) -> MultiwaySystemRecord:
    normalized = _management_no_normalized(management_no)
    if not str(name).strip() or not normalized:
        raise ValueError("Multiway System name and management number are required")
    ensure_multiway_system_db(path)
    with connect_sqlite(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT management_no,management_no_normalized FROM multiway_systems WHERE id=?",
            (str(system_id),),
        ).fetchone()
        if row is None:
            raise ValueError("Multiway System does not exist")
        owner = connection.execute(
            "SELECT id FROM multiway_systems WHERE management_no_normalized=? AND id<>?",
            (normalized, str(system_id)),
        ).fetchone()
        if owner is not None:
            raise ValueError("Multiway System management number is already in use")
        now = utc_now(connection)
        previous_display = str(row[0])
        previous_normalized = str(row[1])
        if previous_normalized and previous_normalized != normalized:
            connection.execute(
                """INSERT OR REPLACE INTO multiway_system_aliases(
                alias_normalized,alias_display,system_id,created_at,retired_at)
                VALUES(?,?,?,?,?)""",
                (previous_normalized, previous_display, str(system_id), now, now),
            )
        connection.execute(
            """UPDATE multiway_systems SET name=?,management_no=?,
            management_no_normalized=?,updated_at=? WHERE id=?""",
            (str(name).strip(), str(management_no).strip(), normalized, now, str(system_id)),
        )
        operation_id = str(uuid.uuid4())
        connection.execute(
            """INSERT INTO change_journal(
            entity_type,entity_id,revision_number,change_kind,content_hash,operation_id,created_at)
            SELECT 'system',id,revision,'system.renamed',content_hash,?,?
            FROM multiway_systems WHERE id=?""",
            (operation_id, now, str(system_id)),
        )
        connection.execute(
            """INSERT INTO audit_log(
            operation_id,action,entity_type,entity_id,before_revision_number,
            after_revision_number,metadata_json,created_at)
            SELECT ?,'system.renamed','system',id,revision,revision,?,?
            FROM multiway_systems WHERE id=?""",
            (
                operation_id,
                canonical_json({
                    "old_management_no": previous_display,
                    "new_management_no": str(management_no).strip(),
                }),
                now, str(system_id),
            ),
        )
    record = get_multiway_system(path, system_id)
    if record is None:
        raise RuntimeError("Renamed Multiway System could not be reloaded")
    return record


def list_multiway_systems(
    path: Path, *, include_archived: bool = False,
) -> list[MultiwaySystemRecord]:
    ensure_multiway_system_db(path)
    with connect_sqlite(path, read_only=True) as connection:
        rows = connection.execute(
            "SELECT id FROM multiway_systems "
            + ("" if include_archived else "WHERE archived=0 ")
            + "ORDER BY updated_at DESC,name COLLATE NOCASE"
        ).fetchall()
    return [
        record for row in rows
        if (record := get_multiway_system(path, str(row[0]))) is not None
    ]


def duplicate_multiway_system(
    path: Path, system_id: str, *, name: str,
) -> MultiwaySystemRecord:
    source = get_multiway_system(path, system_id)
    if source is None:
        raise ValueError("Multiway System does not exist")
    copied_channels = tuple(
        replace(
            channel,
            channel_id=str(uuid.uuid4()),
            latest_assignment_id="",
        )
        for channel in source.system.channels
    )
    copied = replace(
        source.system,
        id=str(uuid.uuid4()), name=str(name).strip(), channels=copied_channels,
        revision=1,
    )
    return save_multiway_system(path, copied)


def set_multiway_system_archived(path: Path, system_id: str, *, archived: bool) -> None:
    ensure_multiway_system_db(path)
    with connect_sqlite(path) as connection:
        now = utc_now(connection)
        cursor = connection.execute(
            "UPDATE multiway_systems SET archived=?,updated_at=? WHERE id=?",
            (int(bool(archived)), now, str(system_id)),
        )
        if cursor.rowcount != 1:
            raise ValueError("Multiway System does not exist")
        operation_id = str(uuid.uuid4())
        change_kind = "system.archived" if archived else "system.restored"
        connection.execute(
            """INSERT INTO change_journal(
            entity_type,entity_id,revision_number,change_kind,content_hash,operation_id,created_at)
            SELECT 'system',id,revision,?,content_hash,?,?
            FROM multiway_systems WHERE id=?""",
            (change_kind, operation_id, now, str(system_id)),
        )
        connection.execute(
            """INSERT INTO audit_log(
            operation_id,action,entity_type,entity_id,before_revision_number,
            after_revision_number,metadata_json,created_at)
            SELECT ?,?,'system',id,revision,revision,?,?
            FROM multiway_systems WHERE id=?""",
            (
                operation_id, change_kind,
                canonical_json({"archived": bool(archived)}), now, str(system_id),
            ),
        )


def update_multiway_channel_assignment(
    path: Path, *, system_id: str, channel_id: str, assignment_id: str,
) -> MultiwaySystemRecord:
    record = get_multiway_system(path, system_id)
    if record is None:
        raise ValueError("Multiway System does not exist")
    found = False
    channels: list[MultiwayChannelReference] = []
    for channel in record.system.channels:
        if channel.channel_id == str(channel_id):
            channel = replace(channel, latest_assignment_id=str(assignment_id).strip())
            found = True
        channels.append(channel)
    if not found:
        raise ValueError("Multiway Channel does not exist")
    return save_multiway_system(path, replace(record.system, channels=tuple(channels)))
