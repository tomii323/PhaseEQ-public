"""Shared revisioned Target repository for PhaseEQ and Multiway System Manager."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal
import uuid

from utils.multiway_system_db import ensure_multiway_system_db
from utils.sqlite_repository import canonical_json, connect_sqlite, content_sha256, utc_now

from .definition import canonical_target_definition


TargetLifecycle = Literal["built_in", "user", "temporary", "archived"]
TargetScope = Literal["system", "group", "channel", "way", "standalone"]
BindingMode = Literal["latest", "pinned"]

VALID_LIFECYCLES = frozenset({"built_in", "user", "temporary", "archived"})
VALID_SCOPES = frozenset({"system", "group", "channel", "way", "standalone"})
VALID_BINDING_MODES = frozenset({"latest", "pinned"})


class TargetRepositoryError(RuntimeError):
    pass


class TargetNotFoundError(TargetRepositoryError):
    pass


class TargetRevisionConflictError(TargetRepositoryError):
    pass


class TargetBindingError(TargetRepositoryError):
    pass


@dataclass(frozen=True)
class TargetRevisionRef:
    target_id: str
    revision_number: int
    content_hash: str


@dataclass(frozen=True)
class TargetRevisionRecord:
    ref: TargetRevisionRef
    name: str
    lifecycle: str
    source: str
    definition: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TargetSummary:
    target_id: str
    name: str
    lifecycle: str
    latest_revision_number: int
    latest_content_hash: str
    updated_at: str


@dataclass(frozen=True)
class TargetBinding:
    system_id: str
    scope: str
    group: str = ""
    channel_id: str = ""
    way: str = ""
    target_id: str = ""
    mode: str = "latest"
    pinned_revision_number: int | None = None

    def validate(self) -> None:
        if not self.system_id.strip() or not self.target_id.strip():
            raise TargetBindingError("System ID and Target ID are required")
        if self.scope not in VALID_SCOPES - {"standalone"}:
            raise TargetBindingError(f"Unsupported Target binding scope: {self.scope}")
        if self.mode not in VALID_BINDING_MODES:
            raise TargetBindingError(f"Unsupported Target binding mode: {self.mode}")
        if self.mode == "latest" and self.pinned_revision_number is not None:
            raise TargetBindingError("Latest Target binding must not specify a pinned revision")
        if self.mode == "pinned" and int(self.pinned_revision_number or 0) < 1:
            raise TargetBindingError("Pinned Target binding requires a revision")
        if self.scope == "group" and not self.group.strip():
            raise TargetBindingError("Group Target binding requires a Group")
        if self.scope == "channel" and not self.channel_id.strip():
            raise TargetBindingError("Channel Target binding requires a Channel ID")
        if self.scope == "way" and not self.way.strip():
            raise TargetBindingError("Way Target binding requires a Way")


@dataclass(frozen=True)
class ResolvedTargetBinding:
    binding: TargetBinding
    resolved: TargetRevisionRecord
    latest_available_revision_number: int
    requires_recalculation: bool


def ensure_target_repository(path: Path) -> None:
    ensure_multiway_system_db(path)
    with connect_sqlite(path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS targets (
            target_id TEXT PRIMARY KEY,
            target_name TEXT NOT NULL,
            lifecycle TEXT NOT NULL CHECK(lifecycle IN ('built_in','user','temporary','archived')),
            latest_revision_number INTEGER NOT NULL CHECK(latest_revision_number>=1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS target_revisions (
            target_id TEXT NOT NULL,
            revision_number INTEGER NOT NULL CHECK(revision_number>=1),
            content_hash TEXT NOT NULL,
            definition_json TEXT NOT NULL,
            source TEXT NOT NULL,
            response_asset_id TEXT,
            response_asset_hash TEXT,
            base_revision_number INTEGER,
            operation_id TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            PRIMARY KEY(target_id,revision_number),
            FOREIGN KEY(target_id) REFERENCES targets(target_id) ON DELETE CASCADE)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS system_target_bindings (
            system_id TEXT NOT NULL,
            scope TEXT NOT NULL CHECK(scope IN ('system','group','channel','way')),
            group_name TEXT NOT NULL DEFAULT '',
            channel_id TEXT NOT NULL DEFAULT '',
            way TEXT NOT NULL DEFAULT '',
            target_id TEXT NOT NULL,
            binding_mode TEXT NOT NULL CHECK(binding_mode IN ('latest','pinned')),
            pinned_revision_number INTEGER,
            resolved_revision_number INTEGER NOT NULL,
            resolved_content_hash TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(system_id,scope,group_name,channel_id,way),
            FOREIGN KEY(system_id) REFERENCES multiway_systems(id) ON DELETE CASCADE,
            FOREIGN KEY(target_id,resolved_revision_number)
              REFERENCES target_revisions(target_id,revision_number),
            CHECK((binding_mode='latest' AND pinned_revision_number IS NULL) OR
                  (binding_mode='pinned' AND pinned_revision_number IS NOT NULL)))"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS target_references (
            target_id TEXT NOT NULL,
            revision_number INTEGER NOT NULL,
            ref_type TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(target_id,revision_number,ref_type,ref_id),
            FOREIGN KEY(target_id,revision_number)
              REFERENCES target_revisions(target_id,revision_number) ON DELETE CASCADE)"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_targets_lifecycle_name "
            "ON targets(lifecycle,target_name COLLATE NOCASE)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_target_revision_hash "
            "ON target_revisions(target_id,content_hash)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_target_bindings_target "
            "ON system_target_bindings(target_id,resolved_revision_number)"
        )


def _definition_payload(definition: dict[str, Any]) -> tuple[dict[str, Any], str]:
    canonical = canonical_target_definition(definition)
    legacy_hash = str(canonical.pop("revision", ""))
    digest = content_sha256(canonical)
    if legacy_hash and legacy_hash != digest:
        raise ValueError("Target canonical content hash is inconsistent")
    return canonical, digest


def create_target(
    path: Path,
    definition: dict[str, Any],
    *,
    lifecycle: str = "temporary",
    source: str = "phaseeq_edit",
    target_id: str | None = None,
    operation_id: str | None = None,
) -> TargetRevisionRecord:
    if lifecycle not in VALID_LIFECYCLES:
        raise ValueError(f"Unsupported Target lifecycle: {lifecycle}")
    ensure_target_repository(path)
    canonical, digest = _definition_payload(definition)
    target_id = str(target_id or uuid.uuid4())
    operation_id = str(operation_id or uuid.uuid4())
    name = str(canonical.get("name", "Target")).strip() or "Target"
    with connect_sqlite(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM targets WHERE target_id=?", (target_id,),
        ).fetchone() is not None:
            raise TargetRevisionConflictError("Target ID already exists")
        now = utc_now(connection)
        connection.execute(
            "INSERT INTO targets VALUES(?,?,?,?,?,?)",
            (target_id, name, lifecycle, 1, now, now),
        )
        connection.execute(
            """INSERT INTO target_revisions(
            target_id,revision_number,content_hash,definition_json,source,
            response_asset_id,response_asset_hash,base_revision_number,operation_id,created_at)
            VALUES(?,?,?,?,?,'','',NULL,?,?)""",
            (target_id, 1, digest, canonical_json(canonical), str(source), operation_id, now),
        )
        _record_change(
            connection, entity_type="target", entity_id=target_id,
            revision_number=1, change_kind="target.created", content_hash=digest,
            operation_id=operation_id, created_at=now,
        )
    record = get_target_revision(path, target_id, 1)
    if record is None:
        raise RuntimeError("Saved Target could not be reloaded")
    return record


def save_target_revision(
    path: Path,
    target_id: str,
    definition: dict[str, Any],
    *,
    expected_revision_number: int,
    source: str = "phaseeq_edit",
    operation_id: str | None = None,
    allow_identical: bool = False,
) -> TargetRevisionRecord:
    ensure_target_repository(path)
    canonical, digest = _definition_payload(definition)
    operation_id = str(operation_id or uuid.uuid4())
    with connect_sqlite(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT latest_revision_number,lifecycle FROM targets WHERE target_id=?",
            (str(target_id),),
        ).fetchone()
        if row is None:
            raise TargetNotFoundError("Target does not exist")
        current = int(row[0])
        if current != int(expected_revision_number):
            raise TargetRevisionConflictError(
                f"Target changed from revision {expected_revision_number} to {current}"
            )
        existing = connection.execute(
            "SELECT revision_number FROM target_revisions WHERE target_id=? AND content_hash=? "
            "ORDER BY revision_number DESC LIMIT 1",
            (str(target_id), digest),
        ).fetchone()
        if existing is not None and not allow_identical:
            revision = int(existing[0])
            connection.rollback()
            record = get_target_revision(path, target_id, revision)
            if record is None:
                raise RuntimeError("Existing Target revision could not be reloaded")
            return record
        next_revision = current + 1
        now = utc_now(connection)
        connection.execute(
            """INSERT INTO target_revisions(
            target_id,revision_number,content_hash,definition_json,source,
            response_asset_id,response_asset_hash,base_revision_number,operation_id,created_at)
            VALUES(?,?,?,?,?,'','',?,?,?)""",
            (
                str(target_id), next_revision, digest, canonical_json(canonical),
                str(source), current, operation_id, now,
            ),
        )
        connection.execute(
            "UPDATE targets SET target_name=?,latest_revision_number=?,updated_at=? "
            "WHERE target_id=?",
            (str(canonical.get("name", "Target")), next_revision, now, str(target_id)),
        )
        _record_change(
            connection, entity_type="target", entity_id=str(target_id),
            revision_number=next_revision, change_kind="target.revision_saved",
            content_hash=digest, operation_id=operation_id, created_at=now,
        )
    record = get_target_revision(path, target_id, next_revision)
    if record is None:
        raise RuntimeError("Saved Target revision could not be reloaded")
    return record


def restore_target_as_new_revision(
    path: Path,
    target_id: str,
    revision_number: int,
    *,
    expected_revision_number: int,
    operation_id: str | None = None,
) -> TargetRevisionRecord:
    source = get_target_revision(path, target_id, revision_number)
    if source is None:
        raise TargetNotFoundError("Target revision does not exist")
    return save_target_revision(
        path, target_id, source.definition,
        expected_revision_number=expected_revision_number,
        source="restore", operation_id=operation_id, allow_identical=True,
    )


def archive_target(
    path: Path,
    target_id: str,
    *,
    expected_revision_number: int,
) -> None:
    ensure_target_repository(path)
    with connect_sqlite(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT latest_revision_number FROM targets WHERE target_id=?",
            (str(target_id),),
        ).fetchone()
        if row is None:
            raise TargetNotFoundError("Target does not exist")
        if int(row[0]) != int(expected_revision_number):
            raise TargetRevisionConflictError("Target changed before Archive")
        now = utc_now(connection)
        operation_id = str(uuid.uuid4())
        connection.execute(
            "UPDATE targets SET lifecycle='archived',updated_at=? WHERE target_id=?",
            (now, str(target_id)),
        )
        revision_row = connection.execute(
            "SELECT content_hash FROM target_revisions WHERE target_id=? AND revision_number=?",
            (str(target_id), int(expected_revision_number)),
        ).fetchone()
        content_hash = str(revision_row[0]) if revision_row is not None else ""
        _record_change(
            connection, entity_type="target", entity_id=str(target_id),
            revision_number=int(expected_revision_number),
            change_kind="target.archived", content_hash=content_hash,
            operation_id=operation_id, created_at=now,
        )


def get_target_revision(
    path: Path, target_id: str, revision_number: int | None = None,
) -> TargetRevisionRecord | None:
    ensure_target_repository(path)
    with connect_sqlite(path, read_only=True) as connection:
        if revision_number is None:
            row = connection.execute(
                """SELECT t.target_name,t.lifecycle,r.target_id,r.revision_number,
                r.content_hash,r.definition_json,r.source,r.created_at
                FROM targets t JOIN target_revisions r
                  ON r.target_id=t.target_id AND r.revision_number=t.latest_revision_number
                WHERE t.target_id=?""",
                (str(target_id),),
            ).fetchone()
        else:
            row = connection.execute(
                """SELECT t.target_name,t.lifecycle,r.target_id,r.revision_number,
                r.content_hash,r.definition_json,r.source,r.created_at
                FROM targets t JOIN target_revisions r ON r.target_id=t.target_id
                WHERE t.target_id=? AND r.revision_number=?""",
                (str(target_id), int(revision_number)),
            ).fetchone()
    if row is None:
        return None
    return TargetRevisionRecord(
        ref=TargetRevisionRef(str(row[2]), int(row[3]), str(row[4])),
        name=str(row[0]), lifecycle=str(row[1]), source=str(row[6]),
        definition=json.loads(str(row[5])), created_at=str(row[7]),
    )


def list_target_revisions(
    path: Path, target_id: str,
) -> tuple[TargetRevisionRecord, ...]:
    ensure_target_repository(path)
    with connect_sqlite(path, read_only=True) as connection:
        rows = connection.execute(
            """SELECT revision_number FROM target_revisions
            WHERE target_id=? ORDER BY revision_number DESC""",
            (str(target_id),),
        ).fetchall()
    return tuple(
        record for row in rows
        if (record := get_target_revision(path, target_id, int(row[0]))) is not None
    )


def list_targets(
    path: Path,
    *,
    lifecycles: Iterable[str] | None = None,
    include_archived: bool = False,
) -> tuple[TargetSummary, ...]:
    ensure_target_repository(path)
    selected = {str(item) for item in lifecycles or ()}
    with connect_sqlite(path, read_only=True) as connection:
        rows = connection.execute(
            """SELECT t.target_id,t.target_name,t.lifecycle,t.latest_revision_number,
            r.content_hash,t.updated_at
            FROM targets t JOIN target_revisions r
              ON r.target_id=t.target_id AND r.revision_number=t.latest_revision_number
            ORDER BY t.updated_at DESC,t.target_name COLLATE NOCASE"""
        ).fetchall()
    return tuple(
        TargetSummary(
            target_id=str(row[0]), name=str(row[1]), lifecycle=str(row[2]),
            latest_revision_number=int(row[3]), latest_content_hash=str(row[4]),
            updated_at=str(row[5]),
        )
        for row in rows
        if (include_archived or str(row[2]) != "archived")
        and (not selected or str(row[2]) in selected)
    )


def find_target_revision_by_hash(
    path: Path, content_hash: str,
) -> TargetRevisionRecord | None:
    ensure_target_repository(path)
    with connect_sqlite(path, read_only=True) as connection:
        row = connection.execute(
            """SELECT target_id,revision_number FROM target_revisions
            WHERE content_hash=? ORDER BY created_at DESC LIMIT 1""",
            (str(content_hash),),
        ).fetchone()
    return None if row is None else get_target_revision(path, str(row[0]), int(row[1]))


def save_target_binding(path: Path, binding: TargetBinding) -> ResolvedTargetBinding:
    binding.validate()
    ensure_target_repository(path)
    with connect_sqlite(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if connection.execute(
            "SELECT 1 FROM multiway_systems WHERE id=?", (binding.system_id,),
        ).fetchone() is None:
            raise TargetBindingError("Multiway System does not exist")
        revision_number = (
            int(binding.pinned_revision_number or 0)
            if binding.mode == "pinned"
            else _latest_target_revision_number(connection, binding.target_id)
        )
        row = connection.execute(
            "SELECT content_hash FROM target_revisions WHERE target_id=? AND revision_number=?",
            (binding.target_id, revision_number),
        ).fetchone()
        if row is None:
            raise TargetBindingError("Target revision does not exist")
        now = utc_now(connection)
        operation_id = str(uuid.uuid4())
        connection.execute(
            """INSERT INTO system_target_bindings(
            system_id,scope,group_name,channel_id,way,target_id,binding_mode,
            pinned_revision_number,resolved_revision_number,resolved_content_hash,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(system_id,scope,group_name,channel_id,way) DO UPDATE SET
            target_id=excluded.target_id,binding_mode=excluded.binding_mode,
            pinned_revision_number=excluded.pinned_revision_number,
            resolved_revision_number=excluded.resolved_revision_number,
            resolved_content_hash=excluded.resolved_content_hash,updated_at=excluded.updated_at""",
            (
                binding.system_id, binding.scope, binding.group, binding.channel_id,
                binding.way, binding.target_id, binding.mode,
                binding.pinned_revision_number, revision_number, str(row[0]), now,
            ),
        )
        _record_change(
            connection, entity_type="binding", entity_id=binding.system_id,
            revision_number=revision_number, change_kind="binding.changed",
            content_hash=str(row[0]), operation_id=operation_id, created_at=now,
        )
    resolved = resolve_target_binding(path, binding)
    if resolved is None:
        raise RuntimeError("Saved Target binding could not be resolved")
    return resolved


def effective_target_binding(
    path: Path,
    *,
    system_id: str,
    group: str = "",
    channel_id: str = "",
    way: str = "",
) -> ResolvedTargetBinding | None:
    ensure_target_repository(path)
    candidates = (
        ("channel", group, channel_id, way),
        ("way", group, "", way),
        ("group", group, "", ""),
        ("system", "", "", ""),
    )
    with connect_sqlite(path, read_only=True) as connection:
        for scope, group_name, candidate_channel, candidate_way in candidates:
            if scope == "channel" and not channel_id:
                continue
            if scope == "way" and not way:
                continue
            if scope == "group" and not group:
                continue
            row = connection.execute(
                """SELECT target_id,binding_mode,pinned_revision_number
                FROM system_target_bindings
                WHERE system_id=? AND scope=? AND group_name=? AND channel_id=? AND way=?""",
                (system_id, scope, group_name, candidate_channel, candidate_way),
            ).fetchone()
            if row is not None:
                return resolve_target_binding(path, TargetBinding(
                    system_id=system_id, scope=scope, group=group_name,
                    channel_id=candidate_channel, way=candidate_way,
                    target_id=str(row[0]), mode=str(row[1]),
                    pinned_revision_number=None if row[2] is None else int(row[2]),
                ))
    return None


def resolve_target_binding(
    path: Path, binding: TargetBinding,
) -> ResolvedTargetBinding | None:
    binding.validate()
    latest = get_target_revision(path, binding.target_id)
    if latest is None:
        return None
    revision = (
        int(binding.pinned_revision_number or 0)
        if binding.mode == "pinned"
        else latest.ref.revision_number
    )
    resolved = get_target_revision(path, binding.target_id, revision)
    if resolved is None:
        return None
    return ResolvedTargetBinding(
        binding=binding, resolved=resolved,
        latest_available_revision_number=latest.ref.revision_number,
        requires_recalculation=(
            binding.mode == "latest" and resolved.ref.revision_number != latest.ref.revision_number
        ),
    )


def read_changes(path: Path, after_sequence: int, *, limit: int = 500) -> tuple[dict[str, Any], ...]:
    ensure_target_repository(path)
    with connect_sqlite(path, read_only=True) as connection:
        rows = connection.execute(
            """SELECT change_sequence,entity_type,entity_id,revision_number,
            change_kind,content_hash,operation_id,created_at
            FROM change_journal WHERE change_sequence>? ORDER BY change_sequence LIMIT ?""",
            (max(0, int(after_sequence)), max(1, min(int(limit), 500))),
        ).fetchall()
    return tuple(dict(row) for row in rows)


def _latest_target_revision_number(connection: Any, target_id: str) -> int:
    row = connection.execute(
        "SELECT latest_revision_number FROM targets WHERE target_id=?", (target_id,),
    ).fetchone()
    if row is None:
        raise TargetNotFoundError("Target does not exist")
    return int(row[0])


def _record_change(
    connection: Any,
    *,
    entity_type: str,
    entity_id: str,
    revision_number: int | None,
    change_kind: str,
    content_hash: str,
    operation_id: str,
    created_at: str,
) -> None:
    connection.execute(
        """INSERT INTO change_journal(
        entity_type,entity_id,revision_number,change_kind,content_hash,operation_id,created_at)
        VALUES(?,?,?,?,?,?,?)""",
        (
            entity_type, entity_id, revision_number, change_kind,
            content_hash, operation_id, created_at,
        ),
    )
    connection.execute(
        """INSERT INTO audit_log(
        operation_id,action,entity_type,entity_id,before_revision_number,
        after_revision_number,metadata_json,created_at)
        VALUES(?,?,?,?,NULL,?,?,?)""",
        (
            operation_id, change_kind, entity_type, entity_id,
            revision_number, canonical_json({"content_hash": content_hash}), created_at,
        ),
    )


@dataclass(frozen=True)
class TargetRevisionSummary:
    ref: TargetRevisionRef
    created_at: str


def list_target_revision_summaries(path, target_id):
    ensure_target_repository(path)
    with connect_sqlite(path, read_only=True) as connection:
        rows = connection.execute(
            "SELECT revision_number,content_hash,created_at FROM target_revisions WHERE target_id=? ORDER BY revision_number DESC",
            (str(target_id),)).fetchall()
    return tuple(TargetRevisionSummary(TargetRevisionRef(str(target_id), int(row[0]), str(row[1])), str(row[2])) for row in rows)
