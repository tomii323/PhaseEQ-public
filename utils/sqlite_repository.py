"""Shared SQLite primitives for PhaseEQ repositories.

The helpers in this module deliberately know nothing about Projects, DSP
Systems, speakers, or targets.  New repositories can share safe connection,
metadata, canonical JSON, digest, and backup behavior without inheriting a
legacy persistence contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


def connect_sqlite(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        if not path.is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def utc_now(connection: sqlite3.Connection) -> str:
    row = connection.execute(
        "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
    ).fetchone()
    return str(row[0])


def ensure_app_meta(
    connection: sqlite3.Connection,
    *,
    schema_version: int,
    db_kind: str,
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT OR REPLACE INTO app_meta(key, value) VALUES('schema_version', ?)",
        (str(int(schema_version)),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO app_meta(key, value) VALUES('db_kind', ?)",
        (str(db_kind),),
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def backup_sqlite(source: Path, destination: Path) -> Path:
    """Create a transactionally consistent SQLite backup without sidecars."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with connect_sqlite(source, read_only=True) as source_connection:
            with connect_sqlite(temporary) as destination_connection:
                source_connection.backup(destination_connection)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
