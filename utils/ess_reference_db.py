"""Persistent ESS reference library for microphone measurement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3


ESS_REFERENCE_DB_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EssReferenceRecord:
    id: str
    name: str
    audio_data: bytes
    manifest_data: bytes = b""
    wav_data: bytes = b""
    flac_data: bytes = b""
    origin: str = "Loaded"
    created_at: str = ""
    updated_at: str = ""


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_ess_reference_db(path: Path) -> None:
    with _connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ess_references (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                audio_data BLOB NOT NULL,
                manifest_data BLOB NOT NULL DEFAULT X'',
                wav_data BLOB NOT NULL DEFAULT X'',
                flac_data BLOB NOT NULL DEFAULT X'',
                origin TEXT NOT NULL DEFAULT 'Loaded',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES('schema_version', ?)",
            (str(ESS_REFERENCE_DB_SCHEMA_VERSION),),
        )


def save_ess_reference(path: Path, record: EssReferenceRecord) -> EssReferenceRecord:
    ensure_ess_reference_db(path)
    with _connect(path) as connection:
        now = str(connection.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0])
        existing = connection.execute(
            "SELECT created_at FROM ess_references WHERE id = ?",
            (str(record.id),),
        ).fetchone()
        created_at = str(existing["created_at"]) if existing is not None else (record.created_at or now)
        connection.execute(
            """
            INSERT OR REPLACE INTO ess_references(
                id, name, audio_data, manifest_data, wav_data, flac_data,
                origin, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.id),
                str(record.name),
                sqlite3.Binary(bytes(record.audio_data)),
                sqlite3.Binary(bytes(record.manifest_data)),
                sqlite3.Binary(bytes(record.wav_data)),
                sqlite3.Binary(bytes(record.flac_data)),
                str(record.origin),
                created_at,
                now,
            ),
        )
    return EssReferenceRecord(
        id=str(record.id),
        name=str(record.name),
        audio_data=bytes(record.audio_data),
        manifest_data=bytes(record.manifest_data),
        wav_data=bytes(record.wav_data),
        flac_data=bytes(record.flac_data),
        origin=str(record.origin),
        created_at=created_at,
        updated_at=now,
    )


def list_ess_references(path: Path, *, limit: int = 64, include_standard: bool = True) -> list[EssReferenceRecord]:
    ensure_ess_reference_db(path)
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT id, name, audio_data, manifest_data, wav_data, flac_data,
                   origin, created_at, updated_at
            FROM ess_references
            WHERE (? OR origin != 'Standard PCM reference')
            ORDER BY updated_at DESC, name COLLATE NOCASE
            LIMIT ?
            """,
            (bool(include_standard), max(1, int(limit))),
        ).fetchall()
    return [
        EssReferenceRecord(
            id=str(row["id"]),
            name=str(row["name"]),
            audio_data=bytes(row["audio_data"]),
            manifest_data=bytes(row["manifest_data"]),
            wav_data=bytes(row["wav_data"]),
            flac_data=bytes(row["flac_data"]),
            origin=str(row["origin"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        for row in rows
    ]
