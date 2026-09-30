from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FIRArtifact:
    content_id: str
    way_id: str
    sample_rate: int
    coefficients: np.ndarray
    design_signature: str
    pinned: bool
    created_at: str


def ensure_fir_artifact_store(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fir_artifacts (
                content_id TEXT PRIMARY KEY,
                way_id TEXT NOT NULL,
                sample_rate INTEGER NOT NULL,
                coefficients_f64le BLOB NOT NULL,
                taps INTEGER NOT NULL,
                design_signature TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_fir_artifact_way ON fir_artifacts(way_id, created_at DESC)"
        )
        connection.execute("""
            CREATE TABLE IF NOT EXISTS fir_artifact_refs (
                way_id TEXT NOT NULL,
                content_id TEXT NOT NULL,
                design_signature TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY(way_id,content_id),
                FOREIGN KEY(content_id) REFERENCES fir_artifacts(content_id)
            )
        """)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_fir_artifact_ref_way ON fir_artifact_refs(way_id, created_at DESC)"
        )


def save_fir_artifact(
    path: Path,
    *,
    way_id: str,
    sample_rate: int,
    coefficients: np.ndarray,
    design_signature: str,
    pinned: bool = False,
) -> FIRArtifact:
    ensure_fir_artifact_store(path)
    values = np.asarray(coefficients, dtype="<f8").ravel()
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("FIR artifact requires finite coefficients")
    digest = hashlib.sha256()
    digest.update(int(sample_rate).to_bytes(8, "little", signed=False))
    digest.update(values.tobytes())
    content_id = digest.hexdigest()
    with sqlite3.connect(path) as connection:
        now = str(connection.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')").fetchone()[0])
        connection.execute(
            """INSERT INTO fir_artifacts(
                content_id,way_id,sample_rate,coefficients_f64le,taps,design_signature,pinned,created_at
            ) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(content_id) DO UPDATE SET
                pinned=MAX(pinned,excluded.pinned), design_signature=excluded.design_signature""",
            (content_id, str(way_id), int(sample_rate), values.tobytes(), values.size,
             str(design_signature), int(bool(pinned)), now),
        )
        connection.execute(
            """INSERT INTO fir_artifact_refs(
                way_id,content_id,design_signature,pinned,created_at
            ) VALUES(?,?,?,?,?) ON CONFLICT(way_id,content_id) DO UPDATE SET
                design_signature=excluded.design_signature,
                pinned=MAX(pinned,excluded.pinned),
                created_at=excluded.created_at""",
            (str(way_id), content_id, str(design_signature), int(bool(pinned)), now),
        )
    artifact = load_fir_artifact(path, content_id)
    assert artifact is not None
    return artifact


def load_fir_artifact(path: Path, content_id: str) -> FIRArtifact | None:
    ensure_fir_artifact_store(path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """SELECT content_id,way_id,sample_rate,coefficients_f64le,taps,
                      design_signature,pinned,created_at
               FROM fir_artifacts WHERE content_id=?""",
            (str(content_id),),
        ).fetchone()
    if row is None:
        return None
    coefficients = np.frombuffer(row[3], dtype="<f8", count=int(row[4])).astype(float, copy=True)
    return FIRArtifact(str(row[0]), str(row[1]), int(row[2]), coefficients, str(row[5]), bool(row[6]), str(row[7]))


def prune_fir_artifacts(path: Path, way_id: str, *, keep_latest: int = 3) -> tuple[str, ...]:
    """Remove unpinned history beyond the newest N artifacts for one Way."""

    ensure_fir_artifact_store(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT content_id,pinned FROM fir_artifact_refs WHERE way_id=? ORDER BY created_at DESC,content_id",
            (str(way_id),),
        ).fetchall()
        removable = [str(content_id) for index, (content_id, pinned) in enumerate(rows) if index >= max(0, int(keep_latest)) and not bool(pinned)]
        connection.executemany(
            "DELETE FROM fir_artifact_refs WHERE way_id=? AND content_id=?",
            [(str(way_id), item) for item in removable],
        )
        connection.execute(
            "DELETE FROM fir_artifacts WHERE content_id NOT IN (SELECT DISTINCT content_id FROM fir_artifact_refs)"
        )
    return tuple(removable)
