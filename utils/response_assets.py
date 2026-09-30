from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
import sqlite3
import uuid
from typing import Any

import numpy as np

from phase_fir_designer import SpeakerResponse


RESPONSE_ASSET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResponseAssetRef:
    id: str
    kind: str
    content_sha256: str
    created_at: str
    updated_at: str


def response_content_digest(response: SpeakerResponse | None) -> str:
    if response is None:
        return "none"
    digest = hashlib.sha256()
    for name, values in (
        ("frequency", response.frequency),
        ("gain_db", response.gain_db),
        ("phase_deg", response.phase_deg),
    ):
        digest.update(name.encode("utf-8"))
        if values is None:
            digest.update(b"none")
            continue
        array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def ensure_response_asset_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS response_assets (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                payload BLOB NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                schema_version INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_response_assets_digest ON response_assets(content_sha256)"
        )


def save_response_asset(
    path: Path,
    response: SpeakerResponse,
    *,
    kind: str,
    current_ref: dict[str, Any] | ResponseAssetRef | None = None,
) -> ResponseAssetRef:
    ensure_response_asset_db(path)
    content_sha256 = response_content_digest(response)
    current_id, current_digest = _ref_identity(current_ref)
    if current_id and current_digest == content_sha256:
        existing = get_response_asset_ref(path, current_id)
        if existing is not None and existing.content_sha256 == content_sha256:
            return existing

    asset_id = str(uuid.uuid4())
    payload = _serialize_response(response)
    with _connect(path) as conn:
        now = _now_sql(conn)
        conn.execute(
            """
            INSERT INTO response_assets (
                id, kind, content_sha256, payload, created_at, updated_at, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                str(kind),
                content_sha256,
                sqlite3.Binary(payload),
                now,
                now,
                RESPONSE_ASSET_SCHEMA_VERSION,
            ),
        )
    loaded = get_response_asset_ref(path, asset_id)
    if loaded is None:
        raise RuntimeError("Response asset save failed.")
    return loaded


def get_response_asset_ref(path: Path, asset_id: str) -> ResponseAssetRef | None:
    ensure_response_asset_db(path)
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT id, kind, content_sha256, created_at, updated_at FROM response_assets WHERE id = ?",
            (str(asset_id),),
        ).fetchone()
    return _ref_from_row(row) if row is not None else None


def load_response_asset(path: Path, ref: dict[str, Any] | ResponseAssetRef | str) -> SpeakerResponse:
    asset_id = ref.id if isinstance(ref, ResponseAssetRef) else str(ref.get("id", "") if isinstance(ref, dict) else ref)
    expected_digest = (
        ref.content_sha256
        if isinstance(ref, ResponseAssetRef)
        else str(ref.get("content_sha256", "") if isinstance(ref, dict) else "")
    )
    if not asset_id:
        raise ValueError("Response asset reference has no id.")
    ensure_response_asset_db(path)
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT content_sha256, payload FROM response_assets WHERE id = ?",
            (asset_id,),
        ).fetchone()
    if row is None:
        raise FileNotFoundError(f"Response asset not found: {asset_id}")
    stored_digest = str(row["content_sha256"])
    if expected_digest and expected_digest != stored_digest:
        raise ValueError(f"Response asset checksum mismatch: {asset_id}")
    response = _deserialize_response(bytes(row["payload"]))
    if response_content_digest(response) != stored_digest:
        raise ValueError(f"Response asset payload is corrupted: {asset_id}")
    return response


def response_asset_ref_payload(ref: ResponseAssetRef | None) -> dict[str, Any] | None:
    return None if ref is None else asdict(ref)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _now_sql(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()
    return str(row[0])


def _serialize_response(response: SpeakerResponse) -> bytes:
    frequency = np.asarray(response.frequency, dtype=np.float64)
    gain_db = np.asarray(response.gain_db, dtype=np.float64)
    if frequency.shape != gain_db.shape:
        raise ValueError("Response frequency and gain arrays must have the same shape.")
    has_phase = response.phase_deg is not None
    phase_deg = np.asarray(response.phase_deg if has_phase else [], dtype=np.float64)
    if has_phase and phase_deg.shape != frequency.shape:
        raise ValueError("Response phase array must match the frequency shape.")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        frequency=frequency,
        gain_db=gain_db,
        phase_deg=phase_deg,
        has_phase=np.asarray([has_phase], dtype=np.bool_),
    )
    return buffer.getvalue()


def _deserialize_response(payload: bytes) -> SpeakerResponse:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        frequency = np.asarray(data["frequency"], dtype=np.float64)
        gain_db = np.asarray(data["gain_db"], dtype=np.float64)
        has_phase = bool(np.asarray(data["has_phase"], dtype=np.bool_).ravel()[0])
        phase_deg = np.asarray(data["phase_deg"], dtype=np.float64) if has_phase else None
    return SpeakerResponse(
        frequency=frequency.tolist(),
        gain_db=gain_db.tolist(),
        phase_deg=None if phase_deg is None else phase_deg.tolist(),
    )


def _ref_identity(ref: dict[str, Any] | ResponseAssetRef | None) -> tuple[str, str]:
    if isinstance(ref, ResponseAssetRef):
        return ref.id, ref.content_sha256
    if isinstance(ref, dict):
        return str(ref.get("id", "")), str(ref.get("content_sha256", ""))
    return "", ""


def _ref_from_row(row: sqlite3.Row) -> ResponseAssetRef:
    return ResponseAssetRef(
        id=str(row["id"]),
        kind=str(row["kind"]),
        content_sha256=str(row["content_sha256"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
