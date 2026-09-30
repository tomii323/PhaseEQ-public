"""Local Speaker DB helpers for PhaseEQ.

The app keeps user-managed speaker specifications and measurement history in
separate SQLite files while exposing a shared set of list/search fields to the
UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Literal


SPEAKER_DB_SCHEMA_VERSION = 5

SpeakerDBKind = Literal["user_specs", "measurements"]
SpeakerSortKey = Literal["updated_at", "brand", "model", "driver_type", "fs_hz", "qts", "source_name"]

SPEC_FIELDS = [
    "nominal_diameter_mm",
    "nominal_impedance_ohm",
    "fs_hz",
    "qts",
    "vas_l",
    "re_ohm",
    "le_mh",
    "sd_cm2",
    "xmax_mm",
    "spl_db",
    "pmax_w",
    "overall_diameter_mm",
    "baffle_cutout_mm",
    "mounting_depth_mm",
]


@dataclass(frozen=True)
class SpeakerSpecRecord:
    id: str
    brand: str
    model: str
    driver_type: str = ""
    nominal_diameter_mm: float | None = None
    nominal_impedance_ohm: float | None = None
    fs_hz: float | None = None
    qts: float | None = None
    vas_l: float | None = None
    re_ohm: float | None = None
    le_mh: float | None = None
    sd_cm2: float | None = None
    xmax_mm: float | None = None
    spl_db: float | None = None
    pmax_w: float | None = None
    overall_diameter_mm: float | None = None
    baffle_cutout_mm: float | None = None
    mounting_depth_mm: float | None = None
    source_name: str = ""
    source_url: str = ""
    source_type: str = "manual"
    redistributable: bool = True
    user_verified: bool = False
    note: str = ""
    raw_text: str = ""
    created_at: str = ""
    updated_at: str = ""
    archived: bool = False


@dataclass(frozen=True)
class SpeakerMeasurementRecord:
    id: str
    name: str
    speaker_spec_id: str = ""
    brand: str = ""
    model: str = ""
    measurement_date: str = ""
    distance: str = ""
    angle: str = ""
    microphone: str = ""
    location: str = ""
    measurement_scope: str = "system_channel"
    system_name: str = ""
    channel: str = ""
    position_name: str = ""
    system_state: str = ""
    intended_use: str = "correction_input"
    correction_note: str = ""
    source_name: str = ""
    source_url: str = ""
    source_type: str = "speaker_input_raw"
    response_payload: dict[str, Any] | None = None
    raw_text: str = ""
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    archived: bool = False


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _now_sql(conn: sqlite3.Connection) -> str:
    return str(conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0])


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_speaker_spec_db(path: Path, *, db_kind: SpeakerDBKind = "user_specs") -> None:
    with _connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS speaker_specs (
                id TEXT PRIMARY KEY,
                brand TEXT NOT NULL,
                model TEXT NOT NULL,
                driver_type TEXT NOT NULL DEFAULT '',
                nominal_diameter_mm REAL,
                nominal_impedance_ohm REAL,
                fs_hz REAL,
                qts REAL,
                vas_l REAL,
                re_ohm REAL,
                le_mh REAL,
                sd_cm2 REAL,
                xmax_mm REAL,
                spl_db REAL,
                pmax_w REAL,
                overall_diameter_mm REAL,
                baffle_cutout_mm REAL,
                mounting_depth_mm REAL,
                source_name TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT 'manual',
                redistributable INTEGER NOT NULL DEFAULT 1,
                user_verified INTEGER NOT NULL DEFAULT 0,
                note TEXT NOT NULL DEFAULT '',
                raw_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        _ensure_columns(
            conn,
            "speaker_specs",
            {
                "nominal_diameter_mm": "REAL",
            },
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_speaker_specs_brand_model ON speaker_specs(brand COLLATE NOCASE, model COLLATE NOCASE)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_speaker_specs_archived_updated ON speaker_specs(archived, updated_at)")
        conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES('schema_version', ?)",
            (str(SPEAKER_DB_SCHEMA_VERSION),),
        )
        conn.execute("INSERT OR REPLACE INTO app_meta(key, value) VALUES('db_kind', ?)", (db_kind,))


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def ensure_measurement_db(path: Path) -> None:
    with _connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS speaker_measurements (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                speaker_spec_id TEXT NOT NULL DEFAULT '',
                brand TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                measurement_date TEXT NOT NULL DEFAULT '',
                distance TEXT NOT NULL DEFAULT '',
                angle TEXT NOT NULL DEFAULT '',
                microphone TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                measurement_scope TEXT NOT NULL DEFAULT 'system_channel',
                system_name TEXT NOT NULL DEFAULT '',
                channel TEXT NOT NULL DEFAULT '',
                position_name TEXT NOT NULL DEFAULT '',
                system_state TEXT NOT NULL DEFAULT '',
                intended_use TEXT NOT NULL DEFAULT 'correction_input',
                correction_note TEXT NOT NULL DEFAULT '',
                source_name TEXT NOT NULL DEFAULT '',
                source_url TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT 'speaker_input_raw',
                response_json TEXT NOT NULL,
                raw_text TEXT NOT NULL DEFAULT '',
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        _ensure_columns(
            conn,
            "speaker_measurements",
            {
                "speaker_spec_id": "TEXT NOT NULL DEFAULT ''",
                "source_url": "TEXT NOT NULL DEFAULT ''",
                "raw_text": "TEXT NOT NULL DEFAULT ''",
                "measurement_scope": "TEXT NOT NULL DEFAULT 'system_channel'",
                "system_name": "TEXT NOT NULL DEFAULT ''",
                "channel": "TEXT NOT NULL DEFAULT ''",
                "position_name": "TEXT NOT NULL DEFAULT ''",
                "system_state": "TEXT NOT NULL DEFAULT ''",
                "intended_use": "TEXT NOT NULL DEFAULT 'correction_input'",
            },
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_measurements_brand_model ON speaker_measurements(brand COLLATE NOCASE, model COLLATE NOCASE)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_measurements_speaker_spec_id ON speaker_measurements(speaker_spec_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_measurements_archived_updated ON speaker_measurements(archived, updated_at)")
        conn.execute(
            "INSERT OR REPLACE INTO app_meta(key, value) VALUES('schema_version', ?)",
            (str(SPEAKER_DB_SCHEMA_VERSION),),
        )
        conn.execute("INSERT OR REPLACE INTO app_meta(key, value) VALUES('db_kind', 'measurements')")


def save_speaker_spec(path: Path, *, record: SpeakerSpecRecord, db_kind: SpeakerDBKind = "user_specs") -> SpeakerSpecRecord:
    ensure_speaker_spec_db(path, db_kind=db_kind)
    with _connect(path) as conn:
        clean_id = record.id or _existing_spec_id(conn, record) or str(uuid.uuid4())
        now = _now_sql(conn)
        existing = conn.execute("SELECT created_at FROM speaker_specs WHERE id = ?", (clean_id,)).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        values = asdict(record) | {
            "id": clean_id,
            "brand": record.brand.strip(),
            "model": record.model.strip(),
            "driver_type": record.driver_type.strip(),
            "created_at": created_at,
            "updated_at": now,
            "archived": int(record.archived),
            "redistributable": int(record.redistributable),
            "user_verified": int(record.user_verified),
        }
        conn.execute(
            f"""
            INSERT INTO speaker_specs({', '.join(values.keys())})
            VALUES ({', '.join('?' for _ in values)})
            ON CONFLICT(id) DO UPDATE SET
                brand=excluded.brand,
                model=excluded.model,
                driver_type=excluded.driver_type,
                {', '.join(f'{field}=excluded.{field}' for field in SPEC_FIELDS)},
                source_name=excluded.source_name,
                source_url=excluded.source_url,
                source_type=excluded.source_type,
                redistributable=excluded.redistributable,
                user_verified=excluded.user_verified,
                note=excluded.note,
                raw_text=excluded.raw_text,
                updated_at=excluded.updated_at,
                archived=excluded.archived
            """,
            tuple(values.values()),
        )
    loaded = get_speaker_spec(path, clean_id, db_kind=db_kind)
    if loaded is None:
        raise RuntimeError("Speaker spec save failed.")
    return loaded


def _existing_spec_id(conn: sqlite3.Connection, record: SpeakerSpecRecord) -> str | None:
    """Return an existing User spec ID for ordinary save-time duplicate cleanup.

    Explicit record IDs always bypass this helper at the caller.  This keeps the
    previous duplicate-cleanup behavior for normal saves while still allowing
    intentional duplicate/history records when the UI supplies a fresh ID.
    """
    source_url = record.source_url.strip()
    source_type = record.source_type.strip()
    if source_url:
        row = conn.execute(
            """
            SELECT id FROM speaker_specs
            WHERE source_url = ? COLLATE NOCASE
              AND source_type = ? COLLATE NOCASE
              AND archived = 0
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (source_url, source_type),
        ).fetchone()
        if row:
            return str(row["id"])
    brand = record.brand.strip()
    model = record.model.strip()
    if brand and model:
        row = conn.execute(
            """
            SELECT id FROM speaker_specs
            WHERE brand = ? COLLATE NOCASE
              AND model = ? COLLATE NOCASE
              AND source_type = ? COLLATE NOCASE
              AND archived = 0
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (brand, model, source_type),
        ).fetchone()
        if row:
            return str(row["id"])
    return None


def get_speaker_spec(path: Path, record_id: str, *, db_kind: SpeakerDBKind = "user_specs") -> SpeakerSpecRecord | None:
    ensure_speaker_spec_db(path, db_kind=db_kind)
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM speaker_specs WHERE id = ?", (record_id,)).fetchone()
    return _spec_from_row(row) if row else None


def list_speaker_specs(
    path: Path,
    *,
    query: str = "",
    brand: str = "",
    driver_type: str = "",
    include_archived: bool = False,
    sort_by: SpeakerSortKey = "updated_at",
    descending: bool = True,
    limit: int = 1000,
    offset: int = 0,
    db_kind: SpeakerDBKind = "user_specs",
) -> list[SpeakerSpecRecord]:
    ensure_speaker_spec_db(path, db_kind=db_kind)
    sort_key = sort_by if sort_by in {"updated_at", "brand", "model", "driver_type", "fs_hz", "qts", "source_name"} else "updated_at"
    clauses = []
    params: list[Any] = []
    if not include_archived:
        clauses.append("archived = 0")
    normalized_query = query.strip()
    if normalized_query:
        like = f"%{normalized_query}%"
        clauses.append(
            "(brand LIKE ? COLLATE NOCASE OR model LIKE ? COLLATE NOCASE OR "
            "driver_type LIKE ? COLLATE NOCASE OR source_name LIKE ? COLLATE NOCASE OR "
            "source_url LIKE ? COLLATE NOCASE OR note LIKE ? COLLATE NOCASE)"
        )
        params.extend([like, like, like, like, like, like])
    normalized_brand = brand.strip()
    if normalized_brand:
        clauses.append("brand = ? COLLATE NOCASE")
        params.append(normalized_brand)
    normalized_driver_type = driver_type.strip()
    if normalized_driver_type:
        clauses.append("driver_type = ? COLLATE NOCASE")
        params.append(normalized_driver_type)
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "DESC" if descending else "ASC"
    with _connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT * FROM speaker_specs
            {where_sql}
            ORDER BY {sort_key} {order}, brand COLLATE NOCASE ASC, model COLLATE NOCASE ASC, id ASC
            LIMIT ? OFFSET ?
            """,
            [*params, max(1, int(limit)), max(0, int(offset))],
        ).fetchall()
    return [_spec_from_row(row) for row in rows]


def speaker_spec_facets(path: Path, *, db_kind: SpeakerDBKind = "user_specs") -> dict[str, list[str]]:
    ensure_speaker_spec_db(path, db_kind=db_kind)
    with _connect(path) as conn:
        brand_rows = conn.execute(
            """
            SELECT brand FROM speaker_specs
            WHERE archived = 0 AND brand <> ''
            GROUP BY brand COLLATE NOCASE
            ORDER BY brand COLLATE NOCASE ASC
            """
        ).fetchall()
        type_rows = conn.execute(
            """
            SELECT driver_type FROM speaker_specs
            WHERE archived = 0 AND driver_type <> ''
            GROUP BY driver_type COLLATE NOCASE
            ORDER BY driver_type COLLATE NOCASE ASC
            """
        ).fetchall()
    return {
        "brands": [str(row["brand"]) for row in brand_rows],
        "driver_types": [str(row["driver_type"]) for row in type_rows],
    }


def delete_speaker_spec(path: Path, record_id: str, *, hard: bool = False) -> None:
    ensure_speaker_spec_db(path)
    with _connect(path) as conn:
        if hard:
            conn.execute("DELETE FROM speaker_specs WHERE id = ?", (record_id,))
        else:
            conn.execute("UPDATE speaker_specs SET archived = 1, updated_at = ? WHERE id = ?", (_now_sql(conn), record_id))


def duplicate_speaker_spec(source: SpeakerSpecRecord, *, new_id: str | None = None) -> SpeakerSpecRecord:
    data = asdict(source)
    data.update(
        {
            "id": new_id or str(uuid.uuid4()),
            "model": f"{source.model} copy".strip(),
            "source_type": "manual",
            "redistributable": True,
            "user_verified": False,
            "created_at": "",
            "updated_at": "",
            "archived": False,
        }
    )
    return SpeakerSpecRecord(**data)


def save_measurement(path: Path, *, record: SpeakerMeasurementRecord, original=None) -> SpeakerMeasurementRecord:
    if (record.response_payload or {}).get("_summary"):
        raise ValueError("測定データの概要だけでは保存できません。主データを読み込んでください。")
    ensure_measurement_db(path)
    clean_id = record.id or str(uuid.uuid4())
    if original is not None:
        from phase_fir_designer.measurement.original import MeasurementOriginal
        from utils.measurement_originals import payload_for_original
        from dataclasses import replace
        provenance = (record.response_payload or {}).get('timing_provenance')
        if isinstance(provenance, dict):
            original = replace(original, metadata={**original.metadata, 'timing_provenance': provenance})
        original_data = original.ir_bytes()
        MeasurementOriginal.from_bytes(original_data, original.recipe, original.calibration, original.storage_metadata)
        record = replace(record, response_payload=payload_for_original(original, record.response_payload))
    response_json = json.dumps(record.response_payload or {}, ensure_ascii=False, sort_keys=True)
    with _connect(path) as conn:
        if original is not None:
            conn.execute('''CREATE TABLE IF NOT EXISTS measurement_ir_originals (
                measurement_id TEXT PRIMARY KEY, ir_npz BLOB NOT NULL,
                recipe_json TEXT NOT NULL, calibration_json TEXT NOT NULL, metadata_json TEXT NOT NULL)''')
            conn.execute('''INSERT INTO measurement_ir_originals VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(measurement_id) DO UPDATE SET ir_npz=excluded.ir_npz,
                recipe_json=excluded.recipe_json, calibration_json=excluded.calibration_json,
                metadata_json=excluded.metadata_json''',
                (clean_id, original_data, json.dumps(original.recipe), json.dumps(original.calibration), json.dumps(original.storage_metadata)))
        now = _now_sql(conn)
        existing = conn.execute("SELECT created_at FROM speaker_measurements WHERE id = ?", (clean_id,)).fetchone()
        created_at = str(existing["created_at"]) if existing else now
        conn.execute(
            """
            INSERT INTO speaker_measurements(
                id, name, speaker_spec_id, brand, model, measurement_date, distance, angle, microphone,
                location, measurement_scope, system_name, channel, position_name, system_state, intended_use,
                correction_note, source_name, source_url, source_type, response_json,
                raw_text, note, created_at, updated_at, archived
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                speaker_spec_id=excluded.speaker_spec_id,
                brand=excluded.brand,
                model=excluded.model,
                measurement_date=excluded.measurement_date,
                distance=excluded.distance,
                angle=excluded.angle,
                microphone=excluded.microphone,
                location=excluded.location,
                measurement_scope=excluded.measurement_scope,
                system_name=excluded.system_name,
                channel=excluded.channel,
                position_name=excluded.position_name,
                system_state=excluded.system_state,
                intended_use=excluded.intended_use,
                correction_note=excluded.correction_note,
                source_name=excluded.source_name,
                source_url=excluded.source_url,
                source_type=excluded.source_type,
                response_json=excluded.response_json,
                raw_text=excluded.raw_text,
                note=excluded.note,
                updated_at=excluded.updated_at,
                archived=excluded.archived
            """,
            (
                clean_id,
                record.name.strip() or "Untitled measurement",
                record.speaker_spec_id.strip(),
                record.brand.strip(),
                record.model.strip(),
                record.measurement_date.strip(),
                record.distance.strip(),
                record.angle.strip(),
                record.microphone.strip(),
                record.location.strip(),
                record.measurement_scope.strip() or "system_channel",
                record.system_name.strip(),
                record.channel.strip(),
                record.position_name.strip(),
                record.system_state.strip(),
                record.intended_use.strip() or "correction_input",
                record.correction_note.strip(),
                record.source_name.strip(),
                record.source_url.strip(),
                record.source_type.strip() or "speaker_input_raw",
                response_json,
                record.raw_text,
                record.note.strip(),
                created_at,
                now,
                int(record.archived),
            ),
        )
    loaded = get_measurement(path, clean_id)
    if loaded is None:
        raise RuntimeError("Speaker measurement save failed.")
    return loaded


def get_measurement(path: Path, record_id: str) -> SpeakerMeasurementRecord | None:
    ensure_measurement_db(path)
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM speaker_measurements WHERE id = ?", (record_id,)).fetchone()
    return _measurement_from_row(row) if row else None


def list_measurements(
    path: Path,
    *,
    query: str = "",
    include_archived: bool = False,
    descending: bool = True,
    limit: int = 1000,
    offset: int = 0,
    summaries: bool = False,
) -> list[SpeakerMeasurementRecord]:
    ensure_measurement_db(path)
    clauses = []
    params: list[Any] = []
    if not include_archived:
        clauses.append("archived = 0")
    normalized_query = query.strip()
    if normalized_query:
        like = f"%{normalized_query}%"
        clauses.append(
            "(name LIKE ? COLLATE NOCASE OR brand LIKE ? COLLATE NOCASE OR "
            "model LIKE ? COLLATE NOCASE OR source_name LIKE ? COLLATE NOCASE OR note LIKE ? COLLATE NOCASE)"
        )
        params.extend([like, like, like, like, like])
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    order = "DESC" if descending else "ASC"
    columns = _measurement_summary_columns() if summaries else "*"
    with _connect(path) as conn:
        rows = conn.execute(
            f"""
            SELECT {columns} FROM speaker_measurements
            {where_sql}
            ORDER BY updated_at {order}, brand COLLATE NOCASE ASC, model COLLATE NOCASE ASC, id ASC
            LIMIT ? OFFSET ?
            """,
            [*params, max(1, int(limit)), max(0, int(offset))],
        ).fetchall()
    return [_measurement_from_row(row) for row in rows]


_MULTIWAY_PROJECT_MEMO_PREFIX = "Multiway project: "


MEASUREMENT_DETAIL_FIELDS = (
    'name', 'brand', 'model', 'measurement_date', 'distance', 'angle', 'microphone',
    'location', 'system_name', 'channel', 'position_name', 'system_state',
    'correction_note', 'speaker_spec_id', 'measurement_scope', 'intended_use',
)


def update_measurement_details(path: Path, record_id: str, *, details: dict[str, str], note: str) -> None:
    """Edit descriptive fields without replacing response/calibration or project history."""
    if set(details) - set(MEASUREMENT_DETAIL_FIELDS):
        raise ValueError("編集できない測定項目が含まれています。")
    if 'name' in details and not details['name'].strip():
        raise ValueError("測定名を入力してください。")
    for field, allowed in {
        'measurement_scope': {'system_channel', 'combined_room', 'driver', 'mic_calibration', 'impedance'},
        'intended_use': {'correction_input', 'verification_only', 'calibration', 'archive'},
    }.items():
        if field in details and details[field] not in allowed:
            raise ValueError('測定対象・用途の指定が無効です。')
    details = dict(details)
    if details.get('measurement_scope') == 'combined_room' or details.get('channel') == 'L+R':
        details['intended_use'] = 'verification_only'
    ensure_measurement_db(path)
    with _connect(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT note FROM speaker_measurements WHERE id = ?', (record_id,)).fetchone()
        if row is None:
            raise ValueError('測定データがDBに見つかりません。')
        history = [line for line in (row['note'] or '').splitlines() if line.startswith(_MULTIWAY_PROJECT_MEMO_PREFIX)]
        free_note = '\n'.join(line for line in note.splitlines() if not line.startswith(_MULTIWAY_PROJECT_MEMO_PREFIX))
        merged = '\n'.join(part for part in (free_note, '\n'.join(history)) if part)
        columns = [f'{key} = ?' for key in details]
        conn.execute(f"UPDATE speaker_measurements SET {', '.join([*columns, 'note = ?', 'updated_at = ?'])} WHERE id = ?",
                     [*[str(value).strip() for value in details.values()], merged, _now_sql(conn), record_id])


def measurement_multiway_projects(record: SpeakerMeasurementRecord) -> list[str]:
    """Project-use history stays portable with the existing editable memo."""
    return list(dict.fromkeys(
        line[len(_MULTIWAY_PROJECT_MEMO_PREFIX):].strip()
        for line in record.note.splitlines()
        if line.startswith(_MULTIWAY_PROJECT_MEMO_PREFIX)
        and line[len(_MULTIWAY_PROJECT_MEMO_PREFIX):].strip()
    ))


def record_measurement_multiway_project(path: Path, record_id: str, project_name: str) -> None:
    name = " ".join(str(project_name).split())
    if not name:
        return
    ensure_measurement_db(path)
    memo = _MULTIWAY_PROJECT_MEMO_PREFIX + name
    # Read and append under the same write transaction: never overwrite a stale
    # response, calibration state, or a concurrently edited user memo.
    with _connect(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT note FROM speaker_measurements WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            raise ValueError("測定データがDBに見つかりません。")
        note = str(row["note"] or "")
        if memo not in note.splitlines():
            note = note + ("\n" if note and not note.endswith("\n") else "") + memo
            conn.execute("UPDATE speaker_measurements SET note = ?, updated_at = ? WHERE id = ?",
                         (note, _now_sql(conn), record_id))


def delete_measurement(path: Path, record_id: str, *, hard: bool = False) -> None:
    ensure_measurement_db(path)
    with _connect(path) as conn:
        if hard:
            conn.execute("DELETE FROM speaker_measurements WHERE id = ?", (record_id,))
        else:
            conn.execute(
                "UPDATE speaker_measurements SET archived = 1, updated_at = ? WHERE id = ?",
                (_now_sql(conn), record_id),
            )


def export_speaker_specs_json(path: Path) -> str:
    records = [asdict(record) for record in list_speaker_specs(path, include_archived=True, limit=100_000)]
    return json.dumps({"kind": "speaker_specs", "schema_version": SPEAKER_DB_SCHEMA_VERSION, "records": records}, ensure_ascii=False, indent=2)


def import_speaker_specs_json(path: Path, payload: dict[str, Any]) -> int:
    if payload.get("kind") != "speaker_specs":
        raise ValueError("JSON kind must be speaker_specs")
    count = 0
    for item in payload.get("records", []):
        if isinstance(item, dict):
            data = _spec_payload_defaults(item)
            save_speaker_spec(path, record=SpeakerSpecRecord(**data))
            count += 1
    return count


def export_measurements_json(path: Path) -> str:
    import base64
    from utils.measurement_originals import load_measurement_original
    records = [asdict(record) for record in list_measurements(path, include_archived=True, limit=100_000)]
    originals = {}
    for record in records:
        if (record.get('response_payload') or {}).get('measurement_session', {}).get('original'):
            original = load_measurement_original(path, record['id'])
            originals[record['id']] = {
                'ir_npz_base64': base64.b64encode(original.ir_bytes()).decode('ascii'),
                'recipe': original.recipe, 'calibration': original.calibration, 'metadata': original.storage_metadata,
            }
    return json.dumps({"kind": "speaker_measurements", "schema_version": SPEAKER_DB_SCHEMA_VERSION,
                       "records": records, 'originals': originals}, ensure_ascii=False, indent=2)


def import_measurements_json(path: Path, payload: dict[str, Any]) -> int:
    if payload.get("kind") != "speaker_measurements":
        raise ValueError("JSON kind must be speaker_measurements")
    import base64
    from phase_fir_designer.measurement.original import MeasurementOriginal
    originals = {
        record_id: MeasurementOriginal.from_bytes(base64.b64decode(item['ir_npz_base64'], validate=True),
                                                  item['recipe'], item['calibration'], item['metadata'])
        for record_id, item in payload.get('originals', {}).items()
    }
    for item in payload.get('records', []):
        if isinstance(item, dict) and (item.get('response_payload') or {}).get('measurement_session', {}).get('original'):
            if item.get('id') not in originals:
                raise ValueError('バックアップに測定原本がありません。変更せずに中止しました。')
    count = 0
    for item in payload.get("records", []):
        if isinstance(item, dict):
            data = _measurement_payload_defaults(item)
            save_measurement(path, record=SpeakerMeasurementRecord(**data), original=originals.get(data['id']))
            count += 1
    return count


def _spec_payload_defaults(item: dict[str, Any]) -> dict[str, Any]:
    data = {field.name: item.get(field.name) for field in SpeakerSpecRecord.__dataclass_fields__.values()}
    data["id"] = str(data.get("id") or uuid.uuid4())
    data["brand"] = str(data.get("brand") or "")
    data["model"] = str(data.get("model") or "")
    for field in SPEC_FIELDS:
        data[field] = _optional_float(data.get(field))
    data["redistributable"] = bool(data.get("redistributable", True))
    data["user_verified"] = bool(data.get("user_verified", False))
    data["archived"] = bool(data.get("archived", False))
    return data


def _measurement_payload_defaults(item: dict[str, Any]) -> dict[str, Any]:
    data = {field.name: item.get(field.name) for field in SpeakerMeasurementRecord.__dataclass_fields__.values()}
    data["id"] = str(data.get("id") or uuid.uuid4())
    data["name"] = str(data.get("name") or "Untitled measurement")
    data["speaker_spec_id"] = str(data.get("speaker_spec_id") or "")
    data["measurement_scope"] = str(data.get("measurement_scope") or "system_channel")
    data["system_name"] = str(data.get("system_name") or "")
    data["channel"] = str(data.get("channel") or "")
    data["position_name"] = str(data.get("position_name") or "")
    data["system_state"] = str(data.get("system_state") or "")
    data["intended_use"] = str(data.get("intended_use") or "correction_input")
    payload = data.get("response_payload")
    data["response_payload"] = payload if isinstance(payload, dict) else {}
    data["archived"] = bool(data.get("archived", False))
    return data


def _spec_from_row(row: sqlite3.Row) -> SpeakerSpecRecord:
    return SpeakerSpecRecord(
        id=str(row["id"]),
        brand=str(row["brand"]),
        model=str(row["model"]),
        driver_type=str(row["driver_type"]),
        **{field: _optional_float(row[field]) for field in SPEC_FIELDS},
        source_name=str(row["source_name"]),
        source_url=str(row["source_url"]),
        source_type=str(row["source_type"]),
        redistributable=bool(row["redistributable"]),
        user_verified=bool(row["user_verified"]),
        note=str(row["note"]),
        raw_text=str(row["raw_text"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        archived=bool(row["archived"]),
    )


def _measurement_from_row(row: sqlite3.Row) -> SpeakerMeasurementRecord:
    try:
        response_payload = json.loads(str(row["response_json"]))
    except json.JSONDecodeError:
        response_payload = {}
    return SpeakerMeasurementRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        speaker_spec_id=str(row["speaker_spec_id"]),
        brand=str(row["brand"]),
        model=str(row["model"]),
        measurement_date=str(row["measurement_date"]),
        distance=str(row["distance"]),
        angle=str(row["angle"]),
        microphone=str(row["microphone"]),
        location=str(row["location"]),
        measurement_scope=str(row["measurement_scope"]),
        system_name=str(row["system_name"]),
        channel=str(row["channel"]),
        position_name=str(row["position_name"]),
        system_state=str(row["system_state"]),
        intended_use=str(row["intended_use"]),
        correction_note=str(row["correction_note"]),
        source_name=str(row["source_name"]),
        source_url=str(row["source_url"]),
        source_type=str(row["source_type"]),
        response_payload=response_payload,
        raw_text=str(row["raw_text"]),
        note=str(row["note"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        archived=bool(row["archived"]),
    )


def list_measurement_summaries(path, **kwargs):
    """SQL projection omits response arrays and original text from list rows."""
    return list_measurements(path, **kwargs, summaries=True)


def _measurement_summary_columns():
    from dataclasses import fields
    columns = [field.name for field in fields(SpeakerMeasurementRecord)
               if field.name not in {"response_payload", "raw_text"}]
    document = "CASE WHEN json_valid(response_json) THEN response_json ELSE '{}' END"
    def value(path):
        return f"json_extract({document}, '$.{path}')"
    def kind(path):
        return f"json_type({document}, '$.{path}')"
    bundle = f"{value('kind')} = 'speaker_input_measurement_bundle'"
    freq = f"CASE WHEN {bundle} THEN '$.speaker_input.frequency' ELSE '$.frequency' END"
    phase = f"CASE WHEN {bundle} THEN '$.speaker_input.phase_deg' ELSE '$.phase_deg' END"
    valid = f"({bundle} OR ({kind('frequency')} = 'array' AND {kind('gain_db')} = 'array' AND {kind('impedance_ohm')} IS NULL))"
    calibration = f"""CASE
        WHEN {value('input_calibration_state')} = 'measurement_calibration_unknown' THEN 'measurement_calibration_unknown'
        WHEN {value('external_mic_calibration_applied')} = 1 THEN 'measurement_calibrated'
        WHEN {value('measurement_session.calibration_application.output_finalized')} = 1 THEN 'measurement_calibrated'
        WHEN {value('measurement_session.calibration_application.magnitude_applied')} = 1 THEN 'measurement_calibrated'
        WHEN {value('measurement_session.calibration_application.magnitude_applied')} = 0 THEN 'raw'
        WHEN trim(coalesce({value('measurement_session.mic_calibration_id')}, '')) != '' THEN 'measurement_calibration_unknown'
        ELSE 'raw' END"""
    summary = f"""json_object('_summary', json_object(
        'rows', CASE WHEN {valid} THEN json_array_length({document}, {freq}) ELSE NULL END,
        'phase', CASE WHEN {valid} THEN json_type({document}, {phase}) = 'array' ELSE 0 END,
        'nf_woofer', ({bundle} AND {kind('near_field_woofer')} = 'object'),
        'nf_port', ({bundle} AND {kind('near_field_port')} = 'object'),
        'calibration', {calibration}),
        'timing_provenance', coalesce({value('timing_provenance')}, {value('measurement_session.timing_provenance')})) AS response_json"""
    return ', '.join([*columns, "'' AS raw_text", summary])
