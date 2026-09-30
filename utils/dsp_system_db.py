from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Any

from phase_fir_designer import DesignConfig, IIRFilter, LinearFIRFilter, SpeakerResponse
from phase_fir_designer.dsp_system import (
    DSPDevice,
    DSPInput,
    DSPInputBinding,
    DSPOutputChannel,
    DSPRoute,
    DSPSystem,
    DSPSystemProcessing,
    DSPSystemTarget,
    DSPWay,
    CrossoverBoundary,
    CrossoverFilterSpec,
    CrossoverPlan,
    SubCrossover,
    DistortionProfile,
    ensure_system_crossover_plan,
    ensure_output_channel_assignments,
)

DSP_SYSTEM_SCHEMA_VERSION = 13


def _field_names(model: type[Any]) -> set[str]:
    return {item.name for item in fields(model)}


def _require_exact_fields(label: str, payload: dict[str, Any], model: type[Any]) -> None:
    expected = _field_names(model)
    missing = sorted(expected - set(payload))
    unexpected = sorted(set(payload) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("retired/unknown " + ", ".join(unexpected))
        raise ValueError(
            f"{label} schema {DSP_SYSTEM_SCHEMA_VERSION} is invalid: {'; '.join(details)}"
        )


@dataclass(frozen=True)
class DSPSystemRecord:
    id: str
    name: str
    system: DSPSystem
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class DSPSystemVariantRecord:
    system_id: str
    scope: str
    slot: int
    name: str
    payload: dict[str, Any]
    content_signature: str
    created_at: str
    updated_at: str
    working_payload: dict[str, Any] | None = None
    working_signature: str = ""


def ensure_dsp_system_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS dsp_systems (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS dsp_system_variants (
            system_id TEXT NOT NULL, scope TEXT NOT NULL, slot INTEGER NOT NULL,
            name TEXT NOT NULL, payload_json TEXT NOT NULL,
            content_signature TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY(system_id, scope, slot))"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_dsp_system_variants_scope "
            "ON dsp_system_variants(system_id, scope, slot)"
        )
        variant_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(dsp_system_variants)")
        }
        if "working_payload_json" not in variant_columns:
            connection.execute(
                "ALTER TABLE dsp_system_variants ADD COLUMN working_payload_json TEXT"
            )
        if "working_signature" not in variant_columns:
            connection.execute(
                "ALTER TABLE dsp_system_variants ADD COLUMN working_signature TEXT NOT NULL DEFAULT ''"
            )
        connection.execute(
            """UPDATE dsp_system_variants
            SET working_payload_json=payload_json,
                working_signature=content_signature
            WHERE working_payload_json IS NULL OR working_signature=''"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS dsp_system_variant_active (
            system_id TEXT NOT NULL, scope TEXT NOT NULL, slot INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(system_id, scope))"""
        )


def save_dsp_system_variant(
    path: Path,
    *,
    system_id: str,
    scope: str,
    slot: int,
    name: str,
    payload: dict[str, Any],
) -> DSPSystemVariantRecord:
    ensure_dsp_system_db(path)
    if not 1 <= int(slot) <= 8:
        raise ValueError("Variant slot must be between 1 and 8")
    clean_scope = str(scope).strip()
    if not clean_scope:
        raise ValueError("Variant scope must not be empty")
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    signature = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT created_at FROM dsp_system_variants WHERE system_id=? AND scope=? AND slot=?",
            (str(system_id), clean_scope, int(slot)),
        ).fetchone()
        now = str(connection.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')").fetchone()[0])
        created_at = str(row[0]) if row else now
        connection.execute(
            """INSERT INTO dsp_system_variants(
            system_id,scope,slot,name,payload_json,content_signature,created_at,updated_at,
            working_payload_json,working_signature
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(system_id,scope,slot) DO UPDATE SET
              name=excluded.name,payload_json=excluded.payload_json,
              content_signature=excluded.content_signature,
              working_payload_json=excluded.working_payload_json,
              working_signature=excluded.working_signature,
              updated_at=excluded.updated_at""",
            (
                str(system_id), clean_scope, int(slot),
                str(name).strip() or f"Variant {int(slot)}",
                payload_json, signature, created_at, now, payload_json, signature,
            ),
        )
    return DSPSystemVariantRecord(
        str(system_id), clean_scope, int(slot),
        str(name).strip() or f"Variant {int(slot)}",
        dict(payload), signature, created_at, now, dict(payload), signature,
    )


def save_dsp_system_variant_working(
    path: Path,
    *,
    system_id: str,
    scope: str,
    slot: int,
    payload: dict[str, Any],
) -> DSPSystemVariantRecord:
    """Update only the recovery Working Copy of one saved Variant."""

    ensure_dsp_system_db(path)
    payload_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    signature = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    with sqlite3.connect(path) as connection:
        current = connection.execute(
            """SELECT working_signature FROM dsp_system_variants
            WHERE system_id=? AND scope=? AND slot=?""",
            (str(system_id), str(scope), int(slot)),
        ).fetchone()
        if current is None:
            raise ValueError("Variant slot is not saved")
        if str(current[0]) == signature:
            records = list_dsp_system_variants(
                path, system_id=system_id, scope=scope,
            )
            return next(item for item in records if item.slot == int(slot))
        now = str(connection.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')"
        ).fetchone()[0])
        cursor = connection.execute(
            """UPDATE dsp_system_variants
            SET working_payload_json=?,working_signature=?,updated_at=?
            WHERE system_id=? AND scope=? AND slot=?""",
            (payload_json, signature, now, str(system_id), str(scope), int(slot)),
        )
        if not cursor.rowcount:
            raise ValueError("Variant Working Copy update failed")
    records = list_dsp_system_variants(path, system_id=system_id, scope=scope)
    return next(item for item in records if item.slot == int(slot))


def set_active_dsp_system_variant(
    path: Path,
    *,
    system_id: str,
    scope: str,
    slot: int | None,
) -> None:
    """Persist the Active Slot independently from Streamlit Session State."""

    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        if slot is None or int(slot) == 0:
            connection.execute(
                "DELETE FROM dsp_system_variant_active WHERE system_id=? AND scope=?",
                (str(system_id), str(scope)),
            )
            return
        exists = connection.execute(
            "SELECT 1 FROM dsp_system_variants WHERE system_id=? AND scope=? AND slot=?",
            (str(system_id), str(scope), int(slot)),
        ).fetchone()
        if exists is None:
            raise ValueError("Active Variant slot is not saved")
        now = str(connection.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')"
        ).fetchone()[0])
        connection.execute(
            """INSERT INTO dsp_system_variant_active(system_id,scope,slot,updated_at)
            VALUES(?,?,?,?) ON CONFLICT(system_id,scope) DO UPDATE SET
            slot=excluded.slot,updated_at=excluded.updated_at""",
            (str(system_id), str(scope), int(slot), now),
        )


def get_active_dsp_system_variant(
    path: Path,
    *,
    system_id: str,
    scope: str,
) -> int:
    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT slot FROM dsp_system_variant_active WHERE system_id=? AND scope=?",
            (str(system_id), str(scope)),
        ).fetchone()
    return 0 if row is None else int(row[0])


def list_dsp_system_variants(
    path: Path,
    *,
    system_id: str,
    scope: str,
) -> list[DSPSystemVariantRecord]:
    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """SELECT system_id,scope,slot,name,payload_json,content_signature,created_at,updated_at,
            working_payload_json,working_signature
            FROM dsp_system_variants WHERE system_id=? AND scope=? ORDER BY slot""",
            (str(system_id), str(scope)),
        ).fetchall()
    return [
        DSPSystemVariantRecord(
            str(row[0]), str(row[1]), int(row[2]), str(row[3]),
            json.loads(str(row[4])), str(row[5]), str(row[6]), str(row[7]),
            json.loads(str(row[8] or row[4])), str(row[9] or row[5]),
        )
        for row in rows
    ]


def delete_dsp_system_variant(
    path: Path,
    *,
    system_id: str,
    scope: str,
    slot: int,
) -> bool:
    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            "DELETE FROM dsp_system_variants WHERE system_id=? AND scope=? AND slot=?",
            (str(system_id), str(scope), int(slot)),
        )
        connection.execute(
            "DELETE FROM dsp_system_variant_active "
            "WHERE system_id=? AND scope=? AND slot=?",
            (str(system_id), str(scope), int(slot)),
        )
    return bool(cursor.rowcount)


def system_to_payload(system: DSPSystem) -> dict[str, Any]:
    normalized = ensure_system_crossover_plan(
        ensure_output_channel_assignments(system)
    )
    return {"schema_version": DSP_SYSTEM_SCHEMA_VERSION, "system": asdict(normalized)}


def system_from_payload(payload: dict[str, Any]) -> DSPSystem:
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != DSP_SYSTEM_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported DSP System schema: {schema_version}. "
            f"Schema {DSP_SYSTEM_SCHEMA_VERSION} is required."
        )
    data = payload.get("system")
    if not isinstance(data, dict):
        raise ValueError("DSP System payload is missing the system object")
    _require_exact_fields("DSP System", data, DSPSystem)
    for item in data.get("devices", []):
        _require_exact_fields("DSP Device", item, DSPDevice)
    for item in data.get("ways", []):
        _require_exact_fields("DSP Way", item, DSPWay)
    for item in data.get("output_channels", []):
        _require_exact_fields("DSP Output", item, DSPOutputChannel)
    for item in data.get("inputs", []):
        _require_exact_fields("DSP Input", item, DSPInput)
    for item in data.get("input_bindings", []):
        _require_exact_fields("DSP Input binding", item, DSPInputBinding)
    for item in data.get("routes", []):
        _require_exact_fields("DSP Route", item, DSPRoute)
    _require_exact_fields("DSP processing", data.get("processing", {}), DSPSystemProcessing)
    devices = tuple(
        DSPDevice(**item)
        for item in data.get("devices", [])
    )
    ways = tuple(
        DSPWay(
            **{
                **item,
                "speaker_package_id": str(item.get("speaker_package_id", "")),
                "linear_fir_filters": tuple(
                    LinearFIRFilter(**filter_payload)
                    for filter_payload in item.get("linear_fir_filters", [])
                ),
                "alignment_iir_filters": tuple(
                    IIRFilter(**filter_payload)
                    for filter_payload in item.get("alignment_iir_filters", [])
                ),
                "manual_iir_filters": tuple(
                    IIRFilter(**filter_payload)
                    for filter_payload in item.get("manual_iir_filters", [])
                ),
                "auto_iir_filters": tuple(
                    IIRFilter(**filter_payload)
                    for filter_payload in item.get("auto_iir_filters", [])
                ),
                "alignment_fir": tuple(float(value) for value in item.get("alignment_fir", [])),
                "distortion_profile": (
                    DistortionProfile(**item["distortion_profile"])
                    if isinstance(item.get("distortion_profile"), dict)
                    else None
                ),
                "correction_config": (
                    DesignConfig.from_dict(item["correction_config"])
                    if isinstance(item.get("correction_config"), dict)
                    else None
                ),
                "output_target_override": (
                    DSPSystemTarget(
                        **{
                            **item["output_target_override"],
                            "response": (
                                SpeakerResponse(**item["output_target_override"]["response"])
                                if isinstance(item["output_target_override"].get("response"), dict)
                                else None
                            ),
                        }
                    )
                    if isinstance(item.get("output_target_override"), dict)
                    else None
                ),
            }
        )
        for item in data.get("ways", [])
    )
    output_channels = tuple(
        DSPOutputChannel(
            **{
                **item,
                "speaker_package_id": str(item.get("speaker_package_id", "")),
                "manual_iir_filters": tuple(
                    IIRFilter(**filter_payload)
                    for filter_payload in item.get("manual_iir_filters", [])
                ),
                "auto_iir_filters": tuple(
                    IIRFilter(**filter_payload)
                    for filter_payload in item.get("auto_iir_filters", [])
                ),
                "distortion_profile": (
                    DistortionProfile(**item["distortion_profile"])
                    if isinstance(item.get("distortion_profile"), dict)
                    else None
                ),
                "correction_config": (
                    DesignConfig.from_dict(item["correction_config"])
                    if isinstance(item.get("correction_config"), dict)
                    else None
                ),
                "target_override": (
                    DSPSystemTarget(
                        **{
                            **item["target_override"],
                            "response": (
                                SpeakerResponse(**item["target_override"]["response"])
                                if isinstance(item["target_override"].get("response"), dict)
                                else None
                            ),
                        }
                    )
                    if isinstance(item.get("target_override"), dict)
                    else None
                ),
            }
        )
        for item in data.get("output_channels", [])
    )
    targets: dict[str, DSPSystemTarget] = {}
    for group, item in data.get("targets", {}).items():
        response_payload = item.get("response")
        response = SpeakerResponse(**response_payload) if isinstance(response_payload, dict) else None
        targets[str(group)] = DSPSystemTarget(**{**item, "response": response})
    crossover_payload = data.get("crossover_plan")
    crossover_plan = None
    if isinstance(crossover_payload, dict):
        boundaries = tuple(
            CrossoverBoundary(
                **{
                    **item,
                    "lower_low_pass": CrossoverFilterSpec(**item.get("lower_low_pass", {})),
                    "upper_high_pass": CrossoverFilterSpec(**item.get("upper_high_pass", {})),
                }
            )
            for item in crossover_payload.get("boundaries", [])
        )
        sub_crossovers = tuple(
            SubCrossover(
                **{
                    **item,
                    "low_pass": CrossoverFilterSpec(**item.get("low_pass", {})),
                }
            )
            for item in crossover_payload.get("sub_crossovers", [])
        )
        crossover_plan = CrossoverPlan(
            main_way_ids=tuple(crossover_payload.get("main_way_ids", [])),
            boundaries=boundaries,
            sub_crossovers=sub_crossovers,
            revision=max(1, int(crossover_payload.get("revision", 1))),
            overlap_convention=crossover_payload.get("overlap_convention", "multiway_fir_studio"),
            main_way_groups=tuple(
                tuple(str(way_id) for way_id in group)
                for group in crossover_payload.get("main_way_groups", [])
            ),
        )
    bindings = tuple(
        DSPInputBinding(**item) for item in data.get("input_bindings", [])
    )
    system = DSPSystem(
        id=str(data.get("id") or uuid.uuid4()),
        name=str(data.get("name") or "DSP System"),
        devices=devices,
        ways=ways,
        inputs=tuple(
            DSPInput(**{
                **item,
                "iir_filters": tuple(
                    IIRFilter(**filter_payload)
                    for filter_payload in item.get("iir_filters", [])
                ),
            })
            for item in data.get("inputs", [])
        ),
        input_bindings=bindings,
        routes=tuple(DSPRoute(**item) for item in data.get("routes", [])),
        processing=DSPSystemProcessing(**data.get("processing", {})),
        targets=targets,
        project_revisions={str(key): str(value) for key, value in data.get("project_revisions", {}).items()},
        crossover_plan=crossover_plan,
        revision=max(1, int(data.get("revision", 1))),
        last_design_signature=str(data.get("last_design_signature", "")),
        way_design_signatures={
            str(key): str(value)
            for key, value in data.get("way_design_signatures", {}).items()
        },
        last_alignment_signature=str(data.get("last_alignment_signature", "")),
        last_export_signature=str(data.get("last_export_signature", "")),
        output_channels=output_channels,
        applied_design_signature=str(data.get("applied_design_signature", "")),
        live_result_signature=str(data.get("live_result_signature", "")),
        saved_design_signature=str(data.get("saved_design_signature", "")),
    )
    system.validate()
    return system


def save_dsp_system(path: Path, system: DSPSystem) -> DSPSystemRecord:
    ensure_dsp_system_db(path)
    system.validate()
    payload = json.dumps(system_to_payload(system), ensure_ascii=False, sort_keys=True)
    with sqlite3.connect(path) as connection:
        now = str(connection.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now')").fetchone()[0])
        existing = connection.execute("SELECT created_at FROM dsp_systems WHERE id=?", (system.id,)).fetchone()
        created = str(existing[0]) if existing else now
        connection.execute(
            """INSERT INTO dsp_systems(id,name,payload_json,created_at,updated_at)
            VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,payload_json=excluded.payload_json,updated_at=excluded.updated_at""",
            (system.id, system.name, payload, created, now),
        )
    return get_dsp_system(path, system.id)  # type: ignore[return-value]


def get_dsp_system(path: Path, system_id: str) -> DSPSystemRecord | None:
    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT id,name,payload_json,created_at,updated_at FROM dsp_systems WHERE id=?", (system_id,)
        ).fetchone()
    if row is None:
        return None
    return DSPSystemRecord(str(row[0]), str(row[1]), system_from_payload(json.loads(row[2])), str(row[3]), str(row[4]))


def list_dsp_systems(path: Path) -> list[DSPSystemRecord]:
    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT id,name,payload_json,created_at,updated_at FROM dsp_systems ORDER BY updated_at DESC,name"
        ).fetchall()
    return [DSPSystemRecord(str(row[0]), str(row[1]), system_from_payload(json.loads(row[2])), str(row[3]), str(row[4])) for row in rows]


def delete_dsp_system(path: Path, system_id: str) -> None:
    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM dsp_system_variants WHERE system_id=?", (system_id,))
        connection.execute("DELETE FROM dsp_system_variant_active WHERE system_id=?", (system_id,))
        connection.execute("DELETE FROM dsp_systems WHERE id=?", (system_id,))


def duplicate_dsp_system(path: Path, system: DSPSystem, name: str) -> DSPSystemRecord:
    copied = DSPSystem(**{
        **system.__dict__,
        "id": str(uuid.uuid4()),
        "name": name,
        "revision": 1,
        "last_design_signature": "",
        "last_alignment_signature": "",
        "last_export_signature": "",
        "way_design_signatures": {},
        "applied_design_signature": "",
        "live_result_signature": "",
        "saved_design_signature": "",
    })
    saved = save_dsp_system(path, copied)
    ensure_dsp_system_db(path)
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """SELECT scope,slot,name,payload_json,content_signature,created_at,updated_at,
            working_payload_json,working_signature
            FROM dsp_system_variants WHERE system_id=?""",
            (system.id,),
        ).fetchall()
        connection.executemany(
            """INSERT INTO dsp_system_variants(
            system_id,scope,slot,name,payload_json,content_signature,created_at,updated_at,
            working_payload_json,working_signature
            ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
            [(copied.id, *row) for row in rows],
        )
        active_rows = connection.execute(
            "SELECT scope,slot,updated_at FROM dsp_system_variant_active WHERE system_id=?",
            (system.id,),
        ).fetchall()
        connection.executemany(
            "INSERT INTO dsp_system_variant_active(system_id,scope,slot,updated_at) VALUES(?,?,?,?)",
            [(copied.id, *row) for row in active_rows],
        )
    return saved
