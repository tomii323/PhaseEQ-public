from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import uuid


@dataclass(frozen=True)
class MicrophoneProfile:
    id: str
    manufacturer: str
    model: str
    device_name_patterns: tuple[str, ...]
    sample_rates_hz: tuple[int, ...]
    preferred_sample_rate_hz: int
    nominal_bit_depth: int
    input_channels: int = 2
    connection_type: str = "usb"
    serial_pattern: str = ""
    serial_format_hint: str = ""
    calibration_angles_deg: tuple[int, ...] = (0,)
    phase_calibration_sample_rates_hz: tuple[int, ...] = ()
    frequency_min_hz: float = 0.0
    frequency_max_hz: float = 0.0
    calibrated_accuracy_db: float = 0.0
    max_spl_db: float = 0.0
    representative_sensitivity_dbfs_94db: float = 0.0
    representative_sensitivity_gain_db: float = 0.0
    noise_level_dbfs_a: float = 0.0
    equivalent_input_noise_db_spl: float = 0.0
    adc_dynamic_range_db: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class MicrophoneUnit:
    id: str
    profile_id: str
    serial_number: str
    label: str = ""
    device_name: str = ""
    calibration_measurement_id: str = ""
    calibration_30_measurement_id: str = ""
    calibration_90_measurement_id: str = ""
    phase_calibration_measurement_id: str = ""
    input_channel: int = 1
    note: str = ""
    created_at: str = ""
    updated_at: str = ""


BUILTIN_PROFILES = (
    MicrophoneProfile(
        id="omnimic",
        manufacturer="Liberty Instruments",
        model="OmniMic",
        device_name_patterns=("omnimic",),
        sample_rates_hz=(48_000,),
        preferred_sample_rate_hz=48_000,
        nominal_bit_depth=16,
        serial_pattern=r"^\d{7}$",
        serial_format_hint="7 digits",
        calibration_angles_deg=(0,),
        phase_calibration_sample_rates_hz=(48_000,),
        note="USB descriptor does not expose a serial number on the tested unit.",
    ),
    MicrophoneProfile(
        id="umik-1",
        manufacturer="miniDSP",
        model="UMIK-1",
        device_name_patterns=("umik-1", "umik 1", "umik1"),
        sample_rates_hz=(48_000,),
        preferred_sample_rate_hz=48_000,
        nominal_bit_depth=24,
        serial_pattern=r"^\d{3}-\d{4}$",
        serial_format_hint="3 digits-4 digits",
        calibration_angles_deg=(0, 90),
        frequency_min_hz=20.0,
        frequency_max_hz=20_000.0,
        calibrated_accuracy_db=1.0,
        max_spl_db=133.0,
        note="Use the product/calibration serial, not the generic USB serial.",
    ),
    MicrophoneProfile(
        id="umik-2",
        manufacturer="miniDSP",
        model="UMIK-2",
        device_name_patterns=("umik-2", "umik 2", "umik2"),
        sample_rates_hz=(48_000, 96_000, 192_000),
        preferred_sample_rate_hz=192_000,
        nominal_bit_depth=32,
        serial_pattern=r"^\d{3}-\d{4}$",
        serial_format_hint="3 digits-4 digits",
        calibration_angles_deg=(0, 90),
        max_spl_db=125.0,
        representative_sensitivity_dbfs_94db=-31.9,
        representative_sensitivity_gain_db=0.0,
        noise_level_dbfs_a=-105.3,
        equivalent_input_noise_db_spl=20.0,
        adc_dynamic_range_db=120.0,
        note="Use the product/calibration serial, not the generic USB serial.",
    ),
    MicrophoneProfile(
        id="emm-6",
        manufacturer="Dayton Audio",
        model="EMM-6",
        device_name_patterns=(),
        sample_rates_hz=(),
        preferred_sample_rate_hz=48_000,
        nominal_bit_depth=0,
        input_channels=1,
        connection_type="analog",
        serial_pattern=r"^\d{4,6}$",
        serial_format_hint="4-6 digits (5 digits on current units)",
        calibration_angles_deg=(0,),
        note="Sample rate and bit depth depend on the connected audio interface.",
    ),
    MicrophoneProfile(
        id="umm-6",
        manufacturer="Dayton Audio",
        model="UMM-6",
        device_name_patterns=("umm-6", "umm 6", "umm6"),
        sample_rates_hz=(48_000,),
        preferred_sample_rate_hz=48_000,
        nominal_bit_depth=16,
        input_channels=1,
        serial_pattern=r"^\d{4,6}$",
        serial_format_hint="4-6 digits (5 digits on current units)",
        calibration_angles_deg=(0,),
        note="Use the individual calibration file from the Dayton Audio calibration tool.",
    ),
    MicrophoneProfile(
        id="xref-20",
        manufacturer="Sonarworks",
        model="XREF 20",
        device_name_patterns=(),
        sample_rates_hz=(),
        preferred_sample_rate_hz=48_000,
        nominal_bit_depth=0,
        input_channels=1,
        connection_type="analog",
        serial_pattern=r"^[A-Za-z0-9]{6}$",
        serial_format_hint="6 alphanumeric characters",
        calibration_angles_deg=(0, 30, 90),
        note="Sample rate and bit depth depend on the connected audio interface.",
    ),
)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_microphone_db(path: Path) -> None:
    with _connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS microphone_profiles (
                id TEXT PRIMARY KEY,
                manufacturer TEXT NOT NULL,
                model TEXT NOT NULL,
                device_name_patterns_json TEXT NOT NULL,
                sample_rates_json TEXT NOT NULL,
                preferred_sample_rate_hz INTEGER NOT NULL,
                nominal_bit_depth INTEGER NOT NULL,
                input_channels INTEGER NOT NULL,
                connection_type TEXT NOT NULL DEFAULT 'usb',
                serial_pattern TEXT NOT NULL DEFAULT '',
                serial_format_hint TEXT NOT NULL DEFAULT '',
                calibration_angles_json TEXT NOT NULL DEFAULT '[0]',
                phase_calibration_sample_rates_json TEXT NOT NULL DEFAULT '[]',
                frequency_min_hz REAL NOT NULL DEFAULT 0,
                frequency_max_hz REAL NOT NULL DEFAULT 0,
                calibrated_accuracy_db REAL NOT NULL DEFAULT 0,
                max_spl_db REAL NOT NULL DEFAULT 0,
                representative_sensitivity_dbfs_94db REAL NOT NULL DEFAULT 0,
                representative_sensitivity_gain_db REAL NOT NULL DEFAULT 0,
                noise_level_dbfs_a REAL NOT NULL DEFAULT 0,
                equivalent_input_noise_db_spl REAL NOT NULL DEFAULT 0,
                adc_dynamic_range_db REAL NOT NULL DEFAULT 0,
                note TEXT NOT NULL DEFAULT ''
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS microphone_units (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                serial_number TEXT NOT NULL,
                label TEXT NOT NULL DEFAULT '',
                device_name TEXT NOT NULL DEFAULT '',
                calibration_measurement_id TEXT NOT NULL DEFAULT '',
                calibration_30_measurement_id TEXT NOT NULL DEFAULT '',
                calibration_90_measurement_id TEXT NOT NULL DEFAULT '',
                phase_calibration_measurement_id TEXT NOT NULL DEFAULT '',
                input_channel INTEGER NOT NULL DEFAULT 1,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(profile_id, serial_number),
                FOREIGN KEY(profile_id) REFERENCES microphone_profiles(id)
            )
            """
        )
        for table, column, definition in (
            ("microphone_profiles", "connection_type", "TEXT NOT NULL DEFAULT 'usb'"),
            ("microphone_profiles", "serial_pattern", "TEXT NOT NULL DEFAULT ''"),
            ("microphone_profiles", "serial_format_hint", "TEXT NOT NULL DEFAULT ''"),
            ("microphone_profiles", "calibration_angles_json", "TEXT NOT NULL DEFAULT '[0]'"),
            ("microphone_profiles", "phase_calibration_sample_rates_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("microphone_profiles", "frequency_min_hz", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "frequency_max_hz", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "calibrated_accuracy_db", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "max_spl_db", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "representative_sensitivity_dbfs_94db", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "representative_sensitivity_gain_db", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "noise_level_dbfs_a", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "equivalent_input_noise_db_spl", "REAL NOT NULL DEFAULT 0"),
            ("microphone_profiles", "adc_dynamic_range_db", "REAL NOT NULL DEFAULT 0"),
            ("microphone_units", "calibration_90_measurement_id", "TEXT NOT NULL DEFAULT ''"),
            ("microphone_units", "calibration_30_measurement_id", "TEXT NOT NULL DEFAULT ''"),
            ("microphone_units", "phase_calibration_measurement_id", "TEXT NOT NULL DEFAULT ''"),
            ("microphone_units", "input_channel", "INTEGER NOT NULL DEFAULT 1"),
        ):
            if column not in {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        for profile in BUILTIN_PROFILES:
            connection.execute(
                """
                INSERT INTO microphone_profiles(
                    id, manufacturer, model, device_name_patterns_json, sample_rates_json,
                    preferred_sample_rate_hz, nominal_bit_depth, input_channels,
                    connection_type, serial_pattern, serial_format_hint,
                    calibration_angles_json, phase_calibration_sample_rates_json,
                    frequency_min_hz, frequency_max_hz, calibrated_accuracy_db, max_spl_db,
                    representative_sensitivity_dbfs_94db, representative_sensitivity_gain_db,
                    noise_level_dbfs_a, equivalent_input_noise_db_spl, adc_dynamic_range_db, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    manufacturer=excluded.manufacturer,
                    model=excluded.model,
                    device_name_patterns_json=excluded.device_name_patterns_json,
                    sample_rates_json=excluded.sample_rates_json,
                    preferred_sample_rate_hz=excluded.preferred_sample_rate_hz,
                    nominal_bit_depth=excluded.nominal_bit_depth,
                    input_channels=excluded.input_channels,
                    connection_type=excluded.connection_type,
                    serial_pattern=excluded.serial_pattern,
                    serial_format_hint=excluded.serial_format_hint,
                    calibration_angles_json=excluded.calibration_angles_json,
                    phase_calibration_sample_rates_json=excluded.phase_calibration_sample_rates_json,
                    frequency_min_hz=excluded.frequency_min_hz,
                    frequency_max_hz=excluded.frequency_max_hz,
                    calibrated_accuracy_db=excluded.calibrated_accuracy_db,
                    max_spl_db=excluded.max_spl_db,
                    representative_sensitivity_dbfs_94db=excluded.representative_sensitivity_dbfs_94db,
                    representative_sensitivity_gain_db=excluded.representative_sensitivity_gain_db,
                    noise_level_dbfs_a=excluded.noise_level_dbfs_a,
                    equivalent_input_noise_db_spl=excluded.equivalent_input_noise_db_spl,
                    adc_dynamic_range_db=excluded.adc_dynamic_range_db,
                    note=excluded.note
                """,
                (
                    profile.id,
                    profile.manufacturer,
                    profile.model,
                    json.dumps(profile.device_name_patterns),
                    json.dumps(profile.sample_rates_hz),
                    profile.preferred_sample_rate_hz,
                    profile.nominal_bit_depth,
                    profile.input_channels,
                    profile.connection_type,
                    profile.serial_pattern,
                    profile.serial_format_hint,
                    json.dumps(profile.calibration_angles_deg),
                    json.dumps(profile.phase_calibration_sample_rates_hz),
                    profile.frequency_min_hz,
                    profile.frequency_max_hz,
                    profile.calibrated_accuracy_db,
                    profile.max_spl_db,
                    profile.representative_sensitivity_dbfs_94db,
                    profile.representative_sensitivity_gain_db,
                    profile.noise_level_dbfs_a,
                    profile.equivalent_input_noise_db_spl,
                    profile.adc_dynamic_range_db,
                    profile.note,
                ),
            )


def list_microphone_profiles(path: Path) -> list[MicrophoneProfile]:
    ensure_microphone_db(path)
    with _connect(path) as connection:
        rows = connection.execute("SELECT * FROM microphone_profiles ORDER BY manufacturer, model").fetchall()
    return [_profile_from_row(row) for row in rows]


def match_microphone_profile(path: Path, device_name: str) -> MicrophoneProfile | None:
    normalized = str(device_name).casefold()
    for profile in list_microphone_profiles(path):
        if any(pattern.casefold() in normalized for pattern in profile.device_name_patterns):
            return profile
    return None


def list_microphone_units(path: Path, *, profile_id: str = "") -> list[MicrophoneUnit]:
    ensure_microphone_db(path)
    with _connect(path) as connection:
        if profile_id:
            rows = connection.execute(
                "SELECT * FROM microphone_units WHERE profile_id = ? ORDER BY label, serial_number",
                (profile_id,),
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM microphone_units ORDER BY label, serial_number").fetchall()
    return [_unit_from_row(row) for row in rows]


def save_microphone_unit(path: Path, unit: MicrophoneUnit) -> MicrophoneUnit:
    ensure_microphone_db(path)
    serial = unit.serial_number.strip()
    if not serial:
        raise ValueError("Microphone serial number is required")
    profile = next((item for item in list_microphone_profiles(path) if item.id == unit.profile_id), None)
    if profile is None:
        raise ValueError("Unknown microphone profile")
    if profile.serial_pattern and re.fullmatch(profile.serial_pattern, serial) is None:
        raise ValueError(f"Serial number must be {profile.serial_format_hint}")
    if int(unit.input_channel) < 1:
        raise ValueError("Input channel must be 1 or greater")
    clean_id = unit.id or str(uuid.uuid4())
    with _connect(path) as connection:
        now = str(connection.execute("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')").fetchone()[0])
        existing = connection.execute(
            "SELECT id, created_at FROM microphone_units WHERE profile_id = ? AND serial_number = ? COLLATE NOCASE",
            (unit.profile_id, serial),
        ).fetchone()
        if existing is not None:
            clean_id = str(existing["id"])
        created_at = str(existing["created_at"]) if existing else now
        connection.execute(
            """
            INSERT INTO microphone_units(
                id, profile_id, serial_number, label, device_name,
                calibration_measurement_id, calibration_90_measurement_id,
                calibration_30_measurement_id, phase_calibration_measurement_id, input_channel,
                note, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                profile_id=excluded.profile_id,
                serial_number=excluded.serial_number,
                label=excluded.label,
                device_name=excluded.device_name,
                calibration_measurement_id=excluded.calibration_measurement_id,
                calibration_30_measurement_id=excluded.calibration_30_measurement_id,
                calibration_90_measurement_id=excluded.calibration_90_measurement_id,
                phase_calibration_measurement_id=excluded.phase_calibration_measurement_id,
                input_channel=excluded.input_channel,
                note=excluded.note,
                updated_at=excluded.updated_at
            """,
            (
                clean_id,
                unit.profile_id,
                serial,
                unit.label.strip(),
                unit.device_name.strip(),
                unit.calibration_measurement_id.strip(),
                unit.calibration_90_measurement_id.strip(),
                unit.calibration_30_measurement_id.strip(),
                unit.phase_calibration_measurement_id.strip(),
                int(unit.input_channel),
                unit.note.strip(),
                created_at,
                now,
            ),
        )
    return next(item for item in list_microphone_units(path) if item.id == clean_id)


def calibration_text_matches_serial(serial_number: str, *values: str) -> bool:
    serial = _identifier(serial_number)
    if len(serial) < 4:
        return False
    return any(serial in _identifier(value) for value in values if value)


def calibration_text_matches_angle(angle_deg: int, *values: str) -> bool:
    combined = " ".join(str(value).casefold() for value in values if value)
    explicit_angles = {
        int(match)
        for match in re.findall(r"(?:^|[^0-9])(0|30|90)\s*(?:deg|degree|degrees|°)(?:[^a-z]|$)", combined)
    }
    if explicit_angles:
        return int(angle_deg) in explicit_angles
    return int(angle_deg) == 0


def _identifier(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).casefold())


def _profile_from_row(row: sqlite3.Row) -> MicrophoneProfile:
    return MicrophoneProfile(
        id=str(row["id"]),
        manufacturer=str(row["manufacturer"]),
        model=str(row["model"]),
        device_name_patterns=tuple(json.loads(row["device_name_patterns_json"])),
        sample_rates_hz=tuple(int(value) for value in json.loads(row["sample_rates_json"])),
        preferred_sample_rate_hz=int(row["preferred_sample_rate_hz"]),
        nominal_bit_depth=int(row["nominal_bit_depth"]),
        input_channels=int(row["input_channels"]),
        connection_type=str(row["connection_type"]),
        serial_pattern=str(row["serial_pattern"]),
        serial_format_hint=str(row["serial_format_hint"]),
        calibration_angles_deg=tuple(int(value) for value in json.loads(row["calibration_angles_json"])),
        phase_calibration_sample_rates_hz=tuple(
            int(value) for value in json.loads(row["phase_calibration_sample_rates_json"])
        ),
        frequency_min_hz=float(row["frequency_min_hz"]),
        frequency_max_hz=float(row["frequency_max_hz"]),
        calibrated_accuracy_db=float(row["calibrated_accuracy_db"]),
        max_spl_db=float(row["max_spl_db"]),
        representative_sensitivity_dbfs_94db=float(row["representative_sensitivity_dbfs_94db"]),
        representative_sensitivity_gain_db=float(row["representative_sensitivity_gain_db"]),
        noise_level_dbfs_a=float(row["noise_level_dbfs_a"]),
        equivalent_input_noise_db_spl=float(row["equivalent_input_noise_db_spl"]),
        adc_dynamic_range_db=float(row["adc_dynamic_range_db"]),
        note=str(row["note"]),
    )


def _unit_from_row(row: sqlite3.Row) -> MicrophoneUnit:
    return MicrophoneUnit(
        id=str(row["id"]),
        profile_id=str(row["profile_id"]),
        serial_number=str(row["serial_number"]),
        label=str(row["label"]),
        device_name=str(row["device_name"]),
        calibration_measurement_id=str(row["calibration_measurement_id"]),
        calibration_30_measurement_id=str(row["calibration_30_measurement_id"]),
        calibration_90_measurement_id=str(row["calibration_90_measurement_id"]),
        phase_calibration_measurement_id=str(row["phase_calibration_measurement_id"]),
        input_channel=int(row["input_channel"]),
        note=str(row["note"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
