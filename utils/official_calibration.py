from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np

from phase_fir_designer import SpeakerResponse
from phase_fir_designer.speaker import load_speaker_response_text
from utils.microphone_db import (
    MicrophoneProfile,
    MicrophoneUnit,
    calibration_text_matches_angle,
    calibration_text_matches_serial,
    list_microphone_units,
    save_microphone_unit,
)
from utils.minidsp_calibration import normalize_umik_serial, parse_minidsp_calibration_pair
from utils.settings_io import speaker_response_payload
from utils.speaker_db import SpeakerMeasurementRecord, list_measurement_summaries, save_measurement


DAYTON_CALIBRATION_URL = "https://support.daytonaudio.com/MicrophoneCalibrationTool"
SONARWORKS_CALIBRATION_URL = "https://www.sonarworks.com/download-center#SoundIDReference"


@dataclass(frozen=True)
class OfficialCalibrationSource:
    profile_id: str
    provider: str
    url: str
    required_angles_deg: tuple[int, ...]
    extensions: tuple[str, ...]
    download_description: str


@dataclass(frozen=True)
class DownloadedCalibration:
    source_name: str
    content: bytes
    angle_deg: int
    sha256: str


OFFICIAL_CALIBRATION_SOURCES = {
    "umik-1": OfficialCalibrationSource(
        "umik-1", "miniDSP", "https://www.minidsp.com/products/acoustic-measurement/umik-1",
        (0, 90), (".txt",), "0° TXT and 90° TXT",
    ),
    "umik-2": OfficialCalibrationSource(
        "umik-2", "miniDSP", "https://www.minidsp.com/products/acoustic-measurement/umik-2",
        (0, 90), (".txt",), "0° TXT and 90° TXT",
    ),
    "omnimic": OfficialCalibrationSource(
        "omnimic", "Dayton Audio", DAYTON_CALIBRATION_URL,
        (0,), (".omm",), "individual OmniMic OMM",
    ),
    "umm-6": OfficialCalibrationSource(
        "umm-6", "Dayton Audio", DAYTON_CALIBRATION_URL,
        (0,), (".txt",), "individual UMM-6 TXT",
    ),
    "emm-6": OfficialCalibrationSource(
        "emm-6", "Dayton Audio", DAYTON_CALIBRATION_URL,
        (0,), (".txt",), "individual EMM-6 TXT",
    ),
    "xref-20": OfficialCalibrationSource(
        "xref-20", "Sonarworks", SONARWORKS_CALIBRATION_URL,
        (0, 30, 90), (".zip", ".txt"), "profile ZIP containing 0° / 30° / 90° TXT",
    ),
}


def normalized_calibration_serial(profile_id: str, serial: str) -> str:
    value = str(serial).strip()
    if profile_id in {"umik-1", "umik-2"}:
        return normalize_umik_serial(value)
    patterns = {
        "omnimic": (r"\d{7}", "7 digits"),
        "umm-6": (r"\d{4,6}", "4-6 digits"),
        "emm-6": (r"\d{4,6}", "4-6 digits"),
        "xref-20": (r"[A-Za-z0-9]{6}", "6 alphanumeric characters"),
    }
    pattern, hint = patterns.get(profile_id, (r".+", "a valid identifier"))
    if re.fullmatch(pattern, value) is None:
        raise ValueError(f"Calibration serial must be {hint}")
    return value.upper() if profile_id == "xref-20" else value


def find_downloaded_calibrations(
    folder: Path,
    *,
    profile_id: str,
    serial: str,
    modified_after: float,
) -> dict[int, DownloadedCalibration]:
    source = OFFICIAL_CALIBRATION_SOURCES.get(profile_id)
    if source is None:
        raise ValueError("Official calibration download is not supported for this microphone")
    normalized = normalized_calibration_serial(profile_id, serial)
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError("Download folder does not exist or is not accessible")
    matches: list[tuple[float, DownloadedCalibration]] = []
    for path in root.iterdir():
        if not path.is_file() or path.suffix.casefold() not in source.extensions:
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if modified + 1.0 < float(modified_after):
            continue
        if path.suffix.casefold() == ".zip":
            matches.extend(_zip_candidates(path, normalized, modified))
            continue
        if _identifier(normalized) not in _identifier(path.name):
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if _candidate_matches(profile_id, normalized, path.name, content):
            matches.append((modified, _downloaded(path.name, content)))
    by_angle: dict[int, tuple[float, DownloadedCalibration]] = {}
    for modified, item in matches:
        previous = by_angle.get(item.angle_deg)
        if previous is None or modified >= previous[0]:
            by_angle[item.angle_deg] = (modified, item)
    return {angle: value[1] for angle, value in by_angle.items()}


def calibrations_from_uploaded_files(
    files: list[tuple[str, bytes]],
    *,
    profile_id: str,
    serial: str,
) -> dict[int, DownloadedCalibration]:
    normalized = normalized_calibration_serial(profile_id, serial)
    matches: list[DownloadedCalibration] = []
    for name, content in files:
        if str(name).casefold().endswith(".zip"):
            matches.extend(item for _, item in _zip_content_candidates(content, normalized, 0.0))
        elif _candidate_matches(profile_id, normalized, name, content):
            matches.append(_downloaded(name, content))
    return {item.angle_deg: item for item in matches}


def validate_official_calibrations(
    profile: MicrophoneProfile,
    serial: str,
    files: dict[int, DownloadedCalibration],
) -> dict[int, tuple[DownloadedCalibration, SpeakerResponse, str]]:
    source = OFFICIAL_CALIBRATION_SOURCES[profile.id]
    normalized = normalized_calibration_serial(profile.id, serial)
    missing = set(source.required_angles_deg) - set(files)
    if missing:
        angles = ", ".join(f"{value}°" for value in sorted(missing))
        raise ValueError(f"Waiting for {angles} calibration file(s)")
    if profile.id in {"umik-1", "umik-2"}:
        pair = parse_minidsp_calibration_pair(
            [(files[angle].source_name, files[angle].content) for angle in source.required_angles_deg],
            expected_serial=normalized,
        )
        return {
            angle: (
                files[angle],
                calibration.response,
                f"Sens Factor {calibration.sensitivity_factor_db:+g} dB"
                + (f" / AGain {calibration.analog_gain_db:+g} dB" if calibration.analog_gain_db is not None else ""),
            )
            for angle, calibration in pair.items()
        }
    validated: dict[int, tuple[DownloadedCalibration, SpeakerResponse, str]] = {}
    for angle in source.required_angles_deg:
        item = files[angle]
        text = _decode_calibration_text(item.content)
        if not _candidate_matches(profile.id, normalized, item.source_name, item.content):
            raise ValueError(f"{item.source_name}: serial/Profile ID does not match {normalized}")
        try:
            response = load_speaker_response_text(io.StringIO(text))
        except Exception as exc:
            raise ValueError(f"{item.source_name}: calibration response could not be parsed: {exc}") from exc
        _validate_response(item.source_name, response)
        validated[angle] = (item, response, _dayton_sensitivity_note(text))
    return validated


def register_official_calibrations(
    *,
    profile: MicrophoneProfile,
    serial: str,
    files: dict[int, DownloadedCalibration],
    measurement_db_path: Path,
    microphone_db_path: Path,
    device_name: str = "",
    input_channel: int = 1,
) -> tuple[MicrophoneUnit, dict[int, SpeakerMeasurementRecord]]:
    source = OFFICIAL_CALIBRATION_SOURCES[profile.id]
    normalized = normalized_calibration_serial(profile.id, serial)
    validated = validate_official_calibrations(profile, normalized, files)
    existing_records = [
        record for record in list_measurement_summaries(measurement_db_path, limit=10_000)
        if record.source_type == "mic_calibration_raw"
    ]
    saved: dict[int, SpeakerMeasurementRecord] = {}
    for angle, (item, response, correction_note) in validated.items():
        matching = [
            record for record in existing_records
            if calibration_text_matches_serial(
                normalized, record.name, record.source_name, record.microphone, record.note
            )
            and calibration_text_matches_angle(
                angle, record.name, record.source_name, record.microphone, record.note
            )
            and profile.model.casefold() in " ".join((record.name, record.microphone)).casefold()
        ]
        saved[angle] = save_measurement(
            measurement_db_path,
            record=SpeakerMeasurementRecord(
                id=matching[0].id if len(matching) == 1 else "",
                name=f"{profile.model} {normalized} {angle}° calibration",
                measurement_date=date.today().isoformat(),
                angle=f"{angle}°",
                microphone=f"{profile.manufacturer} {profile.model} S/N {normalized}",
                correction_note=correction_note,
                source_name=item.source_name,
                source_url=source.url,
                source_type="mic_calibration_raw",
                response_payload=speaker_response_payload(response),
                raw_text=_decode_calibration_text(item.content),
                note=f"{source.provider} official individual calibration / SHA-256 {item.sha256}",
            ),
        )
    base_unit = next(
        (
            unit for unit in list_microphone_units(microphone_db_path, profile_id=profile.id)
            if _identifier(unit.serial_number) == _identifier(normalized)
        ),
        MicrophoneUnit(id="", profile_id=profile.id, serial_number=normalized),
    )
    saved_unit = save_microphone_unit(
        microphone_db_path,
        replace(
            base_unit,
            device_name=device_name or base_unit.device_name,
            input_channel=int(input_channel),
            calibration_measurement_id=(saved[0].id if 0 in saved else base_unit.calibration_measurement_id),
            calibration_30_measurement_id=(saved[30].id if 30 in saved else base_unit.calibration_30_measurement_id),
            calibration_90_measurement_id=(saved[90].id if 90 in saved else base_unit.calibration_90_measurement_id),
            phase_calibration_measurement_id=(
                saved[0].id
                if 0 in saved and validated[0][1].phase_deg is not None
                else base_unit.phase_calibration_measurement_id
            ),
        ),
    )
    return saved_unit, saved


def _zip_candidates(path: Path, serial: str, modified: float) -> list[tuple[float, DownloadedCalibration]]:
    if path.stat().st_size > 20 * 1024 * 1024:
        return []
    try:
        return _zip_content_candidates(path.read_bytes(), serial, modified)
    except OSError:
        return []


def _zip_content_candidates(content: bytes, serial: str, modified: float) -> list[tuple[float, DownloadedCalibration]]:
    if len(content) > 20 * 1024 * 1024:
        return []
    output: list[tuple[float, DownloadedCalibration]] = []
    total_size = 0
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename).name
                if not name or not name.casefold().endswith(".txt") or info.file_size > 2 * 1024 * 1024:
                    continue
                if _identifier(serial) not in _identifier(name):
                    continue
                total_size += info.file_size
                if total_size > 10 * 1024 * 1024 or len(output) >= 20:
                    return []
                output.append((modified, _downloaded(name, archive.read(info))))
    except zipfile.BadZipFile:
        return []
    return output


def _candidate_matches(profile_id: str, serial: str, name: str, content: bytes) -> bool:
    identifier = _identifier(serial)
    if identifier in _identifier(name):
        return True
    if profile_id == "omnimic":
        return identifier in _identifier(_decode_calibration_text(content)[:300])
    if profile_id in {"umik-1", "umik-2"}:
        return identifier in _identifier(_decode_calibration_text(content)[:500])
    return False


def _downloaded(name: str, content: bytes) -> DownloadedCalibration:
    return DownloadedCalibration(
        source_name=Path(name).name,
        content=content,
        angle_deg=_angle_from_name(name),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _angle_from_name(name: str) -> int:
    value = str(name).casefold()
    if re.search(r"(?:^|[^0-9])90\s*(?:deg(?:ree)?|°)", value):
        return 90
    if re.search(r"(?:^|[^0-9])30\s*(?:deg(?:ree)?|°)", value):
        return 30
    return 0


def _decode_calibration_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Calibration file must be UTF-8 or Windows text")


def _validate_response(name: str, response: SpeakerResponse) -> None:
    frequency = np.asarray(response.frequency, dtype=float)
    gain = np.asarray(response.gain_db, dtype=float)
    if frequency.size < 2 or frequency.size != gain.size:
        raise ValueError(f"{name}: calibration response needs at least two rows")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(gain)):
        raise ValueError(f"{name}: calibration response contains a non-finite value")
    if np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise ValueError(f"{name}: frequencies must be positive and strictly increasing")
    if response.phase_deg is not None and not np.all(np.isfinite(response.phase_deg)):
        raise ValueError(f"{name}: phase response contains a non-finite value")


def _dayton_sensitivity_note(text: str) -> str:
    match = re.search(r"\*\s*1000\s*Hz\s+([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    return f"Manufacturer 1 kHz sensitivity entry {float(match.group(1)):+g} dB" if match else ""


def _identifier(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(value).casefold())
