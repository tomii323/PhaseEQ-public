from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from phase_fir_designer import SpeakerResponse
from phase_fir_designer.speaker import load_speaker_response_text


MINIDSP_PRODUCT_URLS = {
    "umik-1": "https://www.minidsp.com/products/acoustic-measurement/umik-1",
    "umik-2": "https://www.minidsp.com/products/acoustic-measurement/umik-2",
}


@dataclass(frozen=True)
class MiniDSPCalibrationFile:
    source_name: str
    serial_number: str
    angle_deg: int
    sensitivity_factor_db: float
    analog_gain_db: float | None
    response: SpeakerResponse
    raw_text: str
    sha256: str


def normalize_umik_serial(value: str) -> str:
    match = re.fullmatch(r"\s*(\d{3})-?(\d{4})\s*", str(value))
    if match is None:
        raise ValueError("UMIK serial number must be 3 digits-4 digits")
    return f"{match.group(1)}-{match.group(2)}"


def parse_minidsp_calibration(
    content: bytes,
    *,
    source_name: str,
    expected_serial: str,
) -> MiniDSPCalibrationFile:
    if not content:
        raise ValueError(f"{source_name}: calibration file is empty")
    text = _decode_text(content)
    serial_match = re.search(r"\bSERNO\s*:\s*([0-9-]+)", text, flags=re.IGNORECASE)
    if serial_match is None:
        raise ValueError(f"{source_name}: miniDSP SERNO header was not found")
    file_serial = normalize_umik_serial(serial_match.group(1))
    wanted_serial = normalize_umik_serial(expected_serial)
    if file_serial != wanted_serial:
        raise ValueError(
            f"{source_name}: serial {file_serial} does not match the selected microphone {wanted_serial}"
        )
    sensitivity_match = re.search(
        r"\bSens\s*Factor\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*dB",
        text,
        flags=re.IGNORECASE,
    )
    if sensitivity_match is None:
        raise ValueError(f"{source_name}: miniDSP Sens Factor header was not found")
    analog_gain_match = re.search(
        r"\bAGain\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*dB",
        text,
        flags=re.IGNORECASE,
    )
    angle_deg = _calibration_angle(source_name, text)
    try:
        response = load_speaker_response_text(io.StringIO(text))
    except Exception as exc:
        raise ValueError(f"{source_name}: calibration response could not be parsed: {exc}") from exc
    frequency = np.asarray(response.frequency, dtype=float)
    gain = np.asarray(response.gain_db, dtype=float)
    if frequency.size < 2 or gain.size != frequency.size:
        raise ValueError(f"{source_name}: calibration response needs at least two frequency rows")
    if not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(gain)):
        raise ValueError(f"{source_name}: calibration response contains a non-finite value")
    if np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise ValueError(f"{source_name}: frequencies must be positive and strictly increasing")
    return MiniDSPCalibrationFile(
        source_name=Path(source_name).name,
        serial_number=file_serial,
        angle_deg=angle_deg,
        sensitivity_factor_db=float(sensitivity_match.group(1)),
        analog_gain_db=float(analog_gain_match.group(1)) if analog_gain_match is not None else None,
        response=response,
        raw_text=text,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def parse_minidsp_calibration_pair(
    files: Iterable[tuple[str, bytes]],
    *,
    expected_serial: str,
) -> dict[int, MiniDSPCalibrationFile]:
    parsed = [
        parse_minidsp_calibration(content, source_name=name, expected_serial=expected_serial)
        for name, content in files
    ]
    if len(parsed) != 2:
        raise ValueError("Select both miniDSP calibration files: 0° and 90°")
    by_angle = {item.angle_deg: item for item in parsed}
    if len(by_angle) != len(parsed):
        raise ValueError("The selected files contain the same calibration angle")
    if set(by_angle) != {0, 90}:
        raise ValueError("Select one 0° file and one 90° file")
    sensitivity = {round(item.sensitivity_factor_db, 6) for item in parsed}
    if len(sensitivity) != 1:
        raise ValueError("Sens Factor differs between the 0° and 90° files")
    analog_gains = {item.analog_gain_db for item in parsed}
    if len(analog_gains) != 1:
        raise ValueError("AGain differs between the 0° and 90° files")
    return by_angle


def minidsp_calibration_header(text: str) -> tuple[float, float | None] | None:
    sensitivity_match = re.search(
        r"\bSens\s*Factor\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*dB",
        str(text),
        flags=re.IGNORECASE,
    )
    if sensitivity_match is None:
        return None
    analog_gain_match = re.search(
        r"\bAGain\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*dB",
        str(text),
        flags=re.IGNORECASE,
    )
    return (
        float(sensitivity_match.group(1)),
        float(analog_gain_match.group(1)) if analog_gain_match is not None else None,
    )


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Calibration file must be UTF-8 or Windows text")


def _calibration_angle(source_name: str, text: str) -> int:
    evidence = f"{source_name}\n{text[:500]}".casefold()
    if re.search(r"(?:90\s*(?:-|_)?\s*(?:deg(?:ree)?|°)|90degree)", evidence):
        return 90
    if re.search(r"(?:30\s*(?:-|_)?\s*(?:deg(?:ree)?|°)|30degree)", evidence):
        raise ValueError(f"{source_name}: 30° is not a UMIK calibration angle")
    return 0
