from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.design_library_db import DesignRecord, SpeakerPackageRecord
from utils.ui_localization import display_text
from utils.speaker_db import SpeakerMeasurementRecord, SpeakerSpecRecord, get_measurement, measurement_multiway_projects
from utils.measurement_payloads import (
    near_field_port_from_measurement_payload,
    near_field_woofer_from_measurement_payload,
    speaker_input_from_measurement_payload,
    speaker_calibration_state_from_measurement_payload,
)


def diameter_label(value_mm: float | None) -> str:
    if value_mm is None:
        return ""
    inch = float(value_mm) / 25.4
    return f"{float(value_mm):g} mm / {inch:.1f} in"


def _number_label(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):g}"


def speaker_spec_label(record: SpeakerSpecRecord) -> str:
    parts = [part for part in (record.brand, record.model, record.driver_type) if part]
    speaker_diameter_label = diameter_label(record.nominal_diameter_mm)
    if speaker_diameter_label:
        parts.append(speaker_diameter_label)
    if record.fs_hz is not None:
        parts.append(f"Fs {record.fs_hz:g} Hz")
    if record.qts is not None:
        parts.append(f"Qts {record.qts:g}")
    return " | ".join(parts) or record.id


def speaker_spec_table(records: list[SpeakerSpecRecord], *, db_label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "DB": db_label,
                "Brand": record.brand,
                "Model": record.model,
                "Type": record.driver_type,
                "Diameter": diameter_label(record.nominal_diameter_mm),
                "Z": _number_label(record.nominal_impedance_ohm),
                "Fs": _number_label(record.fs_hz),
                "Qts": _number_label(record.qts),
                "Vas L": _number_label(record.vas_l),
                "Re": _number_label(record.re_ohm),
                "Sd": _number_label(record.sd_cm2),
                "Xmax": _number_label(record.xmax_mm),
                "SPL": _number_label(record.spl_db),
                "Pmax": _number_label(record.pmax_w),
                "Source": record.source_name,
                "Verified": "yes" if record.user_verified else "",
                "Created": record.created_at,
                "Updated": record.updated_at,
            }
            for record in records
        ]
    )


def _measurement_display_data(record):
    payload = record.response_payload or {}
    if isinstance(payload.get('_summary'), dict):
        return payload['_summary']
    response = speaker_input_from_measurement_payload(payload)
    return {'rows': len(response.frequency) if response is not None else None,
            'phase': response is not None and response.phase_deg is not None,
            'nf_woofer': near_field_woofer_from_measurement_payload(payload) is not None,
            'nf_port': near_field_port_from_measurement_payload(payload) is not None,
            'calibration': speaker_calibration_state_from_measurement_payload(payload)}


def measurement_label(record: SpeakerMeasurementRecord) -> str:
    parts = [record.name]
    meta = " / ".join(part for part in (record.brand, record.model, record.measurement_date) if part)
    if meta:
        parts.append(meta)
    system_meta = " / ".join(
        part for part in (record.system_name, record.channel, record.position_name, record.system_state) if part
    )
    if system_meta:
        parts.append(system_meta)
    data = _measurement_display_data(record)
    if data['rows'] is not None:
        parts.append(f"{data['rows']} rows")
    if data['nf_woofer']:
        parts.append("NF woofer")
    if data['nf_port']:
        parts.append("NF port")
    return " | ".join(part for part in parts if part) or record.id


def measurement_type_label(source_type: str) -> str:
    return {
        "speaker_input_raw": "Speaker/Input",
        "mic_calibration_raw": "Mic Calibration",
        "near_field_woofer_raw": "Near-field woofer",
        "near_field_port_raw": "Near-field port",
        "impedance": "Impedance",
    }.get(source_type, source_type or "Measurement")


def measurement_scope_label(scope: str) -> str:
    return {
        "system_channel": "System / Channel",
        "combined_room": "L+R verification",
        "driver": "Driver / Near-field",
        "mic_calibration": "Mic Calibration",
        "impedance": "Impedance",
    }.get(scope, scope or "System / Channel")


def measurement_use_label(intended_use: str) -> str:
    return {
        "correction_input": "Correction input",
        "verification_only": "Verification only",
        "calibration": "Calibration",
        "archive": "Archive",
    }.get(intended_use, intended_use or "Correction input")


def measurement_is_correction_input(record: SpeakerMeasurementRecord) -> bool:
    if record.source_type != "speaker_input_raw":
        return False
    if record.measurement_scope == "combined_room":
        return False
    if record.intended_use == "verification_only":
        return False
    return True


def _display_scope(record: SpeakerMeasurementRecord) -> str:
    if record.source_type == "mic_calibration_raw" and record.measurement_scope == "system_channel":
        return "mic_calibration"
    if record.source_type == "impedance" and record.measurement_scope == "system_channel":
        return "impedance"
    return record.measurement_scope


def _display_use(record: SpeakerMeasurementRecord) -> str:
    if record.source_type == "mic_calibration_raw" and record.intended_use == "correction_input":
        return "calibration"
    if record.source_type == "impedance" and record.intended_use == "correction_input":
        return "archive"
    return record.intended_use


def measurement_table(records: list[SpeakerMeasurementRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        data = _measurement_display_data(record)
        rows.append(
            {
                "Name": record.name,
                display_text("マルチウェイのプロジェクト"): " / ".join(measurement_multiway_projects(record)),
                "マイク校正": (
                    {"raw": "未校正", "measurement_calibrated": "適用済み",
                     "measurement_calibration_unknown": "履歴あり・状態不明"}.get(
                        data['calibration'], "適用済み")
                    if record.source_type == "speaker_input_raw" else "対象外"
                ),
                "Type": measurement_type_label(record.source_type),
                "Scope": measurement_scope_label(_display_scope(record)),
                "Use": measurement_use_label(_display_use(record)),
                "System": record.system_name,
                "Channel": record.channel,
                "Position": record.position_name,
                "State": record.system_state,
                "Spec ref": "yes" if record.speaker_spec_id else "",
                "Brand": record.brand,
                "Model": record.model,
                "Date": record.measurement_date,
                "Rows": data['rows'] if data['rows'] is not None else '',
                "Phase": "yes" if data['phase'] else "",
                "NF woofer": "yes" if data['nf_woofer'] else "",
                "NF port": "yes" if data['nf_port'] else "",
                "Distance": record.distance,
                "Angle": record.angle,
                "Mic": record.microphone,
                "Source": record.source_name,
                "Created": record.created_at,
                "Updated": record.updated_at,
            }
        )
    return pd.DataFrame(rows)


def measurement_ref_label_from_payload(
    payload: dict[str, Any] | None,
    key: str,
    *,
    measurements_db_path: Path,
) -> str:
    if not isinstance(payload, dict):
        return ""
    ui = payload.get("ui", {})
    if not isinstance(ui, dict):
        return ""
    ref = ui.get(key)
    if isinstance(ref, dict):
        record_id = str(ref.get("id", "")).strip()
        label = str(ref.get("label", "")).strip()
    else:
        record_id = str(ref or "").strip()
        label = ""
    if record_id:
        record = get_measurement(measurements_db_path, record_id)
        if record is not None:
            return measurement_label(record)
    return label


def project_record_label(record: DesignRecord, *, measurements_db_path: Path) -> str:
    parts = [record.name]
    meta = " / ".join(part for part in (record.speaker_type, record.side, record.position) if part)
    if meta:
        parts.append(meta)
    speaker_measurement = measurement_ref_label_from_payload(
        record.payload,
        "current_speaker_measurement",
        measurements_db_path=measurements_db_path,
    )
    if speaker_measurement:
        parts.append(f"Input: {speaker_measurement}")
    if record.sample_rate:
        parts.append(f"{record.sample_rate // 1000} kHz")
    return " | ".join(parts)


def project_record_table(records: list[DesignRecord], *, measurements_db_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Design": record.name,
                "Speaker": record.speaker_type,
                "Side": record.side,
                "Position": record.position,
                "Input measurement": measurement_ref_label_from_payload(
                    record.payload,
                    "current_speaker_measurement",
                    measurements_db_path=measurements_db_path,
                ),
                "Mic Cal": measurement_ref_label_from_payload(
                    record.payload,
                    "current_mic_calibration",
                    measurements_db_path=measurements_db_path,
                ),
                "Sample rate": "" if record.sample_rate is None else f"{record.sample_rate // 1000} kHz",
                "Taps": "" if record.taps is None else record.taps,
                "Updated": record.updated_at,
                "Last opened": record.last_opened_at,
                "Tags": record.tags,
            }
            for record in records
        ]
    )


def speaker_package_record_label(record: SpeakerPackageRecord) -> str:
    parts = [record.name]
    identity = " / ".join(
        part for part in (record.speaker_type, record.side, record.position) if part
    )
    if identity:
        parts.append(identity)
    source = str(record.payload.get("source_name", "")).strip()
    if source:
        parts.append(f"Input: {source}")
    return " | ".join(parts)


def speaker_package_record_table(records: list[SpeakerPackageRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Speaker Package": record.name,
            "Speaker": record.speaker_type,
            "Side": record.side,
            "Position": record.position,
            "Input": str(record.payload.get("source_name", "")),
            "Mic Cal": str(record.payload.get("mic_cal_source_name", "")),
            "保存内容ID": record.content_signature[:10],
            "Updated": record.updated_at,
            "Tags": record.tags,
        }
        for record in records
    ])
