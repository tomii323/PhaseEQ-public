"""Preset-to-engine and original-to-Library contracts, independent of Streamlit."""
from __future__ import annotations

from dataclasses import replace
from datetime import date
import json
from pathlib import Path

import numpy as np

from phase_fir_designer.measurement.models import MeasurementSettings
from phase_fir_designer.measurement.input_gain import read_input_gain_snapshot
from phase_fir_designer.measurement.original import original_from_snapshot
from phase_fir_designer.measurement.standard_audio import load_standard_reference
from utils.measurement_originals import attach_calibration_sources, payload_for_original
from utils.minidsp_calibration import minidsp_calibration_header
from utils.microphone_db import calibration_text_matches_serial
from utils.settings_io import speaker_response_from_payload
from utils.speaker_db import SpeakerMeasurementRecord, save_measurement

POSITIONS = ("近接1 cm", "30 cm", "1 m", "試聴位置")
AVAILABLE_KINDS = {"response", "tweeter", "noise"}
PENDING_REASONS = {
    "distortion": "ESSの次数別歪み解析が未実装のため、このガイドでは取得できません。",
    "decay": "残響指標の専用解析が未実装のため、このガイドでは取得できません。",
    "timing": "初期ディレーと基準・対象取得を結ぶ測距ガイドは未実装です。",
}


def microphone_settings(profile, unit, calibration, device_name, input_channel):
    if profile is None or unit is None or calibration is None:
        raise ValueError("マイク個体と校正ファイルを登録してください。")
    if profile.device_name_patterns and not any(pattern.lower() in device_name.lower() for pattern in profile.device_name_patterns):
        raise ValueError("入力機器と選択したマイク機種が一致しません。接続した測定マイクを選択してください。")
    curve = speaker_response_from_payload(calibration.response_payload)
    if curve is None:
        raise ValueError("周波数偏差の校正データがありません。")
    if not calibration_text_matches_serial(unit.serial_number, calibration.source_name, calibration.name, calibration.raw_text):
        raise ValueError("選択個体に対応する校正ファイルを選んでください（ファイル名で照合できます）。")
    header = minidsp_calibration_header(calibration.raw_text)
    if profile.id not in {"umik-1", "umik-2"} or header is None:
        raise ValueError("この機種・ファイルの個体感度を実行時に解決する処理は未対応です。手動の音量校正は要求しません。")
    gain = read_input_gain_snapshot(device_name, 1)
    adjustment = gain.gain_adjustment_from_max_db
    if not gain.available or adjustment is None:
        raise ValueError("入力Gainを読み取れないため個体感度を確定できません。" + gain.error)
    if header[1] is not None and (gain.analog_gain_db is None or abs(header[1]-gain.analog_gain_db) > .05):
        raise ValueError("校正ファイルの内部Gainと現在の設定を照合できません。")
    return dict(channels=1, input_mode="known_wav_mono", input_channel=input_channel,
        microphone_profile_id=profile.id, microphone_unit_id=unit.id,
        microphone_serial_number=unit.serial_number, mic_calibration_id=calibration.id,
        mic_calibration_frequency_hz=tuple(curve.frequency), mic_calibration_gain_db=tuple(curve.gain_db),
        level_calibration_mode="minidsp_sens_factor", minidsp_sens_factor_db=header[0],
        minidsp_analog_gain_db=header[1], minidsp_gain_adjustment_db=adjustment,
        input_gain_db_channels=tuple(gain.channel_gain_db), input_gain_max_db_channels=tuple(gain.channel_max_gain_db),
        input_gain_device_name=gain.device_name, input_analog_gain_db=gain.analog_gain_db,
        require_stable_input_gain=True)


def settings_for_track(asset, mic, device, position, angle=0):
    reference = load_standard_reference(asset["path"], asset["manifest"], 48000)
    settings = MeasurementSettings(sample_rate=48000, input_device=device,
        start_frequency_hz=reference.start_frequency_hz, end_frequency_hz=reference.end_frequency_hz,
        sweep_duration_s=reference.sweep_duration_s, shot_period_s=reference.shot_period_s,
        shot_capture_duration_s=reference.shot_capture_duration_s, reference_source_name=reference.source_name,
        reference_source_sample_rate=reference.source_sample_rate, reference_repeat_count=reference.repeat_count,
        reference_peak_dbfs=reference.peak_dbfs,
        reference_loop_gap_variable=True, reference_source_channel=reference.source_channel,
        reference_mode="ir_peak", correlation_threshold=.2, noise_preflight_enabled=True,
        automatic_usb_gain=False, guided_measurement=True, wavelet_update_mode="manual",
        measurement_distance_m={"近接1 cm": .01, "30 cm": .3, "1 m": 1, "試聴位置": 1}[position],
        mic_calibration_angle_deg=angle, **mic)
    return settings, reference


def completed(snapshot):
    result = snapshot.standard_result
    return bool(result is not None and result.shot_count >= 3 and
                np.all(np.isfinite(result.merged_complex_response)) and
                snapshot.device_info.get("guide_stage") == "finalizing")


def save_guided_result(path: Path, snapshot, selection: dict, name: str):
    if not completed(snapshot):
        raise ValueError("同条件の3件と統合結果が成立していません。")
    original = attach_calibration_sources(original_from_snapshot(snapshot), path)
    original = replace(original, metadata={**original.metadata, "guided_selection": dict(selection)})
    payload = payload_for_original(original)
    both = selection["output"] == "both"
    return save_measurement(path, record=SpeakerMeasurementRecord(
        id="guided-"+snapshot.session_id, name=name, measurement_date=date.today().isoformat(),
        channel=selection["output"], position_name=selection["position"], distance=selection["position"],
        angle=str(selection.get("angle", 0)), microphone=snapshot.settings.microphone_serial_number,
        measurement_scope="combined" if both else "system_channel",
        intended_use="verification_only" if both else "correction_input", source_type="speaker_input_raw",
        source_name=snapshot.settings.reference_source_name, response_payload=payload,
        note=json.dumps(selection, ensure_ascii=False)), original=original)
