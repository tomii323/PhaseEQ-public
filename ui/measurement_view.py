from __future__ import annotations
from utils.ui_localization import ui_message, display_text, localized_formatter

from dataclasses import replace
from datetime import date
import hashlib
import io
import json
import shutil
import sqlite3
import threading
from pathlib import Path
import time
import uuid
import zipfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils.ui_work_cache import deferred_call

from phase_fir_designer import SpeakerResponse
from phase_fir_designer.cache_coordinator import (
    CacheCoordinator,
    CacheDomain,
    CacheEvent,
    content_digest,
)
from phase_fir_designer.phase_curves import fractional_octave_smooth
from phase_fir_designer.measurement import (
    ContinuousMeasurementEngine,
    EssGeneratorSettings,
    HighPrecisionResult,
    KnownEssReference,
    MeasurementSettings,
    MeasurementState,
    SessionSnapshot,
    ShotStatus,
    average_responses,
    build_inferred_ess_manifest,
    capture_input_level,
    cleanup_measurement_autosave,
    effective_merge_crossover_hz,
    estimate_acoustic_calibration,
    ess_generator_rejection_reasons,
    ess_reference_rejection_reasons,
    extract_known_ess_reference,
    extract_manifest_ess_reference,
    generate_ess_bundle,
    generate_level_check_signal,
    generate_marker_ess_bundle,
    generator_preset,
    list_input_devices,
    load_measurement_session,
    high_precision_result,
    input_settings_error,
    level_calibration_kind,
    level_calibration_offset_db,
    pilot_snr_thresholds,
    read_input_gain_snapshot,
    refresh_audio_devices,
    response_level_dbfs,
    response_level_spl,
    time_align_impulse_to_center,
    tweeter_reference_routing_rejection_reasons,
)
from phase_fir_designer.measurement.persistence import session_zip_bytes
from phase_fir_designer.measurement.original import original_from_snapshot, result_from_original, calibrated_export_zip
from utils.measurement_originals import attach_calibration_sources
from utils.microphone_db import (
    MicrophoneProfile,
    MicrophoneUnit,
    calibration_text_matches_angle,
    calibration_text_matches_serial,
    list_microphone_profiles,
    list_microphone_units,
    match_microphone_profile,
    save_microphone_unit,
)
from utils.minidsp_calibration import (
    MINIDSP_PRODUCT_URLS,
    minidsp_calibration_header,
)
from utils.official_calibration import (
    OFFICIAL_CALIBRATION_SOURCES,
    calibrations_from_uploaded_files,
    find_downloaded_calibrations,
    normalized_calibration_serial,
    register_official_calibrations,
)
from utils.settings_io import speaker_response_from_payload, speaker_response_payload
from utils.measurement_payloads import timing_provenance_from_measurement_payload
from timing_provenance import (
    TimingProvenance,
    reference_arrival_samples_at_rate,
    trusted_reference_provenance,
)
from utils.ess_reference_db import EssReferenceRecord, list_ess_references, save_ess_reference
from utils.speaker_db import (
    SpeakerMeasurementRecord,
    get_measurement,
    list_measurement_summaries,
    save_measurement,
)
from utils.ui_localization import localized_button, localized_segmented_control, tab_hover_translation_css
from ui.wavelet_view import draw_measurement_wavelet_map


ENGINE_KEY = "_continuous_ess_engine"
UNSAVED_KEY = "_continuous_ess_unsaved"
PENDING_START_KEY = "_continuous_ess_pending_start"
START_OPTIONS_KEY = "_continuous_ess_start_options_open"
MEASUREMENT_START_DELAYS_S = (3, 5, 10)
MEASUREMENT_RESULT_CACHE_ALGORITHM_VERSION = "2026-09-15-integrated-original-v2"
MEASUREMENT_DISPLAY_CACHE_ALGORITHM_VERSION = "2026-07-25-measurement-display-v1"
ESS_ANALYSIS_CACHE_ALGORITHM_VERSION = "2026-07-22-ess-analysis-v1"
AUDIO_CAPABILITY_CACHE_ALGORITHM_VERSION = "2026-07-22-audio-capability-v1"
ACTIVE_MEASUREMENT_STATES = frozenset(
    {
        MeasurementState.ARMING,
        MeasurementState.MEASURING_NOISE,
        MeasurementState.PREFLIGHT,
        MeasurementState.SEARCHING,
        MeasurementState.CAPTURING_SHOT,
        MeasurementState.ANALYZING_SHOT,
        MeasurementState.STOPPING,
    }
)


def _analog_level_calibration_fingerprint(
    *,
    input_device: int | None,
    device_name: str,
    input_channel: int,
    sample_rate: int,
    input_gain_db: float,
    reference_spl_db: float,
    calibration_method: str = "acoustic_calibrator",
) -> str:
    """Identify every user-controlled setting that invalidates SPL calibration."""

    payload = {
        "input_device": input_device,
        "device_name": str(device_name),
        "input_channel": int(input_channel),
        "sample_rate": int(sample_rate),
        "input_gain_db": round(float(input_gain_db), 6),
        "reference_spl_db": round(float(reference_spl_db), 6),
        "calibration_method": str(calibration_method),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _measurement_input_mode(ess_reference_label: str, channels: int) -> str:
    """Combine independent ESS-reference and recording-layout dimensions."""

    if ess_reference_label not in {"ESS Library file", "PhaseEQ nominal ESS"}:
        raise ValueError("unsupported ESS reference")
    if int(channels) not in {1, 2}:
        raise ValueError("measurement recording layout must use one or two channels")
    if ess_reference_label == "ESS Library file":
        return "known_wav_mono" if int(channels) == 1 else "known_wav_dual_gain"
    return "mono" if int(channels) == 1 else "dual_gain"


def _trusted_library_reference_arrival(
    measurement_db_path: Path,
    reference_measurement_id: str,
    *,
    timing_reference_session_id: str,
    reference_tweeter_channel_id: str,
    sample_rate_hz: int,
) -> tuple[TimingProvenance, float] | None:
    """Revalidate a Library reference at the final adoption boundary."""

    record_id = str(reference_measurement_id).strip()
    record = get_measurement(measurement_db_path, record_id) if record_id else None
    provenance = (
        trusted_reference_provenance(
            timing_provenance_from_measurement_payload(record.response_payload)
        )
        if record is not None else None
    )
    converted_arrival = (
        reference_arrival_samples_at_rate(
            provenance, sample_rate_hz=int(sample_rate_hz),
        )
        if provenance is not None else None
    )
    if not (
        provenance is not None
        and provenance.source_measurement_id == record_id
        and provenance.reference_measurement_id == record_id
        and provenance.timing_reference_session_id
        == str(timing_reference_session_id)
        and provenance.reference_tweeter_channel_id
        == str(reference_tweeter_channel_id)
        and converted_arrival is not None
    ):
        return None
    return provenance, float(converted_arrival)
MEASUREMENT_CATEGORY_OPTIONS = (
    "Mic input",
    "ESS setup",
    "Calibration",
    "Gate / merge",
    "Capture",
    "Session",
)
MIC_CALIBRATION_HELP = (
    "対応マイクはMeasureのメーカー公式校正アシスタントから、ダウンロード後に自動登録できます。"
    "Manual registration: **Library → Measurements → Import measurement files → "
    "Mic Calibration TXT OMM → Save mic file** を使用します。"
)


def _measurement_result_assessment(
    result: HighPrecisionResult | None,
    *,
    require_clock: bool,
    require_high_precision: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    if result is None:
        return False, ("統合結果がありません",)
    reasons: list[str] = []
    if result.shot_count < 1:
        reasons.append("採用できるショットがありません")
    if require_high_precision and result.shot_count < 3:
        reasons.append(f"高精度に必要な採用数が不足（{result.shot_count}/3）")
    if require_high_precision and result.shot_count >= 2 and result.median_coherence < 0.9:
        reasons.append(f"Median coherence {result.median_coherence:.3f} < 0.900")
    if require_high_precision and result.shot_count >= 2 and result.repeatability_p90_db > 1.5:
        reasons.append(f"Repeatability P90 {result.repeatability_p90_db:.2f} dB > 1.50 dB")
    residual_center = max((abs(value) for value in result.alignment_samples), default=0.0)
    if require_high_precision and residual_center > 0.5:
        reasons.append(f"残留センター誤差 {residual_center:.2f} sample > 0.50 sample")
    if require_high_precision and require_clock and result.shot_count >= 3:
        if result.timing_valid_shot_count < 3:
            reasons.append(f"Timing有効shotが不足（{result.timing_valid_shot_count}/3）")
        if not np.isfinite(result.clock_residual_samples) or result.clock_residual_samples > 1.0:
            reasons.append("連続周期からClockを確定できません")
        if abs(result.clock_drift_ppm) > 5_000.0:
            reasons.append(f"Clock drift {result.clock_drift_ppm:+.1f} ppmが範囲外")
    return not reasons, tuple(reasons)


def _shot_reliability_diagnostics(shots) -> dict[int, tuple[str, float, str]]:
    """Explain why each accepted shot is or is not trustworthy."""

    usable = [shot for shot in shots if shot.combined_raw_ir is not None]
    if not usable:
        return {}

    correlations = np.asarray([shot.correlation for shot in usable], dtype=float)
    peak_levels = np.asarray([
        20.0 * np.log10(max(float(np.max(np.abs(shot.combined_raw_ir))), 1e-15))
        for shot in usable
    ])
    median_correlation = float(np.median(correlations))
    median_peak_level = float(np.median(peak_levels))
    output: dict[int, tuple[str, float, str]] = {}
    for shot, peak_level in zip(usable, peak_levels):
        snr_score = float(np.clip((shot.quality.snr_db - 6.0) / 34.0, 0.05, 1.0))
        reliability = float(np.clip(snr_score * max(shot.correlation, 0.0) ** 2, 0.0, 1.0))
        reasons: list[str] = []
        low_correlation = shot.correlation < max(0.5, 0.75 * median_correlation)
        low_snr = shot.quality.snr_db < 20.0
        if low_correlation:
            reasons.append(f"ESS相関低下 {shot.correlation:.3f}")
        if low_snr:
            reasons.append(f"S/N不足 {shot.quality.snr_db:.1f} dB")
        if np.isfinite(shot.centered_ir_similarity) and shot.centered_ir_similarity < 0.60:
            reasons.append(f"中心合わせ後IR類似度低下 {shot.centered_ir_similarity:.2f}")
        if abs(float(peak_level) - median_peak_level) > 6.0:
            reasons.append(f"IR peakレベル変動 {float(peak_level) - median_peak_level:+.1f} dB")
        if shot.detected_period_s is None:
            reasons.append("連続周期未確認（新anchor）")
        elif abs(shot.timing_error_ms) > 20.0:
            reasons.append(f"再生周期差 {shot.timing_error_ms:+.1f} ms（診断のみ）")
        if not shot.merge_valid:
            reasons.append("Gate / FullのGain・位相が重なる帯域なし（Fullへ退避）")
        low_similarity = np.isfinite(shot.centered_ir_similarity) and shot.centered_ir_similarity < 0.35
        severe = low_correlation or low_snr or low_similarity or reliability < 0.15
        grade = "Low" if severe else "Medium" if reasons else "High"
        output[int(shot.shot_index)] = (
            grade,
            reliability,
            " / ".join(reasons) if reasons else "合格",
        )
    return output


def _preflight_workflow_steps(snapshot: SessionSnapshot) -> tuple[dict[str, str], ...]:
    info = snapshot.device_info
    history = list(info.get("pilot_history", []))
    latest = history[-1] if history else {}
    noise_done = "ambient_noise_duration_s" in info
    ready = bool(info.get("preflight_ready", False))
    thresholds = pilot_snr_thresholds(snapshot.settings)
    snr_ready = bool(latest) and (
        float(latest.get("snr_median_db", -300.0)) >= thresholds.median_db
        and float(latest.get("snr_p10_db", -300.0)) >= thresholds.p10_db
        and float(latest.get("passing_band_fraction", 0.0)) >= thresholds.passing_fraction
    )
    gain_recheck = bool(info.get("gain_recheck_required", False))
    current_gains = tuple(snapshot.settings.input_gain_db_channels)
    gain_detail = (
        "現在 " + " / ".join(f"{value:+.1f} dB" for value in current_gains)
        if current_gains else
        "デバイスGain未取得"
    )

    def step(label: str, status: str, detail: str) -> dict[str, str]:
        return {"label": label, "status": status, "detail": detail}

    noise_status = "完了" if noise_done else "実行中" if snapshot.state == MeasurementState.MEASURING_NOISE else "待機"
    speaker_status = "完了" if snr_ready or ready else "調整中" if noise_done else "待機"
    gain_status = "完了" if gain_recheck or ready else "調整中" if snr_ready else "待機"
    final_status = "完了" if ready else "再測定" if gain_recheck else "待機"
    speaker_detail = str(latest.get("message", "Pilot ESSで帯域別S/Nを確認"))
    final_detail = (
        "正式測定を開始できます" if ready else
        "変更後のGainでPilot ESSを再測定" if gain_recheck else
        "Peak・S/N・ヘッドルームを最終確認"
    )
    return (
        step("1. 暗騒音", noise_status, "5秒間の環境ノイズ"),
        step("2. スピーカー音量", speaker_status, speaker_detail),
        step("3. マイクGain", gain_status, gain_detail),
        step("4. 最終確認", final_status, final_detail),
    )


def _ess_playback_instruction(snapshot: SessionSnapshot) -> tuple[str, str, str]:
    info = snapshot.device_info
    noise_done = "ambient_noise_duration_s" in info
    ready = bool(info.get("preflight_ready", False))
    gain_recheck = bool(info.get("gain_recheck_required", False))
    if snapshot.state == MeasurementState.ARMING:
        return (
            "warning",
            "まだESSを再生しないでください",
            "開始カウントダウン中です。カウントダウン後、最初の5秒間で暗騒音だけを測定します。",
        )
    if snapshot.state == MeasurementState.MEASURING_NOISE:
        return (
            "warning",
            "まだESSを再生しないでください",
            "暗騒音測定中です。5秒間はスピーカーを鳴らさず、現在環境のノイズだけを取得します。",
        )
    if snapshot.state in {MeasurementState.PREFLIGHT, MeasurementState.SEARCHING} and noise_done and not ready:
        detail = (
            "マイクGain変更後の再確認です。スピーカーからESSを1回再生してください。合格後に正式測定へ戻ります。"
            if gain_recheck else
            "暗騒音測定は完了しています。スピーカーからESSを再生してください。最初はPilotとして音量・S/N・ヘッドルームを確認します。"
        )
        return ("success", "ESSを再生してください", detail)
    if snapshot.state in {MeasurementState.SEARCHING, MeasurementState.CAPTURING_SHOT, MeasurementState.ANALYZING_SHOT}:
        return (
            "info",
            "ESS再生をそのまま継続してください",
            "正式測定を取得中です。外部プレーヤーの繰り返し再生を止めず、必要な測定数まで継続してください。",
        )
    if snapshot.state == MeasurementState.STOPPING:
        return (
            "info",
            "ESS再生を停止してかまいません",
            "PhaseEQはStop要求を処理中です。取得済み測定の解析とFinal readyを続けます。",
        )
    if snapshot.state == MeasurementState.STOPPED_WITH_RESULTS:
        return (
            "info",
            "ESS再生は不要です",
            "Measurement is stopped. Select a result row to view graphs or run post processing.",
        )
    if snapshot.state in {MeasurementState.ERROR_RECOVERABLE, MeasurementState.ERROR_FATAL}:
        return (
            "warning",
            "ESS再生を停止してください",
            "入力エラーを解消してから、必要に応じて新しい測定を開始してください。",
        )
    return (
        "info",
        "Start後もすぐにはESSを再生しません",
        "Start後のカウントダウンと5秒間の暗騒音測定が終わり、画面に「ESSを再生してください」と出てから再生します。",
    )


def _render_preflight_workflow(snapshot: SessionSnapshot) -> None:
    instruction_kind, instruction_title, instruction_detail = _ess_playback_instruction(snapshot)
    status_icon = {
        "完了": ":material/check_circle:",
        "実行中": ":material/progress_activity:",
        "調整中": ":material/tune:",
        "再測定": ":material/replay:",
        "待機": ":material/schedule:",
    }
    with st.container(border=True):
        header_cols = st.columns([0.24, 0.76], vertical_alignment="center")
        header_cols[0].markdown(ui_message('ui.fc151424bb80ad'))
        with header_cols[1]:
            instruction_body = f"**{instruction_title}** — {instruction_detail}"
            if instruction_kind == "success":
                st.success(instruction_body, icon=":material/play_circle:")
            elif instruction_kind == "warning":
                st.warning(instruction_body, icon=":material/volume_off:")
            else:
                st.info(instruction_body, icon=":material/info:")
        step_text = "　".join(
            f"{item['label'].split('. ', 1)[-1]} {status_icon[item['status']]} {item['status']}"
            for item in _preflight_workflow_steps(snapshot)
        )
        st.caption(step_text)


def _level_meter_fraction(level_dbfs: float, *, floor_dbfs: float = -60.0) -> float:
    if not np.isfinite(level_dbfs):
        return 0.0
    return float(np.clip((float(level_dbfs) - floor_dbfs) / -floor_dbfs, 0.0, 1.0))


def _render_live_input_level_meter(snapshot: SessionSnapshot) -> None:
    """Render an REW-style RMS bar with peak and headroom diagnostics."""

    info = snapshot.device_info
    peak_levels = tuple(float(value) for value in info.get("live_input_peak_dbfs", ()))
    rms_levels = tuple(float(value) for value in info.get("live_input_rms_dbfs", ()))
    if not peak_levels or len(peak_levels) != len(rms_levels):
        if _measurement_is_running(snapshot):
            with st.container(border=True):
                st.markdown(ui_message('ui.3f09a77063b989'))
                st.caption(ui_message('ui.79a1b2a93eddc3'))
        return
    age_s = info.get("live_input_level_age_s")
    if age_s is not None and float(age_s) > 3.0 and _measurement_is_running(snapshot):
        st.warning(ui_message('ui.a70dc1c3f5afd1'))
    offset_db = level_calibration_offset_db(snapshot.settings)
    with st.container(border=True):
        st.markdown(ui_message('ui.3f09a77063b989'))
        st.caption(
            ui_message('ui.2ff266b8de07d1')
        )
        columns = st.columns(min(2, len(rms_levels)))
        for index, (rms_dbfs, peak_dbfs) in enumerate(zip(rms_levels, peak_levels)):
            if snapshot.settings.input_mode in {"dual_gain", "known_wav_dual_gain"}:
                label = "High-gain input" if index == 0 else "Low-gain input"
            else:
                label = f"Input ch {snapshot.settings.input_channel + index}"
            headroom_db = max(0.0, -peak_dbfs)
            calibrated_level = (
                f" · {rms_dbfs + offset_db:.1f} dB SPL"
                if offset_db is not None else ""
            )
            with columns[index % len(columns)]:
                st.progress(
                    _level_meter_fraction(rms_dbfs),
                    text=f"{label} · RMS {rms_dbfs:.1f} dBFS{calibrated_level}",
                )
                if peak_dbfs >= 20.0 * np.log10(snapshot.settings.clip_threshold):
                    st.error(ui_message('ui.fd35b79068d3bc', p0=f'{peak_dbfs:.1f}'), icon=":material/error:")
                elif headroom_db < snapshot.settings.minimum_headroom_db:
                    st.warning(
                        ui_message('ui.7819fb379c6e19', p0=f'{peak_dbfs:.1f}', p1=f'{headroom_db:.1f}'),
                        icon=":material/warning:",
                    )
                else:
                    st.caption(
                        ui_message('ui.3d0f01712d65be', p0=f'{peak_dbfs:.1f}', p1=f'{headroom_db:.1f}')
                    )


def _render_manual_microphone_gain(engine: ContinuousMeasurementEngine, snapshot: SessionSnapshot) -> None:
    gains = tuple(snapshot.settings.input_gain_db_channels)
    if not gains:
        return
    info = snapshot.device_info
    writable = bool(info.get("input_gain_writable", False))
    preparation_state = snapshot.state in {MeasurementState.PREFLIGHT, MeasurementState.SEARCHING}
    can_apply = writable and preparation_state
    current = float(np.mean(gains))
    minimums = tuple(float(value) for value in info.get("input_gain_min_db_channels", []))
    maximums = tuple(float(value) for value in info.get("input_gain_max_db_channels", []))
    target_min: float | None = None
    target_max: float | None = None
    if len(minimums) == len(gains) and len(maximums) == len(gains):
        target_min = current + max(minimum - gain for minimum, gain in zip(minimums, gains))
        target_max = current + min(maximum - gain for maximum, gain in zip(maximums, gains))

    popover_label = f"マイクGain {current:+.1f} dB"
    with st.popover(popover_label, icon=":material/mic:", width="stretch"):
        st.markdown(ui_message('ui.de4f454b13b610'))
        st.caption(ui_message('ui.e7d2c22928590c'))
        metric_columns = st.columns(3)
        metric_columns[0].metric(ui_message('ui.2acd804610f55e'), f"{current:+.1f} dB")
        metric_columns[1].metric(ui_message('ui.9267e2eb2d3ca4'), " / ".join(f"{value:+.1f}" for value in gains) + " dB")
        metric_columns[2].metric(ui_message('ui.e9ccfa0d73911b'), "CoreAudio書込み可" if writable else "読取りのみ")
        widget_revision = "_".join(f"{value:.2f}" for value in gains)
        with st.form(f"manual_microphone_gain_{snapshot.session_id}_{widget_revision}"):
            number_kwargs: dict[str, object] = {
                "value": current,
                "step": 0.5,
                "format": "%.1f",
                "disabled": not can_apply,
                "help": "絶対Gain値です。内部では現在値との差分へ変換し、全入力チャンネルへ同じ増減量を適用します。",
            }
            if target_min is not None and target_max is not None:
                number_kwargs.update(min_value=float(target_min), max_value=float(target_max))
            target = float(st.number_input(ui_message('ui.28b3e9f66eb08f'), **number_kwargs))
            apply_gain = st.form_submit_button(
                ui_message('ui.a1cf769c7bbf11'),
                type="primary",
                icon=":material/tune:",
                disabled=not can_apply,
                width="stretch",
            )
        if not writable:
            st.caption(ui_message('ui.d775aee659df0a'))
        elif apply_gain:
            try:
                changed = engine.set_microphone_gain_db(target)
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state[UNSAVED_KEY] = True
                st.toast(
                    "マイクGainを " + " / ".join(f"{value:+.1f} dB" for value in changed.channel_gain_db) + " に設定しました。",
                    icon=":material/check_circle:",
                )
                st.rerun(scope="fragment")


def _render_measurement_action_bar(
    engine: ContinuousMeasurementEngine | None,
    snapshot: SessionSnapshot | None,
) -> None:
    """Keep the primary measurement actions next to the preparation flow."""
    context = st.session_state.get("_ess_measure_action_context")
    if not isinstance(context, dict):
        return
    # The controls live in an auto-refresh fragment.  The settings area that
    # stored action_context does not rerun when this fragment starts the
    # engine, so its running flag can be stale.  Always derive button state
    # from the current engine snapshot instead.
    running = _measurement_is_running(snapshot)
    blockers = tuple(str(value) for value in context.get("start_blockers", ()))
    start_kwargs = context.get("start_kwargs")
    if blockers:
        st.session_state.pop(START_OPTIONS_KEY, None)
    pending = st.session_state.get(PENDING_START_KEY)
    pending_start = pending if isinstance(pending, dict) else None
    if pending_start is not None:
        remaining = max(0, int(np.ceil(float(pending_start["starts_at"]) - time.time())))
        if remaining <= 0:
            st.session_state.pop(PENDING_START_KEY, None)
            try:
                _start_engine(**pending_start["start_kwargs"])
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(ui_message('ui.9f1c1d7a32cfdb', p0=f'{exc}'), icon=":material/error:")
            return
    with st.container(border=True):
        st.markdown(ui_message('ui.447f884c57e4a9'))
        if pending_start is not None:
            st.info(
                ui_message('ui.ce3d9e3e56e6a7', p0=f'{remaining}'),
                icon=":material/timer:",
            )
            localized_button(
                st,
                ui_message('ui.0f2d44327bb1e5'),
                icon=":material/cancel:",
                on_click=_cancel_measurement_start,
            )
            return
        if st.session_state.get(START_OPTIONS_KEY, False):
            st.markdown(ui_message('ui.6ecdc60a1764dc'))
            delay_s = localized_segmented_control(
                ui_message('ui.db02b69158d3d9'),
                MEASUREMENT_START_DELAYS_S,
                default=5,
                required=True,
                format_func=lambda seconds: f"{seconds}s",
                key="ess_measure_start_delay_s",
                width="stretch",
            )
            st.caption(
                ui_message('ui.95a42d2bee2c69')
            )
            if _measurement_replacement_requires_confirmation(snapshot):
                save_state = "未保存" if st.session_state.get(UNSAVED_KEY, False) else "保存済み"
                st.warning(
                    ui_message('ui.5e9fa1ef6ea30e', p0=f'{len(snapshot.shots)}', p1=f'{save_state}')
                )
                if snapshot.autosave_directory is not None:
                    st.caption(ui_message('ui.edd1f0c8805e67', p0=f'{snapshot.autosave_directory}'))
            with st.container(horizontal=True, gap="small"):
                start_label = (
                    "Clear and start"
                    if _measurement_replacement_requires_confirmation(snapshot)
                    else "Start"
                )
                localized_button(
                    st,
                    start_label,
                    type="primary",
                    icon=":material/play_arrow:",
                    on_click=_queue_measurement_start,
                    args=(int(delay_s), start_kwargs),
                )
                localized_button(st, ui_message('ui.19766ed6ccb2f4'), on_click=_close_measurement_start_options)
            return
        with st.container(horizontal=True, gap="small"):
            localized_button(
                st,
                ui_message('ui.e4bb9f1ece9af9'),
                type="primary",
                icon=":material/play_arrow:",
                disabled=bool(blockers) or not isinstance(start_kwargs, dict),
                on_click=_open_measurement_start_options,
            )
            localized_button(
                st,
                ui_message('ui.62748653f6f921'),
                icon=":material/stop:",
                disabled=not running or engine is None,
                on_click=_request_measurement_stop,
            )
            localized_button(
                st,
                ui_message('ui.0d4f9a000e2aa9'),
                icon=":material/eject:",
                disabled=not running or engine is None,
                on_click=_request_measurement_abort,
            )
            save_menu = st.popover(
                ui_message('ui.1509f561f24165'),
                icon=":material/save:",
                disabled=snapshot is None or not snapshot.shots,
                help=ui_message('ui.b7a3222ea0690e'),
                width="content",
            )
            with save_menu:
                _render_measurement_file_save(snapshot, terminal=not running, measurement_db_path=context.get('measurement_db_path'))
        if blockers:
            st.warning(
                ui_message('ui.56bcd65707d82c')
                + "\n".join(f"- {reason}" for reason in blockers),
                icon=":material/info:",
            )
        elif not running:
            st.caption(ui_message('ui.3e4d45dd178ee0'))


def _open_measurement_start_options() -> None:
    st.session_state[START_OPTIONS_KEY] = True


def _close_measurement_start_options() -> None:
    st.session_state.pop(START_OPTIONS_KEY, None)


def _queue_measurement_start(delay_s: int, start_kwargs: dict[str, object]) -> None:
    st.session_state[PENDING_START_KEY] = {
        "starts_at": time.time() + int(delay_s),
        "start_kwargs": start_kwargs,
    }
    st.session_state.pop(START_OPTIONS_KEY, None)


def _cancel_measurement_start() -> None:
    st.session_state.pop(PENDING_START_KEY, None)


def _request_measurement_stop() -> None:
    engine = _engine()
    if engine is None:
        return
    request_stop = getattr(engine, "request_stop", None)
    if callable(request_stop):
        request_stop()
    else:
        # Preserve Stop for an engine instance created before a hot reload
        # introduced request_stop().  Its synchronous stop runs off the UI
        # thread so the callback still returns immediately.
        threading.Thread(target=engine.stop, name="ess-stop-legacy", daemon=True).start()
    st.session_state[UNSAVED_KEY] = True


def _request_measurement_abort() -> None:
    engine = _engine()
    if engine is None:
        return
    engine.abort()
    st.session_state[UNSAVED_KEY] = True


def _render_measurement_file_save(
    snapshot: SessionSnapshot | None,
    *,
    terminal: bool,
    measurement_db_path: Path | None = None,
) -> None:
    if snapshot is None or not snapshot.shots:
        st.caption(ui_message('ui.149fcf9d504634'))
        return
    if not terminal:
        st.caption(ui_message('ui.5d83f193bf51b4'))
        return
    shot_count = len(snapshot.shots)
    st.markdown(ui_message('ui.b4132f9b9b787b'))
    st.caption(
        ui_message('ui.173154c42bc2b0', p0=f"{('確定済みセッション' if terminal else '測定中のスナップショット')}", p1=f'{shot_count}')
    )
    st.download_button(
        ui_message('ui.cd839ccf8c731c'),
        data=(deferred_call(_measurement_package_with_sources, snapshot, measurement_db_path)
              if measurement_db_path is not None else deferred_call(session_zip_bytes, snapshot)),
        file_name=f"measurement_session_{snapshot.session_id[:8]}_{shot_count:04d}shots.zip",
        mime="application/zip",
        icon=":material/download:",
        width="stretch",
        key="measurement_session_zip_download",
        on_click="ignore",
    )
    st.caption(ui_message('ui.f8f2cd06673d1d'))


def _measurement_package_with_sources(snapshot: SessionSnapshot, path: Path) -> bytes:
    original = attach_calibration_sources(original_from_snapshot(snapshot), path)
    return session_zip_bytes(replace(snapshot, saved_original=original))


def render_measurement_controls(
    *,
    autosave_root: Path,
    measurement_db_path: Path,
    microphone_db_path: Path,
    ess_reference_db_path: Path,
) -> None:
    _hydrate_ess_reference_library_from_db(ess_reference_db_path)
    pending_restored_settings = st.session_state.pop("_ess_pending_restored_settings", None)
    if st.session_state.get("ess_measure_gate_default_revision") != 1:
        if pending_restored_settings is None:
            if st.session_state.get("ess_measure_gate_end") in {None, 20.0}:
                st.session_state["ess_measure_gate_end"] = 5.0
            if st.session_state.get("ess_measure_reprocess_scope") in {None, "All retained"}:
                st.session_state["ess_measure_reprocess_scope"] = "Latest"
        st.session_state["ess_measure_gate_default_revision"] = 1
    if isinstance(pending_restored_settings, MeasurementSettings):
        _apply_settings_to_measurement_widgets(pending_restored_settings, microphone_db_path)
    for key, default in {
        "ess_measure_gate_start": -2.0,
        "ess_measure_gate_start_widget": -2.0,
        "ess_measure_gate_end": 5.0,
        "ess_measure_merge_crossover": 0.0,
        "ess_measure_reprocess_scope": "Latest",
        "ess_measure_wavelet_n": 3,
        "ess_measure_quality_gate": "Standard",
        "ess_measure_forensic_raw": False,
        "ess_measure_calibration_extrapolation": "Hold edge",
        "ess_measure_clock_mode": "Auto",
        "ess_measure_ess_reference_source": "ESS Library file",
        "ess_measure_reference_mode": "IR peak — external ESS",
        "ess_measure_timing_session_id": "main-tweeter",
        "ess_measure_reference_tweeter_channel": "High",
        "ess_measure_timing_role": "Target Speaker",
        "ess_measure_reference_measurement_id": "",
        "ess_measure_sync_polarity": "Auto ±",
        "ess_measure_distance_m": 1.0,
        "ess_measure_level_calibration_mode": "Uncalibrated dBFS",
        "ess_measure_sensitivity_dbfs_pa_text": "-37.0",
        "ess_measure_sensitivity_mv_pa": 10.0,
        "ess_measure_interface_full_scale_vrms": 2.0,
        "ess_measure_input_gain_db": 0.0,
        "ess_measure_level_reference_spl_db": 94.0,
        "ess_measure_level_reference_dbfs_text": "-40.0",
        "ess_measure_level_calibration_note": "",
    }.items():
        st.session_state.setdefault(key, default)
    if "ess_measure_input_mode" in st.session_state:
        legacy_ess_reference = str(st.session_state.pop("ess_measure_input_mode"))
        st.session_state["ess_measure_ess_reference_source"] = (
            "ESS Library file"
            if legacy_ess_reference == "USB dual-gain mic + known WAV" else
            "PhaseEQ nominal ESS"
        )
    engine = _engine()
    snapshot = engine.snapshot() if engine is not None else None
    running = _measurement_is_running(snapshot)
    status_pilots = list(snapshot.device_info.get("pilot_history", [])) if snapshot else []
    latest_status_pilot = status_pilots[-1] if status_pilots else {}
    pilot_sync_status = (
        f"{latest_status_pilot.get('synchronization', 'ESS correlation')} / "
        f"corr {float(latest_status_pilot.get('ess_correlation', 0.0)):.3f}"
        if latest_status_pilot else "Waiting"
    )
    st.markdown(ui_message('ui.fc05ee8cdec7fc'))
    status_cols = st.columns(3)
    status_cols[0].metric(ui_message('ui.a3b50c476732c7'), str(snapshot.state if snapshot else MeasurementState.IDLE))
    status_cols[1].metric(ui_message('ui.47758c428dbfab'), len(snapshot.shots) if snapshot else 0)
    status_cols[2].metric(ui_message('ui.400b505d77ef9f'), len(snapshot.valid_shots) if snapshot else 0)
    if snapshot and snapshot.device_info.get("last_rejection_reason"):
        st.caption(ui_message('ui.7202847aec7bc5', p0=f"{snapshot.device_info['last_rejection_reason']}"))
    measurement_actions_slot = st.empty()
    next_shot_protection = (
        dict(snapshot.device_info.get("last_next_shot_protection", {}))
        if snapshot else {}
    )
    next_shot_protection_status = (
        f"{float(next_shot_protection.get('predicted_overlap_ms', 0.0)):.1f} msを除外 / "
        f"capture {float(next_shot_protection.get('protected_capture_ms', 0.0)):.1f} ms"
        if next_shot_protection else "No overlap detected"
    )
    with st.expander(ui_message('ui.749f55f2557305'), icon=":material/monitoring:"):
        st.dataframe(
            pd.DataFrame([
                {"Item": "Pilot synchronization", "Value": pilot_sync_status},
                {"Item": "Sweeps processed", "Value": str(snapshot.device_info.get("detected_sweeps", 0) if snapshot else 0)},
                {"Item": "Rejected", "Value": str(snapshot.device_info.get("rejected_shots", 0) if snapshot else 0)},
                {"Item": "Audio warning", "Value": str(snapshot.device_info.get("last_audio_status", "—") or "—") if snapshot else "—"},
                {"Item": "CoreAudio recovery", "Value": (
                    f"{snapshot.device_info.get('audio_recovery_state', 'none')} / "
                    f"{snapshot.device_info.get('audio_recovery_count', 0)} time(s)"
                    if snapshot else "none / 0 time(s)"
                )},
                {"Item": "Expected-period ESS locks", "Value": str(snapshot.device_info.get("expected_ess_locks", 0) if snapshot else 0)},
                {"Item": "Marker validation failures", "Value": str(snapshot.device_info.get("timing_marker_validation_failures", 0) if snapshot else 0)},
                {"Item": "Timing resynchronizations", "Value": str(snapshot.device_info.get("timing_resynchronizations", 0) if snapshot else 0)},
                {"Item": "Last recovery", "Value": str(snapshot.device_info.get("last_recovery_status", "—") or "—") if snapshot else "—"},
                {"Item": "Next-shot protection", "Value": next_shot_protection_status},
                {"Item": "Autosave", "Value": str(snapshot.autosave_directory) if snapshot else "Ready"},
            ]),
            hide_index=True,
            width="stretch",
        )

    category_key = "ess_measure_category_tabs"
    st.markdown(
        f"<style>{tab_hover_translation_css(category_key, MEASUREMENT_CATEGORY_OPTIONS, bordered=True)}</style>",
        unsafe_allow_html=True,
    )
    category_tabs = st.tabs(MEASUREMENT_CATEGORY_OPTIONS, key=category_key)

    with category_tabs[0]:
        st.markdown(ui_message('ui.b37cf520ced7d7'))
        st.caption(ui_message('ui.f4008303cd286d'))
        sample_rate = st.selectbox(
            ui_message('ui.218e0d43275e43'),
            [48_000, 96_000, 192_000],
            key="ess_measure_sample_rate",
            disabled=running,
            format_func=localized_formatter(lambda value: f'{value / 1000:g} kHz'),
        )
        microphone_profiles = list_microphone_profiles(microphone_db_path)
        profile_by_id = {profile.id: profile for profile in microphone_profiles}
        saved_profile_id = str(st.session_state.get("ess_measure_microphone_profile_id", "auto"))
        saved_profile = profile_by_id.get(saved_profile_id)
        st.session_state.setdefault(
            "ess_measure_input_hardware_type",
            (
                "Analog microphone + audio interface"
                if saved_profile is not None and saved_profile.connection_type == "analog" else
                "USB microphone"
            ),
        )
        input_hardware_type = localized_segmented_control(
            ui_message('ui.c70d426b1e780b'),
            ["USB microphone", "Analog microphone + audio interface"],
            key="ess_measure_input_hardware_type",
            disabled=running,
            width="stretch",
            help=(
                ui_message('ui.249a38beeca8db')
            ),
        )
        ess_reference_label = localized_segmented_control(
            ui_message('ui.3bf24cf82d05d9'),
            ["ESS Library file", "PhaseEQ nominal ESS"],
            key="ess_measure_ess_reference_source",
            disabled=running,
            width="stretch",
            help=(
                ui_message('ui.618589c70f8399')
            ),
        )
        known_wav_mode = ess_reference_label == "ESS Library file"
        requested_connection_type = (
            "analog" if input_hardware_type == "Analog microphone + audio interface" else "usb"
        )
        compatible_profile_ids = [
            profile.id for profile in microphone_profiles
            if profile.connection_type == requested_connection_type
        ]
        profile_options = (
            ["auto", *compatible_profile_ids]
            if requested_connection_type == "usb" else
            compatible_profile_ids or [""]
        )
        st.session_state.setdefault("ess_measure_microphone_profile_id", "auto")
        if st.session_state["ess_measure_microphone_profile_id"] not in profile_options:
            st.session_state["ess_measure_microphone_profile_id"] = profile_options[0]
        selected_profile_id = st.selectbox(
            ui_message('ui.ca5e96f58de2c0'),
            profile_options,
            key="ess_measure_microphone_profile_id",
            disabled=running,
            format_func=lambda value: (
                "Auto-detect USB microphone"
                if value == "auto" else
                "No analog microphone profile"
                if not value else
                f"{profile_by_id[value].manufacturer} {profile_by_id[value].model}"
            ),
            help=ui_message('ui.3e8bcacbe0f8ba'),
        )
        analog_profile_selected = requested_connection_type == "analog"

        refresh_requested = localized_button(st,
            ui_message('ui.18ea204d7d0148'),
            key="ess_measure_refresh_input_devices",
            icon=":material/refresh:",
            disabled=running,
            help=ui_message('ui.3f57ecb41e9366'),
        )
        refresh_error = ""
        if refresh_requested:
            try:
                refresh_audio_devices()
                CacheCoordinator(st.session_state).notify(CacheEvent.AUDIO_DEVICES_REFRESHED)
            except RuntimeError as exc:
                refresh_error = str(exc)

        devices = list_input_devices(1)
        device_options: list[int | None] = [int(item["index"]) for item in devices] or [None]
        measurement_mics: list[int] = []
        if requested_connection_type == "usb":
            measurement_mics = [
                value for value in device_options
                if value is not None and any(token in str(next(item["name"] for item in devices if int(item["index"]) == value)).lower() for token in ("omnimic", "umik"))
            ]
            device_options = [*measurement_mics, *[value for value in device_options if value not in measurement_mics]]
        device_labels = {
            int(item["index"]): (
                f'{item["index"]}: {item["name"]} / {int(item["max_input_channels"])} ch'
                f' / default {float(item.get("default_samplerate", 0)) / 1000:g} kHz'
            )
            for item in devices
        }
        device_labels[None] = "No input device"
        if refresh_requested and measurement_mics:
            st.session_state["ess_measure_input_device"] = measurement_mics[0]
        elif st.session_state.get("ess_measure_input_device") not in device_options:
            st.session_state["ess_measure_input_device"] = device_options[0]
        input_device = st.selectbox(
            ui_message('ui.ca4907012e4740') if analog_profile_selected else ui_message('ui.bc2582c9e8809a'),
            device_options,
            key="ess_measure_input_device",
            disabled=running,
            format_func=lambda value: device_labels[value],
            help=ui_message('ui.2c331993ee8365'),
        )
        if not devices:
            st.error(
                ui_message('ui.ac2869d23b2008')
            )
        if refresh_error:
            st.error(refresh_error)
        elif refresh_requested:
            st.success(ui_message('ui.3b5917137abdd2', p0=f'{len(devices)}'))

        selected_device = next(
            (item for item in devices if input_device is not None and int(item["index"]) == int(input_device)),
            None,
        )
        device_name = str(selected_device.get("name", "")) if selected_device else ""
        detected_profile = match_microphone_profile(microphone_db_path, device_name) if device_name else None
        microphone_profile = (
            detected_profile if selected_profile_id == "auto" else profile_by_id.get(str(selected_profile_id))
        )
        analysis_channels = microphone_profile.input_channels if microphone_profile is not None else 2
        max_device_channels = int(selected_device.get("max_input_channels", 0)) if selected_device else 0
        if microphone_profile is not None and microphone_profile.connection_type == "analog":
            input_channel_max = max(1, max_device_channels)
            current_channel = min(
                max(1, int(st.session_state.get("ess_measure_input_channel", 1))),
                input_channel_max,
            )
            st.session_state["ess_measure_input_channel"] = current_channel
            input_channel = int(st.number_input(
                ui_message('ui.61a2a53656ee57'),
                min_value=1,
                max_value=input_channel_max,
                step=1,
                key="ess_measure_input_channel",
                disabled=running or selected_device is None,
                help=ui_message('ui.ff28f8701fa193'),
            ))
        else:
            input_channel = 1
        stream_channels = input_channel if analysis_channels == 1 else analysis_channels
        device_channel_valid = selected_device is not None and max_device_channels >= stream_channels
        format_signature = CacheCoordinator(st.session_state).key(
            CacheDomain.AUDIO_CAPABILITY,
            algorithm_version=AUDIO_CAPABILITY_CACHE_ALGORITHM_VERSION,
            source_digest=content_digest(
                int(input_device) if input_device is not None else "none",
                device_name,
                selected_device.get("hostapi", "") if selected_device else "",
            ),
            settings_digest=content_digest(sample_rate, stream_channels),
        ).digest
        cached_format_check = st.session_state.get("_ess_measure_device_format_check")
        if device_channel_valid and (
            not isinstance(cached_format_check, tuple)
            or cached_format_check[0] != format_signature
        ):
            device_format_error = input_settings_error(input_device, int(sample_rate), stream_channels)
            st.session_state["_ess_measure_device_format_check"] = (format_signature, device_format_error)
        elif device_channel_valid and isinstance(cached_format_check, tuple):
            device_format_error = str(cached_format_check[1])
        else:
            device_format_error = ""
        device_format_valid = device_channel_valid and not device_format_error
        profile_rate_valid = (
            microphone_profile is None
            or not microphone_profile.sample_rates_hz
            or int(sample_rate) in microphone_profile.sample_rates_hz
        )
        if microphone_profile is None and device_name:
            st.info(ui_message('ui.d5d4bb72b2fe41'))
        elif microphone_profile is not None:
            allowed_rates = (
                " / ".join(f"{value / 1000:g} kHz" for value in microphone_profile.sample_rates_hz)
                if microphone_profile.sample_rates_hz else
                "audio interface dependent"
            )
            bit_depth = (
                f"{microphone_profile.nominal_bit_depth} bit"
                if microphone_profile.nominal_bit_depth else
                "audio interface dependent"
            )
            st.caption(
                ui_message('ui.fe85438a0f5711', p0=f'{microphone_profile.manufacturer}', p1=f'{microphone_profile.model}', p2=f'{allowed_rates}', p3=f'{bit_depth}', p4=f'{microphone_profile.input_channels}')
            )
            recording_layout = (
                f"Mono recording · physical input ch {input_channel}"
                if analysis_channels == 1 else
                "2-channel dual-gain recording · same microphone signal"
            )
            st.caption(
                ui_message('ui.5928dc70dcb864', p0=f'{recording_layout}', p1=f'{ess_reference_label}')
            )
            capability_items = []
            if microphone_profile.frequency_min_hz and microphone_profile.frequency_max_hz:
                capability_items.append(
                    f"{microphone_profile.frequency_min_hz:g}–{microphone_profile.frequency_max_hz:g} Hz"
                )
            if microphone_profile.calibrated_accuracy_db:
                capability_items.append(f"校正適用時 ±{microphone_profile.calibrated_accuracy_db:g} dB")
            if microphone_profile.max_spl_db:
                capability_items.append(f"Max {microphone_profile.max_spl_db:g} dB SPL")
            if microphone_profile.adc_dynamic_range_db:
                capability_items.append(f"ADC DR {microphone_profile.adc_dynamic_range_db:g} dB")
            if capability_items:
                st.caption(ui_message('ui.1262fb2658e4b1') + " | ".join(capability_items))
            if not profile_rate_valid:
                st.error(
                    ui_message('ui.4add3b2b95f65c', p0=f'{microphone_profile.model}', p1=f'{allowed_rates}')
                )
            if not device_channel_valid:
                st.error(ui_message('ui.cfbf51b396e839', p0=f'{stream_channels}'))
        if device_format_error:
            st.error(
                ui_message('ui.ca66d0386b8a44', p0=f'{device_name}', p1=f'{stream_channels}', p2=f'{int(sample_rate) / 1000:g}', p3=f'{device_format_error}')
            )

        microphone_units = (
            list_microphone_units(microphone_db_path, profile_id=microphone_profile.id)
            if microphone_profile is not None else []
        )
        unit_by_id = {unit.id: unit for unit in microphone_units}
        unit_ids = ["", *unit_by_id]
        pending_unit_id = st.session_state.pop("_ess_measure_pending_unit_id", "")
        if pending_unit_id in unit_by_id:
            st.session_state["ess_measure_microphone_unit_id"] = pending_unit_id
        if st.session_state.get("ess_measure_microphone_unit_id", "") not in unit_ids:
            st.session_state["ess_measure_microphone_unit_id"] = microphone_units[0].id if len(microphone_units) == 1 else ""
        microphone_unit_id = st.selectbox(
            ui_message('ui.acccddea4d0ead'),
            unit_ids,
            key="ess_measure_microphone_unit_id",
            disabled=running or microphone_profile is None,
            format_func=lambda value: (
                "Not registered"
                if not value else
                f"{unit_by_id[value].label + ' / ' if unit_by_id[value].label else ''}S/N {unit_by_id[value].serial_number}"
            ),
            help=ui_message('ui.8118f916804ec8'),
        )
        microphone_unit = unit_by_id.get(microphone_unit_id)
        with st.expander(ui_message('ui.e8c2cc58c24a44')):
            if st.session_state.pop("_ess_measure_clear_unit_form", False):
                st.session_state["ess_measure_microphone_serial"] = ""
                st.session_state["ess_measure_microphone_label"] = ""
            unit_serial = st.text_input(
                ui_message('ui.ddcd0b81ab927c'),
                key="ess_measure_microphone_serial",
                disabled=running or microphone_profile is None,
                placeholder=(microphone_profile.serial_format_hint if microphone_profile is not None else ""),
            )
            unit_label = st.text_input(
                ui_message('ui.3a282e8fdc7284'),
                key="ess_measure_microphone_label",
                disabled=running or microphone_profile is None,
            )
            if localized_button(st,
                ui_message('ui.e55bd4d9e200fd'),
                disabled=running or microphone_profile is None or not str(unit_serial).strip(),
                width="stretch",
            ) and microphone_profile is not None:
                try:
                    saved_unit = save_microphone_unit(
                        microphone_db_path,
                        MicrophoneUnit(
                            id="",
                            profile_id=microphone_profile.id,
                            serial_number=str(unit_serial),
                            label=str(unit_label),
                            device_name=device_name,
                            input_channel=input_channel,
                        ),
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["_ess_measure_pending_unit_id"] = saved_unit.id
                    st.session_state["_ess_measure_clear_unit_form"] = True
                    st.rerun()

        if microphone_profile is not None and microphone_profile.id in OFFICIAL_CALIBRATION_SOURCES:
            _render_official_calibration_assistant(
                profile=microphone_profile,
                selected_unit=microphone_unit,
                device_name=device_name,
                input_channel=input_channel,
                running=running,
                measurement_db_path=measurement_db_path,
                microphone_db_path=microphone_db_path,
            )

        calibration_records = [
            record for record in list_measurement_summaries(measurement_db_path, limit=10_000)
            if record.source_type == "mic_calibration_raw"
        ]
        calibration_ids = ["", *[record.id for record in calibration_records]]
        calibration_by_id = {record.id: record for record in calibration_records}
        calibration_angles = microphone_profile.calibration_angles_deg if microphone_profile is not None else (0,)
        if (
            "ess_measure_calibration_angle" not in st.session_state
            or st.session_state.get("ess_measure_calibration_angle") not in calibration_angles
        ):
            st.session_state["ess_measure_calibration_angle"] = calibration_angles[0]
        if len(calibration_angles) > 1:
            calibration_angle = int(localized_segmented_control(
                ui_message('ui.6b2d3d97f2c2e3'),
                list(calibration_angles),
                key="ess_measure_calibration_angle",
                disabled=running,
                format_func=lambda value: f"{value}°",
                width="stretch",
                help=MIC_CALIBRATION_HELP,
            ))
        else:
            calibration_angle = int(calibration_angles[0])
        linked_calibration_id = (
            microphone_unit.calibration_90_measurement_id
            if microphone_unit is not None and calibration_angle == 90 else
            microphone_unit.calibration_30_measurement_id
            if microphone_unit is not None and calibration_angle == 30 else
            microphone_unit.calibration_measurement_id if microphone_unit is not None else ""
        )
        auto_match_marker = st.session_state.get("_ess_measure_calibration_auto_unit", "")
        match_marker = f"{microphone_unit.id}:{calibration_angle}" if microphone_unit is not None else ""
        if microphone_unit is not None and auto_match_marker != match_marker:
            matched_ids = [
                record.id for record in calibration_records
                if calibration_text_matches_serial(
                    microphone_unit.serial_number,
                    record.name,
                    record.source_name,
                    record.microphone,
                    record.note,
                )
                and calibration_text_matches_angle(
                    calibration_angle,
                    record.name,
                    record.source_name,
                    record.microphone,
                    record.note,
                )
                and "phase" not in " ".join((record.name, record.source_name, record.microphone, record.note)).casefold()
            ]
            if linked_calibration_id in calibration_by_id:
                st.session_state["ess_measure_mic_calibration_id"] = linked_calibration_id
            elif len(matched_ids) == 1:
                st.session_state["ess_measure_mic_calibration_id"] = matched_ids[0]
            else:
                st.session_state["ess_measure_mic_calibration_id"] = ""
            st.session_state["_ess_measure_calibration_auto_unit"] = match_marker
        if st.session_state.get("ess_measure_mic_calibration_id", "") not in calibration_ids:
            st.session_state["ess_measure_mic_calibration_id"] = ""
        calibration_id = st.selectbox(
            ui_message('ui.17a20b3b7b320c', p0=f'{calibration_angle}'),
            calibration_ids,
            key="ess_measure_mic_calibration_id",
            disabled=running,
            format_func=lambda value: "None" if not value else calibration_by_id[value].name,
            help=MIC_CALIBRATION_HELP,
        )
        # Decode the selected curve and its paired calibration only, not all
        # measurements while populating the microphone menu.
        linked_ids = {calibration_id}
        if microphone_unit is not None:
            linked_ids.update((microphone_unit.calibration_measurement_id,
                               microphone_unit.calibration_90_measurement_id))
        for linked_id in linked_ids:
            if linked_id in calibration_by_id:
                record = get_measurement(measurement_db_path, linked_id)
                if record is not None:
                    calibration_by_id[linked_id] = record
        selected_calibration_response = (
            speaker_response_from_payload(calibration_by_id[calibration_id].response_payload)
            if calibration_id in calibration_by_id else None
        )
        mic_calibration_valid = bool(
            calibration_id
            and selected_calibration_response is not None
            and len(selected_calibration_response.frequency) >= 2
            and len(selected_calibration_response.gain_db) == len(selected_calibration_response.frequency)
            and np.all(np.isfinite(selected_calibration_response.frequency))
            and np.all(np.asarray(selected_calibration_response.frequency) > 0.0)
            and np.all(np.isfinite(selected_calibration_response.gain_db))
        )
        if mic_calibration_valid:
            st.success(
                ui_message('ui.283234c84001db', p0=f'{calibration_by_id[calibration_id].name}', p1=f'{calibration_angle}')
            )
        else:
            st.error(
                ui_message('ui.598f92c99af4ef')
            )
        calibration_extrapolation = localized_segmented_control(
            ui_message('ui.4e0e12d84f5ce1'),
            ["Hold edge", "No correction", "Manual / extended file"],
            key="ess_measure_calibration_extrapolation",
            disabled=running,
            width="stretch",
            help=(
                ui_message('ui.ba15fce2ca0ea2')
            ),
        )
        if microphone_unit is not None:
            linked = linked_calibration_id == calibration_id and bool(calibration_id)
            if linked:
                st.caption(ui_message('ui.193e118ad5298a', p0=f'{microphone_unit.serial_number}', p1=f'{calibration_angle}'))
            elif localized_button(st,
                ui_message('ui.c928e0777f1df8', p0=f'{calibration_angle}'),
                disabled=running or not calibration_id,
                width="stretch",
                help=ui_message('ui.4ea111f1e520c2'),
            ):
                unit_update = (
                    replace(microphone_unit, calibration_90_measurement_id=str(calibration_id), device_name=device_name)
                    if calibration_angle == 90 else
                    replace(microphone_unit, calibration_30_measurement_id=str(calibration_id), device_name=device_name)
                    if calibration_angle == 30 else
                    replace(microphone_unit, calibration_measurement_id=str(calibration_id), device_name=device_name)
                )
                save_microphone_unit(
                    microphone_db_path,
                    unit_update,
                )
                st.success(ui_message('ui.4162ebb6d2b866', p0=f'{microphone_unit.serial_number}', p1=f'{calibration_angle}'))
                st.rerun()

        phase_calibration_id = ""
        phase_supported = (
            microphone_profile is not None
            and int(sample_rate) in microphone_profile.phase_calibration_sample_rates_hz
        )
        if phase_supported and calibration_id:
            phase_response = speaker_response_from_payload(calibration_by_id[calibration_id].response_payload)
            if phase_response is not None and phase_response.phase_deg is not None:
                phase_calibration_id = calibration_id
                st.caption(ui_message('ui.aa21ecde1c9109', p0=f'{sample_rate / 1000:g}'))
            else:
                st.caption(ui_message('ui.599ecd426b7552'))

        minidsp_level = _resolve_minidsp_level_calibration(
            profile=microphone_profile,
            unit=microphone_unit,
            calibration_id=str(calibration_id),
            records=calibration_by_id,
            device_name=device_name,
            channels=analysis_channels,
        )
        if minidsp_level["frequency_status"]:
            st.success(str(minidsp_level["frequency_status"]))
        if minidsp_level["factor"] is not None:
            gain_text = (
                " / ".join(f"{value:+g} dB" for value in minidsp_level["gain_snapshot"].channel_gain_db)
                if minidsp_level["gain_snapshot"].available else
                "unavailable"
            )
            if minidsp_level["available"]:
                st.success(
                    ui_message('ui.43c3d7dfee48e3', p0=f"{float(minidsp_level['factor']):+g}", p1=f'{gain_text}')
                )
            else:
                st.warning(
                    ui_message('ui.b1ebd8ce7ef770', p0=f"{minidsp_level['reason']}")
                )
            gain_snapshot = minidsp_level["gain_snapshot"]
            if gain_snapshot.available:
                st.caption(
                    ui_message('ui.18b8b722d14607')
                    + (ui_message('ui.1ea509f5679495') if gain_snapshot.writable else ui_message('ui.1017c8608ab2b6'))
                )

    with category_tabs[1]:
        known_reference, reference_rejection_reasons = _render_ess_source_manager(
            analysis_sample_rate=int(sample_rate),
            reference_required=known_wav_mode,
            disabled=running,
        )
        st.markdown(ui_message('ui.e573d1867f3235'))
        reference_mode_label = localized_segmented_control(
            ui_message('ui.aa5925c1b52a95'),
            [
                "IR peak — external ESS",
                "Timing marker — external ESS",
                "Tweeter reference — external ESS",
            ],
            key="ess_measure_reference_mode",
            disabled=running,
            width="stretch",
            help=(
                ui_message('ui.741d58886ab462')
            ),
        )
        tweeter_reference_mode = reference_mode_label == "Tweeter reference — external ESS"
        timing_reference_ready = True
        timing_reference_session_id = ""
        reference_tweeter_channel_id = ""
        reference_measurement_id = ""
        reference_arrival_sample = float("nan")
        timing_reference_role = "target_speaker"
        reference_output_channel = ""
        measurement_output_channel = ""
        reference_routing_verified = False
        external_playback_setup_confirmed = False
        reference_protection_confirmed = False
        alignment_processing_bypassed_confirmed = False
        external_playback_setup_note = ""
        if tweeter_reference_mode:
            st.info(
                ui_message('ui.8988e7fef8b423')
            )
            routing_reasons = (
                tweeter_reference_routing_rejection_reasons(known_reference)
                if known_reference is not None else
                ("距離測定用のPhaseEQ ESSとmanifestを選択してください。",)
            )
            reference_routing_verified = not routing_reasons
            if known_reference is not None:
                reference_output_channel = known_reference.timing_marker_output_channels
                measurement_output_channel = known_reference.ess_output_channels
            setup_fingerprint = hashlib.sha256(json.dumps({
                "source": known_reference.source_name if known_reference is not None else "",
                "sample_rate": known_reference.source_sample_rate if known_reference is not None else 0,
                "reference_output": reference_output_channel,
                "measurement_output": measurement_output_channel,
                "routing_verified": reference_routing_verified,
            }, sort_keys=True).encode("utf-8")).hexdigest()
            if st.session_state.get("_ess_tweeter_setup_fingerprint") != setup_fingerprint:
                st.session_state["_ess_tweeter_setup_fingerprint"] = setup_fingerprint
                resumed_route = st.session_state.pop("_ess_tweeter_resumed_route", None)
                preserve_resumed_checks = resumed_route == (
                    known_reference.source_name if known_reference is not None else "",
                    reference_output_channel,
                    measurement_output_channel,
                )
                if not preserve_resumed_checks:
                    for key in (
                        "ess_measure_confirm_external_routing",
                        "ess_measure_confirm_reference_protection",
                        "ess_measure_confirm_alignment_bypass",
                    ):
                        st.session_state[key] = False
            with st.container(border=True):
                st.markdown(ui_message('ui.09418401767c42'))
                if routing_reasons:
                    st.error(
                        ui_message('ui.308c1ac0a9deab')
                        + "\n".join(f"- {reason}" for reason in routing_reasons),
                        icon=":material/block:",
                    )
                else:
                    reference_label = "L" if reference_output_channel == "left" else "R"
                    measurement_label = "L" if measurement_output_channel == "left" else "R"
                    st.success(
                        ui_message('ui.07ff04c34ea2ca', p0=f'{reference_label}', p1=f'{measurement_label}'),
                        icon=":material/cable:",
                    )
                external_playback_setup_confirmed = st.checkbox(
                    ui_message('ui.794108c688dc12'),
                    key="ess_measure_confirm_external_routing",
                    disabled=running or not reference_routing_verified,
                )
                reference_protection_confirmed = st.checkbox(
                    ui_message('ui.ebe8fb948c73e5'),
                    key="ess_measure_confirm_reference_protection",
                    disabled=running or not reference_routing_verified,
                )
                alignment_processing_bypassed_confirmed = st.checkbox(
                    ui_message('ui.9452316f385b54'),
                    key="ess_measure_confirm_alignment_bypass",
                    disabled=running or not reference_routing_verified,
                    help=(
                        ui_message('ui.2877b122f622ed')
                    ),
                )
                external_playback_setup_note = st.text_area(
                    ui_message('ui.085ba50101e855'),
                    key="ess_measure_external_setup_note",
                    disabled=running,
                    placeholder=ui_message('ui.05f572dc5a6cc2'),
                    help=ui_message('ui.785c2479d36e6f'),
                ).strip()
                st.caption(
                    ui_message('ui.d080b6cfca66a3')
                )
            reference_records: dict[
                str, tuple[SpeakerMeasurementRecord, TimingProvenance]
            ] = {}
            rejected_reference_count = 0
            for candidate in list_measurement_summaries(measurement_db_path, limit=10_000):
                provenance = timing_provenance_from_measurement_payload(candidate.response_payload)
                trusted_reference = trusted_reference_provenance(provenance)
                if trusted_reference is not None and (
                    trusted_reference.source_measurement_id == candidate.id
                    and trusted_reference.reference_measurement_id == candidate.id
                ):
                    reference_records[candidate.id] = (candidate, trusted_reference)
                elif provenance is not None and provenance.get(
                    "timing_reference_kind"
                ) == "phaseeq_tweeter_reference":
                    rejected_reference_count += 1
            with st.container(border=True):
                timing_reference_role_label = localized_segmented_control(
                    ui_message('ui.1fe060d98ac071'),
                    ["Reference Tweeter", "Target Speaker"],
                    key="ess_measure_timing_role",
                    disabled=running,
                    width="stretch",
                )
                timing_reference_role = (
                    "reference_tweeter"
                    if timing_reference_role_label == "Reference Tweeter"
                    else "target_speaker"
                )
                if timing_reference_role == "reference_tweeter":
                    timing_reference_session_id = st.text_input(
                        ui_message('ui.e0e62657ad70bc'),
                        key="ess_measure_timing_session_id",
                        disabled=running,
                        help=ui_message('ui.abecb5916eb6ff'),
                    ).strip()
                    reference_tweeter_channel_id = st.text_input(
                        ui_message('ui.cdb694ab31e124'),
                        key="ess_measure_reference_tweeter_channel",
                        disabled=running,
                    ).strip()
                    timing_reference_ready = bool(
                        timing_reference_session_id and reference_tweeter_channel_id
                    )
                    st.caption(ui_message('ui.f05b07710d6e5c'))
                else:
                    candidate_ids = ["", *reference_records]
                    requested_reference_id = str(
                        st.session_state.get("ess_measure_reference_measurement_id", "")
                    ).strip()
                    reference_selection_context = hashlib.sha256(
                        json.dumps({
                            "candidate_ids": list(reference_records),
                            "timing_session_id": str(st.session_state.get(
                                "ess_measure_timing_session_id", "",
                            )).strip(),
                            "reference_channel": str(st.session_state.get(
                                "ess_measure_reference_tweeter_channel", "",
                            )).strip(),
                        }, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    missing_requested_reference = bool(
                        requested_reference_id
                        and requested_reference_id not in candidate_ids
                    )
                    if missing_requested_reference:
                        st.session_state["ess_measure_reference_measurement_id"] = ""
                    reference_auto_selected = False
                    if (
                        not missing_requested_reference
                        and not st.session_state.get("ess_measure_reference_measurement_id")
                        and st.session_state.get(
                            "_ess_reference_auto_selection_context"
                        ) != reference_selection_context
                    ):
                        preferred = [
                            candidate_id for candidate_id, (_record, provenance) in reference_records.items()
                            if provenance.timing_reference_session_id
                            == str(st.session_state.get("ess_measure_timing_session_id", "")).strip()
                            and provenance.reference_tweeter_channel_id
                            == str(st.session_state.get("ess_measure_reference_tweeter_channel", "")).strip()
                        ]
                        safe_candidates = preferred if len(preferred) == 1 else list(reference_records)
                        if len(safe_candidates) == 1:
                            st.session_state["ess_measure_reference_measurement_id"] = safe_candidates[0]
                            reference_auto_selected = True
                    st.session_state[
                        "_ess_reference_auto_selection_context"
                    ] = reference_selection_context
                    selected_reference_id = st.selectbox(
                        ui_message('ui.1aa1d1a9c364df'),
                        candidate_ids,
                        key="ess_measure_reference_measurement_id",
                        disabled=running,
                        format_func=lambda value: (
                            "基準測定を選択"
                            if not value else reference_records[value][0].name
                        ),
                    )
                    if missing_requested_reference:
                        st.warning(
                            ui_message('ui.df112a4c3c92b9')
                        )
                    if selected_reference_id:
                        _record, reference_provenance = reference_records[selected_reference_id]
                        reference_measurement_id = selected_reference_id
                        timing_reference_session_id = (
                            reference_provenance.timing_reference_session_id
                        )
                        reference_tweeter_channel_id = (
                            reference_provenance.reference_tweeter_channel_id
                        )
                        converted_reference_arrival = reference_arrival_samples_at_rate(
                            reference_provenance,
                            sample_rate_hz=int(sample_rate),
                        )
                        reference_arrival_sample = (
                            float(converted_reference_arrival)
                            if converted_reference_arrival is not None else float("nan")
                        )
                        timing_reference_ready = bool(
                            timing_reference_session_id
                            and reference_tweeter_channel_id
                            and np.isfinite(reference_arrival_sample)
                        )
                        if reference_auto_selected:
                            st.success(ui_message('ui.c5670f0c3b0b55'))
                        st.caption(
                            ui_message('ui.d40b78eb2f5663', p0=f'{timing_reference_session_id}', p1=f'{reference_tweeter_channel_id}', p2=f'{reference_arrival_sample:.3f}', p3=f'{int(sample_rate):,}')
                        )
                        if reference_provenance.sample_rate_hz != int(sample_rate):
                            st.caption(
                                ui_message('ui.5a0801c7b81493', p0=f'{reference_provenance.sample_rate_hz:,}', p1=f'{int(sample_rate):,}')
                            )
                    else:
                        timing_reference_ready = False
                        st.warning(ui_message('ui.c45ed45acfa385'))
                    if rejected_reference_count:
                        st.caption(
                            ui_message('ui.46a8ef94601bd6', p0=f'{rejected_reference_count}')
                        )
            timing_reference_ready = bool(
                timing_reference_ready
                and reference_routing_verified
                and external_playback_setup_confirmed
                and reference_protection_confirmed
                and alignment_processing_bypassed_confirmed
            )
        sync_polarity_label = localized_segmented_control(
            ui_message('ui.c3ea77d82425bd'),
            ["Auto ±", "Positive +", "Negative −"],
            key="ess_measure_sync_polarity",
            disabled=running,
            width="stretch",
            help=ui_message('ui.a71eac4be06882'),
        )
        if reference_mode_label == "IR peak — external ESS":
            st.info(ui_message('ui.20a9bb4d41b4e2'))
        elif reference_mode_label != "IR peak — external ESS" and known_reference is not None and not known_reference.has_timing_marker:
            st.warning(ui_message('ui.d884def9560fc9'))

    with category_tabs[2]:
        st.markdown(ui_message('ui.3f31c46c7e2269'))
        measurement_distance_m = float(st.number_input(
            ui_message('ui.89ef7b37a61b74'),
            min_value=0.001,
            step=0.01,
            key="ess_measure_distance_m",
            disabled=running,
            help=ui_message('ui.bc5fb021541d86'),
        ))
        level_mode_options = ["Uncalibrated dBFS"]
        if minidsp_level["available"]:
            level_mode_options.append("UMIK manufacturer Sens Factor")
        level_mode_options.append("USB mic sensitivity dBFS/Pa")
        if microphone_profile is not None and microphone_profile.representative_sensitivity_dbfs_94db:
            level_mode_options.append("Estimated USB representative sensitivity")
        level_mode_options.extend([
            "Analog mic sensitivity mV/Pa",
            "Acoustic calibrator",
            "SPL meter comparison",
        ])
        current_level_label = str(st.session_state.get("ess_measure_level_calibration_mode", "Uncalibrated dBFS"))
        if running and current_level_label not in level_mode_options:
            level_mode_options.append(current_level_label)
        analog_interface_mode = bool(
            microphone_profile is not None
            and microphone_profile.connection_type == "analog"
        )
        auto_level_marker = (
            f"{microphone_profile.id if microphone_profile else ''}:"
            f"{microphone_unit.id if microphone_unit else ''}:{calibration_id}:"
            f"{minidsp_level['available']}:{analog_interface_mode}"
        )
        if not running and st.session_state.get("_ess_measure_minidsp_level_marker") != auto_level_marker:
            if analog_interface_mode:
                st.session_state["ess_measure_level_calibration_mode"] = "Acoustic calibrator"
            elif minidsp_level["available"] and current_level_label in {
                "Uncalibrated dBFS",
                "UMIK manufacturer Sens Factor",
            }:
                st.session_state["ess_measure_level_calibration_mode"] = "UMIK manufacturer Sens Factor"
            elif current_level_label == "UMIK manufacturer Sens Factor" and not minidsp_level["available"]:
                st.session_state["ess_measure_level_calibration_mode"] = "Uncalibrated dBFS"
            st.session_state["_ess_measure_minidsp_level_marker"] = auto_level_marker
        if not running and st.session_state.get("ess_measure_level_calibration_mode") not in level_mode_options:
            st.session_state["ess_measure_level_calibration_mode"] = "Uncalibrated dBFS"
        level_mode_label = st.selectbox(
            ui_message('ui.7494ab2cdcfbb2'),
            level_mode_options,
            key="ess_measure_level_calibration_mode",
            disabled=running,
            help=ui_message('ui.51a594301ad14c'),
        )
        level_mode = {
            "UMIK manufacturer Sens Factor": "minidsp_sens_factor",
            "USB mic sensitivity dBFS/Pa": "digital_sensitivity",
            "Estimated USB representative sensitivity": "estimated_digital_sensitivity",
            "Analog mic sensitivity mV/Pa": "analog_sensitivity",
            "Acoustic calibrator": "acoustic_calibrator",
            "SPL meter comparison": "spl_meter_comparison",
        }.get(str(level_mode_label), "uncalibrated")
        sensitivity_dbfs_pa = 0.0
        sensitivity_mv_pa = 0.0
        interface_full_scale_vrms = 0.0
        input_gain_db = 0.0
        level_reference_spl_db = 94.0
        level_reference_dbfs = 0.0
        level_values_valid = True
        if level_mode in {"digital_sensitivity", "estimated_digital_sensitivity"}:
            if level_mode == "estimated_digital_sensitivity" and microphone_profile is not None:
                marker = f"{microphone_profile.id}:{microphone_profile.representative_sensitivity_dbfs_94db:g}"
                if st.session_state.get("_ess_measure_estimated_sensitivity_marker") != marker:
                    st.session_state["ess_measure_sensitivity_dbfs_pa_text"] = (
                        f"{microphone_profile.representative_sensitivity_dbfs_94db:g}"
                    )
                    st.session_state["_ess_measure_estimated_sensitivity_marker"] = marker
            sensitivity_text = st.text_input(
                (
                    ui_message('ui.6211fc87b0112f')
                    if level_mode == "estimated_digital_sensitivity" else
                    ui_message('ui.6b8c487c896fe5')
                ),
                key="ess_measure_sensitivity_dbfs_pa_text",
                disabled=running,
                help=ui_message('ui.d31e01b9be701d'),
            )
            try:
                sensitivity_dbfs_pa = float(str(sensitivity_text).strip())
                level_values_valid = np.isfinite(sensitivity_dbfs_pa) and sensitivity_dbfs_pa < 0
            except ValueError:
                level_values_valid = False
            if level_mode == "estimated_digital_sensitivity":
                st.warning(ui_message('ui.4c26a267c82c9f'))
        elif level_mode == "analog_sensitivity":
            sensitivity_mv_pa = float(st.number_input(
                ui_message('ui.98cb32a84caf8e'),
                min_value=0.001,
                step=0.1,
                key="ess_measure_sensitivity_mv_pa",
                disabled=running,
            ))
            interface_full_scale_vrms = float(st.number_input(
                ui_message('ui.90ed5a8d0de2a8'),
                min_value=0.001,
                step=0.1,
                key="ess_measure_interface_full_scale_vrms",
                disabled=running,
            ))
            input_gain_db = float(st.number_input(
                ui_message('ui.b2a818e06f81bc'),
                step=0.5,
                key="ess_measure_input_gain_db",
                disabled=running,
            ))
        elif level_mode in {"acoustic_calibrator", "spl_meter_comparison"}:
            if level_mode == "spl_meter_comparison":
                st.info(
                    ui_message('ui.00f6314cac86d3')
                )
            if analog_interface_mode:
                input_gain_db = float(st.number_input(
                    ui_message('ui.a935893bee7312'),
                    step=0.5,
                    key="ess_measure_input_gain_db",
                    disabled=running,
                    help=(
                        ui_message('ui.be7f2a648c27c7')
                    ),
                ))
            level_reference_spl_db = float(st.number_input(
                ui_message('ui.b2502d9a5ec6b2'),
                min_value=40.0,
                max_value=150.0,
                step=0.1,
                key="ess_measure_level_reference_spl_db",
                disabled=running,
                help=ui_message('ui.2ca432b9cad739'),
            ))
            analog_calibration_fingerprint = _analog_level_calibration_fingerprint(
                input_device=input_device,
                device_name=device_name,
                input_channel=int(input_channel),
                sample_rate=int(sample_rate),
                input_gain_db=float(input_gain_db),
                reference_spl_db=float(level_reference_spl_db),
                calibration_method=level_mode,
            )
            capture_button_label = (
                "Capture SPL meter comparison (1 s)"
                if level_mode == "spl_meter_comparison" else
                "Estimate microphone sensitivity (1 s)"
                if analog_interface_mode else
                "Capture calibration level (1 s)"
            )
            if localized_button(st,
                capture_button_label,
                icon=":material/mic:",
                disabled=running or input_device is None or not device_format_valid,
                width="stretch",
                help=(
                    "Pink noiseを止めずに、音圧計の読みをReference sound pressureへ入力してから、"
                    "マイク入力を1秒録音します。入力クリップ時は採用しません。"
                    if level_mode == "spl_meter_comparison" else
                    "定常校正音を1秒録音し、マイク入力のdBFS RMSを取得します。クリップ時は校正しません。"
                ),
            ):
                capture_settings = MeasurementSettings(
                    sample_rate=int(sample_rate),
                    channels=int(analysis_channels),
                    input_mode="mono" if analysis_channels == 1 else "dual_gain",
                    input_device=input_device,
                    input_channel=int(input_channel),
                    end_frequency_hz=min(22_050.0, int(sample_rate) / 2),
                )
                try:
                    reading = capture_input_level(capture_settings)
                except RuntimeError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["ess_measure_level_reference_dbfs_text"] = f"{reading.rms_dbfs:.3f}"
                    st.session_state["_ess_measure_level_capture_message"] = (
                        f"ch {reading.selected_channel}: {reading.rms_dbfs:.2f} dBFS RMS / "
                        f"peak {reading.peak_dbfs:.2f} dBFS"
                    )
                    if analog_interface_mode:
                        candidate = {
                            "fingerprint": analog_calibration_fingerprint,
                            "calibration_method": level_mode,
                            "reference_spl_db": float(level_reference_spl_db),
                            "measured_rms_dbfs": float(reading.rms_dbfs),
                            "peak_dbfs": float(reading.peak_dbfs),
                            "selected_channel": int(reading.selected_channel),
                            "preamp_gain_db": float(input_gain_db),
                            "calibration_offset_db": (
                                float(level_reference_spl_db) - float(reading.rms_dbfs)
                            ),
                        }
                        if level_mode == "acoustic_calibrator":
                            estimate = estimate_acoustic_calibration(
                                reading,
                                reference_spl_db=level_reference_spl_db,
                                preamp_gain_db=input_gain_db,
                            )
                            candidate.update({
                                "chain_sensitivity_dbfs_per_pa": estimate.chain_sensitivity_dbfs_per_pa,
                                "unity_gain_sensitivity_dbfs_per_pa": estimate.unity_gain_sensitivity_dbfs_per_pa,
                            })
                        st.session_state["_ess_measure_analog_level_candidate"] = candidate
                    st.rerun()
            reference_dbfs_text = st.text_input(
                ui_message('ui.a439e0ef2a4721'),
                key="ess_measure_level_reference_dbfs_text",
                disabled=running or analog_interface_mode,
                help=(
                    ui_message('ui.3cbec4ce48eee6')
                    if analog_interface_mode else
                    ui_message('ui.2a1544728683a2')
                ),
            )
            try:
                level_reference_dbfs = float(str(reference_dbfs_text).strip())
                level_values_valid = np.isfinite(level_reference_dbfs) and level_reference_dbfs < 0
            except ValueError:
                level_values_valid = False
            capture_message = st.session_state.get("_ess_measure_level_capture_message", "")
            if capture_message:
                st.caption(ui_message('ui.4e9e8898e12d04', p0=f'{capture_message}'))
            if analog_interface_mode:
                candidate = st.session_state.get("_ess_measure_analog_level_candidate")
                candidate_matches = bool(
                    isinstance(candidate, dict)
                    and candidate.get("fingerprint") == analog_calibration_fingerprint
                )
                if candidate_matches:
                    if level_mode == "spl_meter_comparison":
                        st.info(
                            ui_message('ui.18642f981ac6da', p0=f"{float(candidate['reference_spl_db']):.1f}", p1=f"{float(candidate['measured_rms_dbfs']):.2f}", p2=f"{float(candidate['calibration_offset_db']):+.2f}")
                        )
                    else:
                        st.info(
                            ui_message('ui.a71a89c853a58c', p0=f"{float(candidate['chain_sensitivity_dbfs_per_pa']):.2f}", p1=f"{float(candidate['unity_gain_sensitivity_dbfs_per_pa']):.2f}", p2=f"{float(candidate['calibration_offset_db']):+.2f}")
                        )
                    if localized_button(
                        st,
                        ui_message('ui.5aed40f1b7f780'),
                        type="primary",
                        disabled=running,
                        width="stretch",
                        help=ui_message('ui.b6a6735b2c0e55'),
                    ):
                        st.session_state["_ess_measure_analog_level_accepted"] = dict(candidate)
                        st.rerun()
                accepted = st.session_state.get("_ess_measure_analog_level_accepted")
                analog_level_calibration_accepted = bool(
                    level_mode in {"acoustic_calibrator", "spl_meter_comparison"}
                    and isinstance(accepted, dict)
                    and accepted.get("fingerprint") == analog_calibration_fingerprint
                )
                if analog_level_calibration_accepted:
                    level_reference_dbfs = float(accepted["measured_rms_dbfs"])
                    level_values_valid = True
                    st.success(ui_message('ui.935b2086927509'))
                else:
                    level_values_valid = False
                    if level_mode == "spl_meter_comparison":
                        st.warning(
                            ui_message('ui.fbe3e90005694f')
                        )
                    else:
                        st.warning(
                            ui_message('ui.f31df8afaf8928')
                        )
            else:
                analog_level_calibration_accepted = False
        else:
            analog_level_calibration_accepted = False
        level_calibration_note = st.text_input(
            ui_message('ui.45c84fd34d2369'),
            key="ess_measure_level_calibration_note",
            disabled=running,
            placeholder=ui_message('ui.c109f78d10767f'),
        )
        level_config = {
            "measurement_distance_m": measurement_distance_m,
            "level_calibration_mode": level_mode,
            "mic_sensitivity_dbfs_per_pa": sensitivity_dbfs_pa,
            "mic_sensitivity_mv_pa": sensitivity_mv_pa,
            "interface_full_scale_vrms": interface_full_scale_vrms,
            "input_gain_db": input_gain_db,
            "level_reference_spl_db": level_reference_spl_db,
            "level_reference_dbfs": level_reference_dbfs,
            "level_calibration_note": str(level_calibration_note),
            "input_gain_db_channels": tuple(minidsp_level["gain_snapshot"].channel_gain_db),
            "input_gain_max_db_channels": tuple(minidsp_level["gain_snapshot"].channel_max_gain_db),
            "input_gain_device_name": str(minidsp_level["gain_snapshot"].device_name),
            "input_analog_gain_db": minidsp_level["gain_snapshot"].analog_gain_db,
            "require_stable_input_gain": level_mode == "minidsp_sens_factor",
            "minidsp_sens_factor_db": float(minidsp_level["factor"] or 0.0),
            "minidsp_analog_gain_db": minidsp_level["analog_gain"],
            "minidsp_gain_adjustment_db": float(minidsp_level["gain_adjustment"] or 0.0),
        }
        level_preview_settings = MeasurementSettings(
            end_frequency_hz=22_050.0,
            **level_config,
        )
        level_offset = level_calibration_offset_db(level_preview_settings)
        calibration_kind = level_calibration_kind(level_preview_settings)
        if level_mode == "uncalibrated":
            st.info(ui_message('ui.2aa934d63efa76'))
        elif analog_interface_mode and not analog_level_calibration_accepted:
            st.caption(ui_message('ui.3b97d85a725143'))
        elif not level_values_valid or level_offset is None:
            st.error(ui_message('ui.1ece0048d25593'))
        elif calibration_kind == "estimated":
            st.warning(ui_message('ui.f1524de19380b3', p0=f'{level_offset:+.2f}'))
        else:
            st.success(ui_message('ui.e1e4e533ba92c3', p0=f'{level_offset:+.2f}'))

    with category_tabs[3]:
        st.markdown(ui_message('ui.794926c0c5e1f1'))
        gate_cols = st.columns(2)
        with gate_cols[0]:
            gate_start = float(st.number_input(
                ui_message('ui.630c21fddecee8'),
                min_value=-100.0,
                max_value=2000.0,
                step=0.5,
                format="%.1f",
                key="ess_measure_gate_start_widget",
                disabled=running,
                help=ui_message('ui.9792bcf9aa1a11'),
            ))
            gate_start_valid = np.isfinite(gate_start)
            st.session_state["ess_measure_gate_start"] = gate_start
        with gate_cols[1]:
            gate_end = st.number_input(
                ui_message('ui.b6749e6fbeb147'),
                min_value=0.1,
                max_value=2000.0,
                step=0.5,
                key="ess_measure_gate_end",
                disabled=running,
            )
        merge_crossover = st.number_input(
            ui_message('ui.d5cec646387d65'),
            min_value=0.0,
            step=10.0,
            key="ess_measure_merge_crossover",
            disabled=running,
            help=ui_message('ui.7808d0e4c1ab96'),
        )
        preview_shot = snapshot.shots[-1] if snapshot and snapshot.shots else None
        preview_matches_gate = bool(
            preview_shot is not None
            and preview_shot.settings_snapshot.gate_start_ms == float(gate_start)
            and preview_shot.settings_snapshot.gate_end_ms == float(gate_end)
        )
        effective_crossover = effective_merge_crossover_hz(
            max(float(gate_end), 0.1) / 1000.0,
            float(merge_crossover),
            ungated=(preview_shot.ungated_complex_response if preview_matches_gate else None),
            gated=(preview_shot.gated_complex_response if preview_matches_gate else None),
            frequency=(preview_shot.frequency_hz if preview_matches_gate else None),
        )
        if np.isfinite(effective_crossover):
            transition_ratio = 2.0 ** 0.25
            transition_low = effective_crossover / transition_ratio
            transition_high = effective_crossover * transition_ratio
            crossover_source = "Auto response match" if float(merge_crossover) <= 0.0 else "Manual"
            st.caption(
                ui_message('ui.5293f9df98df2f', p0=f'{crossover_source}', p1=f'{effective_crossover:.0f}', p2=f'{transition_low:.0f}', p3=f'{transition_high:.0f}')
            )
        else:
            st.caption(
                ui_message('ui.dcaa82660ebd79')
            )
        clock_mode = localized_segmented_control(
            ui_message('ui.b7101aaec8ab52'),
            ["Auto", "Force", "Off"],
            key="ess_measure_clock_mode",
            disabled=running or reference_mode_label == "IR peak — external ESS",
            width="stretch",
            help=ui_message('ui.2fdc814a9db0a5'),
        )
        reprocess_scope = localized_segmented_control(
            ui_message('ui.55ecdf8652318b'),
            ["Latest", "All retained"],
            key="ess_measure_reprocess_scope",
            disabled=running,
            width="stretch",
        )
        if localized_button(st,
            ui_message('ui.e693d1dc4deedb'),
            icon=":material/tune:",
            disabled=running or not gate_start_valid or engine is None or not snapshot or not snapshot.shots,
            width="stretch",
        ):
            try:
                with st.spinner(ui_message('ui.13c067d374feff')):
                    engine.update_processing(
                        gate_start_ms=float(gate_start),
                        gate_end_ms=float(gate_end),
                        merge_crossover_hz=float(merge_crossover),
                        scope="all" if reprocess_scope == "All retained" else "latest",
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(ui_message('ui.1c230d54e7eb87', p0=f'{exc}'))
            else:
                st.session_state[UNSAVED_KEY] = True
                st.rerun()

    with category_tabs[4]:
        st.markdown(ui_message('ui.ec72b45a198286'))
        st.info(
            ui_message('ui.54e5bcb5adccd2')
        )
        _render_level_check_signal(
            sample_rate=int(sample_rate),
            measurement_start_hz=float(
                known_reference.start_frequency_hz
                if known_reference is not None and known_reference.start_frequency_hz > 0.0
                else st.session_state.get("ess_gen_start_hz", 20.0)
            ),
            measurement_end_hz=float(
                known_reference.end_frequency_hz
                if known_reference is not None and known_reference.end_frequency_hz > 0.0
                else st.session_state.get("ess_gen_end_hz", min(20_000.0, int(sample_rate) / 2.0 - 1.0))
            ),
            default_rms_dbfs=float(
                known_reference.peak_dbfs
                if known_reference is not None else
                st.session_state.get("ess_gen_peak_dbfs", -18.0)
            )
            - 20.0 * np.log10(np.sqrt(2.0)),
            snapshot=snapshot,
            disabled=running,
        )
        wavelet_mode = st.selectbox(
            ui_message('ui.1c0864fb425846'),
            ["every_shot", "every_n_shots", "manual"],
            key="ess_measure_wavelet_mode",
            disabled=running,
        format_func=localized_formatter(str))
        wavelet_n = st.number_input(
            ui_message('ui.8a58c6523dfdfd'),
            min_value=1,
            max_value=100,
            disabled=running or wavelet_mode != "every_n_shots",
            key="ess_measure_wavelet_n",
        )
        quality_gate = localized_segmented_control(
            ui_message('ui.7d2ef2a2023468'),
            ["Strict", "Standard", "Lenient"],
            key="ess_measure_quality_gate",
            disabled=running,
            width="stretch",
            help=(
                ui_message('ui.40826c9d49b31c')
            ),
        )
        forensic_raw = st.toggle(
            ui_message('ui.55a1a22d99f1d4'),
            key="ess_measure_forensic_raw",
            disabled=running,
            help=ui_message('ui.5f50d34d494988'),
        )
        include_warnings = quality_gate != "Strict"
        if engine is not None and bool(include_warnings) != engine.settings.include_warnings_in_average:
            engine.set_include_warnings(bool(include_warnings))
        start_kwargs = {
            "autosave_root": autosave_root,
            "sample_rate": sample_rate,
            "input_device": input_device,
            "gate_start": gate_start,
            "gate_end": gate_end,
            "merge_crossover": merge_crossover,
            "wavelet_mode": wavelet_mode,
            "wavelet_n": wavelet_n,
            "include_warnings": include_warnings,
            "calibration_id": calibration_id,
            "calibration_by_id": calibration_by_id,
            "ess_reference_label": ess_reference_label,
            "reference_mode_label": reference_mode_label,
            "sync_polarity_label": sync_polarity_label,
            "known_reference": known_reference,
            "microphone_profile": microphone_profile,
            "microphone_unit": microphone_unit,
            "input_channel": input_channel,
            "calibration_angle": calibration_angle,
            "phase_calibration_id": phase_calibration_id,
            "calibration_extrapolation": str(calibration_extrapolation),
            "clock_mode": str(clock_mode),
            "quality_gate": str(quality_gate),
            "forensic_raw": bool(forensic_raw),
            "level_config": level_config,
            "timing_reference_kind": (
                "phaseeq_tweeter_reference" if tweeter_reference_mode else "sequence_marker"
            ),
            "timing_reference_session_id": timing_reference_session_id,
            "reference_tweeter_channel_id": reference_tweeter_channel_id,
            "reference_measurement_id": reference_measurement_id,
            "reference_arrival_sample": reference_arrival_sample,
            "timing_reference_role": timing_reference_role,
            "reference_output_channel": reference_output_channel,
            "measurement_output_channel": measurement_output_channel,
            "reference_routing_verified": reference_routing_verified,
            "external_playback_setup_confirmed": external_playback_setup_confirmed,
            "reference_protection_confirmed": reference_protection_confirmed,
            "alignment_processing_bypassed_confirmed": alignment_processing_bypassed_confirmed,
            "external_playback_setup_note": external_playback_setup_note,
        }
        level_calibration_required = level_mode != "uncalibrated"
        level_calibration_valid = bool(
            analog_interface_mode
            or not level_calibration_required
            or (level_values_valid and level_offset is not None)
        )
        analog_level_calibration_valid = bool(
            not analog_interface_mode or analog_level_calibration_accepted
        )
        start_blockers = _measurement_start_blockers(
            running=running,
            devices_available=bool(devices),
            reference_required=known_wav_mode,
            reference_ready=known_reference is not None,
            reference_rejection_reasons=reference_rejection_reasons,
            profile_rate_valid=profile_rate_valid,
            device_format_valid=device_format_valid,
            device_format_error=device_format_error,
            level_calibration_valid=level_calibration_valid,
            mic_calibration_valid=mic_calibration_valid,
            microphone_connection_known=microphone_profile is not None,
            analog_level_calibration_required=analog_interface_mode,
            analog_level_calibration_valid=analog_level_calibration_valid,
            timing_marker_required=reference_mode_label != "IR peak — external ESS",
            timing_marker_ready=bool(known_reference is not None and known_reference.has_timing_marker),
            tweeter_reference_required=tweeter_reference_mode,
            tweeter_reference_ready=timing_reference_ready,
        )
        st.session_state["_ess_measure_action_context"] = {
            "measurement_db_path": measurement_db_path,
            "running": bool(running),
            "start_blockers": tuple(start_blockers),
            "start_kwargs": start_kwargs,
        }
        st.caption(ui_message('ui.36407eac61d696'))

    with category_tabs[5]:
        st.markdown(ui_message('ui.99b9dc7477512e'))
        upload = st.file_uploader(ui_message('ui.5a1f7a92a04f63'), type=["zip"], key="ess_measure_restore_zip")
        if localized_button(st, ui_message('ui.883c885ec46abb'), disabled=upload is None, width="stretch") and upload is not None:
            if _measurement_replacement_requires_confirmation(snapshot):
                st.session_state["_ess_restore_confirmation"] = True
            else:
                _restore_measurement_upload(upload, autosave_root)
        if st.session_state.get("_ess_restore_confirmation", False):
            if upload is None:
                st.session_state["_ess_restore_confirmation"] = False
            else:
                st.warning(
                    ui_message('ui.83f6e0998e86e0', p0=f'{(len(snapshot.shots) if snapshot else 0)}', p1=f'{upload.name}')
                )
                restore_confirm_row = st.container(horizontal=True, gap="small")
                with restore_confirm_row:
                    if localized_button(st, ui_message('ui.374cd7e0c75593'), type="primary"):
                        st.session_state["_ess_restore_confirmation"] = False
                        _restore_measurement_upload(upload, autosave_root)
                    if localized_button(st, ui_message('ui.337c6879ec657f')):
                        st.session_state["_ess_restore_confirmation"] = False
                        st.rerun()

    with measurement_actions_slot.container():
        render_measurement_actions()
        render_live_input_level()


def _measurement_start_countdown(delay_s: int) -> tuple[int, ...]:
    delay = int(delay_s)
    if delay not in MEASUREMENT_START_DELAYS_S:
        raise ValueError("measurement start delay must be 3, 5, or 10 seconds")
    return tuple(range(delay, 0, -1))


def _measurement_start_blockers(
    *,
    running: bool,
    devices_available: bool,
    reference_required: bool,
    reference_ready: bool,
    profile_rate_valid: bool,
    device_format_valid: bool,
    level_calibration_valid: bool,
    mic_calibration_valid: bool = True,
    microphone_connection_known: bool = True,
    analog_level_calibration_required: bool = False,
    analog_level_calibration_valid: bool = True,
    timing_marker_required: bool = False,
    timing_marker_ready: bool = False,
    tweeter_reference_required: bool = False,
    tweeter_reference_ready: bool = False,
    device_format_error: str = "",
    reference_rejection_reasons: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if running:
        return ("測定が進行中です。新しい測定はStopまたはAbort後に開始できます。",)
    reasons: list[str] = []
    if reference_required and reference_rejection_reasons:
        reasons.extend(f"ESS Library: {reason}" for reason in reference_rejection_reasons)
    elif reference_required and not reference_ready:
        reasons.append("ESS → ESS Libraryで、実際に再生するESSを追加・選択してください。")
    if timing_marker_required and not timing_marker_ready:
        reasons.append("Timing marker方式には、manifestでmarkerが確認できるESSを選択してください。")
    if tweeter_reference_required and not tweeter_reference_ready:
        reasons.append("Tweeter基準Session、Reference Channel、基準測定を確認してください。")
    if not devices_available:
        reasons.append("入力機器がありません。マイクを接続してRefresh input devicesを実行してください。")
    if not profile_rate_valid:
        reasons.append("選択マイクが現在のSample rateに対応していません。対応Sample rateへ変更してください。")
    if not microphone_connection_known:
        reasons.append(
            "入力機器だけではUSBマイクとオーディオインターフェースを判定できません。"
            "Microphone modelを選択してください。"
        )
    if not mic_calibration_valid:
        reasons.append(
            "Mic Calibrationを選択し、周波数／Gain校正を適用してから測定してください。"
        )
    if devices_available and not device_format_valid:
        detail = str(device_format_error).strip()
        reasons.append(
            "選択した入力チャンネル／Sample rateをCoreAudioで開始できません。"
            + (f" 詳細: {detail}" if detail else "入力機器とチャンネル数を確認してください。")
        )
    if not level_calibration_valid:
        reasons.append("Calibrationで選択した絶対レベル校正の数値が未入力または無効です。")
    if analog_level_calibration_required and not analog_level_calibration_valid:
        reasons.append(
            "アナログマイク＋オーディオインターフェースでは、Acoustic calibratorまたは"
            "SPL meter comparisonで音量校正し、Accept sound pressure calibrationを実行してください。"
        )
    return tuple(reasons)


def _start_engine(
    autosave_root: Path,
    sample_rate: int,
    input_device: int | None,
    gate_start: float,
    gate_end: float,
    merge_crossover: float,
    wavelet_mode: str,
    wavelet_n: int,
    include_warnings: bool,
    calibration_id: str,
    calibration_by_id: dict[str, SpeakerMeasurementRecord],
    ess_reference_label: str,
    reference_mode_label: str,
    sync_polarity_label: str,
    known_reference: KnownEssReference | None,
    microphone_profile: MicrophoneProfile | None,
    microphone_unit: MicrophoneUnit | None,
    input_channel: int,
    calibration_angle: int,
    phase_calibration_id: str,
    calibration_extrapolation: str,
    clock_mode: str,
    quality_gate: str,
    forensic_raw: bool,
    level_config: dict[str, object],
    timing_reference_kind: str = "sequence_marker",
    timing_reference_session_id: str = "",
    reference_tweeter_channel_id: str = "",
    reference_measurement_id: str = "",
    reference_arrival_sample: float = float("nan"),
    timing_reference_role: str = "target_speaker",
    reference_output_channel: str = "",
    measurement_output_channel: str = "",
    reference_routing_verified: bool = False,
    external_playback_setup_confirmed: bool = False,
    reference_protection_confirmed: bool = False,
    alignment_processing_bypassed_confirmed: bool = False,
    external_playback_setup_note: str = "",
) -> None:
    cal_frequency: tuple[float, ...] = ()
    cal_gain: tuple[float, ...] = ()
    record = calibration_by_id.get(calibration_id)
    if record is not None:
        response = speaker_response_from_payload(record.response_payload)
        if response is not None:
            cal_frequency = tuple(float(value) for value in response.frequency)
            cal_gain = tuple(float(value) for value in response.gain_db)
    phase_frequency: tuple[float, ...] = ()
    phase_deg: tuple[float, ...] = ()
    phase_record = calibration_by_id.get(phase_calibration_id)
    if phase_record is not None:
        phase_response = speaker_response_from_payload(phase_record.response_payload)
        if phase_response is not None and phase_response.phase_deg is not None:
            phase_frequency = tuple(float(value) for value in phase_response.frequency)
            phase_deg = tuple(float(value) for value in phase_response.phase_deg)
    known_wav_mode = ess_reference_label == "ESS Library file"
    channels = microphone_profile.input_channels if microphone_profile is not None else 2
    settings_kwargs = {
        "channels": channels,
        "input_mode": _measurement_input_mode(ess_reference_label, channels),
        "correlation_threshold": 0.20 if known_wav_mode else 0.35,
        "reference_mode": (
            "timing_marker" if reference_mode_label != "IR peak — external ESS" else "ir_peak"
        ),
        "sync_polarity": {
            "Positive +": "positive",
            "Negative −": "negative",
        }.get(sync_polarity_label, "auto"),
    }
    if known_reference is not None:
        settings_kwargs.update(
            sweep_duration_s=float(known_reference.sweep_duration_s),
            shot_period_s=float(known_reference.shot_period_s),
            reference_source_name=str(known_reference.source_name),
            reference_source_sample_rate=int(known_reference.source_sample_rate),
            reference_peak_dbfs=float(
                20.0 * np.log10(max(float(np.max(np.abs(known_reference.samples))), 1e-15))
            ),
            reference_repeat_count=int(known_reference.repeat_count),
            reference_loop_gap_variable=bool(known_reference.variable_loop_gap),
            reference_source_channel=int(known_reference.source_channel),
            shot_capture_duration_s=float(known_reference.shot_capture_duration_s),
            reference_has_timing_marker=bool(known_reference.has_timing_marker),
            timing_marker_to_ess_s=float(known_reference.timing_marker_to_ess_s),
        )
        if known_reference.start_frequency_hz > 0:
            settings_kwargs["start_frequency_hz"] = float(known_reference.start_frequency_hz)
        if known_reference.end_frequency_hz > 0:
            settings_kwargs["end_frequency_hz"] = min(
                float(known_reference.end_frequency_hz), int(sample_rate) / 2
            )
    settings = MeasurementSettings(
        sample_rate=int(sample_rate),
        input_device=input_device,
        gate_start_ms=float(gate_start),
        gate_end_ms=float(gate_end),
        merge_crossover_hz=float(merge_crossover),
        wavelet_update_mode=str(wavelet_mode),
        wavelet_every_n=int(wavelet_n),
        include_warnings_in_average=bool(include_warnings),
        mic_calibration_id=str(calibration_id),
        mic_calibration_angle_deg=int(calibration_angle),
        mic_calibration_frequency_hz=cal_frequency,
        mic_calibration_gain_db=cal_gain,
        mic_calibration_extrapolation={
            "No correction": "no_correction",
            "Manual / extended file": "manual_extension",
        }.get(calibration_extrapolation, "hold_edge"),
        clock_correction_mode={"Force": "force", "Off": "off"}.get(clock_mode, "auto"),
        quality_gate_mode=str(quality_gate).lower(),
        raw_audio_storage="session_wav" if forensic_raw else "accepted_shots",
        mic_phase_calibration_id=str(phase_calibration_id),
        mic_phase_calibration_frequency_hz=phase_frequency,
        mic_phase_calibration_deg=phase_deg,
        noise_preflight_enabled=True,
        ambient_noise_duration_s=5.0,
        target_snr_db=25.0,
        minimum_snr_db=18.0,
        minimum_passing_band_fraction=0.80,
        preflight_mask_relative_db=-25.0,
        preflight_mask_min_bands=12,
        preflight_mask_min_octaves=1.0,
        preflight_mask_edge_trim_bands=2,
        preflight_mask_max_gap_bands=2,
        target_headroom_db=10.0,
        minimum_headroom_db=6.0,
        automatic_usb_gain=True,
        microphone_profile_id=microphone_profile.id if microphone_profile is not None else "",
        microphone_unit_id=microphone_unit.id if microphone_unit is not None else "",
        microphone_serial_number=microphone_unit.serial_number if microphone_unit is not None else "",
        microphone_nominal_bit_depth=(
            microphone_profile.nominal_bit_depth if microphone_profile is not None else 0
        ),
        input_channel=int(input_channel),
        timing_reference_kind=str(timing_reference_kind),
        timing_reference_session_id=str(timing_reference_session_id),
        reference_tweeter_channel_id=str(reference_tweeter_channel_id),
        reference_measurement_id=str(reference_measurement_id),
        reference_arrival_sample=float(reference_arrival_sample),
        timing_reference_role=str(timing_reference_role),
        reference_output_channel=str(reference_output_channel),
        measurement_output_channel=str(measurement_output_channel),
        reference_routing_verified=bool(reference_routing_verified),
        external_playback_setup_confirmed=bool(external_playback_setup_confirmed),
        reference_protection_confirmed=bool(reference_protection_confirmed),
        alignment_processing_bypassed_confirmed=bool(alignment_processing_bypassed_confirmed),
        external_playback_setup_note=str(external_playback_setup_note),
        **level_config,
        **settings_kwargs,
    )
    from ui.guided_measurement_view import release_audio_downloads_before_capture
    release_audio_downloads_before_capture()
    engine = ContinuousMeasurementEngine(
        settings,
        autosave_root,
        reference=known_reference.samples if known_reference is not None else None,
        timing_marker=(
            known_reference.timing_marker_samples
            if (
                known_reference is not None
                and reference_mode_label != "IR peak — external ESS"
            ) else None
        ),
    )
    st.session_state[ENGINE_KEY] = engine
    st.session_state[UNSAVED_KEY] = True
    engine.start()


def _render_shot_diagnostics(
    engine: ContinuousMeasurementEngine,
    snapshot: SessionSnapshot,
) -> None:
    if snapshot.saved_original is not None:
        st.caption(ui_message('ui.f857859605bd9d'))
        return
    reliability = _shot_reliability_diagnostics(snapshot.shots)
    rows = []
    for shot in snapshot.shots:
        rating, score, analysis = reliability.get(shot.shot_index, ("N/A", 0.0, "解析情報なし"))
        rows.append({
            "Use": shot.included_in_average,
            "Sweep": shot.sweep_index or shot.shot_index,
            "Measurement": shot.shot_index,
            "Status": shot.status,
            "Correlation": shot.correlation,
            "S/N dB": shot.quality.snr_db,
            "Gain diff dB": shot.quality.gain_difference_db,
            "L peak dBFS": 20.0 * np.log10(max(shot.quality.left_peak, 1e-15)),
            "R peak dBFS": 20.0 * np.log10(max(shot.quality.right_peak, 1e-15)),
            "Center": shot.center_sample,
            "Center tracking": shot.center_tracking_state,
            "IR similarity": shot.centered_ir_similarity,
            "Timing valid": shot.timing_valid,
            "Merge valid": shot.merge_valid,
            "Reliability": f"{rating} {score:.2f}",
            "Failure analysis": analysis,
        })
    edited = st.data_editor(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        disabled=[column for column in rows[0] if column != "Use"],
        key=f"ess_shot_table_{snapshot.session_id}_{len(snapshot.shots)}",
    )
    for shot, included in zip(snapshot.shots, edited["Use"].tolist()):
        if bool(included) != shot.included_in_average:
            engine.set_shot_included(shot.shot_index, bool(included))
            st.session_state[UNSAVED_KEY] = True


def _latest_display_shot(snapshot: SessionSnapshot) -> object | None:
    """Return the latest usable shot for the user-facing live graph."""
    allowed = {ShotStatus.VALID}
    if snapshot.settings.include_warnings_in_average:
        allowed.add(ShotStatus.WARNING)
    for shot in reversed(snapshot.shots):
        if (
            shot.included_in_average
            and shot.status in allowed
            and shot.frequency_hz is not None
            and shot.merged_complex_response is not None
        ):
            return shot
    for shot in reversed(snapshot.shots):
        if shot.frequency_hz is not None and shot.merged_complex_response is not None:
            return shot
    return snapshot.latest


def _shot_by_index(snapshot: SessionSnapshot, shot_index: int) -> object | None:
    return next((shot for shot in snapshot.shots if int(shot.shot_index) == int(shot_index)), None)


def _reference_shot_for_result(snapshot: SessionSnapshot, result: HighPrecisionResult | None) -> object | None:
    if result is not None:
        shot = _shot_by_index(snapshot, int(result.reference_shot_index))
        if shot is not None:
            return shot
    return _latest_display_shot(snapshot)


def _measurement_result_selector(
    snapshot: SessionSnapshot,
    *,
    default_id: str,
) -> str:
    rows: list[dict[str, object]] = []
    option_ids: list[str] = []
    if snapshot.standard_result is not None:
        rows.append({
            "Kind": "Final",
            "Result": "Final merged",
            "Measurements": ", ".join(str(value) for value in snapshot.standard_result.used_shot_indices),
            "Status": "OK",
            "S/N dB": "",
            "Sync": snapshot.standard_result.reference_mode,
            "Center": f"max {max((abs(value) for value in snapshot.standard_result.alignment_samples), default=0.0):.2f} sample",
            "Merge": "Merged",
        })
        option_ids.append("standard")
    if snapshot.denoised_result is not None:
        rows.append({
            "Kind": "Denoised",
            "Result": "Denoised Merged",
            "Measurements": ", ".join(str(value) for value in snapshot.denoised_result.used_shot_indices),
            "Status": "OK",
            "S/N dB": "",
            "Sync": snapshot.denoised_result.reference_mode,
            "Center": f"max {max((abs(value) for value in snapshot.denoised_result.alignment_samples), default=0.0):.2f} sample",
            "Merge": "Merged",
        })
        option_ids.append("denoised")
    for shot in reversed(snapshot.shots):
        rows.append({
            "Kind": "Measurement",
            "Result": f"Measurement {shot.shot_index}",
            "Measurements": str(shot.shot_index),
            "Status": str(shot.status),
            "S/N dB": f"{shot.quality.snr_db:.1f}",
            "Sync": f"{shot.correlation:.3f}",
            "Center": f"{shot.center_polarity} / {shot.center_sample:.2f}",
            "Merge": "OK" if shot.merge_valid else "Full fallback",
        })
        option_ids.append(f"shot:{shot.shot_index}")
    if not rows:
        return ""
    selected_key = f"ess_result_selected_id_{snapshot.session_id}"
    selected_id = str(st.session_state.get(selected_key, ""))
    if selected_id not in option_ids:
        selected_id = default_id if default_id in option_ids else option_ids[0]
        st.session_state[selected_key] = selected_id
    default_index = option_ids.index(selected_id)
    st.markdown(ui_message('ui.2233481dd96ae0'))
    st.caption(ui_message('ui.11ceb32f8a34fb'))
    event = st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        height=min(180, 36 + 30 * len(rows)),
        key=f"ess_result_selector_{snapshot.session_id}_{len(snapshot.shots)}",
        on_select="rerun",
        selection_mode="single-row",
        selection_default={"selection": {"rows": [default_index]}},
        column_config={
            "Kind": st.column_config.TextColumn("Kind", pinned=True, width="small"),
            "Result": st.column_config.TextColumn("Result", pinned=True, width="medium"),
            "S/N dB": st.column_config.TextColumn("S/N dB", width="small"),
        },
    )
    selected_rows = list(event.selection.rows)
    if selected_rows and 0 <= int(selected_rows[0]) < len(option_ids):
        selected_id = option_ids[int(selected_rows[0])]
        st.session_state[selected_key] = selected_id
    return selected_id


def _render_live_measurement_adjustments(
    engine: ContinuousMeasurementEngine,
    snapshot: SessionSnapshot,
) -> None:
    with st.container(border=True):
        st.markdown(ui_message('ui.c73d2cec4270f4'))
        live_marker = (
            f"{snapshot.session_id}:"
            f"{snapshot.settings.gate_start_ms:.3f}:"
            f"{snapshot.settings.gate_end_ms:.3f}:"
            f"{snapshot.settings.sync_polarity}"
        )
        if st.session_state.get("_ess_live_adjustment_marker") != live_marker:
            st.session_state["_ess_live_adjustment_marker"] = live_marker
            st.session_state["ess_live_gate_start_ms"] = float(snapshot.settings.gate_start_ms)
            st.session_state["ess_live_gate_end_ms"] = float(snapshot.settings.gate_end_ms)
            st.session_state["ess_live_center_polarity"] = {
                "positive": "Positive +",
                "negative": "Negative −",
            }.get(snapshot.settings.sync_polarity, "Auto ±")
        live_cols = st.columns(2, vertical_alignment="bottom")
        with live_cols[0]:
            gate_start = float(st.number_input(
                ui_message('ui.cf73a52934e256'),
                min_value=-100.0,
                max_value=2000.0,
                step=0.5,
                format="%.1f",
                key="ess_live_gate_start_ms",
                help=ui_message('ui.5f9c5ca7699e80'),
            ))
        with live_cols[1]:
            gate_end = float(st.number_input(
                ui_message('ui.26de15c085b1da'),
                min_value=0.1,
                max_value=2000.0,
                step=0.5,
                format="%.1f",
                key="ess_live_gate_end_ms",
            ))
        polarity_label = localized_segmented_control(
            ui_message('ui.d3286dc24d92b0'),
            ["Auto ±", "Positive +", "Negative −"],
            key="ess_live_center_polarity",
            default={
                "positive": "Positive +",
                "negative": "Negative −",
            }.get(snapshot.settings.sync_polarity, "Auto ±"),
            width="stretch",
            help=ui_message('ui.7d027bf4b6ae9b'),
        )
        apply_live = localized_button(
            st,
            ui_message('ui.7377e14780c385'),
            icon=":material/tune:",
            width="stretch",
            disabled=not snapshot.shots or gate_start >= gate_end,
            help=ui_message('ui.99a476ce4a3af8'),
        )
        if apply_live:
            polarity = {
                "Positive +": "positive",
                "Negative −": "negative",
            }.get(str(polarity_label), "auto")
            try:
                engine.update_processing(
                    gate_start_ms=gate_start,
                    gate_end_ms=gate_end,
                    merge_crossover_hz=float(snapshot.settings.merge_crossover_hz),
                    scope="latest" if _measurement_is_running(snapshot) else "all",
                    sync_polarity=polarity,
                    force_recenter=True,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(ui_message('ui.29329c7b3d8287', p0=f'{exc}'))
            else:
                st.session_state["ess_measure_gate_start"] = gate_start
                st.session_state["ess_measure_gate_start_widget"] = gate_start
                st.session_state["ess_measure_gate_end"] = gate_end
                st.session_state["ess_measure_sync_polarity"] = str(polarity_label)
                st.session_state[UNSAVED_KEY] = True
                st.rerun(scope="fragment")


def _measurement_result_refresh_token(snapshot: SessionSnapshot | None) -> str:
    if snapshot is None:
        return content_digest(MEASUREMENT_DISPLAY_CACHE_ALGORITHM_VERSION, "no-session")
    device_info = snapshot.device_info
    pilot_history = device_info.get("pilot_history", [])
    latest_pilot = pilot_history[-1] if pilot_history else None
    return content_digest(
        MEASUREMENT_DISPLAY_CACHE_ALGORITHM_VERSION,
        snapshot.session_id,
        int(snapshot.revision),
        str(snapshot.state),
        snapshot.stopped_at,
        snapshot.error_message,
        device_info.get("preflight_message", ""),
        device_info.get("preflight_ready", False),
        device_info.get("gain_recheck_required", False),
        device_info.get("low_snr_continue", False),
        device_info.get("last_audio_status", ""),
        device_info.get("audio_status_events", 0),
        device_info.get("audio_recovery_state", ""),
        device_info.get("last_recovery_status", ""),
        device_info.get("rejected_shots", 0),
        device_info.get("last_rejection_reason", ""),
        len(pilot_history),
        latest_pilot,
    )


def _rerun_measurement_results_when_source_changes(snapshot: SessionSnapshot | None) -> None:
    """Poll lightweight engine state and refresh heavy results only on change."""

    state_key = "_measurement_results_poll_token"
    current = _measurement_result_refresh_token(snapshot)
    previous = st.session_state.get(state_key)
    st.session_state[state_key] = current
    if previous is not None and previous != current:
        st.rerun()


def _measurement_display_cache_key(
    snapshot: SessionSnapshot,
    *,
    selected_result_id: str,
    selected_shot_index: int,
    view: str,
    graph_points: int,
    gain_smoothing_fraction: float = 0.0,
) -> str:
    return CacheCoordinator(st.session_state).key(
        CacheDomain.MEASUREMENT_RESULT,
        algorithm_version=MEASUREMENT_DISPLAY_CACHE_ALGORITHM_VERSION,
        source_digest=content_digest(
            snapshot.session_id,
            selected_result_id,
            int(selected_shot_index),
        ),
        settings_digest=content_digest(
            view,
            int(graph_points),
            float(gain_smoothing_fraction),
        ),
        stage="Measurement results",
        revision=int(snapshot.revision),
    ).digest


@st.fragment(run_every=1.0)
def render_measurement_actions() -> None:
    """Render live preparation and actions in the left operation pane."""
    engine = _engine()
    snapshot = engine.snapshot() if engine is not None else None
    _rerun_measurement_results_when_source_changes(snapshot)
    st.markdown(ui_message('ui.778e5330fe4c8c'))
    if snapshot is not None:
        _render_preflight_workflow(snapshot)
    _render_measurement_action_bar(engine, snapshot)
    if snapshot is None or engine is None:
        st.caption(ui_message('ui.5e32ccbf1fb8d2'))
        return
    _render_live_measurement_adjustments(engine, snapshot)
    _render_manual_microphone_gain(engine, snapshot)


@st.fragment(run_every=0.1)
def render_live_input_level() -> None:
    """Refresh only the lightweight input meter at an interactive rate."""

    engine = _engine()
    if engine is not None:
        _render_live_input_level_meter(engine.snapshot())


@st.fragment
def render_measurement_results(*, measurement_db_path: Path) -> None:
    engine = _engine()
    snapshot = engine.snapshot() if engine is not None else None
    if snapshot is not None:
        measurement_token = (snapshot.session_id, int(snapshot.revision))
        if st.session_state.get("_cache_measurement_source_revision") != measurement_token:
            CacheCoordinator(st.session_state).notify(CacheEvent.MEASUREMENT_CHANGED)
            st.session_state["_cache_measurement_source_revision"] = measurement_token
    st.markdown("<span class='rf-window-marker-live'></span>", unsafe_allow_html=True)
    st.session_state.setdefault("ess_measure_graph_mode", "Light")
    st.session_state.setdefault("ess_measure_graph_points", 1024)
    st.session_state.setdefault("ess_measure_gain_smoothing", "1/12 oct")
    title_col, controls_col = st.columns([0.72, 0.28], gap="medium")
    with title_col:
        st.markdown(ui_message('ui.0cafc50b714b8e'))
        st.caption(ui_message('ui.d9019dc1082793'))
    with controls_col:
        with st.popover(ui_message('ui.2302d21f4bef22'), icon=":material/tune:", width="stretch"):
            graph_mode = st.selectbox(
                ui_message('ui.5e23ec6a300dc6'),
                ["Light", "Interactive"],
                key="ess_measure_graph_mode",
                help=ui_message('ui.acf9549354c8ed'),
            format_func=localized_formatter(str))
            graph_points = int(st.selectbox(
                ui_message('ui.77deabe238de2f'),
                [1024, 2048, 4096],
                key="ess_measure_graph_points",
                format_func=localized_formatter(lambda value: f'{value} points / series'),
            ))
            smoothing_label = st.selectbox(
                ui_message('ui.51437e9c53940b'),
                ["Off", "1/24 oct", "1/12 oct", "1/6 oct", "1/3 oct"],
                key="ess_measure_gain_smoothing",
                help=ui_message('ui.46df884bc132cf'),
            format_func=localized_formatter(str))
    gain_smoothing_fraction = {
        "1/24 oct": 24.0,
        "1/12 oct": 12.0,
        "1/6 oct": 6.0,
        "1/3 oct": 3.0,
    }.get(str(smoothing_label), 0.0)
    if snapshot is None:
        st.info(ui_message('ui.ca04f03cec6c5b'))
        return
    preflight_message = str(snapshot.device_info.get("preflight_message", "") or "")
    if snapshot.state == MeasurementState.MEASURING_NOISE:
        st.info(ui_message('ui.6451b221f7ece6'))
    elif snapshot.device_info.get("gain_recheck_required", False):
        st.warning(
            preflight_message or "マイクGain変更後の再確認待ちです。Pilot ESSをもう一度再生してください。",
            icon=":material/replay:",
        )
    elif preflight_message and not snapshot.device_info.get("preflight_ready", False):
        st.warning(preflight_message)
    elif preflight_message and snapshot.device_info.get("low_snr_continue", False):
        st.warning(preflight_message, icon=":material/signal_cellular_alt_1_bar:")
    elif preflight_message:
        st.success(preflight_message)
    pilot_history = snapshot.device_info.get("pilot_history", [])
    if pilot_history:
        latest_pilot = pilot_history[-1]
        playback_spl = latest_pilot.get("playback_spl_db")
        if playback_spl is not None and np.isfinite(float(playback_spl)):
            playback_spl = float(playback_spl)
            if playback_spl >= snapshot.settings.playback_red_warning_spl_db:
                st.error(
                    ui_message('ui.d7a1ca455811c0', p0=f'{playback_spl:.1f}'),
                    icon=":material/volume_up:",
                )
            elif playback_spl >= snapshot.settings.playback_warning_spl_db:
                st.warning(
                    ui_message('ui.348d7022ac8523', p0=f'{playback_spl:.1f}'),
                    icon=":material/hearing:",
                )
        with st.expander(ui_message('ui.560ba2548cdca0'), icon=":material/manage_search:"):
            if playback_spl is not None and np.isfinite(float(playback_spl)):
                st.caption(
                    ui_message('ui.448308bc914ec8', p0=f'{float(playback_spl):.1f}', p1=f'{snapshot.settings.playback_target_spl_db:.0f}')
                )
            else:
                st.caption(ui_message('ui.24b04d9ac19faa'))
            active_start = float(latest_pilot.get("active_band_start_hz", 0.0) or 0.0)
            active_end = float(latest_pilot.get("active_band_end_hz", 0.0) or 0.0)
            active_count = int(latest_pilot.get("active_band_count", 0) or 0)
            ess_correlation = float(latest_pilot.get("ess_correlation", 0.0) or 0.0)
            synchronization = str(latest_pilot.get("synchronization", "ESS correlation"))
            st.caption(
                ui_message('ui.2bc33dda8023d7', p0=f'{synchronization}', p1=f'{ess_correlation:.3f}')
            )
            if active_start > 0.0 and active_end > active_start:
                st.caption(
                    ui_message('ui.073cf73f80a5ae', p0=f'{active_start:g}', p1=f'{active_end:g}', p2=f'{active_count}')
                )
                st.caption(
                    ui_message('ui.daff7f0a72f70d')
                )
            st.dataframe(pd.DataFrame(pilot_history), hide_index=True, width="stretch")
    if snapshot.device_info.get("last_audio_status"):
        st.warning(
            ui_message('ui.4e8850ea5e0d23', p0=f"{snapshot.device_info['last_audio_status']}", p1=f"{snapshot.device_info.get('audio_status_events', 1)}"),
            icon=":material/warning:",
        )
    audio_recovery_state = str(snapshot.device_info.get("audio_recovery_state", "") or "")
    if audio_recovery_state == "reconnecting":
        st.warning(
            ui_message('ui.3f9118476f04b9'),
            icon=":material/sync:",
        )
    elif audio_recovery_state == "recovered":
        st.info(
            str(snapshot.device_info.get("last_recovery_status", "CoreAudio入力を再接続しました。")),
            icon=":material/settings_backup_restore:",
        )
    if snapshot.error_message:
        retained_message = "Unsaved results available" if snapshot.shots else "No completed Measurements"
        st.error(
            ui_message('ui.c0cf9967533d5f', p0=f'{snapshot.error_message}', p1=f'{len(snapshot.shots)}', p2=f'{retained_message}')
        )
    if not snapshot.shots:
        if snapshot.error_message:
            st.caption(ui_message('ui.71754ca28fd674'))
        elif snapshot.device_info.get("rejected_shots", 0):
            recovery = str(snapshot.device_info.get("last_recovery_status", "") or "")
            if recovery:
                st.info(ui_message('ui.36025e031a0564', p0=f'{recovery}'))
            else:
                st.warning(
                    ui_message('ui.ee63cc4fe744fe', p0=f"{snapshot.device_info.get('last_rejection_reason', 'quality gate')}")
                )
        else:
            st.caption(ui_message('ui.7861fb575679ed'))
        return

    terminal = snapshot.state in {
        MeasurementState.STOPPED_WITH_RESULTS,
        MeasurementState.ERROR_RECOVERABLE,
        MeasurementState.ERROR_FATAL,
    }
    if terminal and snapshot.standard_result is None:
        engine.finalize_standard()
        snapshot = engine.snapshot()

    # During measurement, keep the graph on the latest usable result.  If the
    # newest shot is rejected, leave the last accepted/warning response visible.
    # After Stop, expose an explicit table selector for Final/Denoised/shots.
    selected = _latest_display_shot(snapshot) or snapshot.shots[-1]
    selected_result_id = f"shot:{selected.shot_index}"
    result_view_key = f"ess_result_view_{snapshot.session_id}"
    st.session_state.setdefault(result_view_key, "Gain / Phase")
    result_view = localized_segmented_control(
        ui_message('ui.b2f96cd9f07c3f'),
        ["Gain / Phase", "Impulse", "Wavelet", "App diagnostics"],
        key=result_view_key,
        width="stretch",
    )
    if terminal:
        selected_result_id = _measurement_result_selector(
            snapshot,
            default_id="standard" if snapshot.standard_result is not None else selected_result_id,
        )
        if selected_result_id.startswith("shot:"):
            selected = _shot_by_index(snapshot, int(selected_result_id.split(":", 1)[1])) or selected
        elif selected_result_id == "standard":
            selected = _reference_shot_for_result(snapshot, snapshot.standard_result) or selected
        elif selected_result_id == "denoised":
            selected = _reference_shot_for_result(snapshot, snapshot.denoised_result) or selected
    wavelet_center = (
        getattr(selected.wavelet_map, "alignment_center_sample", None)
        if selected.wavelet_map is not None else None
    )
    wavelet_needs_alignment = (
        selected.wavelet_map is None
        or wavelet_center is None
        or not np.isfinite(wavelet_center)
        or not np.isclose(float(wavelet_center), float(selected.center_sample), atol=0.01)
    )
    if (
        result_view == "Wavelet"
        and wavelet_needs_alignment
        and selected.combined_raw_ir is not None
    ):
        engine.update_wavelet(selected.shot_index)
        snapshot = engine.snapshot()
        selected = _shot_by_index(snapshot, selected.shot_index) or _latest_display_shot(snapshot) or snapshot.shots[-1]
        st.session_state[UNSAVED_KEY] = True
    if terminal:
        st.caption(ui_message('ui.e7fb78cf42741b', p0=f'{selected_result_id}', p1=f'{selected.shot_index}'))
    else:
        st.caption(ui_message('ui.58b50ae857de14', p0=f'{selected.shot_index}'))
    if not selected.merge_valid:
        gate_limit_hz = 1.0 / max(snapshot.settings.gate_end_ms / 1000.0, 1e-4)
        st.warning(
            ui_message('ui.f4f5a137b0d259', p0=f'{snapshot.settings.gate_end_ms:g}', p1=f'{gate_limit_hz:.0f}'),
            icon=":material/warning:",
        )
    average_shots = list(snapshot.valid_shots)
    average = None
    precision = (
        snapshot.standard_result
        if terminal and snapshot.standard_result is not None else
        _cached_high_precision(snapshot.session_id, snapshot.revision, average_shots)
    )
    if precision is not None:
        comparison = (precision.frequency_hz, precision.merged_complex_response)
    elif terminal:
        comparison = None
    else:
        average = average_responses(average_shots)
        comparison = average
    comparison_label = "Final" if terminal and precision is snapshot.standard_result else "Live preview"
    measurement_ready = False
    high_precision_ready = False
    readiness_reasons: list[str] = []
    high_precision_reasons: list[str] = []
    st.caption(
        ui_message('ui.fc4c5aa17da4a4')
        if terminal and snapshot.standard_result is not None else
        ui_message('ui.0fb5f3472abe90')
    )
    if precision is None:
        readiness_reasons = ["統合に使用できるAccepted Measurementがありません"]
    else:
        measurement_ready, readiness_reasons = _measurement_result_assessment(
            precision,
            require_clock=False,
        )
        high_precision_ready, high_precision_reasons = _measurement_result_assessment(
            precision,
            require_clock=(
                terminal and snapshot.settings.reference_mode == "timing_marker"
            ),
            require_high_precision=True,
        )
    if terminal:
        if selected_result_id == "standard" and snapshot.standard_result is not None:
            comparison = (snapshot.standard_result.frequency_hz, snapshot.standard_result.merged_complex_response)
            comparison_label = "Selected final"
        elif selected_result_id == "denoised" and snapshot.denoised_result is not None:
            comparison = (snapshot.denoised_result.frequency_hz, snapshot.denoised_result.merged_complex_response)
            comparison_label = "Selected Denoised"
        elif selected_result_id.startswith("shot:"):
            comparison = None
            comparison_label = "Selected Measurement"
        with st.popover(ui_message('ui.f3f51a2f2a34c0'), icon=":material/auto_fix_high:", width="stretch"):
            if localized_button(st, ui_message('ui.4246174e38bdf3'), icon=":material/restart_alt:", width="stretch"):
                selected_clock_mode = {
                    "Force": "force",
                    "Off": "off",
                }.get(str(st.session_state.get("ess_measure_clock_mode", "Auto")), "auto")
                engine.finalize_standard(clock_mode=selected_clock_mode, force=True)
                st.session_state[UNSAVED_KEY] = True
                st.rerun(scope="fragment")
            st.session_state.setdefault(f"ess_denoise_max_{snapshot.session_id}", 3.0)
            denoise_max = float(st.number_input(
                ui_message('ui.7d2c6f07655ee8'),
                min_value=0.0,
                step=0.5,
                key=f"ess_denoise_max_{snapshot.session_id}",
                help=ui_message('ui.53244e99b688ae'),
            ))
            if localized_button(st, ui_message('ui.b828e59e7b383b'), icon=":material/auto_fix_high:", width="stretch"):
                engine.create_denoised(max_attenuation_db=denoise_max)
                st.session_state[UNSAVED_KEY] = True
                st.rerun(scope="fragment")
    if result_view == "Gain / Phase":
        display_cache_key = _measurement_display_cache_key(
            snapshot,
            selected_result_id=selected_result_id,
            selected_shot_index=selected.shot_index,
            view="Gain / Phase",
            graph_points=graph_points,
            gain_smoothing_fraction=gain_smoothing_fraction,
        )
        _response_charts(
            selected,
            comparison,
            comparison_label=comparison_label,
            graph_mode=str(graph_mode),
            graph_points=graph_points,
            gain_smoothing_fraction=gain_smoothing_fraction,
            settings=snapshot.settings,
            display_cache_key=display_cache_key,
        )
    elif result_view == "Impulse":
        _ir_chart(
            selected,
            graph_mode=str(graph_mode),
            graph_points=graph_points,
            display_cache_key=_measurement_display_cache_key(
                snapshot,
                selected_result_id=selected_result_id,
                selected_shot_index=selected.shot_index,
                view="Impulse",
                graph_points=graph_points,
            ),
        )
    elif result_view == "Wavelet":
        _wavelet_chart(selected, graph_mode=str(graph_mode), graph_points=graph_points)
    else:
        st.caption(ui_message('ui.7b1c52eca71928'))
        diag_cols = st.columns(4)
        diag_cols[0].metric(ui_message('ui.a3b50c476732c7'), snapshot.state)
        diag_cols[1].metric(ui_message('ui.a00fb0c50741f8'), len(snapshot.shots))
        diag_cols[2].metric(ui_message('ui.f4ed8fa656b74c'), len(snapshot.valid_shots))
        diag_cols[3].metric(ui_message('ui.aea4a04a80426e'), snapshot.device_info.get("rejected_shots", 0))
        if precision is not None:
            quality_cols = st.columns(4)
            quality_cols[0].metric(ui_message('ui.5eb7be0ba62c3d'), precision.shot_count)
            quality_cols[1].metric(ui_message('ui.299c42ee05cc68'), "N/A" if precision.shot_count == 1 else f"{precision.median_coherence:.3f}")
            quality_cols[2].metric(ui_message('ui.7d53d7dce7d14a'), "N/A" if precision.shot_count == 1 else f"{precision.repeatability_p90_db:.2f} dB")
            quality_cols[3].metric(ui_message('ui.58a5bdb3dc5b07'), f"{max((abs(value) for value in precision.alignment_samples), default=0.0):.2f} sample")
        _render_shot_diagnostics(engine, snapshot)
        _quality_chart(snapshot.standard_result or precision, graph_mode=str(graph_mode), graph_points=graph_points)

    with st.expander(ui_message('ui.268f14bbfe119c'), icon=":material/rule:"):
        if precision is None:
            st.warning(ui_message('ui.91a96664b2ef22'))
        elif measurement_ready:
            st.success(
                ui_message('ui.80aff31c89380b'),
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                ui_message('ui.58765c273e8cc4') + " / ".join(readiness_reasons),
                icon=":material/pending_actions:",
            )
        if precision is not None:
            st.caption(
                ui_message('ui.9ff87da642015a') if high_precision_ready else
                ui_message('ui.d9404139fbb482') + " / ".join(high_precision_reasons)
            )

    if terminal:
        db_precision = snapshot.standard_result or precision or _cached_high_precision(
            snapshot.session_id,
            snapshot.revision,
            list(snapshot.valid_shots),
        )
        st.markdown(ui_message('ui.b43c09f277c002'))
        st.caption(ui_message('ui.fb9bc26e40d42f'))
        measurement_name_key = f"ess_db_name_{snapshot.session_id}"
        st.session_state.setdefault(measurement_name_key, f"ESS {date.today().isoformat()}")
        measurement_name = st.text_input(
            ui_message('ui.e98017ab1725d0'),
            key=measurement_name_key,
        )
        st.session_state.setdefault(f"ess_db_scope_{snapshot.session_id}", "system_channel")
        st.session_state.setdefault(f"ess_db_channel_{snapshot.session_id}", "")
        st.session_state.setdefault(f"ess_db_use_{snapshot.session_id}", "correction_input")
        scope_cols = st.columns([0.28, 0.28, 0.18, 0.26])
        with scope_cols[0]:
            measurement_scope = localized_segmented_control(
                ui_message('ui.978354db0c00fc'),
                ["system_channel", "combined_room"],
                format_func=lambda value: {
                    "system_channel": "System",
                    "combined_room": "L+R",
                }.get(str(value), str(value)),
                key=f"ess_db_scope_{snapshot.session_id}",
                help=ui_message('ui.c888060da93398'),
                width="stretch",
            )
        with scope_cols[1]:
            system_name = st.text_input(
                ui_message('ui.6725e7bbcd28f3'),
                key=f"ess_db_system_{snapshot.session_id}",
                placeholder=ui_message('ui.0a63c2e462f0f8'),
            )
        with scope_cols[2]:
            channel = st.selectbox(
                ui_message('ui.ce4683e7013a18'),
                ["", "Left", "Right", "Center", "Mono", "L+R"],
                key=f"ess_db_channel_{snapshot.session_id}",
                help=ui_message('ui.7acadeeac7f931'),
            format_func=localized_formatter(str))
        with scope_cols[3]:
            position_name = st.text_input(
                ui_message('ui.6d031af10da7a2'),
                key=f"ess_db_position_{snapshot.session_id}",
                placeholder=ui_message('ui.c5ad34d720f0c8'),
            )
        state_cols = st.columns([0.5, 0.5])
        with state_cols[0]:
            system_state = st.text_input(
                ui_message('ui.ea70af6da7f40e'),
                key=f"ess_db_state_{snapshot.session_id}",
                placeholder=ui_message('ui.e49800da50b425'),
            )
        with state_cols[1]:
            intended_use = localized_segmented_control(
                ui_message('ui.c36d819e7bc6d2'),
                ["correction_input", "verification_only", "archive"],
                format_func=lambda value: {
                    "correction_input": "Correction",
                    "verification_only": "Check",
                    "archive": "Archive",
                }.get(str(value), str(value)),
                key=f"ess_db_use_{snapshot.session_id}",
                help=ui_message('ui.a48d8639a8395b'),
                width="stretch",
            )
            if measurement_scope == "combined_room" or channel == "L+R":
                intended_use = "verification_only"
                st.caption(ui_message('ui.cf487b6eb6e92e'))
        if localized_button(
            st,
            ui_message('ui.d8e0a9cad5af76'),
            type="primary",
            width="stretch",
            disabled=not measurement_ready,
            help=None if measurement_ready else "測定OKになってから登録できます。",
        ):
            try:
                original = attach_calibration_sources(original_from_snapshot(snapshot), measurement_db_path)
                db_precision = result_from_original(original)
            except (ValueError, OSError) as exc:
                st.error(ui_message('ui.b19dcb00ba2aa8', p0=f'{exc}'))
                return
            response = (db_precision.frequency_hz, db_precision.merged_complex_response)
            if response is None:
                st.error(ui_message('ui.5d9e9a5d50d09f'))
            else:
                frequency, complex_response = response
                calibrated_spl = response_level_spl(complex_response, snapshot.settings)
                stored_gain = (
                    calibrated_spl
                    if calibrated_spl is not None else
                    response_level_dbfs(complex_response, snapshot.settings)
                )
                stored_gain_unit = (
                    "Estimated dB SPL"
                    if calibrated_spl is not None and level_calibration_kind(snapshot.settings) == "estimated" else
                    "dB SPL" if calibrated_spl is not None else
                    "dBFS RMS"
                )
                speaker = SpeakerResponse(
                    frequency=frequency.tolist(),
                    gain_db=np.asarray(stored_gain).tolist(),
                    phase_deg=np.rad2deg(np.unwrap(np.angle(complex_response))).tolist(),
                )
                payload = speaker_response_payload(speaker)
                measurement_record_id = str(uuid.uuid4())
                is_precision = high_precision_ready
                timing_provenance: dict[str, object] | None = None
                if snapshot.settings.timing_reference_kind == "phaseeq_tweeter_reference":
                    target_arrival = db_precision.arrival_sample
                    is_reference = snapshot.settings.timing_reference_role == "reference_tweeter"
                    reference_contract_valid = is_reference
                    if is_reference:
                        reference_arrival = target_arrival
                        reference_measurement_id = measurement_record_id
                    else:
                        reference_arrival = float("nan")
                        reference_measurement_id = str(
                            snapshot.settings.reference_measurement_id
                        )
                        validated_reference = _trusted_library_reference_arrival(
                            measurement_db_path,
                            reference_measurement_id,
                            timing_reference_session_id=(
                                snapshot.settings.timing_reference_session_id
                            ),
                            reference_tweeter_channel_id=(
                                snapshot.settings.reference_tweeter_channel_id
                            ),
                            sample_rate_hz=int(snapshot.settings.sample_rate),
                        )
                        reference_contract_valid = validated_reference is not None
                        if validated_reference is not None:
                            _reference_provenance, reference_arrival = validated_reference
                    sequence_valid = bool(
                        snapshot.settings.reference_mode == "timing_marker"
                        and db_precision.absolute_timing_valid
                        and snapshot.settings.tweeter_reference_setup_ready
                        and reference_contract_valid
                    )
                    arrival_valid = bool(
                        sequence_valid
                        and np.isfinite(target_arrival)
                        and np.isfinite(reference_arrival)
                    )
                    relative_samples = (
                        0.0 if is_reference
                        else target_arrival - reference_arrival
                    ) if arrival_valid else float("nan")
                    source_ir_hash = original.sha256
                    timing_provenance = TimingProvenance(
                        timing_reference_kind="phaseeq_tweeter_reference",
                        timing_reference_session_id=snapshot.settings.timing_reference_session_id,
                        reference_tweeter_channel_id=snapshot.settings.reference_tweeter_channel_id,
                        reference_measurement_id=reference_measurement_id,
                        source_measurement_id=measurement_record_id,
                        sample_rate_hz=int(snapshot.settings.sample_rate),
                        target_arrival_sample=target_arrival,
                        reference_arrival_sample=reference_arrival,
                        relative_arrival_samples=relative_samples,
                        relative_arrival_ms=(
                            relative_samples * 1_000.0 / snapshot.settings.sample_rate
                            if np.isfinite(relative_samples) else float("nan")
                        ),
                        removed_bulk_delay_samples=float(target_arrival),
                        clock_adjustment_ppm=float(
                            db_precision.clock_drift_ppm if db_precision is not None else 0.0
                        ),
                        sequence_timing_valid=sequence_valid,
                        common_reference_timing_valid=arrival_valid,
                        arrival_time_valid=arrival_valid,
                        confidence=(
                            "trusted"
                            if arrival_valid and db_precision.minimum_center_confidence >= 0.2
                            else "invalid"
                        ),
                        source_ir_hash=source_ir_hash,
                        measurement_recipe_hash=hashlib.sha256(
                            repr(snapshot.settings).encode("utf-8")
                        ).hexdigest(),
                    ).to_dict()
                    payload["timing_provenance"] = timing_provenance
                    if not arrival_valid:
                        st.warning(
                            ui_message('ui.76f37b0e3f9e57')
                        )
                payload["measurement_session"] = {
                    "session_id": snapshot.session_id,
                    "selected_shot": selected.shot_index,
                    "valid_shots": len(snapshot.valid_shots),
                    "excluded_shots": len(snapshot.shots) - len(snapshot.valid_shots),
                    "gate_start_ms": snapshot.settings.gate_start_ms,
                    "gate_end_ms": snapshot.settings.gate_end_ms,
                    "merge_crossover_hz": snapshot.settings.merge_crossover_hz,
                    "reference_mode": snapshot.settings.reference_mode,
                    "sync_polarity": snapshot.settings.sync_polarity,
                    "reference_plane": selected.reference_plane,
                    "absolute_timing_valid": selected.absolute_timing_valid,
                    "timing_provenance": timing_provenance,
                    "center_method": selected.center_method,
                    "center_polarity": selected.center_polarity,
                    "center_sample": selected.center_sample,
                    "center_tracking_state": selected.center_tracking_state,
                    "center_search_samples": [
                        selected.center_search_start_sample,
                        selected.center_search_end_sample,
                    ],
                    "input_gain_normalization_db": selected.input_gain_normalization_db,
                    "merge_valid": selected.merge_valid,
                    "effective_merge_crossover_hz": selected.effective_merge_crossover_hz,
                    "mic_calibration_id": snapshot.settings.mic_calibration_id,
                    "calibration_application": {
                        "schema_version": 1,
                        "magnitude_applied": bool(
                            snapshot.settings.mic_calibration_frequency_hz
                            and snapshot.settings.mic_calibration_gain_db
                        ),
                        "phase_applied": bool(
                            snapshot.settings.mic_phase_calibration_frequency_hz
                            and snapshot.settings.mic_phase_calibration_deg
                        ),
                        "curve_semantics": "microphone_deviation",
                        "operation": "subtract",
                        "extrapolation": snapshot.settings.mic_calibration_extrapolation,
                    },
                    "mic_calibration_angle_deg": snapshot.settings.mic_calibration_angle_deg,
                    "mic_phase_calibration_id": snapshot.settings.mic_phase_calibration_id,
                    "microphone_profile_id": snapshot.settings.microphone_profile_id,
                    "microphone_unit_id": snapshot.settings.microphone_unit_id,
                    "microphone_serial_number": snapshot.settings.microphone_serial_number,
                    "microphone_nominal_bit_depth": snapshot.settings.microphone_nominal_bit_depth,
                    "input_channel": snapshot.settings.input_channel,
                    "level": {
                        "gain_unit": stored_gain_unit,
                        "distance_m": snapshot.settings.measurement_distance_m,
                        "calibration_mode": snapshot.settings.level_calibration_mode,
                        "calibration_kind": level_calibration_kind(snapshot.settings),
                        "calibration_offset_db": level_calibration_offset_db(snapshot.settings),
                        "mic_sensitivity_dbfs_per_pa": snapshot.settings.mic_sensitivity_dbfs_per_pa,
                        "mic_sensitivity_mv_pa": snapshot.settings.mic_sensitivity_mv_pa,
                        "interface_full_scale_vrms": snapshot.settings.interface_full_scale_vrms,
                        "input_gain_db": snapshot.settings.input_gain_db,
                        "input_gain_db_channels": list(snapshot.settings.input_gain_db_channels),
                        "input_gain_max_db_channels": list(snapshot.settings.input_gain_max_db_channels),
                        "input_gain_device_name": snapshot.settings.input_gain_device_name,
                        "input_analog_gain_db": snapshot.settings.input_analog_gain_db,
                        "minidsp_sens_factor_db": snapshot.settings.minidsp_sens_factor_db,
                        "minidsp_analog_gain_db": snapshot.settings.minidsp_analog_gain_db,
                        "minidsp_gain_adjustment_db": snapshot.settings.minidsp_gain_adjustment_db,
                        "reference_spl_db": snapshot.settings.level_reference_spl_db,
                        "reference_dbfs_rms": snapshot.settings.level_reference_dbfs,
                        "reference_peak_dbfs": snapshot.settings.reference_peak_dbfs,
                        "note": snapshot.settings.level_calibration_note,
                    },
                    "ess_reference": {
                        "input_mode": snapshot.settings.input_mode,
                        "source_name": snapshot.settings.reference_source_name,
                        "source_sample_rate_hz": snapshot.settings.reference_source_sample_rate,
                        "start_frequency_hz": snapshot.settings.start_frequency_hz,
                        "end_frequency_hz": snapshot.settings.end_frequency_hz,
                        "sweep_duration_s": snapshot.settings.sweep_duration_s,
                        "shot_period_s": snapshot.settings.shot_period_s,
                        "reference_output_channel": snapshot.settings.reference_output_channel,
                        "measurement_output_channel": snapshot.settings.measurement_output_channel,
                        "routing_verified": snapshot.settings.reference_routing_verified,
                        "external_playback_setup_confirmed": (
                            snapshot.settings.external_playback_setup_confirmed
                        ),
                        "reference_protection_confirmed": (
                            snapshot.settings.reference_protection_confirmed
                        ),
                        "alignment_processing_bypassed_confirmed": (
                            snapshot.settings.alignment_processing_bypassed_confirmed
                        ),
                        "external_playback_setup_note": (
                            snapshot.settings.external_playback_setup_note
                        ),
                    },
                    "quality": {
                        "status": "high_precision" if is_precision else "basic",
                        "snr_db": selected.quality.snr_db,
                        "correlation": selected.correlation,
                        "median_coherence": db_precision.median_coherence if is_precision else None,
                        "repeatability_p90_db": db_precision.repeatability_p90_db if is_precision else None,
                    },
                    "derivation": (
                        {
                            "method": "subsample_aligned_quality_weighted_robust_complex_average",
                            "used_shot_indices": list(db_precision.used_shot_indices),
                            "reference_shot_index": db_precision.reference_shot_index,
                            "alignment_samples": list(db_precision.alignment_samples),
                        }
                        if is_precision else None
                    ),
                    "ir_sample_rate_hz": snapshot.settings.sample_rate,
                }
                try:
                    save_measurement(
                        measurement_db_path,
                        original=original,
                        record=SpeakerMeasurementRecord(
                            id=measurement_record_id,
                            name=measurement_name,
                            measurement_date=date.today().isoformat(),
                            distance=f"{snapshot.settings.measurement_distance_m:g} m",
                            measurement_scope=str(measurement_scope or "system_channel"),
                            system_name=str(system_name).strip(),
                            channel=str(channel).strip(),
                            position_name=str(position_name).strip(),
                            system_state=str(system_state).strip(),
                            intended_use=str(intended_use or "correction_input"),
                            microphone=(
                                f"{snapshot.device_info.get('name', 'USB measurement microphone')} "
                                f"S/N {snapshot.settings.microphone_serial_number}"
                                if snapshot.settings.microphone_serial_number else
                                str(snapshot.device_info.get("name", "USB measurement microphone"))
                            ),
                            correction_note=(
                                f"{stored_gain_unit} / {snapshot.settings.level_calibration_mode} / "
                                f"Gate {snapshot.settings.gate_start_ms:g}..{snapshot.settings.gate_end_ms:g} ms"
                            ),
                            source_name=f"ESS session {snapshot.session_id[:8]}",
                            source_type="speaker_input_raw",
                            response_payload=payload,
                            note=f"{len(snapshot.valid_shots)} valid shots / {len(snapshot.shots)} total",
                        ),
                    )
                except (OSError, ValueError, sqlite3.Error) as exc:
                    st.error(ui_message('ui.8d44fa6d846e02', p0=f'{exc}'))
                    return
                st.session_state[UNSAVED_KEY] = False
                try:
                    released_bytes = cleanup_measurement_autosave(snapshot)
                except (OSError, ValueError) as exc:
                    st.warning(
                        ui_message('ui.e17f2f216bb0f6', p0=f'{exc}')
                    )
                else:
                    released_mb = released_bytes / (1024 * 1024)
                    st.success(
                        ui_message('ui.9625d985d6ae9c', p0=f'{released_mb:.1f}')
                    )


def _resolve_minidsp_level_calibration(
    *,
    profile: MicrophoneProfile | None,
    unit: MicrophoneUnit | None,
    calibration_id: str,
    records: dict[str, SpeakerMeasurementRecord],
    device_name: str,
    channels: int,
) -> dict[str, object]:
    gain_snapshot = read_input_gain_snapshot(device_name, channels) if device_name else read_input_gain_snapshot("", channels)
    result: dict[str, object] = {
        "available": False,
        "factor": None,
        "analog_gain": None,
        "gain_adjustment": None,
        "gain_snapshot": gain_snapshot,
        "reason": "個体校正ファイルを選択してください",
        "frequency_status": "",
    }
    if profile is None or profile.id not in MINIDSP_PRODUCT_URLS or unit is None:
        return result
    record = records.get(calibration_id)
    if record is None:
        return result
    if record.source_url != MINIDSP_PRODUCT_URLS[profile.id]:
        result["reason"] = "miniDSP公式取得元として登録されたファイルではありません"
        return result
    if not calibration_text_matches_serial(
        unit.serial_number,
        record.name,
        record.source_name,
        record.microphone,
        record.note,
        record.raw_text,
    ):
        result["reason"] = "校正ファイルのSERNOとマイク個体が一致しません"
        return result
    if profile.id == "umik-1":
        result["frequency_status"] = (
            "Manufacturer calibrated: 20 Hz–20 kHz ±1 dB（校正ファイル適用時）。"
            "範囲外も測定・保存・FIR補正を継続します。"
        )
    else:
        response = speaker_response_from_payload(record.response_payload)
        coverage = (
            f" File coverage {min(response.frequency):g}–{max(response.frequency):g} Hz。"
            if response is not None and response.frequency else
            ""
        )
        result["frequency_status"] = (
            "Manufacturer individually calibrated。公開精度・保証周波数範囲は未指定。"
            + coverage
            + "範囲外も測定・保存・FIR補正を継続します。"
        )
    linked_ids = [unit.calibration_measurement_id, unit.calibration_90_measurement_id]
    linked_records = [records.get(value) for value in linked_ids]
    if any(value is None for value in linked_records):
        result["reason"] = "0°と90°の両校正ファイルが個体へリンクされていません"
        return result
    headers = [minidsp_calibration_header(value.raw_text) for value in linked_records if value is not None]
    if len(headers) != 2 or any(value is None for value in headers):
        result["reason"] = "Sens Factorを校正ファイルから読み取れません"
        return result
    factor_values = [float(value[0]) for value in headers if value is not None]
    analog_values = [value[1] for value in headers if value is not None]
    result["factor"] = factor_values[0]
    result["analog_gain"] = analog_values[0]
    if abs(factor_values[0] - factor_values[1]) > 0.01:
        result["reason"] = "0°と90°のSens Factorが一致しません"
        return result
    if analog_values[0] != analog_values[1]:
        result["reason"] = "0°と90°のAGainが一致しません"
        return result
    if not gain_snapshot.available:
        result["reason"] = gain_snapshot.error or "入力Gainを取得できません"
        return result
    gain_adjustment = gain_snapshot.gain_adjustment_from_max_db
    result["gain_adjustment"] = gain_adjustment
    if gain_adjustment is None:
        result["reason"] = "左右チャンネルの入力Gain条件を一意に補正できません"
        return result
    expected_analog_gain = analog_values[0]
    if expected_analog_gain is not None:
        if gain_snapshot.analog_gain_db is None:
            result["reason"] = "校正ファイルのAGainと現在の内部Gainを照合できません"
            return result
        if abs(float(expected_analog_gain) - float(gain_snapshot.analog_gain_db)) > 0.05:
            result["reason"] = (
                f"校正AGain {float(expected_analog_gain):g} dBと現在の内部Gain "
                f"{float(gain_snapshot.analog_gain_db):g} dBが一致しません"
            )
            return result
    result["available"] = True
    result["reason"] = ""
    return result


def _render_official_calibration_assistant(
    *,
    profile: MicrophoneProfile,
    selected_unit: MicrophoneUnit | None,
    device_name: str,
    input_channel: int,
    running: bool,
    measurement_db_path: Path,
    microphone_db_path: Path,
) -> None:
    source = OFFICIAL_CALIBRATION_SOURCES[profile.id]
    identity = selected_unit.id if selected_unit is not None else "new"
    serial_key = f"ess_official_calibration_serial_{profile.id}_{identity}"
    folder_key = f"ess_official_calibration_folder_{profile.id}_{identity}"
    watch_key = f"_ess_official_calibration_watch_{profile.id}_{identity}"
    upload_nonce_key = f"_ess_official_calibration_upload_nonce_{profile.id}_{identity}"
    st.session_state.setdefault(
        serial_key,
        selected_unit.serial_number if selected_unit is not None else "",
    )
    st.session_state.setdefault(folder_key, str(Path.home() / "Downloads"))
    st.session_state.setdefault(upload_nonce_key, 0)
    with st.expander(ui_message('ui.9a3d5d69d72bbf', p0=f'{source.provider}')):
        st.caption(
            ui_message('ui.e159ae47db2497', p0=f'{source.download_description}')
        )
        serial = st.text_input(
            ui_message('ui.34531c6a3b2df8'),
            key=serial_key,
            disabled=running,
            placeholder=profile.serial_format_hint,
        )
        folder = st.text_input(
            ui_message('ui.5cbcb993983c50'),
            key=folder_key,
            disabled=running,
            help=ui_message('ui.8895931ffddcc8'),
        )
        with st.container(horizontal=True, gap="small"):
            if localized_button(st,
                ui_message('ui.60b14f6b881d1d'),
                key=f"ess_official_calibration_start_{profile.id}_{identity}",
                icon=":material/folder_open:",
                disabled=running or not str(serial).strip() or not str(folder).strip(),
            ):
                try:
                    normalized = normalized_calibration_serial(profile.id, str(serial))
                    download_folder = Path(str(folder)).expanduser()
                    if not download_folder.is_dir():
                        raise ValueError("Download folder does not exist or is not accessible")
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state[watch_key] = {
                        "serial": normalized,
                        "folder": str(download_folder),
                        "started_at": time.time(),
                    }
                    st.rerun()
            st.link_button(
                ui_message('ui.3f17b7dc0729d9'),
                source.url,
                icon=":material/open_in_new:",
                disabled=running,
            )
            if localized_button(st,
                ui_message('ui.fd92bab1a6f462'),
                key=f"ess_official_calibration_stop_{profile.id}_{identity}",
                disabled=watch_key not in st.session_state,
            ):
                st.session_state.pop(watch_key, None)
                st.rerun()
        _render_official_calibration_watch(
            profile=profile,
            identity=identity,
            watch_key=watch_key,
            device_name=device_name,
            input_channel=input_channel,
            running=running,
            measurement_db_path=measurement_db_path,
            microphone_db_path=microphone_db_path,
        )
        with st.container(border=True):
            st.markdown(ui_message('ui.84b47cfa7436b1'))
            uploaded = st.file_uploader(
                ui_message('ui.b51db72d909f4b'),
                type=[extension.removeprefix(".") for extension in source.extensions],
                accept_multiple_files=True,
                key=(
                    f"ess_official_calibration_files_{profile.id}_{identity}_"
                    f"{st.session_state[upload_nonce_key]}"
                ),
                disabled=running,
            )
            files = list(uploaded or [])
            if localized_button(st,
                ui_message('ui.6c1da61f816385'),
                key=f"ess_official_calibration_manual_{profile.id}_{identity}",
                icon=":material/verified:",
                disabled=running or not str(serial).strip() or not files,
                width="stretch",
            ):
                try:
                    normalized = normalized_calibration_serial(profile.id, str(serial))
                    detected = calibrations_from_uploaded_files(
                        [(item.name, item.getvalue()) for item in files],
                        profile_id=profile.id,
                        serial=normalized,
                    )
                    saved_unit, records = register_official_calibrations(
                        profile=profile,
                        serial=normalized,
                        files=detected,
                        measurement_db_path=measurement_db_path,
                        microphone_db_path=microphone_db_path,
                        device_name=device_name,
                        input_channel=input_channel,
                    )
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    st.error(str(exc))
                else:
                    _select_registered_calibration(saved_unit, records)
                    st.session_state[upload_nonce_key] += 1
                    st.success(ui_message('ui.1f725238de8769', p0=f'{profile.model}', p1=f'{normalized}'))
                    st.rerun()


@st.fragment(run_every=1.0)
def _render_official_calibration_watch(
    *,
    profile: MicrophoneProfile,
    identity: str,
    watch_key: str,
    device_name: str,
    input_channel: int,
    running: bool,
    measurement_db_path: Path,
    microphone_db_path: Path,
) -> None:
    watch = st.session_state.get(watch_key)
    if not isinstance(watch, dict):
        st.caption(ui_message('ui.fa06410e613a31'))
        return
    if running:
        st.warning(ui_message('ui.537c54610a3bde'))
        return
    elapsed = time.time() - float(watch.get("started_at", 0.0))
    if elapsed > 300.0:
        st.session_state.pop(watch_key, None)
        st.warning(ui_message('ui.44522cfe8636fc'))
        return
    try:
        detected = find_downloaded_calibrations(
            Path(str(watch["folder"])),
            profile_id=profile.id,
            serial=str(watch["serial"]),
            modified_after=float(watch["started_at"]),
        )
        required = set(OFFICIAL_CALIBRATION_SOURCES[profile.id].required_angles_deg)
        missing = required - set(detected)
        if missing:
            found_text = ", ".join(f"{angle}°" for angle in sorted(detected)) or "none"
            missing_text = ", ".join(f"{angle}°" for angle in sorted(missing))
            st.info(
                ui_message('ui.9db79e8ab91f79', p0=f'{max(0, 300 - int(elapsed))}', p1=f'{found_text}', p2=f'{missing_text}')
            )
            return
        saved_unit, records = register_official_calibrations(
            profile=profile,
            serial=str(watch["serial"]),
            files=detected,
            measurement_db_path=measurement_db_path,
            microphone_db_path=microphone_db_path,
            device_name=device_name,
            input_channel=input_channel,
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        st.error(ui_message('ui.9bfda9dbb4c614', p0=f'{exc}'))
        return
    st.session_state.pop(watch_key, None)
    _select_registered_calibration(saved_unit, records)
    st.success(
        ui_message('ui.978def1d8f28f8', p0=f'{profile.model}', p1=f'{saved_unit.serial_number}', p2=f"{', '.join((f'{angle}°' for angle in sorted(records)))}")
    )
    st.rerun()


def _select_registered_calibration(
    unit: MicrophoneUnit,
    records: dict[int, SpeakerMeasurementRecord],
) -> None:
    st.session_state["_ess_measure_pending_unit_id"] = unit.id
    preferred_angle = 0 if 0 in records else min(records)
    st.session_state["ess_measure_calibration_angle"] = preferred_angle
    st.session_state["ess_measure_mic_calibration_id"] = records[preferred_angle].id
    st.session_state["_ess_measure_calibration_auto_unit"] = ""


def _automatic_gain_axis_domain(
    gain_frame: pd.DataFrame,
    *,
    span_db: float = 40.0,
) -> tuple[float, float]:
    """Return a stable 10 dB-aligned window around the visible Gain data."""
    values = pd.to_numeric(gain_frame.get("Value", pd.Series(dtype=float)), errors="coerce")
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -float(span_db), 0.0
    low, high = np.percentile(finite, [5.0, 95.0])
    center = (float(low) + float(high)) / 2.0
    aligned_center = 10.0 * round(center / 10.0)
    half_span = float(span_db) / 2.0
    return aligned_center - half_span, aligned_center + half_span


def _response_charts(
    selected,
    comparison,
    *,
    comparison_label: str,
    graph_mode: str,
    graph_points: int,
    gain_smoothing_fraction: float,
    settings: MeasurementSettings,
    display_cache_key: str | None = None,
) -> None:
    if selected.frequency_hz is None or selected.merged_complex_response is None:
        st.info(ui_message('ui.df9a156d7e7857'))
        return
    response_series = []
    for label, response in (
        ("Selected full", selected.ungated_complex_response),
        ("Selected gated", selected.gated_complex_response),
        ("Selected Merged", selected.merged_complex_response),
    ):
        if response is not None:
            response_series.append((label, np.asarray(selected.frequency_hz), np.asarray(response)))
    if comparison is not None:
        frequency, response = comparison
        response_series.append((f"{comparison_label} Merged", np.asarray(frequency), np.asarray(response)))
    st.caption(
        ui_message('ui.12ffb1d13591c6', p0=f'{comparison_label}')
    )

    calibration_kind = level_calibration_kind(settings)
    spl_available = calibration_kind != "uncalibrated"
    spl_label = "Estimated dB SPL" if calibration_kind == "estimated" else "dB SPL"
    level_unit_options = [spl_label, "dBFS RMS"] if spl_available else ["dBFS RMS"]
    if st.session_state.get("ess_measure_level_unit") not in level_unit_options:
        st.session_state["ess_measure_level_unit"] = level_unit_options[0]
    level_unit = localized_segmented_control(
        ui_message('ui.01fa6be1bcb20b'),
        level_unit_options,
        key="ess_measure_level_unit",
        width="stretch",
        help=ui_message('ui.e91f27de6b5b12'),
    )
    if level_unit in {"dB SPL", "Estimated dB SPL"}:
        gain_offset_db = float(response_level_spl(np.ones(1), settings)[0])
        gain_axis_title = (
            "Estimated sound pressure level (dB SPL)"
            if level_unit == "Estimated dB SPL" else
            "Sound pressure level (dB SPL)"
        )
    else:
        gain_offset_db = float(response_level_dbfs(np.ones(1), settings)[0])
        gain_axis_title = "Microphone input level (dBFS RMS)"
    if display_cache_key is None:
        gain_frame, phase_frame = _measurement_response_frames(
            response_series,
            graph_points=graph_points,
            phase_gain_threshold_db=-80.0,
            gain_smoothing_fraction=gain_smoothing_fraction,
            gain_offset_db=gain_offset_db,
        )
    else:
        gain_frame, phase_frame = _cached_measurement_response_frames(
            content_digest(display_cache_key, gain_offset_db),
            response_series,
            graph_points=graph_points,
            phase_gain_threshold_db=-80.0,
            gain_smoothing_fraction=gain_smoothing_fraction,
            gain_offset_db=gain_offset_db,
        )
    axis_unit_key = "_ess_measure_gain_axis_unit"
    if st.session_state.get(axis_unit_key) != level_unit:
        st.session_state[axis_unit_key] = level_unit
        st.session_state["ess_measure_gain_axis_shift_db"] = 0.0
    st.session_state.setdefault("ess_measure_gain_axis_shift_db", 0.0)
    axis_controls = st.columns([1, 1, 1], gap="small")
    with axis_controls[0]:
        if localized_button(st, ui_message('ui.216e78a9a3a741'), icon=":material/arrow_downward:", width="stretch"):
            st.session_state["ess_measure_gain_axis_shift_db"] -= 10.0
    with axis_controls[1]:
        if localized_button(st, ui_message('ui.60b3cfa21839a4'), icon=":material/center_focus_strong:", width="stretch"):
            st.session_state["ess_measure_gain_axis_shift_db"] = 0.0
    with axis_controls[2]:
        if localized_button(st, ui_message('ui.02908bc5e12ee2'), icon=":material/arrow_upward:", width="stretch"):
            st.session_state["ess_measure_gain_axis_shift_db"] += 10.0
    auto_y_min, auto_y_max = _automatic_gain_axis_domain(gain_frame, span_db=40.0)
    axis_shift_db = float(st.session_state["ess_measure_gain_axis_shift_db"])
    gain_y_min = auto_y_min + axis_shift_db
    gain_y_max = auto_y_max + axis_shift_db
    st.caption(
        ui_message('ui.587cdade99bb00', p0=f'{level_unit}', p1=f'{gain_y_min:+.0f}', p2=f'{gain_y_max:+.0f}', p3=f'{axis_shift_db:+.0f}', p4=f'{settings.measurement_distance_m:g}')
    )
    st.markdown(ui_message('ui.0540656d429ced'))
    _measurement_line_chart(
        gain_frame,
        y_title=gain_axis_title,
        y_domain=[float(gain_y_min), float(gain_y_max)],
        graph_mode=graph_mode,
        height=430,
    )
    st.markdown(ui_message('ui.4e7301dde15566'))
    st.caption(ui_message('ui.40d706262e417a'))
    _measurement_line_chart(
        phase_frame,
        y_title="Phase (deg)",
        y_domain=[-180.0, 180.0],
        graph_mode=graph_mode,
        height=430,
    )


def _quality_chart(result, *, graph_mode: str, graph_points: int) -> None:
    if result is None:
        st.info(ui_message('ui.9a360683564ff2'))
        return
    frequency = np.asarray(result.frequency_hz, dtype=float)
    if frequency.size > graph_points:
        indices = np.linspace(0, frequency.size - 1, graph_points).astype(int)
    else:
        indices = np.arange(frequency.size)
    rows = []
    for label, values in (("Coherence", result.coherence), ("Confidence", result.confidence)):
        for f, value in zip(frequency[indices], np.asarray(values)[indices]):
            if f > 0 and np.isfinite(value):
                rows.append({"Frequency": f, "Value": float(value), "Series": label})
    st.markdown(ui_message('ui.f8230fffa61624'))
    if rows:
        chart = alt.Chart(pd.DataFrame(rows)).mark_line().encode(
            x=alt.X("Frequency:Q", scale=alt.Scale(type="log"), axis=alt.Axis(title=ui_message('ui.5f9597e24414b2'))),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(title=None)),
            color=alt.Color("Series:N", title=None),
            tooltip=["Series:N", alt.Tooltip("Frequency:Q", format=".1f"), alt.Tooltip("Value:Q", format=".3f")]
            if graph_mode == "Interactive" else [],
        ).properties(height=280)
        if graph_mode == "Interactive":
            chart = chart.interactive()
        st.altair_chart(chart, width="stretch")
    else:
        st.caption(ui_message('ui.b60cc221e700f6'))
    repeatability = np.asarray(result.repeatability_db)[indices]
    repeat_frame = pd.DataFrame({"Frequency": frequency[indices], "Repeatability dB": repeatability})
    repeat_frame = repeat_frame.replace([np.inf, -np.inf], np.nan).dropna()
    if not repeat_frame.empty:
        repeat_chart = alt.Chart(repeat_frame).mark_line(color="#98652f").encode(
            x=alt.X("Frequency:Q", scale=alt.Scale(type="log"), axis=alt.Axis(title=ui_message('ui.5f9597e24414b2'))),
            y=alt.Y("Repeatability dB:Q", axis=alt.Axis(title=ui_message('ui.baa933f4c7a47f'))),
            tooltip=[alt.Tooltip("Frequency:Q", format=".1f"), alt.Tooltip("Repeatability dB:Q", format=".3f")]
            if graph_mode == "Interactive" else [],
        ).properties(height=260)
        if graph_mode == "Interactive":
            repeat_chart = repeat_chart.interactive()
        st.markdown(ui_message('ui.d4a9a377d4292b'))
        st.altair_chart(repeat_chart, width="stretch")


def _measurement_response_frames(
    response_series,
    *,
    graph_points: int,
    phase_gain_threshold_db: float,
    gain_smoothing_fraction: float = 0.0,
    gain_offset_db: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gain_frames: list[pd.DataFrame] = []
    phase_frames: list[pd.DataFrame] = []
    for label, frequency_values, response_values in response_series:
        frequency = np.asarray(frequency_values, dtype=float)
        response = np.asarray(response_values, dtype=complex)
        valid = np.isfinite(frequency) & (frequency >= 10.0) & np.isfinite(response.real) & np.isfinite(response.imag)
        frequency = frequency[valid]
        response = response[valid]
        if frequency.size == 0:
            continue
        raw_gain = 20.0 * np.log10(np.maximum(np.abs(response), 1e-15))
        gain = raw_gain + float(gain_offset_db)
        if gain_smoothing_fraction > 0:
            gain = fractional_octave_smooth(frequency, gain, gain_smoothing_fraction)
        phase = (np.rad2deg(np.angle(response)) + 180.0) % 360.0 - 180.0
        phase[raw_gain < float(phase_gain_threshold_db)] = np.nan
        if frequency.size > int(graph_points):
            indices = np.linspace(0, frequency.size - 1, int(graph_points)).astype(int)
            frequency = frequency[indices]
            gain = gain[indices]
            phase = phase[indices]
        jumps = np.zeros(phase.size, dtype=int)
        if phase.size > 1:
            phase_delta = np.abs(np.diff(phase))
            discontinuity = (np.isfinite(phase_delta) & (phase_delta > 180.0)) | ~np.isfinite(phase[:-1])
            jumps[1:] = np.where(discontinuity, 1, 0)
        segments = np.cumsum(jumps)
        gain_frames.append(pd.DataFrame({"Frequency": frequency, "Value": gain, "Series": label, "Segment": 0}))
        phase_frames.append(
            pd.DataFrame({"Frequency": frequency, "Value": phase, "Series": label, "Segment": segments})
        )
    empty = pd.DataFrame(columns=["Frequency", "Value", "Series", "Segment"])
    return (
        pd.concat(gain_frames, ignore_index=True) if gain_frames else empty.copy(),
        pd.concat(phase_frames, ignore_index=True) if phase_frames else empty.copy(),
    )


@st.cache_data(max_entries=24, ttl=1800, show_spinner=False)
def _cached_measurement_response_frames(
    cache_key: str,
    _response_series,
    *,
    graph_points: int,
    phase_gain_threshold_db: float,
    gain_smoothing_fraction: float,
    gain_offset_db: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = cache_key
    return _measurement_response_frames(
        _response_series,
        graph_points=graph_points,
        phase_gain_threshold_db=phase_gain_threshold_db,
        gain_smoothing_fraction=gain_smoothing_fraction,
        gain_offset_db=gain_offset_db,
    )


def _measurement_line_chart(
    frame: pd.DataFrame,
    *,
    y_title: str,
    y_domain: list[float],
    graph_mode: str,
    height: int,
) -> None:
    data = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["Frequency", "Value"])
    if data.empty:
        st.warning(ui_message('ui.097282056558b3'))
        return
    data = data.copy()
    data["Line group"] = data["Series"].astype(str) + ":" + data["Segment"].astype(str)
    series_names = list(dict.fromkeys(data["Series"].astype(str).tolist()))
    palette = ["#2563eb", "#d97706", "#059669", "#7c3aed"]
    dash_map = [[], [7, 4], [2, 3], []]
    width_map = [1.8, 2.2, 2.4, 3.0]
    tooltip = (
        ["Series:N", alt.Tooltip("Frequency:Q", format=".1f"), alt.Tooltip("Value:Q", format=".2f")]
        if graph_mode == "Interactive" else []
    )
    chart = alt.Chart(data).mark_line(clip=True).encode(
        x=alt.X(
            "Frequency:Q",
            scale=alt.Scale(type="log", domain=[10.0, float(data["Frequency"].max())], nice=False),
            axis=alt.Axis(
                title=ui_message('ui.5f9597e24414b2'),
                format=".0f",
                gridColor="#d8d0c3",
                labelFontSize=16,
                titleFontSize=17,
                labelPadding=8,
                titlePadding=12,
            ),
        ),
        y=alt.Y(
            "Value:Q",
            scale=alt.Scale(domain=y_domain, zero=False, nice=False, clamp=True),
            axis=alt.Axis(title=y_title, format=".2f", gridColor="#d8d0c3", labelFontSize=16, titleFontSize=17),
        ),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=series_names, range=palette[: len(series_names)]),
            legend=alt.Legend(
                title=ui_message('ui.2a45c07314b2cd'),
                orient="top",
                direction="horizontal",
                columns=2,
                labelLimit=260,
                symbolStrokeWidth=3,
            ),
        ),
        strokeDash=alt.StrokeDash(
            "Series:N",
            scale=alt.Scale(domain=series_names, range=dash_map[: len(series_names)]),
            legend=alt.Legend(title=ui_message('ui.2a45c07314b2cd'), orient="top", direction="horizontal", columns=2),
        ),
        strokeWidth=alt.StrokeWidth("Series:N", legend=None, scale=alt.Scale(domain=series_names, range=width_map[: len(series_names)])),
        detail="Line group:N",
        tooltip=tooltip,
    ).properties(height=height)
    if graph_mode == "Interactive":
        chart = chart.interactive()
    st.altair_chart(chart, width="stretch")


def _measurement_impulse_plot_data(selected, graph_points: int):
    if selected.combined_raw_ir is None:
        return None
    raw_ir = np.asarray(selected.combined_raw_ir)
    gated_ir = np.asarray(selected.gated_ir) if selected.gated_ir is not None else None
    settings = selected.settings_snapshot
    sample_rate = selected.settings_snapshot.sample_rate
    center_sample = (
        float(selected.center_sample)
        if np.isfinite(selected.center_sample) else
        float(np.argmax(np.abs(raw_ir)))
    )
    # Deconvolution IRs are circular.  If the direct peak is near the end of
    # the buffer, slicing peak..end makes the visible tail only a few ms long.
    # Rotate both traces so the direct peak is always followed by the complete
    # post-peak response while retaining 5 ms of pre-peak context.
    pre_samples = min(int(round(0.005 * sample_rate)), max(0, len(raw_ir) - 1))
    raw_ir = time_align_impulse_to_center(
        raw_ir,
        center_sample,
        target_sample=float(pre_samples),
    )
    if gated_ir is not None:
        gated_ir = time_align_impulse_to_center(
            gated_ir,
            center_sample,
            target_sample=float(pre_samples),
        )
    peak = pre_samples
    lo = 0
    hi = len(raw_ir)
    available_indices = np.arange(lo, hi)
    if available_indices.size > graph_points:
        # Preserve positive/negative impulse peaks throughout the full tail;
        # uniform point picking can entirely miss a narrow reflection.
        bucket_count = max(1, graph_points // 2)
        boundaries = np.linspace(lo, hi, bucket_count + 1).astype(int)
        envelope_indices: list[int] = []
        for bucket_start, bucket_end in zip(boundaries[:-1], boundaries[1:]):
            bucket_end = max(bucket_start + 1, bucket_end)
            segment = raw_ir[bucket_start:bucket_end]
            envelope_indices.extend((
                bucket_start + int(np.argmin(segment)),
                bucket_start + int(np.argmax(segment)),
            ))
        indices = np.asarray(envelope_indices, dtype=int)
        gate_indices = peak + np.rint(
            np.asarray([settings.gate_start_ms, 0.0, settings.gate_end_ms])
            * sample_rate / 1000.0
        ).astype(int)
        indices = np.unique(np.clip(np.concatenate((indices, gate_indices)), lo, hi - 1))
    else:
        indices = available_indices
    frame = pd.DataFrame(
        {
            "Time ms": (indices - peak) / sample_rate * 1000,
            "Full IR (including after Gate)": raw_ir[indices],
            "Gated IR": gated_ir[indices] if gated_ir is not None else np.nan,
        }
    ).melt("Time ms", var_name="Series", value_name="Amplitude")
    gate_min_ms = float(min(settings.gate_start_ms, settings.gate_end_ms))
    gate_max_ms = float(max(settings.gate_start_ms, settings.gate_end_ms))
    initial_x_min = min(-5.0, gate_min_ms - 2.0)
    initial_x_max = max(50.0, gate_max_ms + 10.0)
    amplitude_limit = max(float(np.nanmax(np.abs(raw_ir))) * 1.05, 1e-9)
    return (
        frame,
        _time_markers(selected),
        gate_min_ms,
        gate_max_ms,
        initial_x_min,
        initial_x_max,
        amplitude_limit,
    )


@st.cache_data(max_entries=24, ttl=1800, show_spinner=False)
def _cached_measurement_impulse_plot_data(cache_key: str, _selected, graph_points: int):
    _ = cache_key
    return _measurement_impulse_plot_data(_selected, graph_points)


def _ir_chart(
    selected,
    *,
    graph_mode: str,
    graph_points: int,
    display_cache_key: str | None = None,
) -> None:
    if display_cache_key is None:
        plot_data = _measurement_impulse_plot_data(selected, graph_points)
    else:
        plot_data = _cached_measurement_impulse_plot_data(display_cache_key, selected, graph_points)
    if plot_data is None:
        return
    (
        frame,
        markers,
        gate_min_ms,
        gate_max_ms,
        initial_x_min,
        initial_x_max,
        amplitude_limit,
    ) = plot_data
    tooltip = (
        ["Series:N", alt.Tooltip("Time ms:Q", format=".3f"), alt.Tooltip("Amplitude:Q", format=".5f")]
        if graph_mode == "Interactive" else []
    )
    lines = alt.Chart(frame).mark_line(clip=True).encode(
        x=alt.X(
            "Time ms:Q",
            scale=alt.Scale(domain=[initial_x_min, initial_x_max], nice=False),
            axis=alt.Axis(title=ui_message('ui.48693e0569ffab'), labelFontSize=16, titleFontSize=17),
        ),
        y=alt.Y(
            "Amplitude:Q",
            scale=alt.Scale(domain=[-amplitude_limit, amplitude_limit], nice=False),
            axis=alt.Axis(title=None, labelFontSize=16),
        ),
        color=alt.Color("Series:N", title=None, scale=alt.Scale(range=["#365e80", "#98652f"])),
        strokeDash=alt.StrokeDash("Series:N", title=None, scale=alt.Scale(range=[[], [7, 4]])),
        tooltip=tooltip,
    )
    gate_band = alt.Chart(pd.DataFrame([{
        "Gate min ms": gate_min_ms,
        "Gate max ms": gate_max_ms,
    }])).mark_rect(color="#d6a84b", opacity=0.12).encode(
        x=alt.X("Gate min ms:Q"),
        x2=alt.X2("Gate max ms:Q"),
        tooltip=[
            alt.Tooltip("Gate min ms:Q", title=ui_message('ui.cb50a1ab45b6b5'), format=".2f"),
            alt.Tooltip("Gate max ms:Q", title=ui_message('ui.69b011f4ff6442'), format=".2f"),
        ],
    )
    rules = alt.Chart(markers).mark_rule(strokeDash=[5, 3]).encode(
        x="Time ms:Q",
        color=alt.Color("Marker:N", legend=alt.Legend(title=ui_message('ui.e1e4a528c6566f'))),
        tooltip=["Marker:N", alt.Tooltip("Time ms:Q", format=".2f")],
    )
    chart = alt.layer(gate_band, lines, rules).properties(height=430)
    if graph_mode == "Interactive":
        chart = chart.interactive(bind_x=True, bind_y=False)
    st.markdown(ui_message('ui.54185cce876321'))
    st.caption(
        ui_message('ui.a2d8e95c5fb1e7')
    )
    st.altair_chart(chart, width="stretch")


def _wavelet_chart(selected, *, graph_mode: str, graph_points: int) -> None:
    wavelet = selected.wavelet_map
    if wavelet is None or wavelet.level_db.size == 0:
        st.caption(ui_message('ui.0ffb0fcf81bbdf'))
        return
    st.markdown(ui_message('ui.6efffa62e74991'))
    st.caption(
        ui_message('ui.49ba702c2cd78c')
    )
    draw_measurement_wavelet_map(wavelet, graph_mode=graph_mode)


def _time_markers(selected) -> pd.DataFrame:
    settings = selected.settings_snapshot
    rows = [
        {"Time ms": 0.0, "Marker": "Direct sound"},
        {"Time ms": float(settings.gate_start_ms), "Marker": "Gate start"},
        {"Time ms": float(settings.gate_end_ms), "Marker": "Gate end"},
    ]
    impulse = selected.combined_raw_ir
    if impulse is not None:
        peak = int(np.argmax(np.abs(impulse)))
        sample_rate = settings.sample_rate
        search_start = min(len(impulse), peak + max(1, int(0.001 * sample_rate)))
        search_end = min(len(impulse), peak + int(0.05 * sample_rate))
        segment = np.abs(np.asarray(impulse)[search_start:search_end])
        if segment.size:
            from scipy import signal

            peaks, properties = signal.find_peaks(
                segment,
                height=max(float(np.max(np.abs(impulse))) * 0.05, 1e-9),
                distance=max(1, int(0.0005 * sample_rate)),
            )
            strongest = sorted(peaks, key=lambda index: float(properties["peak_heights"][np.where(peaks == index)[0][0]]), reverse=True)[:3]
            for number, index in enumerate(sorted(strongest), start=1):
                rows.append({"Time ms": (search_start + int(index) - peak) / sample_rate * 1000, "Marker": f"Reflection {number}"})
    return pd.DataFrame(rows)


def _selected_response(snapshot, selected, choice: str, average, precision=None, denoised=None):
    if choice == "Final merged":
        if precision is None:
            return None
        return precision.frequency_hz, precision.merged_complex_response
    if choice == "Denoised Merged":
        if denoised is None:
            return None
        return denoised.frequency_hz, denoised.merged_complex_response
    if choice == "Averaged Merged":
        return average
    attribute = {
        "Selected Merged": "merged_complex_response",
        "Selected full": "ungated_complex_response",
        "Selected gated": "gated_complex_response",
    }.get(choice, "merged_complex_response")
    value = getattr(selected, attribute)
    return None if selected.frequency_hz is None or value is None else (selected.frequency_hz, value)


def _cached_high_precision(session_id: str, revision: int, shots: list) -> HighPrecisionResult | None:
    """Keep the derived result by measurement revision, never object identity."""

    source_digest = content_digest(
        session_id,
        revision,
        ",".join(f"{shot.shot_index}:{int(shot.included_in_average)}" for shot in shots),
    )
    cache_key = CacheCoordinator(st.session_state).key(
        CacheDomain.MEASUREMENT_RESULT,
        algorithm_version=MEASUREMENT_RESULT_CACHE_ALGORITHM_VERSION,
        source_digest=source_digest,
        revision=int(revision),
    ).digest
    state_key = "_ess_high_precision_cache"
    cached = st.session_state.get(state_key)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == cache_key:
        return cached[1]
    result = high_precision_result(shots)
    st.session_state[state_key] = (cache_key, result)
    return result


def _apply_settings_to_measurement_widgets(
    settings: MeasurementSettings,
    microphone_db_path: Path,
) -> None:
    st.session_state["ess_measure_sample_rate"] = settings.sample_rate
    st.session_state["ess_measure_input_device"] = settings.input_device
    st.session_state["ess_measure_distance_m"] = settings.measurement_distance_m
    st.session_state["ess_measure_level_calibration_mode"] = {
        "minidsp_sens_factor": "UMIK manufacturer Sens Factor",
        "digital_sensitivity": "USB mic sensitivity dBFS/Pa",
        "estimated_digital_sensitivity": "Estimated USB representative sensitivity",
        "analog_sensitivity": "Analog mic sensitivity mV/Pa",
        "acoustic_calibrator": "Acoustic calibrator",
        "spl_meter_comparison": "SPL meter comparison",
    }.get(settings.level_calibration_mode, "Uncalibrated dBFS")
    st.session_state["ess_measure_sensitivity_dbfs_pa_text"] = f"{settings.mic_sensitivity_dbfs_per_pa:g}"
    st.session_state["ess_measure_sensitivity_mv_pa"] = settings.mic_sensitivity_mv_pa
    st.session_state["ess_measure_interface_full_scale_vrms"] = settings.interface_full_scale_vrms
    st.session_state["ess_measure_input_gain_db"] = settings.input_gain_db
    st.session_state["ess_measure_level_reference_spl_db"] = settings.level_reference_spl_db
    st.session_state["ess_measure_level_reference_dbfs_text"] = f"{settings.level_reference_dbfs:g}"
    st.session_state["ess_measure_level_calibration_note"] = settings.level_calibration_note
    st.session_state["ess_measure_gate_start"] = settings.gate_start_ms
    st.session_state["ess_measure_gate_start_widget"] = settings.gate_start_ms
    st.session_state["ess_measure_gate_end"] = settings.gate_end_ms
    st.session_state["ess_measure_merge_crossover"] = settings.merge_crossover_hz
    st.session_state["ess_measure_wavelet_mode"] = settings.wavelet_update_mode
    st.session_state["ess_measure_wavelet_n"] = settings.wavelet_every_n
    st.session_state["ess_measure_quality_gate"] = settings.quality_gate_mode.capitalize()
    st.session_state["ess_measure_forensic_raw"] = settings.raw_audio_storage == "session_wav"
    st.session_state["ess_measure_mic_calibration_id"] = settings.mic_calibration_id
    st.session_state["ess_measure_calibration_extrapolation"] = {
        "no_correction": "No correction",
        "manual_extension": "Manual / extended file",
    }.get(settings.mic_calibration_extrapolation, "Hold edge")
    st.session_state["ess_measure_clock_mode"] = {
        "force": "Force",
        "off": "Off",
    }.get(settings.clock_correction_mode, "Auto")
    st.session_state["ess_measure_mic_phase_calibration_id"] = settings.mic_phase_calibration_id
    st.session_state["ess_measure_calibration_angle"] = settings.mic_calibration_angle_deg
    st.session_state["ess_measure_input_channel"] = settings.input_channel
    st.session_state["ess_measure_microphone_profile_id"] = settings.microphone_profile_id or "auto"
    restored_profile = next(
        (
            profile for profile in list_microphone_profiles(microphone_db_path)
            if profile.id == settings.microphone_profile_id
        ),
        None,
    )
    st.session_state["ess_measure_input_hardware_type"] = (
        "Analog microphone + audio interface"
        if restored_profile is not None and restored_profile.connection_type == "analog" else
        "USB microphone"
    )
    st.session_state["ess_measure_microphone_unit_id"] = settings.microphone_unit_id
    st.session_state["ess_measure_ess_reference_source"] = (
        "ESS Library file"
        if settings.input_mode.startswith("known_wav")
        else "PhaseEQ nominal ESS"
    )
    st.session_state["ess_measure_reference_mode"] = (
        "Tweeter reference — external ESS"
        if settings.timing_reference_kind == "phaseeq_tweeter_reference"
        else "Timing marker — external ESS"
        if settings.reference_mode == "timing_marker"
        else "IR peak — external ESS"
    )
    st.session_state["ess_measure_timing_session_id"] = settings.timing_reference_session_id
    st.session_state["ess_measure_reference_tweeter_channel"] = settings.reference_tweeter_channel_id
    st.session_state["ess_measure_reference_measurement_id"] = settings.reference_measurement_id
    st.session_state["ess_measure_timing_role"] = (
        "Reference Tweeter"
        if settings.timing_reference_role == "reference_tweeter"
        else "Target Speaker"
    )
    st.session_state["ess_measure_confirm_external_routing"] = bool(
        settings.external_playback_setup_confirmed
    )
    st.session_state["ess_measure_confirm_reference_protection"] = bool(
        settings.reference_protection_confirmed
    )
    st.session_state["ess_measure_confirm_alignment_bypass"] = bool(
        settings.alignment_processing_bypassed_confirmed
    )
    st.session_state["ess_measure_external_setup_note"] = settings.external_playback_setup_note
    st.session_state["_ess_tweeter_resumed_route"] = (
        settings.reference_source_name,
        settings.reference_output_channel,
        settings.measurement_output_channel,
    )
    st.session_state["ess_measure_sync_polarity"] = {
        "positive": "Positive +",
        "negative": "Negative −",
    }.get(settings.sync_polarity, "Auto ±")


ESS_REFERENCE_LIBRARY_KEY = "_ess_reference_library"
ESS_REFERENCE_SELECTION_KEY = "ess_measure_reference_library_id"
ESS_REFERENCE_PENDING_SELECTION_KEY = "_ess_reference_pending_selection_id"
ESS_REFERENCE_DB_PATH_KEY = "_ess_reference_db_path"
ESS_REFERENCE_DB_LOADED_KEY = "_ess_reference_db_loaded_path"
ESS_REFERENCE_LIBRARY_LIMIT = 64
ESS_GENERATOR_PRESET_DESCRIPTIONS = {
    "General measurement": "推奨: 一般的な室内・スピーカー測定。48 kHz、10 Hz–22 kHz、2秒×8回、Peak -18 dBFS。",
    "Quick check": "短時間の接続・音量確認。48 kHz、20 Hz–20 kHz、1秒×5回、低めの再生レベル。",
    "Compatibility": "48 kHz機器向けの広帯域測定。2秒×8回、Peak -12 dBFS。再生系の余裕を確認してください。",
    "Precision": "96 kHzで40 kHzまで測る高精度設定。対応する再生機器とマイクが必要です。",
    "Tweeter extended": "ツイーター専用の192 kHz拡張測定。保護回路と低い再生レベルが必須です。",
    "Noisy environment": "暗騒音が多い環境向け。96 kHz、4秒×12回で測定時間を使って安定性を高めます。",
    "Custom": "各項目を手動設定します。迷う場合はGeneral measurementを選択してください。",
}


def _ess_reference_source_id(audio_data: bytes) -> str:
    return hashlib.sha256(bytes(audio_data)).hexdigest()[:20]


def _register_ess_reference_source(
    audio_data: bytes,
    name: str,
    *,
    manifest_data: bytes = b"",
    wav_data: bytes = b"",
    flac_data: bytes = b"",
    origin: str,
    select: bool = False,
    defer_selection: bool = False,
) -> str:
    source_id = _ess_reference_source_id(audio_data)
    library = st.session_state.setdefault(ESS_REFERENCE_LIBRARY_KEY, {})
    is_new = source_id not in library
    previous = library.get(source_id, {})
    previous_manifest = bytes(previous.get("manifest_data", b""))
    previous_wav = bytes(previous.get("wav_data", b""))
    previous_flac = bytes(previous.get("flac_data", b""))
    suffix = Path(str(name)).suffix.lower()
    source_wav = bytes(audio_data) if suffix in {".wav", ".wave"} else b""
    source_flac = bytes(audio_data) if suffix == ".flac" else b""
    library[source_id] = {
        "name": str(name),
        "audio_data": bytes(audio_data),
        "manifest_data": bytes(manifest_data) or previous_manifest,
        "wav_data": bytes(wav_data) or source_wav or previous_wav,
        "flac_data": bytes(flac_data) or source_flac or previous_flac,
        "origin": str(origin),
    }
    source_changed = (
        is_new
        or previous_manifest != bytes(library[source_id]["manifest_data"])
        or previous_wav != bytes(library[source_id]["wav_data"])
        or previous_flac != bytes(library[source_id]["flac_data"])
        or str(previous.get("name", "")) != str(name)
    )
    if source_changed:
        CacheCoordinator(st.session_state).notify(CacheEvent.ESS_SOURCE_CHANGED)
    db_path_value = st.session_state.get(ESS_REFERENCE_DB_PATH_KEY)
    if db_path_value:
        save_ess_reference(
            Path(str(db_path_value)),
            EssReferenceRecord(
                id=source_id,
                name=str(name),
                audio_data=bytes(audio_data),
                manifest_data=bytes(library[source_id]["manifest_data"]),
                wav_data=bytes(library[source_id]["wav_data"]),
                flac_data=bytes(library[source_id]["flac_data"]),
                origin=str(origin),
            ),
        )
    while len(library) > ESS_REFERENCE_LIBRARY_LIMIT:
        oldest_id = next(iter(library))
        if oldest_id == source_id and len(library) > 1:
            oldest_id = next(item for item in library if item != source_id)
        library.pop(oldest_id, None)
    if select or is_new or st.session_state.get(ESS_REFERENCE_SELECTION_KEY) not in library:
        if defer_selection:
            st.session_state[ESS_REFERENCE_PENDING_SELECTION_KEY] = source_id
        else:
            st.session_state[ESS_REFERENCE_SELECTION_KEY] = source_id
    return source_id


def _hydrate_ess_reference_library_from_db(path: Path) -> None:
    db_path = str(Path(path))
    st.session_state[ESS_REFERENCE_DB_PATH_KEY] = db_path
    if st.session_state.get(ESS_REFERENCE_DB_LOADED_KEY) == db_path:
        return
    library = st.session_state.setdefault(ESS_REFERENCE_LIBRARY_KEY, {})
    for record in reversed(list_ess_references(Path(path), limit=ESS_REFERENCE_LIBRARY_LIMIT, include_standard=False)):
        if record.origin == "Standard PCM reference":
            continue  # Compact references are resolved by the standard guide, not the legacy file reader.
        library.setdefault(
            record.id,
            {
                "name": record.name,
                "audio_data": record.audio_data,
                "manifest_data": record.manifest_data,
                "wav_data": record.wav_data,
                "flac_data": record.flac_data,
                "origin": f"DB / {record.origin}",
            },
        )
    st.session_state[ESS_REFERENCE_DB_LOADED_KEY] = db_path


def _ess_reference_label(source_id: str, entry: dict[str, object]) -> str:
    origin = str(entry.get("origin", "Loaded"))
    return f"{entry.get('name', source_id)} — {origin}"


def _ess_output_file_names(name: str) -> tuple[str, str, str]:
    stem = Path(str(name)).stem
    return f"{stem}.wav", f"{stem}.flac", f"{stem}.json"


def _render_ess_source_manager(
    *,
    analysis_sample_rate: int,
    reference_required: bool,
    disabled: bool,
) -> tuple[KnownEssReference | None, tuple[str, ...]]:
    st.markdown(ui_message('ui.ed58ce63efaa76'))
    st.caption(ui_message('ui.c29cc7463fd1ea'))
    ess_tabs_key = "ess_measure_source_tabs"
    ess_tab_labels = ["ESS Generator", "ESS Library"]
    st.markdown(
        f"<style>{tab_hover_translation_css(ess_tabs_key, ess_tab_labels)}</style>",
        unsafe_allow_html=True,
    )
    generator_tab, library_tab = st.tabs(ess_tab_labels, key=ess_tabs_key)
    with generator_tab:
        _render_ess_generator(disabled=disabled)
    with library_tab:
        upload_cols = st.columns(2)
        with upload_cols[0]:
            reference_upload = st.file_uploader(
                ui_message('ui.4eca8585d80b0e'),
                type=["wav", "wave", "flac"],
                key="ess_measure_reference_wav",
                disabled=disabled,
                help=ui_message('ui.a0cc3a52c61de6'),
            )
        with upload_cols[1]:
            manifest_upload = st.file_uploader(
                ui_message('ui.10edf370903009'),
                type=["json"],
                key="ess_measure_reference_manifest",
                disabled=disabled,
                help=ui_message('ui.1f0f28c9743f94'),
            )
        if reference_upload is not None:
            uploaded_audio = reference_upload.getvalue()
            uploaded_manifest = manifest_upload.getvalue() if manifest_upload is not None else b""
            try:
                _, effective_manifest, inferred_manifest = _extract_uploaded_reference_with_manifest(
                    uploaded_audio,
                    reference_upload.name,
                    int(analysis_sample_rate),
                    uploaded_manifest,
                    _ess_analysis_cache_key(
                        uploaded_audio,
                        reference_upload.name,
                        int(analysis_sample_rate),
                        uploaded_manifest,
                    ),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(ui_message('ui.27f6efdf9e3fe9', p0=f'{exc}'))
            else:
                _register_ess_reference_source(
                    uploaded_audio,
                    reference_upload.name,
                    manifest_data=effective_manifest,
                    origin="Analyzed input" if inferred_manifest else "Loaded + manifest",
                )
                if inferred_manifest:
                    st.info(ui_message('ui.8b6859ec414259'))

        library = st.session_state.setdefault(ESS_REFERENCE_LIBRARY_KEY, {})
        if not library:
            st.selectbox(
                ui_message('ui.0bc39da64a1932'),
                [],
                key=ESS_REFERENCE_SELECTION_KEY,
                disabled=True,
                placeholder=ui_message('ui.afccb01aa762fe'),
                help=ui_message('ui.2b9db2023761f5'),
            format_func=localized_formatter(str))
            st.info(ui_message('ui.15e69959b7d6fd'))
            return None, ()
        source_ids = list(library)
        pending_selection = st.session_state.pop(ESS_REFERENCE_PENDING_SELECTION_KEY, None)
        if pending_selection in source_ids:
            st.session_state[ESS_REFERENCE_SELECTION_KEY] = pending_selection
        if st.session_state.get(ESS_REFERENCE_SELECTION_KEY) not in source_ids:
            st.session_state[ESS_REFERENCE_SELECTION_KEY] = source_ids[-1]
        selected_id = st.selectbox(
            ui_message('ui.0bc39da64a1932'),
            source_ids,
            key=ESS_REFERENCE_SELECTION_KEY,
            disabled=disabled or not reference_required,
            format_func=lambda value: _ess_reference_label(value, library[value]),
            help=ui_message('ui.1860c9e48cb5a6'),
        )
        if st.session_state.get("_cache_selected_ess_source") != selected_id:
            CacheCoordinator(st.session_state).notify(CacheEvent.ESS_SOURCE_CHANGED)
            st.session_state["_cache_selected_ess_source"] = selected_id
        if not reference_required:
            st.caption(ui_message('ui.c31b7c5e5849a8'))
            return None, ()
        entry = library[str(selected_id)]
        try:
            known_reference, effective_manifest, inferred_manifest = _extract_uploaded_reference_with_manifest(
                bytes(entry["audio_data"]),
                str(entry["name"]),
                int(analysis_sample_rate),
                bytes(entry.get("manifest_data", b"")),
                _ess_analysis_cache_key(
                    bytes(entry["audio_data"]),
                    str(entry["name"]),
                    int(analysis_sample_rate),
                    bytes(entry.get("manifest_data", b"")),
                ),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(ui_message('ui.5cce6a444ec06c', p0=f'{exc}'))
            return None, (f"選択したESS基準を解析できません: {exc}",)
        if inferred_manifest:
            entry["manifest_data"] = effective_manifest
            _register_ess_reference_source(
                bytes(entry["audio_data"]),
                str(entry["name"]),
                manifest_data=effective_manifest,
                wav_data=bytes(entry.get("wav_data", b"")),
                flac_data=bytes(entry.get("flac_data", b"")),
                origin="Analyzed input",
            )
            st.info(ui_message('ui.81fb983461f1a6'))
        reference_rejection_reasons = ess_reference_rejection_reasons(known_reference)
        st.success(
            ui_message('ui.f136d7020e6b5b', p0=f'{known_reference.source_name}', p1=f'{known_reference.sweep_duration_s:.4f}', p2=f'{known_reference.shot_period_s:.4f}', p3=f'{known_reference.repeat_count}', p4=f"{('L' if known_reference.source_channel == 1 else 'R')}", p5=f"{','.join(('L' if channel == 1 else 'R' for channel in known_reference.ess_channels))}", p6=f'{known_reference.source_sample_rate / 1000:g}', p7=f'{known_reference.analysis_sample_rate / 1000:g}')
        )
        marker_status = (
            f"Timing marker detected ({known_reference.timing_marker_to_ess_s * 1000:.0f} ms before ESS)"
            if known_reference.has_timing_marker else
            "Legacy ESS without timing marker — ESS correlation fallback is enabled"
        )
        st.caption(
            ui_message('ui.becccd6f2fc43a', p0=f'{marker_status}', p1=f'{known_reference.start_frequency_hz:.0f}', p2=f'{known_reference.end_frequency_hz:.0f}', p3=f'{known_reference.silence_duration_s:.3f}', p4=f'{known_reference.peak_dbfs:.1f}', p5=f'{known_reference.intro_duration_s:.2f}')
        )
        if reference_rejection_reasons:
            st.error(
                ui_message('ui.66751ab69131f9')
                + "\n".join(f"- {reason}" for reason in reference_rejection_reasons),
                icon=":material/block:",
            )
        else:
            st.success(ui_message('ui.5e474e215c8dad'), icon=":material/check_circle:")
        selected_name = str(entry["name"])
        wav_name, flac_name, manifest_name = _ess_output_file_names(selected_name)
        wav_data = bytes(entry.get("wav_data", b""))
        flac_data = bytes(entry.get("flac_data", b""))
        selected_manifest = bytes(entry.get("manifest_data", b""))
        st.markdown(ui_message('ui.d69572d433f8f2'))
        st.caption(
            ui_message('ui.7ab4f3eef21231', p0=f'{selected_name}', p1=f"{entry.get('origin', 'Loaded')}", p2=f"{(wav_name if wav_data else '—')}", p3=f"{(flac_name if flac_data else '—')}", p4=f"{(manifest_name if selected_manifest else '—')}")
        )
        download_columns = st.columns(3)
        if wav_data:
            download_columns[0].download_button(
                ui_message('ui.dc413fa06bd917'),
                wav_data,
                wav_name,
                "audio/wav",
                key=f"ess_library_download_wav_{selected_id}",
                width="stretch",
            )
        if flac_data:
            download_columns[1].download_button(
                ui_message('ui.358fcb4b8d0b8a'),
                flac_data,
                flac_name,
                "audio/flac",
                key=f"ess_library_download_flac_{selected_id}",
                width="stretch",
            )
        if selected_manifest:
            download_columns[2].download_button(
                ui_message('ui.68af93a3757849'),
                selected_manifest,
                manifest_name,
                "application/json",
                key=f"ess_library_download_manifest_{selected_id}",
                width="stretch",
            )
        if not known_reference.has_timing_marker:
            converted_route_label = localized_segmented_control(
                ui_message('ui.5f291c21209a91'),
                ["Both", "Left only", "Right only"],
                key="ess_legacy_conversion_output_channels",
                disabled=disabled or bool(reference_rejection_reasons),
                width="stretch",
                help=ui_message('ui.b44c8a1e3cf475'),
            )
            converted_route = {
                "Left only": "left",
                "Right only": "right",
            }.get(str(converted_route_label), "both")
            if localized_button(
                st,
                ui_message('ui.b9a04d8af6f3e8'),
                type="primary",
                icon=":material/graphic_eq:",
                    disabled=disabled or bool(reference_rejection_reasons),
                width="stretch",
            ):
                source_reference = _extract_uploaded_reference(
                    bytes(entry["audio_data"]),
                    str(entry["name"]),
                    int(known_reference.source_sample_rate),
                    bytes(entry.get("manifest_data", b"")),
                )
                converted = generate_marker_ess_bundle(
                    source_reference.samples,
                    source_reference.source_sample_rate,
                    silence_duration_s=max(0.05, source_reference.silence_duration_s),
                    repeats=max(1, source_reference.repeat_count),
                    output_channels=converted_route,
                    timing_marker=True,
                    settings_manifest={
                        "sample_rate": source_reference.source_sample_rate,
                        "bit_depth": 24,
                        "start_frequency_hz": source_reference.start_frequency_hz,
                        "end_frequency_hz": source_reference.end_frequency_hz,
                        "sweep_duration_s": source_reference.sweep_duration_s,
                        "silence_duration_s": max(0.05, source_reference.silence_duration_s),
                        "repeats": max(1, source_reference.repeat_count),
                        "output_channels": converted_route,
                        "timing_marker": True,
                        "converted_from": str(entry["name"]),
                        "conversion_mode": "exact_extracted_ess_waveform",
                        "preserved_intro": source_reference.intro_samples is not None,
                    },
                    actual_end_frequency_hz=source_reference.end_frequency_hz,
                    leader_audio=source_reference.intro_samples,
                )
                suffix = Path(str(entry["name"])).stem
                converted_name = f"{suffix}_timing_marker_{converted_route}.flac"
                st.session_state["_ess_generated_bundle"] = converted
                st.session_state["_ess_generated_name"] = converted_name
                _register_ess_reference_source(
                    converted.flac_bytes,
                    converted_name,
                    manifest_data=converted.manifest_bytes,
                    wav_data=converted.wav_bytes,
                    flac_data=converted.flac_bytes,
                    origin="Marker conversion",
                    select=True,
                    defer_selection=True,
                )
                st.rerun()
        return known_reference, reference_rejection_reasons


def _apply_generator_routing_preset(marker_side: str) -> None:
    side = str(marker_side).casefold()
    if side == "left":
        ess_output, marker_output = "Right only", "Left only"
    elif side == "right":
        ess_output, marker_output = "Left only", "Right only"
    else:
        ess_output, marker_output = "Both", "Same as ESS"
    st.session_state["ess_gen_output_channels"] = ess_output
    st.session_state["ess_gen_timing_marker"] = True
    st.session_state["ess_gen_timing_marker_output_channels"] = marker_output
    st.session_state.pop("_ess_generated_bundle", None)
    st.session_state.pop("_ess_generated_name", None)


def _render_level_check_signal(
    *,
    sample_rate: int,
    measurement_start_hz: float,
    measurement_end_hz: float,
    default_rms_dbfs: float,
    snapshot: SessionSnapshot | None,
    disabled: bool,
) -> None:
    """Create a short REW-inspired level-check file without ESS metadata."""

    with st.container(border=True):
        st.markdown(ui_message('ui.5fac6b558a0e0a'))
        st.caption(
            ui_message('ui.802ba5eb1246f3')
        )
        band_label = st.selectbox(
            ui_message('ui.df6540b6d27176'),
            [
                "Pink · Speaker 500–2,000 Hz",
                "Pink · Subwoofer 30–80 Hz",
                "Pink · Measurement range",
            ],
            key="ess_level_check_band",
            disabled=disabled,
        format_func=localized_formatter(str))
        route_label = localized_segmented_control(
            ui_message('ui.bb6fd5a4149fe7'),
            ["Both", "Left only", "Right only"],
            key="ess_level_check_output",
            disabled=disabled,
            width="stretch",
            help=ui_message('ui.86a130bf8aec60'),
        )
        route = {"Left only": "left", "Right only": "right"}.get(
            str(route_label), "both"
        )
        st.session_state.setdefault(
            "ess_level_check_rms_dbfs",
            float(np.clip(default_rms_dbfs, -60.0, -18.0)),
        )
        rms_dbfs = float(st.number_input(
            ui_message('ui.7200c04f76d336'),
            min_value=-60.0,
            max_value=-18.0,
            step=1.0,
            key="ess_level_check_rms_dbfs",
            disabled=disabled,
            help=(
                ui_message('ui.9de4dc4fd0347e')
            ),
        ))
        if band_label == "Pink · Speaker 500–2,000 Hz":
            low_hz, high_hz = 500.0, 2_000.0
        elif band_label == "Pink · Subwoofer 30–80 Hz":
            low_hz, high_hz = 30.0, 80.0
        else:
            low_hz = max(1.0, measurement_start_hz)
            high_hz = min(measurement_end_hz, sample_rate / 2.0 - 1.0)
        invalid_band = not (0.0 < low_hz < high_hz < sample_rate / 2.0)
        st.caption(
            ui_message('ui.ab08788790638d', p0=f'{low_hz:g}', p1=f'{high_hz:g}', p2=f'{rms_dbfs:.1f}')
        )
        if localized_button(
            st,
            ui_message('ui.902e4463fd2dd8'),
            icon=":material/graphic_eq:",
            disabled=disabled or invalid_band,
            width="stretch",
        ):
            try:
                st.session_state["_ess_level_check_bundle"] = generate_level_check_signal(
                    sample_rate=sample_rate,
                    duration_s=3.0,
                    rms_dbfs=rms_dbfs,
                    output_channels=route,
                    start_frequency_hz=low_hz,
                    end_frequency_hz=high_hz,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(ui_message('ui.061f7d6bbe47bd', p0=f'{exc}'))
        bundle = st.session_state.get("_ess_level_check_bundle")
        if bundle is not None:
            file_name = (
                f"phaseeq_level_check_pink_{bundle.start_frequency_hz:g}-"
                f"{bundle.end_frequency_hz:g}Hz_{bundle.sample_rate // 1000}k.wav"
            )
            st.download_button(
                ui_message('ui.b5ed3d4154ff3a'),
                bundle.wav_bytes,
                file_name,
                "audio/wav",
                key="ess_level_check_download",
                width="stretch",
            )
        if snapshot is None:
            st.caption(ui_message('ui.f22ff3985a741d'))
        elif snapshot.state == MeasurementState.MEASURING_NOISE:
            st.warning(ui_message('ui.06e73da6520442'), icon=":material/volume_off:")
        elif _measurement_is_running(snapshot):
            st.success(
                ui_message('ui.c6ecce1687623b'),
                icon=":material/volume_up:",
            )


def _render_ess_generator(*, disabled: bool) -> None:
    st.info(
        ui_message('ui.ca2205cd4220d1'),
        icon=":material/recommend:",
    )
    preset_names = list(ESS_GENERATOR_PRESET_DESCRIPTIONS)
    st.session_state.setdefault("ess_gen_preset", "General measurement")
    preset = st.selectbox(ui_message('ui.cf681c636f47b0'), preset_names, key="ess_gen_preset", disabled=disabled)
    st.caption(ESS_GENERATOR_PRESET_DESCRIPTIONS[str(preset)])
    if localized_button(st, ui_message('ui.83e8d9c95611db'), disabled=disabled or preset == "Custom", width="stretch"):
        _apply_generator_preset(str(preset))
        st.rerun()

    defaults = generator_preset("General measurement")
    for key, value in {
        "ess_gen_sample_rate": defaults.sample_rate,
        "ess_gen_start_hz": defaults.start_frequency_hz,
        "ess_gen_end_hz": defaults.end_frequency_hz,
        "ess_gen_sweep_s": defaults.sweep_duration_s,
        "ess_gen_silence_s": defaults.silence_duration_s,
        "ess_gen_repeats": defaults.repeats,
        "ess_gen_leader_s": defaults.leader_silence_s,
        "ess_gen_trailer_s": defaults.trailer_silence_s,
        "ess_gen_peak_dbfs": defaults.peak_dbfs,
        "ess_gen_fade_s": defaults.fade_s,
        "ess_gen_start_cue": defaults.start_cue,
        "ess_gen_output_channels": "Both",
        "ess_gen_timing_marker": defaults.timing_marker,
        "ess_gen_timing_marker_output_channels": "Same as ESS",
    }.items():
        st.session_state.setdefault(key, value)
    st.markdown(ui_message('ui.2921e18d72b2c7'))
    st.caption(
        ui_message('ui.3e8442ade0144d')
    )
    with st.container(horizontal=True):
        st.button(
            ui_message('ui.a4bc01708acb85'), disabled=disabled,
            key="ess_gen_routing_standard",
            on_click=_apply_generator_routing_preset, args=("standard",),
        )
        st.button(
            ui_message('ui.0c709396b4f5c0'), icon=":material/swap_horiz:", disabled=disabled,
            key="ess_gen_routing_marker_left",
            on_click=_apply_generator_routing_preset, args=("left",),
        )
        st.button(
            ui_message('ui.e8f7fd28127d0a'), icon=":material/swap_horiz:", disabled=disabled,
            key="ess_gen_routing_marker_right",
            on_click=_apply_generator_routing_preset, args=("right",),
        )
    sample_rate = int(st.selectbox(
        ui_message('ui.d474feb6dfd775'),
        [48_000, 96_000, 192_000],
        key="ess_gen_sample_rate",
        disabled=disabled,
        format_func=localized_formatter(lambda value: f'{value / 1000:g} kHz'),
    ))
    output_channels_label = localized_segmented_control(
        ui_message('ui.eeb006dac20471'),
        ["Both", "Left only", "Right only"],
        key="ess_gen_output_channels",
        disabled=disabled,
        width="stretch",
        help=ui_message('ui.f58ff9d98c4fd1'),
    )
    output_channels = {
        "Left only": "left",
        "Right only": "right",
    }.get(str(output_channels_label), "both")
    nyquist = sample_rate / 2
    if float(st.session_state["ess_gen_end_hz"]) > nyquist:
        st.session_state["ess_gen_end_hz"] = nyquist
    frequency_cols = st.columns(2)
    with frequency_cols[0]:
        start_hz = float(st.number_input(
            ui_message('ui.a44ddf93a26a38'), min_value=1.0, max_value=max(1.0, nyquist - 1.0), step=10.0,
            key="ess_gen_start_hz", disabled=disabled,
        ))
    with frequency_cols[1]:
        end_hz = float(st.number_input(
            ui_message('ui.b33d6cc717566c'), min_value=2.0, max_value=float(nyquist), step=100.0,
            key="ess_gen_end_hz", disabled=disabled,
            help=ui_message('ui.0b7fa0a5a4f267'),
        ))
    timing_cols = st.columns(3)
    with timing_cols[0]:
        sweep_s = float(st.number_input(
            ui_message('ui.511bfd5aeae01e'), min_value=0.1, max_value=60.0, step=0.5,
            key="ess_gen_sweep_s", disabled=disabled,
        ))
    with timing_cols[1]:
        silence_s = float(st.number_input(
            ui_message('ui.98554584b8e7b8'), min_value=0.05, max_value=30.0, step=0.25,
            key="ess_gen_silence_s", disabled=disabled,
            help=ui_message('ui.d73ef1e3609e0a'),
        ))
    with timing_cols[2]:
        repeats = int(st.number_input(
            ui_message('ui.92ee0233f01396'), min_value=3, max_value=1000, step=1,
            key="ess_gen_repeats", disabled=disabled,
        ))
    level_cols = st.columns(3)
    with level_cols[0]:
        peak_dbfs = float(st.number_input(
            ui_message('ui.7569b255465d20'), min_value=-60.0, max_value=-3.0, step=1.0,
            key="ess_gen_peak_dbfs", disabled=disabled,
        ))
    with level_cols[1]:
        leader_s = float(st.number_input(
            ui_message('ui.3c502a8d110740'), min_value=0.0, max_value=60.0, step=0.5,
            key="ess_gen_leader_s", disabled=disabled,
        ))
    with level_cols[2]:
        trailer_s = float(st.number_input(
            ui_message('ui.f91a9b2285b5f7'), min_value=0.0, max_value=60.0, step=0.5,
            key="ess_gen_trailer_s", disabled=disabled,
        ))
    start_cue = bool(st.toggle(ui_message('ui.afbcf21b61720e'), key="ess_gen_start_cue", disabled=disabled))
    timing_marker = bool(st.toggle(
        ui_message('ui.c190ca6003c5a5'),
        key="ess_gen_timing_marker",
        disabled=disabled,
        help=ui_message('ui.83b9286f72ed62'),
    ))
    marker_output_label = st.selectbox(
        ui_message('ui.f699ae79f26128'),
        ["Same as ESS", "Left only", "Right only", "Both"],
        key="ess_gen_timing_marker_output_channels",
        disabled=disabled or not timing_marker,
        help=(
            ui_message('ui.4e831cfe32e022')
        ),
    format_func=localized_formatter(str))
    marker_output_channels = {
        "Left only": "left",
        "Right only": "right",
        "Both": "both",
    }.get(str(marker_output_label), "same")
    tweeter_routed = bool(
        timing_marker and marker_output_channels not in {"same", output_channels}
    )
    if tweeter_routed:
        st.info(
            ui_message('ui.8ccbb96717b81f')
        )
    if start_hz < 500 and str(preset) == "Tweeter extended":
        st.warning(ui_message('ui.0598322d0cdc95'))
    if end_hz > 20_000:
        st.info(ui_message('ui.41266fbfa6a4a8'))
    marker_overhead_s = (
        defaults.timing_marker_duration_s + defaults.timing_marker_gap_s
        if timing_marker else 0.0
    )
    estimated_duration = (
        leader_s + repeats * (marker_overhead_s + sweep_s + silence_s) + trailer_s
        + (defaults.timing_marker_duration_s if tweeter_routed else 0.0)
    )
    st.caption(ui_message('ui.8a9643fc8dfcf7', p0=f'{estimated_duration:.1f}'))
    settings: EssGeneratorSettings | None = None
    try:
        settings = EssGeneratorSettings(
            sample_rate=sample_rate,
            start_frequency_hz=start_hz,
            end_frequency_hz=end_hz,
            sweep_duration_s=sweep_s,
            silence_duration_s=silence_s,
            repeats=repeats,
            leader_silence_s=leader_s,
            trailer_silence_s=trailer_s,
            peak_dbfs=peak_dbfs,
            fade_s=float(st.session_state["ess_gen_fade_s"]),
            start_cue=start_cue,
            output_channels=output_channels,
            timing_marker=timing_marker,
            timing_marker_output_channels=marker_output_channels,
            terminal_timing_marker=tweeter_routed,
        )
        generator_rejection_reasons = ess_generator_rejection_reasons(settings)
    except ValueError as exc:
        generator_rejection_reasons = (str(exc),)
    if generator_rejection_reasons:
        st.error(
            ui_message('ui.d8d9603283d8b5')
            + "\n".join(f"- {reason}" for reason in generator_rejection_reasons),
            icon=":material/block:",
        )
    else:
        st.success(ui_message('ui.972637aa560895'), icon=":material/check_circle:")
    if localized_button(
        st,
        ui_message('ui.407c65e2a4c8e5'),
        type="primary",
        icon=":material/audio_file:",
        disabled=disabled or bool(generator_rejection_reasons),
        width="stretch",
    ):
        try:
            assert settings is not None
            with st.spinner(ui_message('ui.5b9825a0a695f7')):
                bundle = generate_ess_bundle(settings)
                st.session_state["_ess_generated_bundle"] = bundle
                preset_slug = str(preset).lower().replace(" ", "_")
                generated_name = (
                    f"phaseeq_ess_{sample_rate // 1000}k_{preset_slug}_{output_channels}_"
                    f"{'tweeter_reference' if tweeter_routed else 'marker' if timing_marker else 'legacy'}.flac"
                )
                st.session_state["_ess_generated_name"] = generated_name
                _register_ess_reference_source(
                    bundle.flac_bytes,
                    generated_name,
                    manifest_data=bundle.manifest_bytes,
                    wav_data=bundle.wav_bytes,
                    flac_data=bundle.flac_bytes,
                    origin="Generated",
                    select=True,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(ui_message('ui.2ede8fd56a8115', p0=f'{exc}'))
    bundle = st.session_state.get("_ess_generated_bundle")
    if bundle is not None:
        last_generated_name = str(
            st.session_state.get(
                "_ess_generated_name",
                f"phaseeq_ess_{bundle.manifest['settings']['sample_rate'] // 1000}k_"
                f"{bundle.manifest['settings'].get('output_channels', 'both')}.wav",
            )
        )
        wav_name, flac_name, manifest_name = _ess_output_file_names(last_generated_name)
        route_label = {
            "both": "L + R",
            "left": "L (R is silent)",
            "right": "R (L is silent)",
        }.get(str(bundle.manifest["settings"].get("output_channels", "both")), "L + R")
        sample_rate_hz = int(bundle.manifest["settings"]["sample_rate"])
        duration_s = len(bundle.stereo_samples) / sample_rate_hz
        st.markdown(ui_message('ui.29d537f5519ea2'))
        st.caption(
            ui_message('ui.a31e76877cb1c9', p0=f'{wav_name}', p1=f'{flac_name}', p2=f'{manifest_name}')
        )
        st.caption(
            ui_message('ui.96152be4646039', p0=f'{sample_rate_hz / 1000:g}', p1=f'{route_label}', p2=f'{len(bundle.shot_start_samples)}', p3=f'{duration_s:.2f}', p4=f"{bundle.manifest['pcm_sha256'][:16]}")
        )
        downloads = st.columns(3)
        downloads[0].download_button(
            ui_message('ui.dc413fa06bd917'), bundle.wav_bytes, wav_name, "audio/wav",
            key="ess_generator_download_wav", width="stretch",
        )
        downloads[1].download_button(
            ui_message('ui.358fcb4b8d0b8a'), bundle.flac_bytes, flac_name, "audio/flac",
            key="ess_generator_download_flac", width="stretch",
        )
        downloads[2].download_button(
            ui_message('ui.68af93a3757849'), bundle.manifest_bytes, manifest_name, "application/json",
            key="ess_generator_download_manifest", width="stretch",
        )
        st.success(
            ui_message('ui.fcce9451cfba32', p0=f'{len(bundle.shot_start_samples)}', p1=f"{bundle.manifest['pcm_sha256'][:16]}")
        )


def _apply_generator_preset(name: str) -> None:
    settings = generator_preset(name)
    st.session_state["ess_gen_sample_rate"] = settings.sample_rate
    st.session_state["ess_gen_start_hz"] = settings.start_frequency_hz
    st.session_state["ess_gen_end_hz"] = settings.end_frequency_hz
    st.session_state["ess_gen_sweep_s"] = settings.sweep_duration_s
    st.session_state["ess_gen_silence_s"] = settings.silence_duration_s
    st.session_state["ess_gen_repeats"] = settings.repeats
    st.session_state["ess_gen_leader_s"] = settings.leader_silence_s
    st.session_state["ess_gen_trailer_s"] = settings.trailer_silence_s
    st.session_state["ess_gen_peak_dbfs"] = settings.peak_dbfs
    st.session_state["ess_gen_fade_s"] = settings.fade_s
    st.session_state["ess_gen_start_cue"] = settings.start_cue
    st.session_state["ess_gen_output_channels"] = {
        "left": "Left only",
        "right": "Right only",
    }.get(settings.output_channels, "Both")
    st.session_state["ess_gen_timing_marker"] = settings.timing_marker
    st.session_state["ess_gen_timing_marker_output_channels"] = {
        "left": "Left only",
        "right": "Right only",
        "both": "Both",
    }.get(settings.timing_marker_output_channels, "Same as ESS")
    st.session_state.pop("_ess_generated_bundle", None)
    st.session_state.pop("_ess_generated_name", None)


def _ess_analysis_cache_key(
    data: bytes,
    name: str,
    analysis_sample_rate: int,
    manifest_data: bytes = b"",
) -> str:
    return CacheCoordinator(st.session_state).key(
        CacheDomain.ESS_ANALYSIS,
        algorithm_version=ESS_ANALYSIS_CACHE_ALGORITHM_VERSION,
        source_digest=content_digest(data, manifest_data),
        settings_digest=content_digest(analysis_sample_rate),
        stage=name,
    ).digest


@st.cache_data(max_entries=8, ttl=86_400, show_spinner="ESS基準ファイルを解析しています…")
def _extract_uploaded_reference_with_manifest(
    data: bytes,
    name: str,
    analysis_sample_rate: int,
    manifest_data: bytes = b"",
    cache_key: str = "",
) -> tuple[KnownEssReference, bytes, bool]:
    _ = cache_key
    import soundfile as sf

    audio, source_sample_rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    if manifest_data:
        reference = extract_manifest_ess_reference(
            audio,
            int(source_sample_rate),
            int(analysis_sample_rate),
            manifest_data,
            source_name=str(name),
        )
        return reference, bytes(manifest_data), False
    reference = extract_known_ess_reference(
        audio,
        int(source_sample_rate),
        int(analysis_sample_rate),
        source_name=str(name),
    )
    inferred_manifest = build_inferred_ess_manifest(
        audio,
        int(source_sample_rate),
        reference,
        source_name=str(name),
    )
    return reference, inferred_manifest, True


def _extract_uploaded_reference(
    data: bytes,
    name: str,
    analysis_sample_rate: int,
    manifest_data: bytes = b"",
) -> KnownEssReference:
    reference, _, _ = _extract_uploaded_reference_with_manifest(
        data,
        name,
        analysis_sample_rate,
        manifest_data,
        _ess_analysis_cache_key(data, name, analysis_sample_rate, manifest_data),
    )
    return reference


def _measurement_replacement_requires_confirmation(snapshot) -> bool:
    return snapshot is not None and bool(snapshot.shots)


def _restore_measurement_upload(upload, autosave_root: Path) -> None:
    try:
        restore_dir = autosave_root / f"restored_{Path(upload.name).stem}"
        _safe_extract_zip(upload.getvalue(), restore_dir)
        manifests = list(restore_dir.rglob("measurement.json"))
        if not manifests:
            raise ValueError("measurement.jsonが見つかりません。")
        loaded = load_measurement_session(manifests[0].parent)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        st.error(ui_message('ui.e3e5661aa44394', p0=f'{exc}'))
        return
    st.session_state[ENGINE_KEY] = ContinuousMeasurementEngine.from_snapshot(loaded, autosave_root)
    st.session_state["_ess_pending_restored_settings"] = loaded.settings
    st.session_state[UNSAVED_KEY] = False
    st.rerun()


def _safe_extract_zip(data: bytes, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError("ZIP contains an unsafe path")
        archive.extractall(destination)


def _engine() -> ContinuousMeasurementEngine | None:
    value = st.session_state.get(ENGINE_KEY)
    if isinstance(value, ContinuousMeasurementEngine):
        return value
    # Streamlit keeps session_state across source hot reloads.  An engine made
    # by the previous class object is still valid even though isinstance()
    # against the reloaded class returns False.
    if (
        value is not None
        and callable(getattr(value, "snapshot", None))
        and callable(getattr(value, "stop", None))
        and callable(getattr(value, "abort", None))
    ):
        return value
    return None


def _measurement_is_running(snapshot: SessionSnapshot | None) -> bool:
    return snapshot is not None and str(snapshot.state) in {
        str(state) for state in ACTIVE_MEASUREMENT_STATES
    }
