"""Preset-first external playback guide, using the existing acquisition engine."""
from __future__ import annotations
from utils.ui_localization import ui_message, display_text, localized_formatter

from datetime import datetime
import io
import json
import os
from pathlib import Path
import time
from threading import Lock
import uuid
import weakref
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from phase_fir_designer.measurement import ContinuousMeasurementEngine, list_input_devices, refresh_audio_devices, level_calibration_offset_db
from phase_fir_designer.measurement.original import calibrated_output, calibrated_export_zip
from phase_fir_designer.measurement.persistence import cleanup_measurement_autosave
from phase_fir_designer.measurement.standard_audio import TRACKS, track_for
from phase_fir_designer.speaker import load_speaker_response_text
from utils.guided_measurement import POSITIONS, PENDING_REASONS, microphone_settings, settings_for_track, completed, save_guided_result
from utils.microphone_db import list_microphone_profiles, list_microphone_units, save_microphone_unit, MicrophoneUnit
from utils.settings_io import speaker_response_payload
from utils.speaker_db import get_measurement, save_measurement, SpeakerMeasurementRecord
from utils.measurement_originals import load_measurement_original
from utils.standard_audio_assets import StandardAudioAssets
from utils.ui_work_cache import deferred_call
from utils.ui_localization import required_segmented_control

ENGINE_KEY = "_continuous_ess_engine"
CONFIG_KEY = "_guided_selection"
ACTIVE = {"ARMING", "MEASURING_NOISE", "PREFLIGHT", "SEARCHING", "CAPTURING_SHOT", "ANALYZING_SHOT", "STOPPING"}
KINDS = {"response": "フルレンジ", "tweeter": "ツイーター", "distortion": "歪み", "decay": "残響", "timing": "相対測距"}
OUTPUTS = {"both": "左右同時", "left": "左のみ", "right": "右のみ"}


@st.cache_resource(show_spinner=False)
def standard_audio_service(root: str, db: str):
    service = StandardAudioAssets(Path(root), Path(db))
    if os.environ.get("PHASEEQ_STANDARD_AUDIO_AUTOPREPARE", "1") != "0":
        service.start()
    return service


def _move(step):
    if step != "select":
        release_audio_downloads_on_navigation("setup")
    st.session_state["_guided_step"] = step
    st.rerun()


def _download_tracks(root, tracks, media):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_STORED) as archive:
        lines = []
        for track in tracks:
            path = Path(root) / "set1" / media / track.filename(media)
            from phase_fir_designer.measurement.standard_audio import file_hash
            manifest = json.loads(path.with_suffix(".json").read_text())
            if file_hash(path) != manifest["file_sha256"]:
                raise ValueError(f"No.{track.number:02d}のファイル整合性を確認できません")
            archive.write(path, arcname=path.name)
            lines.append(f"{track.number:02d}  {track.title}")
        archive.writestr("曲順表.txt", "\n".join(lines))
    return data.getvalue()


class _PreparedAudioZip:
    """Session-owned bytes; the worker never touches Streamlit session state."""

    def __init__(self, root, tracks, media):
        self.root, self.tracks, self.media = root, tracks, media
        self.data = None
        self.error = ""
        self.lock = Lock()
        self.closed = False
        self.filename = None
        self._transfer_finalizer = None

    def __call__(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("Measurement download session has ended. Reopen Measure and retry.")
            if self.data is None:
                try:
                    self.data = _download_tracks(self.root, self.tracks, self.media)
                    self.error = ""
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
                    self.error = str(exc)
                    raise
            self._pin_transfer()
            return self.data

    def _pin_transfer(self):
        # execute_deferred leaves the generated file unreferenced. Timer reruns
        # can collect it before the browser's first GET (or a retry). Pin its
        # exact ID before returning from the callable, not on a later UI rerun.
        from streamlit import runtime
        if self._transfer_finalizer is not None or not self.filename or not runtime.exists():
            return
        from streamlit.runtime.memory_media_file_storage import _calculate_file_id
        manager = runtime.get_instance().media_file_mgr
        file_id = _calculate_file_id(self.data, "application/zip", self.filename)
        owner = "phaseeq-audio-zip-" + uuid.uuid4().hex
        with manager._lock:
            manager._files_by_session_and_coord[owner]["zip"] = file_id
        # Finalization also handles a closed browser session. The callback must
        # not retain this package, otherwise abandoned downloads would leak.
        self._transfer_finalizer = weakref.finalize(self, _unpin_audio_transfer, manager, owner)

    def release(self):
        with self.lock:
            self.data = None
            self.closed = True
            if self._transfer_finalizer is not None:
                self._transfer_finalizer()


def _unpin_audio_transfer(manager, owner):
    with manager._lock:
        manager._files_by_session_and_coord.pop(owner, None)


def release_audio_downloads_on_navigation(active_page):
    """Release preparation buffers before setup or when leaving Measure."""
    if active_page == "Measure":
        return
    engine = st.session_state.get(ENGINE_KEY)
    if engine is not None and str(engine.snapshot().state) in ACTIVE:
        return
    for package in st.session_state.pop("_guided_audio_downloads", {}).values():
        # Remember only our ZIP IDs so later cleanup cannot remove unrelated media.
        with package.lock:
            if package.data is not None and package.filename:
                from streamlit.runtime.memory_media_file_storage import _calculate_file_id
                file_id = _calculate_file_id(package.data, "application/zip", package.filename)
                st.session_state.setdefault("_guided_audio_retired_ids", set()).add(file_id)
        package.release()


def release_audio_downloads_before_capture():
    """Called before allocating/starting capture, after source widgets disappear."""
    engine = st.session_state.get(ENGINE_KEY)
    if engine is not None and str(engine.snapshot().state) in ACTIVE:
        return
    release_audio_downloads_on_navigation("capture")
    from streamlit import runtime
    if not runtime.exists():
        return
    manager = runtime.get_instance().media_file_mgr
    retired = st.session_state.pop("_guided_audio_retired_ids", set())
    # Streamlit has no public per-file retirement API. Keep this adapter scoped
    # to our known ZIPs and respect references from all other browser sessions.
    with manager._lock:
        inactive = manager._get_inactive_file_ids()
        for file_id in retired & inactive:
            manager._delete_file(file_id)


@st.fragment(run_every=1)
def _watch_download_ready(package):
    # Register completed bytes as an ordinary download. The browser may reuse
    # its resolved URL, so that URL must stay referenced after the first click.
    if package.data is not None or package.error or package.closed:
        st.rerun()


@st.fragment
def _audio_download_control(root, tracks, media, *, key, all_tracks=False):
    """One click prepares and downloads; no timer re-registers this callback."""
    engine = st.session_state.get(ENGINE_KEY)
    if engine is not None and str(engine.snapshot().state) in ACTIVE:
        return
    root = Path(root)
    tracks = tuple(tracks)
    try:
        files = [root / "set1" / media / track.filename(media) for track in tracks]
        signature = (str(root), media, tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns,
             path.with_suffix(".json").stat().st_mtime_ns) for path in files
        ))
    except OSError as exc:
        st.error(ui_message('ui.c87b416f051644', error=str(exc)))
        return
    packages = st.session_state.setdefault("_guided_audio_downloads", {})
    if signature not in packages:
        packages[signature] = _PreparedAudioZip(root, tracks, media)
    package = packages[signature]
    if package.closed:
        return
    if package.error:
        st.error(ui_message('ui.c87b416f051644', error=package.error))
        if st.button(display_text('ZIP生成を再試行'), key=key + '_retry'):
            with package.lock:
                package.error = ''
            st.rerun()
        return
    name = f"PhaseEQ_standard_set1_{media}.zip" if all_tracks else f"PhaseEQ_{tracks[-1].key}_{media}.zip"
    package.filename = name
    st.download_button(
        ui_message('ui.3050ec236b97ca'),
        data=package.data if package.data is not None else package,
        file_name=name, mime="application/zip",
        on_click="ignore", key=key + "_save",
    )
    if package.data is None:
        _watch_download_ready(package)


@st.fragment(run_every=1)
def _watch_audio_preparation(service, media, previous_states):
    # Only preparation status needs polling on the source-selection screen.
    # Keep download callbacks out of timer reruns to avoid expiring their IDs.
    states = tuple(service.status(track, media)["state"] for track in TRACKS)
    if states != previous_states:
        st.rerun()


def _config():
    config = st.session_state.setdefault(CONFIG_KEY, dict(kind="response", output="left", media="cd", position="1 m", angle=0,
        target="左スピーカー", unit_id="", purpose="スピーカー特性", path_kind="システム経由"))
    for key, choices, fallback in (("kind", KINDS, "response"), ("output", OUTPUTS, "left"), ("media", ("cd",), "cd"), ("position", POSITIONS, "1 m")):
        if config.get(key) not in choices:
            config[key] = fallback
    return config


def _save_preferences(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.json")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    os.replace(temporary, path)


def _audio_preparation_menu(service, config):
    st.info(ui_message('ui.944a4ea9c1b20d'))
    with st.expander(ui_message('ui.b242e31abc40ed'), expanded=False):
        all_ready = all(service.status(t, config["media"])["state"] == "ready" for t in TRACKS)
        if not all_ready:
            st.caption(ui_message('ui.ca9c6ae89cc234', p0=f"{sum((service.status(t, config['media'])['state'] == 'ready' for t in TRACKS))}"))
            for pending in TRACKS:
                state = service.status(pending, config["media"])
                if state["state"] == "failed":
                    st.warning(ui_message('ui.8fc3d3ef54e839', p0=f'{pending.number:02d}', p1=f"{state['message']}"))
        if all_ready:
            _audio_download_control(service.root, TRACKS, config["media"], key="guide_all_audio", all_tracks=True)
        cd_tab, wav_tab = st.tabs([ui_message('ui.c779a49b7e11e6'), ui_message('ui.aea62bda8f7b04')])
        with cd_tab:
            st.markdown(ui_message('ui.bbd68eb50c447a'))
        with wav_tab:
            st.markdown(ui_message('ui.6077dc58a92ecb'))


def _selector(service, config):
    states = tuple(service.status(track, config["media"])["state"] for track in TRACKS)
    if "preparing" in states:
        _watch_audio_preparation(service, config["media"], states)
    _audio_preparation_menu(service, config)
    st.markdown(ui_message('ui.dfac149d3a0a49'))
    if st.session_state.get("guide_kind") not in KINDS:
        st.session_state["guide_kind"] = config["kind"]
    kind = required_segmented_control(ui_message('ui.67347a8012651f'), list(KINDS), format_func=KINDS.get,
        key="guide_kind", required=True, width="stretch", persist_state="session") or config["kind"]
    config["kind"] = kind
    st.caption(display_text({"response": "スピーカー全体や試聴位置の応答を測ります。", "tweeter": "システム経由またはユニット単体の高域応答を測ります。",
        "distortion": "音の歪みをグラフで確認します。", "decay": "部屋の音の減衰を確認します。", "timing": "基準ツイーターに対する音の到達差を測ります。"}[kind]))
    if kind in PENDING_REASONS:
        st.info(PENDING_REASONS[kind])
    if kind == "timing":
        st.caption(ui_message('ui.2bdccd859f334b'))
    st.session_state.setdefault("guide_output", config["output"])
    output = required_segmented_control(ui_message('ui.01e3b3e54b3ab7'), list(OUTPUTS), format_func=OUTPUTS.get,
        default=config["output"], key="guide_output", width="stretch", persist_state="session")
    config["output"] = output
    st.caption(ui_message('ui.558d6e661e9295'))
    st.info(ui_message('ui.5c6abc4c4e542d'))
    if output is not None and kind != "timing":
        track = track_for(kind, output)
        status = service.status(track, config["media"])
        noise = service.status(TRACKS[0], config["media"])
        st.info(ui_message('ui.0767915cbdfd59', p0=f'{track.number:02d}', p1=display_text(track.title)))
        st.caption(display_text(status["message"]))
        ready = status["state"] == noise["state"] == "ready"
        if status["state"] == "failed" or noise["state"] == "failed":
            st.error(status["message"] if status["state"] == "failed" else noise["message"])
            if st.button(ui_message('ui.6d5f2ea1fb5d1d'), key='i18n_17b8fd216f164c44'):
                service.start()
        if st.button(ui_message('ui.89b210c0115767'), type="primary", disabled=not ready or kind in PENDING_REASONS, key='i18n_af6d5da0108cb519'):
            config["target"] = {"both": "左右スピーカー", "left": "左スピーカー", "right": "右スピーカー"}[output]
            st.session_state["guide_target"] = config["target"]
            config["level_only"] = False
            _move("setup")
    if st.button(ui_message('ui.a6c9a11c0c67fe'), disabled=service.status(TRACKS[0], config["media"])["state"] != "ready" or service.status(TRACKS[1], config["media"])["state"] != "ready", key='i18n_54be608f23aae389'):
        config.update(kind="response", output="both", level_only=True)
        _move("setup")


def _register_microphone(microphone_db_path, measurement_db_path, profiles, *, expanded=False):
    with st.expander(ui_message('ui.8f8a2cd0aa0724'), expanded=expanded):
        profile_id = st.selectbox(ui_message('ui.65d31324c43cbc'), list(profiles), format_func=lambda p: profiles[p].model, key="guide_register_profile")
        serial = st.text_input(ui_message('ui.e82e2009302fa5'), key="guide_register_serial")
        angle = st.selectbox(ui_message('ui.e854770143fc22'), [0, 90], key="guide_register_angle", format_func=localized_formatter(str))
        upload = st.file_uploader(ui_message('ui.a7c8d5f8e90383'), type=["txt", "cal", "omm"], key="guide_register_cal")
        if st.button(ui_message('ui.54f2bceee3f522'), disabled=not serial.strip() or upload is None, key='i18n_51096c41e6b52160'):
            try:
                text = upload.getvalue().decode("utf-8-sig")
                response = load_speaker_response_text(text)
                existing = next((u for u in list_microphone_units(microphone_db_path) if u.profile_id == profile_id and u.serial_number == serial.strip()), None)
                from dataclasses import replace
                unit = existing or MicrophoneUnit(id=uuid.uuid4().hex, profile_id=profile_id, serial_number=serial.strip())
                calibration_id = uuid.uuid4().hex
                save_measurement(measurement_db_path, record=SpeakerMeasurementRecord(id=calibration_id,
                    name=upload.name, source_name=upload.name, microphone=serial.strip(), source_type="mic_calibration_raw",
                    raw_text=text, response_payload=speaker_response_payload(response)))
                unit = replace(unit, **{("calibration_90_measurement_id" if angle == 90 else "calibration_measurement_id"): calibration_id})
                save_microphone_unit(microphone_db_path, unit)
                _config()["unit_id"] = unit.id
                st.success(ui_message('ui.5576e94fbb8200'))
            except (ValueError, OSError, UnicodeError) as exc:
                st.error(ui_message('ui.f83d09aa502682', p0=f'{exc}'))


def _setup(service, config, autosave_root, microphone_db_path, measurement_db_path):
    st.markdown(ui_message('ui.a545da7a975887'))
    profiles = {p.id: p for p in list_microphone_profiles(microphone_db_path) if p.connection_type == "usb"}
    units = {u.id: u for u in list_microphone_units(microphone_db_path) if u.profile_id in profiles}
    st.session_state.setdefault("guide_unit", config.get("unit_id") if config.get("unit_id") in units else None)
    if st.session_state.get("guide_unit") not in units:
        st.session_state["guide_unit"] = None
    selected = st.selectbox(ui_message('ui.fd52f9a64d608a'), list(units), index=None, key="guide_unit",
        format_func=lambda k: f"{profiles[units[k].profile_id].model} · {units[k].serial_number}", persist_state="session")
    config["unit_id"] = selected
    st.caption(ui_message('ui.2d7a6d9f817604'))
    _register_microphone(microphone_db_path, measurement_db_path, profiles, expanded=not units)
    if st.button(ui_message('ui.e7482f987c9845'), key='i18n_0ef76a1d181cf56c'):
        refresh_audio_devices()
    try:
        devices = list_input_devices(min_channels=1)
    except Exception as exc:
        st.error(ui_message('ui.c4c5449daa9514', p0=f'{exc}'))
        devices = []
    by_device = {d["index"]: d for d in devices}
    unit = units.get(selected)
    if "guide_device" not in st.session_state and unit is not None:
        match = next((d["index"] for d in devices if d["name"] == unit.device_name), None)
        if match is not None:
            st.session_state["guide_device"] = match
    device = st.selectbox(ui_message('ui.6ec6b11696e26e'), list(by_device), format_func=lambda k: by_device[k]["name"], key="guide_device", persist_state="session")
    st.session_state.setdefault("guide_angle", config["angle"])
    angle = st.selectbox(ui_message('ui.8822ec2aaba34d'), [0, 90], key="guide_angle", persist_state="session", format_func=localized_formatter(str))
    config["angle"] = angle
    unit = units.get(selected)
    calibration_id = (unit.calibration_90_measurement_id if angle == 90 else unit.calibration_measurement_id) if unit else ""
    calibration = get_measurement(measurement_db_path, calibration_id) if calibration_id else None
    if calibration is not None and calibration.source_type != "mic_calibration_raw":
        calibration = None
    if calibration:
        st.caption(ui_message('ui.cc971474a68b07', p0=f'{calibration.source_name}'))
    st.session_state.setdefault("guide_target", config["target"])
    config["target"] = st.text_input(ui_message('ui.fc0d322cf40fd8'), key="guide_target", persist_state="session")
    st.session_state.setdefault("guide_position", config["position"])
    config["position"] = required_segmented_control(ui_message('ui.1332819ff4b653'), list(POSITIONS), default=config["position"], key="guide_position", required=True, persist_state="session")
    st.session_state.setdefault("guide_purpose", config["purpose"])
    st.session_state.setdefault("guide_path_kind", config["path_kind"])
    config["purpose"] = st.selectbox(ui_message('ui.c79bfdb0efca73'), ["スピーカー特性", "室内を含む特性"], key="guide_purpose", persist_state="session", format_func=localized_formatter(str))
    config["path_kind"] = st.selectbox(ui_message('ui.c0b2921eec9280'), ["システム経由", "ユニット単体"], key="guide_path_kind", persist_state="session", format_func=localized_formatter(str))
    output = config["output"]
    if output == "both":
        for key, label in (("left_name", "左スピーカー"), ("right_name", "右スピーカー")):
            st.session_state.setdefault("guide_"+key, config.get(key, label))
            config[key] = st.text_input(label+"の登録名", key="guide_"+key)
    for ch, key in (("L", "left"), ("R", "right")):
        target = config[key+"_name"] if output == "both" else config["target"] if output == key else "非対象側：個別ミュート"
        st.write(ui_message('ui.2eafd653279533', p0=f'{ch}', p1=f'{target}'))
    st.caption(ui_message('ui.086f6cbae272e6'))
    placement = {"近接1 cm": "対象振動板の中央からマイクのカプセルまで1 cm。",
        "30 cm": "指定した測定軸上で、基準点からカプセルまで30 cm。",
        "1 m": "指定した測定軸上で、基準点からカプセルまで1 m。",
        "試聴位置": "普段聴く位置の耳の高さにカプセルを置きます。"}[config["position"]]
    st.info(placement + (ui_message('ui.e810eebaf76a22') if angle == 0 else ui_message('ui.d108124638e4e7')) + ui_message('ui.35eb0aba4efa2c'))
    with st.container(border=True):
        st.write(ui_message('ui.96ac29cfea538d', p0=f"{config['position']}"))
        st.caption(ui_message('ui.d0a2407d5f277b'))
    config["origin"] = st.text_input(ui_message('ui.02beaf2fb264eb'), value=config.get("origin", "対象のバッフル面・ツイーター軸"), key="guide_origin")
    protected = True
    if config["path_kind"] == "ユニット単体":
        st.warning(ui_message('ui.2f3c803dc718d2'))
        protected = st.checkbox(ui_message('ui.019d97546d2c22'), key="guide_protected")
    if output == "both":
        st.caption(ui_message('ui.9ec7512d3960bc'))
    if output == "both" and config["position"] == "近接1 cm":
        st.warning(ui_message('ui.adbefed0bb998b'))
    mic = None
    try:
        mic = microphone_settings(profiles.get(unit.profile_id) if unit else None, unit, calibration,
            by_device[device]["name"] if device in by_device else "", unit.input_channel if unit else 1)
    except ValueError as exc:
        st.info(str(exc))
    st.caption(ui_message('ui.3b4a37f0fed7e9'))
    if st.button(ui_message('ui.8049c0c89a49c6'), type="primary", disabled=mic is None or device is None or not protected or not config["target"].strip(), key='i18n_9ac7dccb89a079e1'):
        try:
            from dataclasses import replace
            save_microphone_unit(microphone_db_path, replace(unit, device_name=by_device[device]["name"]))
            _save_preferences(Path(autosave_root).parent.parent / "measurement_guide.json", config)
            release_audio_downloads_before_capture()
            track = track_for(config["kind"], output)
            settings, reference = settings_for_track(service.status(track, config["media"]), mic, device, config["position"], angle)
            engine = ContinuousMeasurementEngine(settings, Path(autosave_root), reference=reference.samples)
            st.session_state[ENGINE_KEY] = engine
            st.session_state["_guided_capture_selection"] = dict(config)
            st.session_state["_guided_wait_started"] = time.monotonic()
            engine.start()
            _move("capture")
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(ui_message('ui.dfa9207485df7a', p0=f'{exc}'))
    if st.button(ui_message('ui.6d01df0b8a206f'), key='i18n_5e3e12552e098a89'):
        _move("select")


@st.fragment(run_every=1)
def _capture(config, engine):
    if engine is None:
        _move("setup")
    snap = engine.snapshot()
    stage = snap.device_info.get("guide_stage", "ambient")
    track = track_for(config["kind"], config["output"])
    active = str(snap.state) in ACTIVE
    if stage == "ambient" and active:
        st.markdown(ui_message('ui.67c0da35b1c991'))
        st.info(ui_message('ui.fda246cd3bdd3e'))
    elif stage == "noise" and active:
        st.markdown(ui_message('ui.df3247f3b8366b'))
        st.info(ui_message('ui.b21179cec3fbda'))
        st.caption(ui_message('ui.7def4fa5abf9bf'))
        levels = snap.device_info.get("live_input_rms_dbfs", [])
        peaks = snap.device_info.get("live_input_peak_dbfs", [])
        ambient = snap.device_info.get("ambient_noise_p90_dbfs", [-150])
        offset = level_calibration_offset_db(snap.settings)
        if levels and offset is not None:
            st.metric(ui_message('ui.edddcf02b67f72'), f"{levels[0]+offset:.1f} dB SPL")
        st.caption(ui_message('ui.f633e86b95e3f1'))
        level_ok = bool(levels and peaks and peaks[0] < -23 and levels[0] > ambient[0]+6)
        st.write("ノイズを確認できました" if level_ok else "ノイズとクリップ余裕を確認中です")
        if config.get("level_only"):
            if st.button(ui_message('ui.3419eda97f70ed'), type="primary", key='i18n_35beb4918970dc17'):
                engine.guide_recheck("音量確認を終了しました。外部プレーヤーを停止してください。")
                st.rerun()
        elif st.button(ui_message('ui.22730e744bc92f'), type="primary", disabled=not level_ok, key='i18n_1b0ef38a52039018'):
            _move("switch")
    elif stage in {"pilot", "measuring"} and active:
        st.markdown(ui_message('ui.f140e2c95f5545'))
        st.info(ui_message('ui.79d7505bd2b842', p0=f'{track.number:02d}', p1=display_text(track.title)))
        count = min(3, len(snap.valid_shots))
        st.progress(count/3, text=f"採用できた測定 {count} / 3")
        st.write("ESSの音量・同期を確認しています" if stage == "pilot" else "次の測定音を待っています")
        if snap.device_info.get("last_rejection_reason"):
            st.caption(ui_message('ui.ede63060fb8034') + str(snap.device_info["last_rejection_reason"]))
        if time.monotonic()-st.session_state.get("_guided_wait_started", time.monotonic()) > max(60, track.period_s*6):
            st.warning(ui_message('ui.3ffb6271ecd835'))
    elif stage == "finalizing":
        if not snap.device_info.get("guide_finalized"):
            st.info(ui_message('ui.c6018ab9fe092a'))
        elif completed(snap):
            st.success(ui_message('ui.99a5b8acde62fd'))
            if st.button(ui_message('ui.5f7a4a2ebd845e'), type="primary", key='i18n_c27efa24d8a5bd04'):
                _move("result")
        else:
            st.error(ui_message('ui.d854bac35c1822'))
    else:
        st.warning(ui_message('ui.54c0f97079ba8c'))
        st.caption(str(snap.device_info.get("guide_reason") or snap.error_message or "停止しました"))
    if active and st.button(ui_message('ui.c499bf1b0976b5'), icon=":material/stop:", key='i18n_34626e55b57b5dc1'):
        engine.guide_recheck("利用者が中断しました。")
        st.rerun()
    if not active and not completed(snap) and (stage != "finalizing" or snap.device_info.get("guide_finalized")) and st.button(ui_message('ui.bbb3efc04bc66f'), key='i18n_431755426fcb1c99'):
        st.session_state.setdefault("_guided_previous_sessions", []).append(str(snap.autosave_directory))
        _move("setup")


@st.fragment
def render_guided_measurement(*, autosave_root, measurement_db_path, microphone_db_path, ess_reference_db_path, apply_result=None):
    preferences = Path(autosave_root).parent.parent / "measurement_guide.json"
    if CONFIG_KEY not in st.session_state and preferences.exists():
        try:
            restored = json.loads(preferences.read_text())
            if restored.get("kind") in KINDS and restored.get("output") in OUTPUTS and restored.get("position") in POSITIONS and restored.get("media") in {"cd", "file"}:
                st.session_state[CONFIG_KEY] = {**_config(), **restored}
        except (ValueError, OSError) as exc:
            st.warning(ui_message('ui.44339c281287ec', p0=f'{exc}'))
    config = _config()
    service = standard_audio_service(str(Path(autosave_root).parent.parent / "measurement_audio"), str(ess_reference_db_path))
    step = st.session_state.setdefault("_guided_step", "select")
    engine = st.session_state.get(ENGINE_KEY)
    st.caption(ui_message('ui.d8f809096bccf7'))
    try:
        if step == "select":
            _selector(service, config)
        elif step == "setup":
            _setup(service, config, autosave_root, microphone_db_path, measurement_db_path)
        elif step == "switch":
            track = track_for(config["kind"], config["output"])
            st.markdown(ui_message('ui.e1a982a659af22'))
            st.info(ui_message('ui.3e433958cf5405', p0=f'{track.number:02d}', p1=display_text(track.title)))
            st.caption(ui_message('ui.831c2fca10bcf6'))
            if st.button(ui_message('ui.06c6b11a9bbe0b'), type="primary", key='i18n_8b164489695c3637'):
                engine.guide_begin_ess()
                st.session_state["_guided_wait_started"] = time.monotonic()
                _move("capture")
            if st.button(ui_message('ui.c499bf1b0976b5'), key='i18n_c0eac8f6f28e8eb2'):
                engine.guide_recheck("曲切り替え中に中断しました。")
                _move("capture")
        elif step == "capture":
            _capture(config, engine)
        elif step == "result":
            _result(config, engine, measurement_db_path, apply_result)
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(ui_message('ui.256a11ed76cde0', p0=f'{exc}'))


def _result(config, engine, db, apply_result):
    snapshot = engine.snapshot()
    st.markdown(ui_message('ui.cf0976d9f9e897'))
    if not completed(snapshot):
        st.error(ui_message('ui.bddbd7b9fd9b2f'))
        return
    st.success(ui_message('ui.ce4310f2e6917f', p0=f"{config['target']}", p1=f"{config['position']}"))
    st.session_state.setdefault("guide_result_name", f"{config['target']}_{config['position']}_{KINDS[config['kind']]}_{datetime.now():%Y%m%d_%H%M%S}")
    name = st.text_input(ui_message('ui.12b3c2452d93a6'), key="guide_result_name")
    saved_id = st.session_state.get("_guided_saved_id")
    is_saved = saved_id == "guided-"+snapshot.session_id
    if st.button(ui_message('ui.a1a4245ad1a583'), type="primary", disabled=is_saved or not name.strip(), key='i18n_2c5f771366103b8d'):
        record = save_guided_result(Path(db), snapshot, st.session_state["_guided_capture_selection"], name)
        st.session_state["_guided_saved_id"] = record.id
        st.session_state["_guided_saved_record"] = record
        cleanup_measurement_autosave(snapshot)
        st.rerun()
    if is_saved:
        st.success(ui_message('ui.dd23b03c04ea3c'))
        from utils.channel_input_policy import channel_input_locked, CHANNEL_INPUT_HELP
        if channel_input_locked(st.session_state):
            st.caption(display_text(CHANNEL_INPUT_HELP))
        if config["output"] != "both" and apply_result and st.button(ui_message('ui.d45fd8f997f067'), type="primary", key='i18n_91909a1b7d83e150', disabled=channel_input_locked(st.session_state)):
            ok, message = apply_result(st.session_state["_guided_saved_record"])
            (st.success if ok else st.error)(message)
        elif config["output"] == "both":
            st.caption(ui_message('ui.9a166709a85d99'))
        original = load_measurement_original(Path(db), saved_id)
        st.download_button(ui_message('ui.365c232f127d05'), deferred_call(calibrated_export_zip, original),
            file_name=f"{name}_calibrated.zip", mime="application/zip", on_click="ignore", key='i18n_f7ee37c37012034b')
        if st.button(ui_message('ui.8620c7874b7e3d'), key='i18n_485370f15800750a'):
            st.session_state.pop("guide_result_name", None)
            _move("select")


@st.fragment(run_every=1)
def render_guided_results():
    engine = st.session_state.get(ENGINE_KEY)
    config = _config()
    st.markdown(ui_message('ui.0ad65d4f5dabed'))
    if config.get("output"):
        st.write(ui_message('ui.61791a3671b0d7', p0=display_text(KINDS[config['kind']]), p1=display_text(OUTPUTS[config['output']]), p2=f"{config['position']}"))
    if engine is None or not engine.settings.guided_measurement:
        st.info(ui_message('ui.99fd61a4f8530e') if st.session_state.get("_guided_step") == "setup" else ui_message('ui.43bd4cdfca4434'))
        return
    snap = engine.snapshot()
    if snap.standard_result is not None:
        frame, metadata = _result_frame(snap.session_id, snap.revision, snap)
        low, high = metadata["valid_frequency_range_hz"]
        import altair as alt
        st.altair_chart(alt.Chart(frame).mark_line().encode(x=alt.X("周波数 Hz:Q", scale=alt.Scale(type="log")), y="レベル dB:Q"))
        st.caption(ui_message('ui.e03115252b9f32', p0=f'{low:g}', p1=f'{high:g}', p2=f"{metadata['gain_unit']}"))
    else:
        st.info(ui_message('ui.3073303fb4618c'))


@st.cache_data(max_entries=4, ttl=300, show_spinner=False)
def _result_frame(session_id, revision, _snapshot):
    from phase_fir_designer.measurement.original import original_from_snapshot
    frequency, response, metadata = calibrated_output(original_from_snapshot(_snapshot))
    low, high = metadata["valid_frequency_range_hz"]
    mask = (frequency >= low) & (frequency <= high)
    stride = max(1, int(np.sum(mask))//2000)
    frame = pd.DataFrame({"周波数 Hz": frequency[mask][::stride], "レベル dB": 20*np.log10(np.maximum(np.abs(response[mask][::stride]), 1e-15))})
    return frame, metadata
