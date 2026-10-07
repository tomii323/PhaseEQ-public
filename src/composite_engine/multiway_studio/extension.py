from __future__ import annotations
from utils.ui_localization import ui_message, display_text, localized_formatter, display_notice

from utils.exchange_retry import record_failure, record_success, retry_ready, reset_retries

import base64
import hashlib
import html
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from urllib.parse import urlencode
import zipfile
import uuid
from dataclasses import dataclass
import datetime
from collections.abc import Callable

import numpy as np
import pandas as pd
import streamlit as st
from utils.list_menu_ui import selectbox as shared_selectbox, radio as shared_radio, segmented_control as shared_segmented_control

from crossover_engine.recipe import from_studio as band_split_from_studio
from composite_engine.multiway_studio.processing import alignment_target, alignment_structure

from composite_engine.adapter import (
    ChannelPipelineInput,
    CompositePackage,
    GroupTargetInput,
    PhaseEQAssignmentRequest,
    ResponseStage,
    attach_group_targets,
    build_channel_pipelines,
    latest_phaseeq_assignment_id,
    latest_published_phaseeq_assignment_id,
    phaseeq_channel_id,
    read_phaseeq_assignment_status,
    write_phaseeq_assignment,
    analyze_phase_alignment,
)
from utils.ui_work_cache import deferred_call, first_result
from composite_engine.graph import phase_gain_masked_values
from composite_engine.core import MultichannelCompositeResult, result_from_responses
from composite_engine.display_projection import (
    DISPLAY_LEFT,
    DISPLAY_MAIN,
    DISPLAY_RIGHT,
    DISPLAY_STEREO,
    build_display_graph_series,
    normalize_display_value,
    resolve_display_projection,
)
from composite_engine.iir_crossover import (
    IIRCrossoverConfig,
    iir_crossover_sos,
    realized_exclusive_iir_config,
)
from target_engine import (
    TargetBinding,
    canonical_target_definition,
    create_target,
    create_target_edit_session,
    find_target_revision_by_hash,
    read_target_edit_session,
    shared_target_preset_catalog,
    materialize_target_preset,
    save_target_binding,
    target_definition_frd_bytes,
    target_definition_from_asset,
)


PHASEEQ_WORKSPACE_WINDOW_NAME = "phaseeq_assignment_workspace"


def _build_multichannel_export_zip(*args, **kwargs):
    """Resolve export-only dependencies when a download is requested."""
    from composite_engine.export import build_multichannel_export_zip
    return build_multichannel_export_zip(*args, **kwargs)


def _build_dsp_resume_zip(*args, **kwargs):
    """Resolve resume-package dependencies when a download is requested."""
    from composite_engine.export import build_dsp_resume_zip
    return build_dsp_resume_zip(*args, **kwargs)


def _build_dsp_export_zip(*args, **kwargs):
    """Resolve adapter packaging only after Streamlit requests the payload."""
    from composite_engine.dsp_export import build_dsp_export_zip
    return build_dsp_export_zip(*args, **kwargs)


def _render_phaseeq_workspace_link(label: str, url: str) -> None:
    """Keep the launch control mounted while connection state changes."""
    from ui.browser_presence import launch_button
    launch_button(_exchange_root(), "phaseeq", label, url)


def _assignment_status_display(state: str) -> str:
    return {
        "Assigned": "送信済み",
        "Editing": "PhaseEQ編集中",
        "Ready": "返却済み",
    }.get(str(state), str(state) or "未送信")


def _assignment_reference_state(connected_id: str, latest_id: str) -> str:
    connected = str(connected_id).strip()
    latest = str(latest_id).strip()
    if connected and connected == latest:
        return "最新と一致"
    if connected and latest:
        return "別Assignmentを参照"
    if not connected:
        return "未接続"
    return "最新ID不明"


def _connected_assignment_id(channel_id, latest_id, sessions):
    """Resolve live editor identity independently of retained DSP results."""
    ids = [str(getattr(item, "assignment_id", "")) for item in sessions
           if getattr(item, "mode", "") == "Assignment"
           and getattr(item, "channel_id", "") == str(channel_id)]
    if latest_id in ids:
        return str(latest_id)
    return next((value for value in ids if value), "")


def _phaseeq_screen_state(channel_id: str, latest_id: str, sessions: object) -> str:
    channel_sessions = [
        session for session in sessions
        if getattr(session, "mode", "") == "Assignment"
        and getattr(session, "channel_id", "") == str(channel_id)
    ]
    if not channel_sessions:
        return "画面なし"
    if any(not getattr(session, "assignment_id", "") for session in channel_sessions):
        return "参照中（ID不明）"
    if not str(latest_id):
        return "参照中（最新ID不明）"
    if any(getattr(session, "assignment_id", "") == str(latest_id) for session in channel_sessions):
        return "参照中（最新）"
    return "参照中（別ID）"


def _phaseeq_inbox_transition(
    previous_result: str | None,
    result_signature: str,
) -> tuple[bool, bool]:
    """Only a DSP-result revision may trigger a full Studio rerun."""
    result_changed = previous_result is not None and previous_result != result_signature
    return result_changed, result_changed


@dataclass(frozen=True)
class _PhaseEQConnectionView:
    level: str
    message: str
    assignment_sessions: tuple[object, ...]
    incomplete_assignment_count: int


def _phaseeq_connection_view(sessions: object) -> _PhaseEQConnectionView:
    records = tuple(sessions)
    assignment_records = tuple(
        session for session in records if getattr(session, "mode", "") == "Assignment"
    )
    usable_assignments = tuple(
        session for session in assignment_records
        if getattr(session, "assignment_id", "") and getattr(session, "channel_id", "")
    )
    incomplete_count = len(assignment_records) - len(usable_assignments)
    if usable_assignments:
        level = "success"
        message = f"PhaseEQ Assignment: {len(usable_assignments)}画面が編集中です。"
    elif any(getattr(session, "mode", "") == "Standalone" for session in records):
        level = "info"
        message = "PhaseEQはStandaloneで開いています。このChannelには未接続です。"
    else:
        level = "caption"
        message = "PhaseEQ Assignment画面は開いていません。"
    return _PhaseEQConnectionView(
        level=level,
        message=message,
        assignment_sessions=usable_assignments,
        incomplete_assignment_count=incomplete_count,
    )
from composite_engine.multiway_studio.utils.display_rounding import format_for_display
from composite_engine.validation import CompositeValidationError
from composite_engine.phase_alignment import AllPassSection, allpass_response
from composite_engine.phase_alignment import (
    AlignmentBranch,
    analyze_system_phase_alignment,
)
from composite_engine.distance_timing import (
    NOMINAL_SOUND_SPEED_M_S,
    external_distance_timing,
)
from composite_engine.speaker_timing import (
    resolve_speaker_timing_projection,
    speaker_timing_samples_for_row,
)
from utils.multiway_system_db import (
    MultiwayChannelReference,
    MultiwaySystem,
    duplicate_multiway_system,
    get_multiway_system,
    list_multiway_system_revisions,
    list_multiway_systems,
    new_multiway_system,
    rename_multiway_system,
    restore_multiway_system_revision,
    save_multiway_system,
    save_multiway_system_revision,
    set_multiway_system_archived,
    update_multiway_channel_assignment,
)
from utils.design_library_db import list_speaker_package_summaries, get_speaker_package
from utils.settings_io import speaker_response_from_payload
from utils.composite_exchange import (
    assignment_status_signature,
    list_active_phaseeq_sessions,
    write_multiway_workspace,
)
from composite_engine.multiway_studio.processing.speaker_source import (
    assess_speaker_alignment_readiness,
    resolve_speaker_source,
)
from composite_engine.multiway_studio.processing.dsp_input import studio_dsp_input_signature
from composite_engine.multiway_studio.processing.delay import (
    resolve_output_timing,
)
from composite_engine.multiway_studio.components.visualization import (
    DEFAULT_GAIN_Y_MAX_DB,
    DEFAULT_GAIN_Y_MIN_DB,
    DEFAULT_PHASE_GAIN_MASK_DB,
    DEFAULT_GRAPH_MODE,
    DEFAULT_GRAPH_MAX_POINTS,
    TIME_RESPONSE_DISPLAY_MAX_MS,
    TIME_RESPONSE_DISPLAY_MIN_MS,
    log_spaced_sample_indices,
    render_studio_figure,
    set_group_delay_ylim,
    set_time_response_xlim,
)


CROSSOVER_METHODS = ("Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR", "LR2", "LR4", "Through")
CROSSOVER_METHOD_LABELS = {
    "Through": "スルー（外部で帯域分割済み）",
    "Kaiser FIR": "Kaiser-window linear-phase FIR",
    "Linear-phase LR4 FIR": "LR4 linear-phase FIR",
    "Linear-phase LR2 FIR": "LR2 linear-phase FIR",
    "LR2": "LR2 IIR(Biquad)",
    "LR4": "LR4 IIR(2 Biquads)",
}


@dataclass(frozen=True)
class _CachedUpload:
    name: str
    data: bytes

    def getvalue(self) -> bytes:
        return self.data


def _resume_display_value(
    workspace: dict[str, object], settings: dict[str, object], *, stereo: bool,
) -> str:
    """Resolve display state without trusting legacy non-scalar selected_group data."""
    saved_display = workspace.get("selected_display")
    if not isinstance(saved_display, str):
        saved_display = settings.get("selected_display")
    if not isinstance(saved_display, str):
        # Early DSP Export v3 packages could accidentally store graph records
        # in selected_group. Channel layout remains authoritative in that case.
        saved_display = "L+R" if stereo else "Main"
    return normalize_display_value(saved_display, stereo=stereo)


def _capture_channel_upload(widget_key: str, asset_key: str) -> None:
    """Persist a user upload outside widget-owned state across forced reruns."""
    upload = st.session_state.get(widget_key)
    cache = st.session_state.setdefault("_composite_channel_upload_cache", {})
    if upload is None:
        cache.pop(asset_key, None)
        return
    cache[asset_key] = {
        "name": Path(str(upload.name)).name,
        "data": bytes(upload.getvalue()),
    }


def _channel_upload(widget_value: object, asset_key: str) -> object | None:
    if widget_value is not None:
        return widget_value
    cache = st.session_state.get("_composite_channel_upload_cache", {})
    value = cache.get(asset_key) if isinstance(cache, dict) else None
    if not isinstance(value, dict) or not value.get("data"):
        return None
    return _CachedUpload(Path(str(value.get("name", "upload.bin"))).name, bytes(value["data"]))


def _downsample_graph_frame(
    frame: pd.DataFrame, *, max_points_per_series: int = 4096, wrapped_phase: bool = False,
) -> pd.DataFrame:
    """Bound browser chart payload without changing the computed DSP result."""
    if frame.empty:
        return frame
    sampled: list[pd.DataFrame] = []
    for _name, group in frame.groupby("series", sort=False, dropna=False):
        from response_display import sample_series
        indices = log_spaced_sample_indices(
            group["x"].to_numpy(dtype=float), int(max_points_per_series),
        )
        selected = group.iloc[indices].copy()
        _, selected_values = sample_series(group["x"].to_numpy(dtype=float), group["value"].to_numpy(dtype=float),
                                           max_points_per_series, wrapped_phase=wrapped_phase)
        selected["value"] = selected_values
        sampled.append(selected)
    return pd.concat(sampled, ignore_index=True) if sampled else frame.iloc[0:0]


def _time_response_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep every computed tap inside the visible impulse/step time window."""
    if frame.empty:
        return frame
    x_values = frame["x"].to_numpy(dtype=float)
    return frame[
        (x_values >= TIME_RESPONSE_DISPLAY_MIN_MS)
        & (x_values <= TIME_RESPONSE_DISPLAY_MAX_MS)
    ].reset_index(drop=True)


def _finite_axis_domain(values: pd.Series) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1.0)
    else:
        padding = max((upper - lower) * 0.02, 1e-9)
    return (lower - padding, upper + padding)


def normalize_crossover_method(value: object) -> str:
    normalized = str(value).strip()
    return normalized if normalized in CROSSOVER_METHODS else "Kaiser FIR"


def _clear_applied_phase_alignment() -> None:
    """Invalidate automatic alignment when the realized crossover changes."""
    st.session_state["_alignment_refresh_requested"] = True
    for key in list(st.session_state):
        if key.endswith("_auto_alignment_delay"):
            st.session_state[key] = 0.0
        elif key.endswith("_auto_alignment_allpass"):
            st.session_state[key] = []
        elif key.startswith("composite_alignment_") and any(
            marker in key
            for marker in (
                "_applied_", "_undo_", "_preview_", "_residual_",
                "_removed_allpass_", "_removed_delays_", "_signature_",
                "_caution_confirmed_",
                "system_applied", "system_undo", "system_signature",
            )
        ):
            st.session_state.pop(key, None)


def render_shared_iir_crossover_ui(
    mode_key: str,
    crossover_frequencies_hz: tuple[float, ...],
    *,
    initial_methods: tuple[str, ...] = (),
    initial_lr2_auto_polarity: bool = True,
    on_method_change: Callable[[], None] | None = None,
    on_acoustic_target_change: Callable[[], None] | None = None,
    render_methods: bool = True,
) -> object:
    """Render Studio-owned IIR controls beside the shared Kaiser boundaries."""
    ordered_ways = MODE_WAYS[str(mode_key)]
    if not crossover_frequencies_hz:
        return st.container()
    _prefix = f"composite_studio_{mode_key}"
    polarity_key = f"{_prefix}_lr2_auto_polarity"
    st.session_state.setdefault(polarity_key, bool(initial_lr2_auto_polarity))
    location = (
        st.container(border=True)
        if render_methods else
        st.expander(ui_message('ui.2b6ef034f57451'), expanded=False)
    )
    with location:
        if render_methods:
            st.markdown(ui_message('ui.d23a182d5bd1e6'))
        st.toggle(
            ui_message('ui.bd30318ab45dac'),
            key=polarity_key,
            help=ui_message('ui.667b94b8e1eb5b'),
        )
        if render_methods:
            st.caption(ui_message('ui.7c3a63e441572c'))
            # Match the surrounding Studio layout: show the highest-frequency
            # boundary first while retaining the low-to-high boundary index.
            for index in reversed(range(len(crossover_frequencies_hz))):
                common_fc = crossover_frequencies_hz[index]
                label = f"{ordered_ways[index]} / {ordered_ways[index + 1]}"
                method = render_crossover_method_control(
                    mode_key, index, label,
                    initial_method=(
                        initial_methods[index]
                        if index < len(initial_methods) else "Kaiser FIR"
                    ),
                    on_method_change=on_method_change,
                    on_acoustic_target_change=on_acoustic_target_change,
                )
                overlap = _boundary_overlap_from_state(str(mode_key), index)
                lp = float(common_fc) * 2.0 ** (float(overlap) / 2.0)
                hp = float(common_fc) * 2.0 ** (-float(overlap) / 2.0)
                if method != "Through":
                    st.caption(ui_message('ui.25adf695ed2f1c', p0=f'{CROSSOVER_METHOD_LABELS[method]}', p1=f'{float(common_fc):,.1f}', p2=f'{lp:,.1f}', p3=f'{hp:,.1f}'))
    # The realized responses needed by the analyzer are built later in the
    # Streamlit run. Keep the previous controls mounted until their replacement
    # arrives: st.empty() collapses this area on every rerun and scrolls the
    # downstream tap inputs out of position while calculation is in progress.
    return st.container()


def render_crossover_method_control(
    mode_key: str,
    index: int,
    label: str,
    *,
    initial_method: str = "Kaiser FIR",
    on_method_change: Callable[[], None] | None = None,
    on_acoustic_target_change: Callable[[], None] | None = None,
    fir_enabled: bool = True,
) -> str:
    """Render the primary method selector inside one boundary design block."""
    method_key = f"composite_studio_{mode_key}_crossover_method_{index}"
    target_key = f"composite_studio_{mode_key}_acoustic_target_{index}"
    target_methods = {"Linear-phase LR2 FIR", "Linear-phase LR4 FIR", "LR2", "LR4"}
    normalized_initial = normalize_crossover_method(initial_method)
    if method_key not in st.session_state:
        st.session_state[method_key] = normalized_initial
    elif normalize_crossover_method(st.session_state[method_key]) != st.session_state[method_key]:
        st.session_state[method_key] = normalized_initial

    def _on_change() -> None:
        # A method transition may choose a target default, but it must not
        # impersonate an explicit target-toggle action or rewrite tap settings.
        st.session_state[target_key] = st.session_state[method_key] in target_methods
        _clear_applied_phase_alignment()
        if on_method_change is not None:
            on_method_change()

    from utils.band_split_capability import convert_studio_methods, studio_method_options
    methods = studio_method_options(fir_enabled=fir_enabled)
    st.session_state[method_key] = convert_studio_methods(
        [st.session_state[method_key]], fir_enabled=fir_enabled,
    )[0]
    method = shared_segmented_control(
        ui_message('ui.b47dba7d9b82bd', p0=f'{label}'), methods,
        format_func=localized_formatter(lambda value: CROSSOVER_METHOD_LABELS[value]),
        key=method_key, width="stretch", required=True, on_change=_on_change,
    ) or "Kaiser FIR"
    target_allowed = method in target_methods
    initial_targets = st.session_state.get("settings", {}).get(mode_key, {}).get("boundary_acoustic_targets", [])
    if target_key not in st.session_state:
        st.session_state[target_key] = bool(initial_targets[index]) if index < len(initial_targets) else target_allowed
    if not target_allowed:
        st.session_state[target_key] = False
    st.checkbox(ui_message('ui.6af081827d6a12', p0=f'{label}'), key=target_key, disabled=not target_allowed,
                on_change=on_acoustic_target_change or on_method_change,
                help=(
                    ui_message('ui.a45fc68c20a621')
                ))
    if target_allowed:
        with st.expander(ui_message('ui.2bbd78a88b6990', p0=f'{label}'), expanded=False):
            st.markdown(
                ui_message('ui.d83ea615703db4')
            )
    if method == "Through":
        st.caption(
            ui_message('ui.df06f2d052f642')
        )
    return method


def _boundary_overlap_from_state(mode_key: str, index: int) -> float:
    legacy_keys = {
        "2Way": ("kaiser_overlap_oct",),
        "3Way": ("kaiser_overlap_low_mid_oct", "kaiser_overlap_mid_high_oct"),
        "3Way+SUB": ("kaiser_overlap_sub_low_oct", "kaiser_overlap_low_mid_oct", "kaiser_overlap_mid_high_oct"),
    }
    key = legacy_keys.get(str(mode_key), ())[index] if str(mode_key) in legacy_keys else f"boundary_overlap_{mode_key}_{index}"
    return float(st.session_state.get(key, 0.0))


def _shared_iir_configs(
    mode_key: str, crossover_frequencies_hz: tuple[float, ...],
) -> dict[str, IIRCrossoverConfig]:
    ordered_ways = MODE_WAYS.get(str(mode_key), ())
    prefix = f"composite_studio_{mode_key}"
    methods = tuple(
        normalize_crossover_method(st.session_state.get(f"{prefix}_crossover_method_{index}"))
        for index in range(len(crossover_frequencies_hz))
    )
    methods = tuple("Through" if st.session_state.get(f"{prefix}_acoustic_target_{i}", False) else method
                    for i, method in enumerate(methods))
    overlaps = tuple(
        _boundary_overlap_from_state(str(mode_key), index)
        for index in range(len(crossover_frequencies_hz))
    )
    configs = {
        way: realized_exclusive_iir_config(
            way=way, ordered_ways=ordered_ways,
            crossover_frequencies_hz=tuple(float(value) for value in crossover_frequencies_hz),
            methods=methods, overlap_oct=overlaps,
            lr2_auto_polarity=bool(st.session_state.get(f"{prefix}_lr2_auto_polarity", True)),
        )
        for way in ordered_ways
    }
    return configs


def _exchange_root() -> Path:
    return Path(os.environ.get(
        "PHASEEQ_COMPOSITE_EXCHANGE_DIR",
        Path(os.environ.get("PHASEEQ_DATA_DIR", "data")) / "tmp" / "composite_exchange",
    ))


def _multiway_system_db_path() -> Path:
    return Path(os.environ.get("PHASEEQ_DATA_DIR", "data")) / "databases" / "multiway_systems.sqlite3"


def _design_library_paths() -> tuple[Path, Path]:
    root = Path(os.environ.get("PHASEEQ_DATA_DIR", "data")) / "databases"
    return root / "phaseeq_design_library.sqlite3", root / "response_assets.sqlite3"


def _speaker_package_options() -> tuple[list[str], dict[str, object]]:
    design_path, response_path = _design_library_paths()
    records = list_speaker_package_summaries(design_path, response_path)
    return ["", *(record.id for record in records)], {record.id: record for record in records}


def _speaker_package_frd(record: object) -> bytes | None:
    runtime = getattr(record, "runtime_payload", {})
    config = runtime.get("config", {}) if isinstance(runtime, dict) else {}
    response = speaker_response_from_payload(
        config.get("speaker_response") if isinstance(config, dict) else None
    )
    if response is None:
        return None
    if response.phase_deg is None:
        lines = (
            f"{frequency:.16g} {gain:.16g}"
            for frequency, gain in zip(response.frequency, response.gain_db, strict=True)
        )
    else:
        lines = (
            f"{frequency:.16g} {gain:.16g} {phase_value:.16g}"
            for frequency, gain, phase_value in zip(
                response.frequency, response.gain_db, response.phase_deg, strict=True,
            )
        )
    return ("\n".join(lines) + "\n").encode()


def _channel_state_prefix(band: str, group: str, *, stereo: bool) -> str:
    if not stereo:
        return f"composite_studio_{band.casefold()}"
    side = str(group).strip().casefold() or "main"
    return f"composite_studio_{side}_{band.casefold()}"


def _alignment_state_prefix(row: dict[str, object]) -> str:
    """Resolve the same state owner used when the Channel row was rendered."""
    return _channel_state_prefix(
        str(row.get("band", "")),
        str(row.get("group", "")),
        stereo=str(st.session_state.get("composite_studio_output_layout", "Mono")) == "Stereo",
    )


def _default_additional_input_group() -> str:
    if str(st.session_state.get("composite_studio_output_layout", "Mono")) != "Stereo":
        return "Main"
    displayed = normalize_display_value(
        st.session_state.get("composite_studio_result_group", "L+R"), stereo=True,
    )
    return "Right" if displayed == DISPLAY_RIGHT else "Left"


def _adjust_additional_input_count(delta: int) -> None:
    current = int(st.session_state.get("composite_studio_extra_count", 0) or 0)
    updated = min(32, max(0, current + int(delta)))
    if updated > current:
        for index in range(current, updated):
            prefix = f"composite_studio_extra_{index}"
            st.session_state.setdefault(f"{prefix}_name", f"Input {index + 1}")
            st.session_state.setdefault(f"{prefix}_way", "Fullrange")
            st.session_state.setdefault(f"{prefix}_group", _default_additional_input_group())
            st.session_state.setdefault(f"{prefix}_gain", 0.0)
            st.session_state.setdefault(f"{prefix}_polarity", "Normal (+)")
            st.session_state.setdefault(f"{prefix}_delay", 0.0)
    st.session_state["composite_studio_extra_count"] = updated


def _parse_external_distances_mm(text: str, expected_count: int) -> tuple[float, ...]:
    normalized = str(text).translate(str.maketrans({"，": ",", "；": ",", ";": ","}))
    tokens = [
        token for token in normalized.replace("\t", " ").replace(
            "\n", " ",
        ).replace(",", " ").split() if token
    ]
    if len(tokens) != int(expected_count):
        raise ValueError(f"Channel数と同じ{int(expected_count)}個の距離を入力してください。")
    try:
        values = tuple(float(token) for token in tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("距離はmm単位の数値で入力してください。") from exc
    if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
        raise ValueError("全ての距離を0より大きい有限値で入力してください。")
    return values


def _apply_external_distance_bulk(channel_ids: tuple[str, ...]) -> None:
    try:
        values = _parse_external_distances_mm(
            st.session_state.get("composite_alignment_external_distance_bulk", ""),
            len(channel_ids),
        )
    except ValueError as exc:
        st.session_state["composite_alignment_external_distance_bulk_error"] = str(exc)
        return
    for channel_id, distance_mm in zip(channel_ids, values, strict=True):
        st.session_state[f"composite_alignment_external_distance_{channel_id}"] = distance_mm
    st.session_state.pop("composite_alignment_external_distance_bulk_error", None)


def _alignment_timing_source_label(source: str) -> str:
    return {
        "phaseeq_tweeter_reference": "PhaseEQ Tweeter基準",
        "external_distance": "外部測定距離",
        "phase_fit": "Crossover位相推定",
    }.get(str(source), str(source) or "不明")


def _timing_export_metadata(row: dict[str, object]) -> dict[str, object] | None:
    metadata = (
        dict(row["timing_provenance"])
        if isinstance(row.get("timing_provenance"), dict) else {}
    )
    distance_m = float(row.get("external_distance_m", 0.0) or 0.0)
    applied = st.session_state.get("composite_alignment_system_applied")
    external_distance_is_current = (
        getattr(applied, "timing_source", "") == "external_distance"
        and st.session_state.get("composite_alignment_system_signature")
        == st.session_state.get("composite_alignment_system_current_signature")
    )
    if distance_m > 0.0 and external_distance_is_current and str(
        st.session_state.get("composite_alignment_system_timing_source", "自動選択")
    ) == "外部測定距離":
        metadata["external_distance_alignment"] = {
            "distance_m": distance_m,
            "sound_speed_m_s": NOMINAL_SOUND_SPEED_M_S,
            "source": "user_entered_external_measurement",
        }
    return metadata or None


def _current_library_system() -> MultiwaySystem | None:
    record = _current_library_record()
    return record.system if record is not None and not record.archived else None


def _current_library_record():
    system_id = str(st.session_state.get("composite_studio_current_system_id", "")).strip()
    if not system_id:
        return None
    return get_multiway_system(_multiway_system_db_path(), system_id)


OUTPUT_FIR_SETTINGS = (
    ("composite_output_cosine_taper", "output_fir_cosine_taper_enabled", False, bool),
    ("composite_output_remove_nyquist", "output_fir_remove_nyquist_enabled", False, bool),
    ("composite_output_nyquist_strength", "output_fir_remove_nyquist_strength", 1.0, float),
)


def restore_output_fir_settings(settings, *, overwrite=True) -> None:
    for widget, field, default, cast in OUTPUT_FIR_SETTINGS:
        value = cast(settings.get(field, default))
        if cast is float:
            value = min(1.0, max(0.0, value)) if np.isfinite(value) else default
        if overwrite or widget not in st.session_state:
            st.session_state[widget] = value


def _commit_output_taper() -> None:
    # Commit before the main script builds/saves its settings snapshot.
    settings = dict(st.session_state.get("settings", {}))
    for widget, field, default, cast in OUTPUT_FIR_SETTINGS:
        settings[field] = cast(st.session_state.get(widget, settings.get(field, default)))
    st.session_state["settings"] = settings


def _apply_library_system(system: MultiwaySystem) -> None:
    """Stage persisted settings and restore editable channel state on the next rerun."""
    st.session_state["_pending_uploaded_settings"] = system.settings
    restore_output_fir_settings(system.settings)
    st.session_state["_pending_uploaded_name"] = f"System Library · {system.name}"
    st.session_state["_pending_uploaded_missing_eq"] = []
    st.session_state["composite_studio_current_system_id"] = system.id
    st.session_state["composite_studio_system_name"] = system.name
    generated = [item for item in system.channels if item.band in VALID_GENERATED_BANDS]
    additional = [item for item in system.channels if item.band not in VALID_GENERATED_BANDS]
    generated_groups = {item.group for item in generated if item.band != "SUB"}
    stereo = {"Left", "Right"}.issubset(generated_groups)
    st.session_state["composite_studio_output_layout"] = "Stereo" if stereo else "Mono"
    st.session_state["composite_studio_shared_sub"] = bool(
        stereo and any(item.band == "SUB" and item.group == "Sub" for item in generated)
    )
    for key in tuple(st.session_state):
        if str(key).startswith("composite_alignment_external_distance_"):
            st.session_state.pop(key, None)
    distance_config = system.settings.get("external_distance_timing", {})
    if not isinstance(distance_config, dict):
        distance_config = {}
    st.session_state["composite_alignment_system_timing_source"] = (
        "外部測定距離" if bool(distance_config.get("enabled", False)) else "自動選択"
    )
    st.session_state.pop("composite_alignment_external_temperature_c", None)
    for item in generated:
        prefix = _channel_state_prefix(item.band, item.group, stereo=stereo)
        st.session_state[f"{prefix}_enabled"] = bool(item.enabled)
        st.session_state[f"{prefix}_name"] = item.name
        st.session_state[f"{prefix}_group"] = item.group
        st.session_state[f"{prefix}_gain"] = float(item.gain_db)
        st.session_state[f"{prefix}_dc_gain_normalize"] = bool(item.dc_gain_normalize)
        st.session_state[f"{prefix}_polarity"] = "Normal (+)" if item.polarity == 1 else "Invert (-)"
        st.session_state[f"{prefix}_delay"] = float(item.delay_samples)
        st.session_state[f"{prefix}_auto_alignment_delay"] = float(
            item.auto_alignment_delay_samples
        )
        st.session_state[f"{prefix}_auto_alignment_allpass"] = [
            dict(section) for section in item.auto_alignment_allpass
        ]
        st.session_state[f"{prefix}_channel_id"] = item.channel_id
        st.session_state[f"{prefix}_speaker_package_id"] = item.speaker_package_id
        st.session_state[f"{prefix}_latest_assignment_id"] = item.latest_assignment_id
        st.session_state[f"composite_studio_channel_target_{item.channel_id}"] = item.target_preset_id
    for key in tuple(st.session_state):
        if str(key).startswith("composite_studio_extra_"):
            st.session_state.pop(key, None)
    st.session_state["composite_studio_extra_count"] = len(additional)
    for index, item in enumerate(additional):
        prefix = f"composite_studio_extra_{index}"
        st.session_state[f"{prefix}_name"] = item.name
        st.session_state[f"{prefix}_way"] = item.way
        st.session_state[f"{prefix}_group"] = item.group
        st.session_state[f"{prefix}_gain"] = float(item.gain_db)
        st.session_state[f"{prefix}_dc_gain_normalize"] = bool(item.dc_gain_normalize)
        st.session_state[f"{prefix}_polarity"] = "Normal (+)" if item.polarity == 1 else "Invert (-)"
        st.session_state[f"{prefix}_delay"] = float(item.delay_samples)
        st.session_state[f"{prefix}_channel_id"] = item.channel_id
        st.session_state[f"{prefix}_speaker_package_id"] = item.speaker_package_id
        st.session_state[f"{prefix}_latest_assignment_id"] = item.latest_assignment_id
        st.session_state[f"composite_studio_channel_target_{item.channel_id}"] = item.target_preset_id
    st.session_state["composite_studio_result_group"] = normalize_display_value(
        system.settings.get("selected_display", system.selected_group), stereo=stereo,
    )


MODE_WAYS = {
    "Fullrange": ("Fullrange",),
    "Fullrange+SUB": ("SUB", "Fullrange"),
    "2Way": ("Low", "High"),
    "2Way+SUB": ("SUB", "Low", "High"),
    "3Way": ("Low", "Mid", "High"),
    "3Way+SUB": ("SUB", "Low", "Mid", "High"),
    "4Way": ("Low", "Low-Mid", "High-Mid", "High"),
    "4Way+SUB": ("SUB", "Low", "Low-Mid", "High-Mid", "High"),
}
VALID_GENERATED_BANDS = frozenset(
    band for ways in MODE_WAYS.values() for band in ways
)


def _library_channel_id(row: dict[str, object], current: MultiwaySystem | None) -> str:
    restored_id = str(row.get("channel_id", "")).strip()
    if restored_id:
        return restored_id
    if current is not None:
        band = str(row.get("band", ""))
        identity = (str(row.get("name", "")), str(row.get("way", "")), str(row.get("group", "")))
        for item in current.channels:
            if (item.name, item.way, item.group) == identity:
                return item.channel_id
        same_band = [item for item in current.channels if item.band == band]
        if len(same_band) == 1:
            return same_band[0].channel_id
    return phaseeq_channel_id(
        str(row.get("name", "")), str(row.get("way", "")), str(row.get("group", "")),
    )


def _render_system_library_header() -> None:
    st.markdown(ui_message('ui.2256b5f3fcd9f5'))
    st.caption(ui_message('ui.30ec164ed241e4'))
    records = list_multiway_systems(_multiway_system_db_path(), include_archived=True)
    if not records:
        st.info(ui_message('ui.ca10f96f5be670'))
        return
    record_by_id = {record.system.id: record for record in records}
    options = list(record_by_id)
    selected_key = "composite_studio_library_selection"
    current_id = str(st.session_state.get("composite_studio_current_system_id", ""))
    if st.session_state.get(selected_key) not in options:
        st.session_state[selected_key] = current_id if current_id in options else options[0]
    selected_id = shared_selectbox(
        ui_message('ui.15cdae0ccf990b'), options, key=selected_key,
        format_func=lambda value: (
            f"{record_by_id[value].management_no} · {record_by_id[value].system.name} · "
            f"{record_by_id[value].system.mode} · "
            f"{record_by_id[value].system.sample_rate_hz:,} Hz · rev {record_by_id[value].system.revision}"
            + (" · Archived" if record_by_id[value].archived else "")
        ),
    )
    selected_record = record_by_id[str(selected_id)]
    selected = selected_record.system
    open_col, duplicate_col, archive_col = st.columns(3)
    if open_col.button(
        ui_message('ui.3aa692fcaae417'), key="composite_studio_library_open", width="stretch",
        disabled=selected_record.archived,
    ):
        _apply_library_system(selected)
        st.rerun()
    if duplicate_col.button(ui_message('ui.bb6dad578c8142'), key="composite_studio_library_duplicate", width="stretch"):
        copied = duplicate_multiway_system(
            _multiway_system_db_path(), selected.id, name=f"{selected.name} copy",
        )
        _apply_library_system(copied.system)
        st.rerun()
    archive_label = "復元" if selected_record.archived else "保管"
    if archive_col.button(
        archive_label, key="composite_studio_library_archive", width="stretch",
    ):
        set_multiway_system_archived(
            _multiway_system_db_path(), selected.id, archived=not selected_record.archived,
        )
        if not selected_record.archived and current_id == selected.id:
            st.session_state["composite_studio_current_system_id"] = ""
        st.rerun()
    with st.expander(ui_message('ui.901c58fef536dc'), expanded=False):
        identity_name_key = f"composite_studio_identity_name_{selected.id}"
        identity_no_key = f"composite_studio_identity_no_{selected.id}"
        st.session_state.setdefault(identity_name_key, selected.name)
        st.session_state.setdefault(identity_no_key, selected_record.management_no)
        identity_name = st.text_input(ui_message('ui.69186998727cbc'), key=identity_name_key)
        identity_no = st.text_input(ui_message('ui.4dffb0cff59a2f'), key=identity_no_key)
        if st.button(
            ui_message('ui.9ba41992d5c9bd'), key=f"composite_studio_identity_save_{selected.id}",
            icon=":material/badge:",
        ):
            try:
                renamed = rename_multiway_system(
                    _multiway_system_db_path(), selected.id,
                    name=identity_name, management_no=identity_no,
                )
            except ValueError as exc:
                st.error(ui_message('ui.2351202cd1dc4f', p0=f'{exc}'))
            else:
                st.success(ui_message('ui.028204cd2db9b7', p0=f'{renamed.management_no}'))
                st.rerun()
        revisions = list_multiway_system_revisions(
            _multiway_system_db_path(), selected.id,
        )
        revision_numbers = [record.revision_number for record in revisions]
        restore_revision = shared_selectbox(
            ui_message('ui.84b18256a3c4a5'), revision_numbers,
            key=f"composite_studio_restore_revision_{selected.id}",
            format_func=lambda value: f"rev {value}",
        ) if revision_numbers else None
        if st.button(
            ui_message('ui.b5a5664049a6b9'),
            key=f"composite_studio_restore_apply_{selected.id}",
            disabled=restore_revision is None or selected_record.archived,
            icon=":material/history:",
        ):
            assert restore_revision is not None
            try:
                restored = restore_multiway_system_revision(
                    _multiway_system_db_path(), selected.id, int(restore_revision),
                    expected_revision=selected.revision,
                )
            except (ValueError, RuntimeError) as exc:
                st.error(ui_message('ui.96148551108e15', p0=f'{exc}'))
            else:
                _apply_library_system(restored.system)
                st.success(
                    ui_message('ui.cfe7db341afc98', p0=f'{restore_revision}', p1=f'{restored.system.revision}')
                )
                st.rerun()


def _design_history_path():
    return _multiway_system_db_path().with_name("design_history.sqlite3")


def _studio_history_payload(system, rows):
    from utils.multiway_system_db import _system_snapshot
    from utils.export_bundle import config_payload_from_project_zip
    from utils.design_history import compact_settings, HistoryPending
    from utils.composite_exchange import read_composite_assignment, read_multiway_workspace
    from utils.multiway_measurements import read_measurement_inputs
    payload = _system_snapshot(system)
    workspace = read_multiway_workspace(_exchange_root())
    measurements = read_measurement_inputs(_exchange_root(), workspace) if workspace else {}
    payload["speaker_inputs"] = {channel: {
        "speaker_source_name": record.get("name", ""),
        "speaker_response_raw": record.get("response"),
    } for channel, record in measurements.items()}
    files_before = {}
    payload["phaseeq"] = {}
    payload["targets"] = {}
    for row in rows:
        channel = str(row["channel_id"])
        assignment = str(row.get("latest_assignment_id", ""))
        workspace = row.get("phaseeq_working_session")
        if assignment:
            if not workspace or not Path(workspace).is_file():
                raise HistoryPending("PhaseEQの設定同期を待っています。")
            if str(row.get("phaseeq_assignment_id", "")) != assignment:
                raise HistoryPending("PhaseEQの設定同期を待っています。")
            live_assignment = read_composite_assignment(_exchange_root(), assignment, require_current=True)
            status = read_phaseeq_assignment_status(_exchange_root(), assignment_id=assignment)
            if live_assignment is None or status is None or status.state != "Ready":
                raise HistoryPending("PhaseEQの設定同期を待っています。")
            expected_recipe = band_split_from_studio(system.mode, system.sample_rate_hz,
                system.settings[system.mode], str(row["way"]), fir_enabled=live_assignment.fir_enabled, phase_alignment=alignment_target.recipe_definition(row))
            if live_assignment.band_split_recipe != expected_recipe:
                raise HistoryPending("帯域分割の設定同期を待っています。")
            if live_assignment.target_definition != _selected_group_target_definition(str(row["group"]), system.sample_rate_hz):
                raise HistoryPending("Targetの設定同期を待っています。")
            files_before[str(workspace)] = Path(workspace).stat().st_mtime_ns
            payload["phaseeq"][channel] = compact_settings(
                config_payload_from_project_zip(io.BytesIO(Path(workspace).read_bytes())))
        group = str(row["group"])
        payload["targets"][group] = _selected_group_target_definition(group, system.sample_rate_hz)
    if any(Path(name).stat().st_mtime_ns != stamp for name, stamp in files_before.items()):
        raise HistoryPending("PhaseEQの設定同期を待っています。")
    for row in rows:
        assignment = str(row.get("latest_assignment_id", ""))
        if assignment and read_composite_assignment(_exchange_root(), assignment, require_current=True) is None:
            raise HistoryPending("PhaseEQの設定同期を待っています。")
    current_workspace = read_multiway_workspace(_exchange_root())
    if current_workspace and read_measurement_inputs(_exchange_root(), current_workspace) != measurements:
        raise HistoryPending("測定入力の設定同期を待っています。")
    return payload


def _restore_studio_history(payload, identifier):
    from utils.design_history import validate_settings_restore
    # Validate every channel before staging any change. The existing Assignment
    # publisher applies the restored EQ through the normal pipeline on next send.
    for config_payload in payload.get("phaseeq", {}).values():
        validate_settings_restore(config_payload)
    current = _current_library_system()
    if current is None:
        raise ValueError("プロジェクトを選択してください。")
    existing = {item.channel_id: item for item in current.channels}
    channels = []
    for row in payload["channels"]:
        values = dict(row)
        previous = existing.get(str(values["channel_id"]))
        values["latest_assignment_id"] = previous.latest_assignment_id if previous else ""
        values["speaker_package_id"] = previous.speaker_package_id if previous else ""
        channels.append(MultiwayChannelReference(**values))
    restored_settings = dict(payload["settings"])
    restored_settings["history_target_definitions"] = payload.get("targets", {})
    restored = replace(current, mode=payload["mode"], sample_rate_hz=int(payload["sample_rate_hz"]),
                       settings=restored_settings, channels=tuple(channels))
    restored.validate()
    # Staging is reversible; no generated coefficients are taken from history.
    restored_phaseeq = dict(payload.get("phaseeq", {}))
    current_candidate = st.session_state.get("_history_studio_candidate", current)
    current_payload = _studio_history_payload(current_candidate, st.session_state.get("composite_studio_channels", []))
    from utils.design_history import unedited_channel_settings
    for channel in channels:
        if channel.channel_id not in restored_phaseeq and channel.channel_id in current_payload["phaseeq"]:
            restored_phaseeq[channel.channel_id] = unedited_channel_settings(current_payload["phaseeq"][channel.channel_id])
    st.session_state["_history_pending_phaseeq"] = restored_phaseeq
    st.session_state["_history_restored_from"] = identifier
    st.session_state["_history_apply_system"] = restored
    st.rerun()


@st.fragment(run_every=30)
def _studio_history_tick(system, rows, generation):
    from utils.design_history import consider, HistoryPending
    try:
        consider(_design_history_path(), "multiway", system.id, generation,
                 lambda: _studio_history_payload(system, rows),
                 settled=not bool(st.session_state.get("_history_pending_phaseeq")),
                 busy=any(session.active_page == "Measure" for session in list_active_phaseeq_sessions(_exchange_root())),
                 app_version=_history_app_version())
        st.session_state.pop("_studio_history_error", None)
        st.session_state.pop("_studio_history_error_count", None)
    except HistoryPending:
        st.session_state.pop("_studio_history_error", None)
    except (OSError, ValueError) as exc:
        count = int(st.session_state.get("_studio_history_error_count", 0)) + 1
        st.session_state["_studio_history_error_count"] = count
        if count >= 3:
            st.session_state["_studio_history_error"] = str(exc)


def _clone_studio_history(payload, name):
    from copy import deepcopy
    copied = deepcopy(payload)
    channels = []
    remapped = {}
    channel_ids = {}
    for item in copied["channels"]:
        old_id = item["channel_id"]
        item["channel_id"] = str(uuid.uuid4())
        item["latest_assignment_id"] = ""
        channel_ids[old_id] = item["channel_id"]
        channels.append(MultiwayChannelReference(**item))
        if old_id in copied.get("phaseeq", {}):
            remapped[item["channel_id"]] = copied["phaseeq"][old_id]
    from utils.design_history import validate_settings_restore
    for item in remapped.values():
        validate_settings_restore(item)
    settings = dict(copied["settings"])
    settings["history_target_definitions"] = copied.get("targets", {})
    created = save_multiway_system(_multiway_system_db_path(), new_multiway_system(
        name, mode=copied["mode"], sample_rate_hz=copied["sample_rate_hz"],
        settings=settings, channels=tuple(channels)))
    from utils.composite_exchange import read_multiway_workspace, _atomic_json
    from utils.multiway_measurements import read_measurement_inputs, mapping_path, workspace_key
    source_workspace = read_multiway_workspace(_exchange_root())
    if source_workspace:
        source_inputs = read_measurement_inputs(_exchange_root(), source_workspace)
        target_workspace = {"system_id": created.system.id, "sample_rate_hz": created.system.sample_rate_hz,
                            "channel_ids": [item.channel_id for item in channels if item.enabled]}
        copied_inputs = {channel_ids[key]: {**value, "request_id": uuid.uuid4().hex, "project_name": name}
                         for key, value in source_inputs.items() if key in channel_ids}
        _atomic_json(mapping_path(_exchange_root(), target_workspace),
                     {"workspace_key": workspace_key(target_workspace), "channels": copied_inputs})
    st.session_state["_history_pending_phaseeq"] = remapped
    st.session_state["_history_apply_system"] = created.system
    st.rerun()


def _history_app_version():
    return (Path(__file__).resolve().parents[3] / "VERSION").read_text().strip()


def _render_studio_history(rows):
    from utils.design_history import digest, save
    from utils.design_history_ui import render_history
    awaiting = st.session_state.get("_history_awaiting_return", {})
    for row in rows:
        channel = str(row.get("channel_id", ""))
        assignment = awaiting.get(channel)
        if assignment and str(row.get("phaseeq_assignment_id", "")) == assignment:
            status = read_phaseeq_assignment_status(_exchange_root(), assignment_id=assignment)
            if status is not None and status.state == "Ready":
                awaiting.pop(channel, None)
    system = st.session_state.get("_history_studio_candidate")
    if system is None:
        return
    # File metadata is only an event signal; contents are compared once due.
    generation = digest([system.settings, [
        [str(row.get("channel_id")), str(row.get("latest_assignment_id")),
         Path(row["phaseeq_working_session"]).stat().st_mtime_ns
         if row.get("phaseeq_working_session") and Path(row["phaseeq_working_session"]).is_file() else 0]
        for row in rows], [dict(vars(item)) for item in system.channels]])
    _studio_history_tick(system, rows, generation)
    ready = not any(session.active_page == "Measure" for session in list_active_phaseeq_sessions(_exchange_root())) and not st.session_state.get("_history_pending_phaseeq") and all(
        not row.get("latest_assignment_id") or
        (row.get("phaseeq_working_session") and str(row.get("phaseeq_assignment_id", "")) == str(row.get("latest_assignment_id")))
        for row in rows)
    factory = lambda: _studio_history_payload(system, rows)
    if st.session_state.get("_studio_history_error"):
        with st.expander(display_text("履歴の詳細"), expanded=False):
            st.caption(st.session_state["_studio_history_error"])
    render_history(_design_history_path(), "multiway", system.id, factory,
                   _restore_studio_history, app_version=_history_app_version(), ready=ready, clone=_clone_studio_history)
    try:
        if ready and st.session_state.get("_history_manual_pending"):
            save(_design_history_path(), "multiway", system.id, factory(), kind="manual",
                 app_version=_history_app_version())
            st.session_state.pop("_history_manual_pending", None)
        if ready and st.session_state.get("_history_restored_from"):
            save(_design_history_path(), "multiway", system.id, factory(), kind="restore",
                 restored_from=st.session_state["_history_restored_from"], app_version=_history_app_version())
            st.session_state.pop("_history_restored_from", None)
    except (ValueError, OSError) as exc:
        from utils.design_history import HistoryPending
        if not isinstance(exc, HistoryPending):
            st.session_state["_studio_history_error"] = str(exc)


def _render_system_library_save(
    channel_rows: list[dict[str, object]], *, mode_key: str, sample_rate_hz: int,
    current_settings: dict[str, object],
) -> MultiwaySystem | None:
    current = _current_library_system()
    default_name = current.name if current is not None else f"{mode_key} System"
    st.session_state.setdefault("composite_studio_system_name", default_name)
    st.caption(
        ui_message('ui.980090d64fdf82')
    )
    name = st.text_input(ui_message('ui.03857f0c1902b8'), key="composite_studio_system_name")
    references = tuple(
        MultiwayChannelReference(
            channel_id=_library_channel_id(row, current),
            name=str(row.get("name", "")).strip(),
            way=str(row.get("way", "")).strip(),
            group=str(row.get("group", "")).strip(),
            band=str(row.get("band", "")).strip(),
            enabled=bool(row.get("enabled", True)),
            gain_db=float(row.get("gain_db", 0.0)),
            dc_gain_normalize=bool(row.get("dc_gain_normalize", False)),
            polarity=int(row.get("polarity", 1)),
            delay_samples=float(row.get("delay_samples", 0.0)),
            auto_alignment_delay_samples=float(
                row.get("auto_alignment_delay_samples", 0.0)
            ),
            auto_alignment_allpass=tuple(
                section.to_dict() if isinstance(section, AllPassSection) else dict(section)
                for section in row.get("auto_alignment_allpass", ())
                if isinstance(section, (AllPassSection, dict))
            ),
            speaker_package_id=str(row.get("speaker_package_id", "")).strip(),
            latest_assignment_id=str(row.get("latest_assignment_id", "")).strip(),
            # Retain the schema field for backward-compatible reads, but new
            # systems no longer assign a Target per Channel.
            target_preset_id="",
            sort_order=index,
        )
        for index, row in enumerate(channel_rows)
        if str(row.get("name", "")).strip() and str(row.get("group", "")).strip()
    )
    if current is not None:
        st.session_state["_history_studio_candidate"] = replace(
            current, settings=dict(current_settings), channels=references,
            sample_rate_hz=sample_rate_hz, mode=mode_key)
    else:
        st.session_state.pop("_history_studio_candidate", None)
    save_new_col, update_col = st.columns(2)
    if save_new_col.button(
        ui_message('ui.3c6657e9b91f9f'), key="composite_studio_system_save_new",
        width="stretch", icon=":material/save_as:",
    ):
        try:
            new_references = tuple(
                replace(
                    reference, channel_id=str(uuid.uuid4()), latest_assignment_id="",
                )
                for reference in references
            )
            selected_display = normalize_display_value(
                st.session_state.get("composite_studio_result_group"),
                stereo=str(current_settings.get("output_layout", "Mono")) == "Stereo",
            )
            saved_settings = {**dict(current_settings), "selected_display": selected_display}
            created = save_multiway_system(_multiway_system_db_path(), new_multiway_system(
                name, mode=mode_key, sample_rate_hz=sample_rate_hz,
                settings=saved_settings, channels=new_references,
                selected_group=_physical_group_for_display(
                    selected_display, stereo=selected_display != DISPLAY_MAIN,
                ),
            ))
        except (ValueError, RuntimeError) as exc:
            st.error(ui_message('ui.e169772b4f88b8', p0=f'{exc}'))
        else:
            # The name widget already exists in this callback. Select the new
            # record without mutating that widget's state during the same run.
            st.session_state["_history_manual_pending"] = True
            st.session_state["composite_studio_current_system_id"] = created.system.id
            st.success(ui_message('ui.e2879846f43161', p0=f'{created.system.name}'))
            st.rerun()
    if update_col.button(
        ui_message('ui.4966bca7b8d93e'), key="composite_studio_system_save_current",
        width="stretch", disabled=current is None, icon=":material/save:",
    ):
        assert current is not None
        try:
            selected_display = normalize_display_value(
                st.session_state.get("composite_studio_result_group"),
                stereo=str(current_settings.get("output_layout", "Mono")) == "Stereo",
            )
            saved_settings = {**dict(current_settings), "selected_display": selected_display}
            saved = save_multiway_system_revision(
                _multiway_system_db_path(),
                MultiwaySystem(
                    id=current.id, name=str(name).strip(), mode=mode_key,
                    sample_rate_hz=sample_rate_hz,
                    settings=saved_settings, channels=references,
                    selected_group=_physical_group_for_display(
                        selected_display, stereo=selected_display != DISPLAY_MAIN,
                    ),
                    note=current.note, revision=current.revision,
                ), expected_revision=current.revision,
            )
        except (ValueError, RuntimeError) as exc:
            st.error(ui_message('ui.e169772b4f88b8', p0=f'{exc}'))
        else:
            st.session_state["_history_manual_pending"] = True
            st.success(ui_message('ui.d67d0e5777ebc0', p0=f'{saved.system.name}', p1=f'{saved.system.revision}'))
            st.rerun()
    if current is not None:
        current_record = _current_library_record()
        management_no = current_record.management_no if current_record is not None else ""
        st.caption(
            ui_message('ui.eb116877955b63', p0=f'{management_no}', p1=f'{current.name}', p2=f'{current.revision}')
        )
    return current


def _fir_coefficients(filename: str, data: bytes, sample_rate_hz: int) -> np.ndarray:
    suffix = Path(str(filename)).suffix.casefold()
    if suffix == ".wav":
        from scipy.io import wavfile
        source_rate, values = wavfile.read(io.BytesIO(data))
        if int(source_rate) != int(sample_rate_hz):
            raise ValueError(f"FIR sample rate mismatch: {source_rate} != {sample_rate_hz}")
        array = np.asarray(values, dtype=float)
    else:
        delimiter = "," if suffix == ".csv" else None
        array = np.genfromtxt(io.BytesIO(data), delimiter=delimiter)
        if array.ndim == 2:
            array = array[:, -1]
    array = np.asarray(array, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("FIR coefficients must be finite and non-empty")
    return array


def _fir_stage_from_upload(label: str, upload: object, sample_rate_hz: int) -> ResponseStage:
    from scipy.io import wavfile
    values = _fir_coefficients(str(upload.name), upload.getvalue(), sample_rate_hz)
    buffer = io.BytesIO()
    wavfile.write(buffer, int(sample_rate_hz), values.astype(np.float64))
    return ResponseStage(label, f"{label.casefold().replace(' ', '_')}.wav", buffer.getvalue())


def _phaseeq_status_label(row: dict[str, object]) -> str:
    connected_state = str(row.get("phaseeq_status_state", "")).strip()
    if connected_state:
        return connected_state
    channel_id = str(row.get("channel_id") or phaseeq_channel_id(
        str(row.get("name", "")), str(row.get("way", "")), str(row.get("group", "")),
    ))
    assignment_id = latest_phaseeq_assignment_id(_exchange_root(), channel_id=channel_id)
    if not assignment_id:
        return "Not assigned"
    status = read_phaseeq_assignment_status(_exchange_root(), assignment_id=assignment_id)
    return status.state if status is not None else "Assigned"


@st.fragment(run_every=2.0)
def _watch_phaseeq_result_inbox() -> None:
    """Rerun Studio when a connected PhaseEQ channel publishes a new revision."""
    from ui.browser_presence import ensure_browser_presence, refresh_peer_presence
    ensure_browser_presence(_exchange_root(), "multiway")
    refresh_peer_presence(_exchange_root(), "phaseeq")
    configured = st.session_state.get("composite_studio_channels", [])
    current_system = _current_library_system()
    channel_ids = tuple(sorted({
        _library_channel_id(row, current_system)
        for row in configured
        if isinstance(row, dict) and row.get("source") == "generated_band"
    }))
    from utils.multiway_measurements import mapping_path
    from utils.composite_exchange import read_multiway_workspace
    try:
        measurement_workspace = read_multiway_workspace(_exchange_root())
        if measurement_workspace is not None:
            path = mapping_path(_exchange_root(), measurement_workspace)
            signature = path.stat().st_mtime_ns if path.exists() else 0
            previous = st.session_state.get('_measurement_inbox_signature', 0)
            st.session_state['_measurement_inbox_signature'] = signature
            if previous != signature:
                st.rerun()
        st.session_state.pop('_measurement_inbox_error', None)
    except (OSError, ValueError) as exc:
        st.session_state['_measurement_inbox_error'] = str(exc)
        st.warning('連携先の構成を確認できません。前回結果を保持します：' + str(exc))
        return
    if not channel_ids:
        return
    result_signature = assignment_status_signature(
        _exchange_root(), channel_ids=channel_ids, states=("Ready", "Editing"),
    )
    previous_result = st.session_state.get("_phaseeq_result_inbox_signature")
    st.session_state["_phaseeq_result_inbox_signature"] = result_signature
    notify_result, rerun_required = _phaseeq_inbox_transition(
        previous_result, result_signature,
    )
    if notify_result:
        st.session_state["_phaseeq_result_auto_received"] = True
    from utils.composite_exchange import phaseeq_session_status_signature
    connection_signature = phaseeq_session_status_signature(_exchange_root())
    previous_connection = st.session_state.get("_phaseeq_connection_signature")
    st.session_state["_phaseeq_connection_signature"] = connection_signature
    if retry_ready(_exchange_root(), st.session_state, channel_ids) or rerun_required or (previous_connection is not None and previous_connection != connection_signature):
        st.rerun()


def _restore_fir_off_assignment_result(
    row: dict[str, object], current_system: MultiwaySystem | None,
) -> bool:
    """Restore an IIR-only Assignment without inventing an Exchange FIR asset."""
    channel_id = _library_channel_id(row, current_system)
    assignment_id = latest_published_phaseeq_assignment_id(
        _exchange_root(), channel_id=channel_id,
    )
    if not assignment_id:
        return False
    status = read_phaseeq_assignment_status(
        _exchange_root(), assignment_id=assignment_id,
    )
    result = (
        status.result
        if status is not None and status.state in {"Ready", "Editing"}
        and isinstance(status.result, dict) else None
    )
    if result is None or int(result.get("tap_count", -1)) != 0:
        return False
    if not _assignment_result_provenance_valid(
        assignment_id, result, current_system,
    ):
        return False

    from composite_engine.assignment_snapshot import validated_speaker_asset, timing_for_asset
    try:
        validated_speaker = validated_speaker_asset(result, _exchange_root(), status.revision)
        from utils.composite_exchange import assignment_working_session_path
        from utils.exchange_publication import working_session_from_result
        working_path = (working_session_from_result(_exchange_root(), assignment_id, result)
            if result.get('asset_schema') == 1 else assignment_working_session_path(_exchange_root(), assignment_id))
    except (OSError, ValueError) as exc:
        from utils.exchange_retry import record_failure
        record_failure(st.session_state, assignment_id, status.revision, exc)
        st.warning(ui_message('ui.0cb0262cf0de6a', p0=f"{row.get('name', channel_id)}", p1=f'{exc}'))
        return False
    row.pop("phaseeq_fir_response", None)
    row.pop("phaseeq_fir_target_excluded", None)
    row["phaseeq_iir"] = list(result.get("phaseeq_iir", []))
    row["phaseeq_iir_sos"] = tuple(
        tuple(float(value) for value in section)
        for section in result.get("phaseeq_iir_sos", [])
    )
    if validated_speaker is not None:
        row["phaseeq_speaker_response"] = validated_speaker
    alignment_target.receive(row, result)
    row["phaseeq_assignment_id"] = assignment_id
    row["phaseeq_assignment_revision"] = status.revision
    if isinstance(result.get("timing_provenance"), dict):
        row["timing_provenance"] = timing_for_asset(result, validated_speaker)
    else:
        row.pop("timing_provenance", None)
    row["phaseeq_status_state"] = status.state
    row["phaseeq_working_session"] = working_path
    _cache_phaseeq_bundle(row, assignment_id)
    return True


def _assignment_target_reference(
    assignment_id: str,
) -> tuple[dict[str, object] | None, str]:
    normalized = str(assignment_id).strip()
    if not normalized or not all(
        character.isalnum() or character in "_-" for character in normalized
    ):
        return None, ""
    path = _exchange_root() / "assignments" / normalized / "assignment.json"
    if not path.is_file():
        return None, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, ""
    target_payload = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    target = (
        target_payload.get("definition_snapshot")
        if isinstance(target_payload.get("definition_snapshot"), dict)
        else payload.get("target_definition")
    )
    return (
        dict(target) if isinstance(target, dict) else None,
        str(target_payload.get("application", payload.get("target_application", "legacy_apply"))),
    )


def _assignment_result_provenance_valid(
    assignment_id: str, result: dict[str, object], current_system: MultiwaySystem | None,
) -> bool:
    normalized = str(assignment_id).strip()
    path = _exchange_root() / "assignments" / normalized / "assignment.json"
    if not normalized or not path.is_file():
        return False
    try:
        assignment = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if int(assignment.get("format_version", 2)) < 3:
        return True
    system_payload = assignment.get("system") if isinstance(assignment.get("system"), dict) else {}
    target_payload = assignment.get("target") if isinstance(assignment.get("target"), dict) else {}
    # Standalone/legacy-compatible v3 assignments have no System or Target
    # identity to validate. Preserve their established result path.
    if not system_payload and not target_payload:
        return True
    expected = (
        ("base_assignment_signature", assignment.get("assignment_signature", "")),
        ("system_id", system_payload.get("system_id", "")),
        ("system_revision_number", system_payload.get("revision_number")),
        ("system_content_hash", system_payload.get("content_hash", "")),
        ("target_id", target_payload.get("target_id", "")),
        ("target_revision_number", target_payload.get("revision_number")),
        ("target_content_hash", target_payload.get("content_hash", "")),
        ("sample_rate_hz", assignment.get("sample_rate_hz")),
        ("tap_count", assignment.get("tap_count")),
    )
    if any(result.get(key) != value for key, value in expected):
        return False
    if current_system is not None and str(system_payload.get("system_id", "")):
        if current_system.id != str(system_payload.get("system_id", "")):
            return False
    return bool(str(result.get("result_generation_signature", "")).strip())


def _group_target_catalog(target_scope):
    return list(shared_target_preset_catalog(
        Path(__file__).resolve().parents[3], scope="composite_group", summaries=True))


def _commit_group_target_choice(settings_key, widget_key, target_scope):
    selected = st.session_state.get(widget_key)
    if selected == st.session_state.get(settings_key):
        return
    st.session_state[settings_key] = selected
    settings = dict(st.session_state.get("settings", {}))
    targets = dict(settings.get("group_target_presets", {}))
    targets[target_scope] = selected
    settings["group_target_presets"] = targets
    settings.pop("history_target_definitions", None)
    st.session_state["settings"] = settings
    # An explicit new source supersedes the previous edit return, while its
    # saved session remains available. Do not let the URL reattach it next run.
    session_id = st.session_state.pop(f"_composite_group_target_edit_session_{target_scope}", "")
    if session_id:
        ignored = set(st.session_state.get("_composite_ignored_target_edit_sessions", ()))
        ignored.add(session_id)
        st.session_state["_composite_ignored_target_edit_sessions"] = tuple(ignored)


def _group_target_choice(label, options, settings_key, target_scope, *, kind="selectbox", **kwargs):
    options = list(options)
    current = st.session_state.get(settings_key)
    if current not in options:
        current = options[0]
    st.session_state[settings_key] = current
    widget_key = f"{settings_key}_widget"
    st.session_state[widget_key] = current
    renderer = getattr(st, kind)
    # The restored widget state is the sole source of the selected value.
    result = renderer(display_text(label), options, key=widget_key,
        on_change=_commit_group_target_choice, args=(settings_key, widget_key, target_scope), **kwargs)
    return result


def _selected_group_target_definition(
    group: str, sample_rate_hz: int,
) -> dict[str, object] | None:
    """Resolve the Group Target shared by graphs and a PhaseEQ Assignment."""
    normalized_group = str(group).strip()
    frozen = st.session_state.get("settings", {}).get("history_target_definitions", {})
    if normalized_group in frozen:
        return frozen[normalized_group]
    target_scope = _group_target_scope(normalized_group)
    _migrate_legacy_stereo_group_target_state()
    source = st.session_state.get(f"composite_studio_target_source_{target_scope}", "PhaseEQ preset")
    if source not in {"PhaseEQ preset", "Global Target", "Built-in"}:
        return None
    catalog = _group_target_catalog(target_scope)
    selected_id = str(st.session_state.get(
        f"composite_studio_group_target_preset_{target_scope}",
        st.session_state.get("settings", {}).get("group_target_presets", {}).get(
            target_scope, str(catalog[0]["id"]) if catalog else "")))
    selected = next((item for item in catalog if str(item["id"]) == selected_id), None)
    if selected is not None and selected.get("_summary"):
        selected = materialize_target_preset(Path(__file__).resolve().parents[3], selected)
    return canonical_target_definition(selected) if selected is not None else None


def _fir_coefficients_are_identity(
    filename: str, data: bytes, sample_rate_hz: int,
) -> bool:
    values = _fir_coefficients(filename, data, sample_rate_hz)
    if values.size % 2 == 0:
        return False
    center = values.size // 2
    expected = np.zeros(values.shape, dtype=float)
    expected[center] = 1.0
    return bool(np.allclose(values, expected, rtol=0.0, atol=1e-9))


def _phaseeq_speaker_cache_key(row: dict[str, object]) -> str:
    return str(row.get("channel_id") or phaseeq_channel_id(
        str(row.get("name", "")), str(row.get("way", "")),
        str(row.get("group", "")),
    ))


def _enabled_phaseeq_iir_definitions(row: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Return only IIR definitions that should produce realized SOS sections."""
    definitions = row.get("phaseeq_iir", ())
    if not isinstance(definitions, (list, tuple)):
        return ()
    return tuple(
        item for item in definitions
        if isinstance(item, dict) and bool(item.get("enabled", True))
    )


def _restore_cached_phaseeq_bundle(row: dict[str, object]) -> None:
    cache = st.session_state.get("_composite_phaseeq_bundle_cache", {})
    if not isinstance(cache, dict):
        return
    cached = cache.get(_phaseeq_speaker_cache_key(row))
    if not isinstance(cached, dict):
        return
    expected_assignment = str(row.get("latest_assignment_id", "")).strip()
    cached_assignment = str(cached.get("assignment_id", "")).strip()
    if expected_assignment and cached_assignment != expected_assignment:
        return
    for key in (
        "phaseeq_speaker_response", "phaseeq_fir_response", "phaseeq_iir",
        "phaseeq_iir_sos", "phaseeq_assignment_id", "phaseeq_assignment_revision",
        "phaseeq_alignment_definition", "phaseeq_alignment_target_applied", "phaseeq_alignment_recipe",
        "phaseeq_status_state", "phaseeq_working_session", "timing_provenance",
    ):
        if key in cached:
            row[key] = cached[key]


def _cache_phaseeq_bundle(row: dict[str, object], assignment_id: str) -> None:
    if not assignment_id:
        return
    from utils.exchange_retry import record_success
    record_success(st.session_state, assignment_id)
    cache = st.session_state.setdefault("_composite_phaseeq_bundle_cache", {})
    cache[_phaseeq_speaker_cache_key(row)] = {
        "assignment_id": str(assignment_id),
        **{
            key: row[key] for key in (
                "phaseeq_speaker_response", "phaseeq_fir_response", "phaseeq_iir",
                "phaseeq_iir_sos", "phaseeq_assignment_id", "phaseeq_assignment_revision",
                "phaseeq_alignment_definition", "phaseeq_alignment_target_applied", "phaseeq_alignment_recipe",
                "phaseeq_status_state", "phaseeq_working_session", "timing_provenance",
            ) if key in row
        },
    }


def _restore_cached_phaseeq_speaker(row: dict[str, object]) -> None:
    """Compatibility wrapper retained for focused state tests."""
    _restore_cached_phaseeq_bundle(row)


def _cache_phaseeq_speaker(row: dict[str, object], assignment_id: str) -> None:
    """Compatibility wrapper retained for focused state tests."""
    _cache_phaseeq_bundle(row, assignment_id)


def _phaseeq_assignment_rows(channel_rows):
    # Imported/returned audio is an independent input, not a channel owned by
    # the current crossover recipe. Keep it in the mix without reassigning it.
    return {
        str(row["channel_id"]): row for row in channel_rows
        if row.get("enabled", True) and row.get("source") == "generated_band"
    }


def _phaseeq_return_summary(row):
    """Describe the accepted processing stages, independently of screen presence."""
    returned = bool(row.get("phaseeq_assignment_id"))
    fir = row.get("phaseeq_fir_response")
    if alignment_target.pending(row):
        fir_state = display_text("再生成待ち")
    elif row.get("phaseeq_fir_target_excluded") is not None:
        fir_state = display_text("Target未検証・除外")
    elif isinstance(fir, dict):
        fir_state = f"{int(fir.get('tap_count', 0))} taps"
    else:
        fir_state = display_text("なし／バイパス") if returned else display_text("未返却")
    sos = row.get("phaseeq_iir_sos", ())
    return {
        display_text("FIR反映"): fir_state,
        display_text("IIR反映"): f"{len(sos)} SOS" if sos else display_text("なし") if returned else display_text("未返却"),
        display_text("測定反映"): display_text("反映済み") if isinstance(row.get("phaseeq_speaker_response"), dict)
                    else display_text("なし") if returned else display_text("未返却"),
    }


def render_phaseeq_send_controls(assignment_rows, assignment_controls_host=None):
    """One sidebar action publishes every enabled channel; no destination picker."""
    return st.button(
        ui_message('ui.782b8178306c4c'), type="primary", width="stretch",
        icon=":material/send:", key="composite_studio_send_assignment",
        disabled=not assignment_rows,
        help=ui_message('ui.5658f5c2200b11'),
    )


def render_composite_sidebar(
    bands: list[str],
    sample_rate_hz: int,
    *,
    mode_key: str = "",
    crossover_frequencies_hz: tuple[float, ...] = (),
    band_tap_lengths: dict[str, int] | None = None,
    band_linear_fir_filters: dict[str, tuple[dict[str, object], ...]] | None = None,
    current_settings: dict[str, object] | None = None,
    resume_upload: object | None = None,
    resume_feedback: object | None = None,
    assignment_controls_host: object | None = None,
) -> None:
    """Append PhaseEQ-owned transport controls without replacing Studio controls."""
    with st.sidebar:
        st.markdown(ui_message('ui.c5910f25faa7ae'))
        st.caption(
            ui_message('ui.fdc4b992027ba1')
        )
        current_record = _current_library_record()
        with st.container(border=True):
            st.caption(ui_message('ui.3bb10874e336dd'))
            if current_record is None or current_record.archived:
                st.markdown(ui_message('ui.6e1d1bde8446a2'))
                st.caption(ui_message('ui.fe04be61485884', p0=f"{mode_key or 'Multiway'}", p1=f'{int(sample_rate_hz):,}'))
            else:
                st.markdown(
                    ui_message('ui.3b3723cc974cfa', p0=f'{current_record.management_no}', p1=f'{current_record.system.name}')
                )
                saved_stereo = str(
                    current_record.system.settings.get("output_layout", "Mono")
                ) == "Stereo"
                saved_display = normalize_display_value(
                    current_record.system.settings.get(
                        "selected_display", current_record.system.selected_group,
                    ),
                    stereo=saved_stereo,
                )
                st.caption(
                    ui_message('ui.47d23e5922aa75', p0=f'{current_record.system.revision}', p1=f'{current_record.system.mode}', p2=f'{saved_display}')
                )
        if resume_upload is not None:
            resume_digest = str(hash(resume_upload.getvalue()))
            if st.session_state.get("_composite_resume_digest") != resume_digest:
                try:
                    from composite_engine.export import load_dsp_resume_zip
                    _resume_manifest, resume_workspace = load_dsp_resume_zip(resume_upload.getvalue())
                except (ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
                    feedback = resume_feedback if resume_feedback is not None else st
                    feedback.error(ui_message('ui.46f99654e2a9f2', p0=f'{exc}'))
                else:
                    st.session_state["composite_studio_current_system_id"] = ""
                    imported_settings = resume_workspace.get("multiway_settings")
                    restore_output_fir_settings(imported_settings if isinstance(imported_settings, dict) else {})
                    if isinstance(imported_settings, dict):
                        st.session_state["_pending_uploaded_settings"] = imported_settings
                        st.session_state["_pending_uploaded_name"] = str(resume_upload.name)
                        st.session_state["_pending_uploaded_missing_eq"] = []
                        st.session_state["_pending_uploaded_kind"] = "dsp_package"
                    saved_channels = resume_workspace.get("channels", [])
                    if not isinstance(saved_channels, list):
                        saved_channels = []
                    else:
                        for key in tuple(st.session_state):
                            if str(key).startswith("composite_studio_extra_"):
                                st.session_state.pop(key, None)
                        st.session_state["composite_studio_extra_count"] = sum(
                            1 for item in saved_channels
                            if isinstance(item, dict)
                            and str(item.get("band", "")) not in VALID_GENERATED_BANDS
                        )
                    saved_groups = {
                        str(item.get("group", "")) for item in saved_channels
                        if isinstance(item, dict)
                        and str(item.get("band", "")) in VALID_GENERATED_BANDS
                        and str(item.get("band", "")) != "SUB"
                    }
                    saved_stereo = {"Left", "Right"}.issubset(saved_groups)
                    st.session_state["composite_studio_output_layout"] = (
                        "Stereo" if saved_stereo else "Mono"
                    )
                    st.session_state["composite_studio_shared_sub"] = bool(
                        saved_stereo and any(
                            isinstance(item, dict)
                            and str(item.get("band", "")) == "SUB"
                            and str(item.get("group", "")) == "Sub"
                            for item in saved_channels
                        )
                    )
                    additional_index = 0
                    for saved in saved_channels:
                        if not isinstance(saved, dict):
                            continue
                        if str(saved.get("band", "")) in VALID_GENERATED_BANDS:
                            prefix = _channel_state_prefix(
                                str(saved.get("band", "")), str(saved.get("group", "")),
                                stereo=saved_stereo,
                            )
                        else:
                            prefix = f"composite_studio_extra_{additional_index}"
                            additional_index += 1
                        for saved_key, state_suffix in (
                            ("name", "name"), ("way", "way"), ("group", "group"),
                            ("enabled", "enabled"), ("gain_db", "gain"),
                            ("delay_samples", "delay"), ("iir_xover_enabled", "iir_xover_enabled"),
                            ("auto_alignment_delay_samples", "auto_alignment_delay"),
                            ("auto_alignment_allpass", "auto_alignment_allpass"),
                            ("iir_xover_family", "iir_xover_family"), ("iir_xover_order", "iir_xover_order"),
                            ("iir_xover_hp", "iir_xover_hp"), ("iir_xover_lp", "iir_xover_lp"),
                            ("channel_id", "channel_id"),
                            ("speaker_package_id", "speaker_package_id"),
                            ("latest_assignment_id", "latest_assignment_id"),
                        ):
                            if saved_key in saved:
                                st.session_state[f"{prefix}_{state_suffix}"] = saved[saved_key]
                        st.session_state[f"{prefix}_dc_gain_normalize"] = bool(saved.get("dc_gain_normalize", False))
                        if "polarity" in saved:
                            st.session_state[f"{prefix}_polarity"] = (
                                "Normal (+)" if int(saved["polarity"]) == 1 else "Invert (-)"
                            )
                        if saved.get("target_preset_id") and saved.get("channel_id"):
                            st.session_state[
                                f"composite_studio_channel_target_{saved['channel_id']}"
                            ] = str(saved["target_preset_id"])
                        if isinstance(saved.get("target_preset_definition"), dict) and saved.get("channel_id"):
                            st.session_state[
                                f"_composite_resumed_channel_target_{saved['channel_id']}"
                            ] = canonical_target_definition(saved["target_preset_definition"])
                    resume_settings = resume_workspace.get("multiway_settings", {})
                    resume_stereo = bool(
                        isinstance(resume_settings, dict)
                        and str(resume_settings.get("output_layout", "Mono")) == "Stereo"
                    )
                    selected_group = _resume_display_value(
                        resume_workspace,
                        resume_settings if isinstance(resume_settings, dict) else {},
                        stereo=resume_stereo,
                    )
                    st.session_state["composite_studio_result_group"] = selected_group
                    saved_target = resume_workspace.get("target")
                    if isinstance(saved_target, dict) and selected_group:
                        target_group = str(saved_target.get("group", "")).strip() or (
                            _physical_group_for_display(selected_group, stereo=resume_stereo)
                        )
                        target_group = _group_target_scope(target_group)
                        if saved_target.get("source") in {"PhaseEQ preset", "File"}:
                            st.session_state[
                                f"composite_studio_target_source_{target_group}"
                            ] = saved_target["source"]
                        if isinstance(saved_target.get("file"), dict):
                            st.session_state["_composite_resumed_group_target"] = saved_target["file"]
                        if saved_target.get("preset_id"):
                            st.session_state[
                                f"composite_studio_group_target_preset_{target_group}"
                            ] = str(saved_target["preset_id"])
                        if isinstance(saved_target.get("preset_definition"), dict):
                            restored_definition = canonical_target_definition(
                                saved_target["preset_definition"]
                            )
                            st.session_state[
                                f"_composite_resumed_group_target_{target_group}"
                            ] = restored_definition
                            st.session_state[
                                f"composite_studio_group_target_phase_enabled_{target_group}"
                            ] = bool(restored_definition.get("phase_enabled", True))
                            st.session_state[
                                f"composite_studio_group_target_gain_db_{target_group}"
                            ] = float(restored_definition.get("application_gain_db", 0.0))
                    saved_targets = resume_workspace.get("targets")
                    if isinstance(saved_targets, dict):
                        target_items = sorted(
                            saved_targets.items(),
                            key=lambda item: (
                                0 if str(item[0]).strip() == "Left" else
                                1 if str(item[0]).strip() == "Right" else 0
                            ),
                        )
                        restored_target_scopes: set[str] = set()
                        for target_group, target_payload in target_items:
                            if not isinstance(target_payload, dict):
                                continue
                            normalized_target_group = str(target_group).strip()
                            if normalized_target_group not in {"Main", "Left", "Right"}:
                                continue
                            normalized_target_group = _group_target_scope(
                                normalized_target_group
                            )
                            if normalized_target_group in restored_target_scopes:
                                continue
                            restored_target_scopes.add(normalized_target_group)
                            if target_payload.get("source") in {"PhaseEQ preset", "File"}:
                                st.session_state[
                                    f"composite_studio_target_source_{normalized_target_group}"
                                ] = target_payload["source"]
                            if target_payload.get("preset_id"):
                                st.session_state[
                                    f"composite_studio_group_target_preset_{normalized_target_group}"
                                ] = str(target_payload["preset_id"])
                            if isinstance(target_payload.get("preset_definition"), dict):
                                restored_definition = canonical_target_definition(
                                    target_payload["preset_definition"]
                                )
                                st.session_state[
                                    f"_composite_resumed_group_target_{normalized_target_group}"
                                ] = restored_definition
                                st.session_state[
                                    f"composite_studio_group_target_phase_enabled_{normalized_target_group}"
                                ] = bool(restored_definition.get("phase_enabled", True))
                                st.session_state[
                                    f"composite_studio_group_target_gain_db_{normalized_target_group}"
                                ] = float(
                                    restored_definition.get("application_gain_db", 0.0)
                                )
                    saved_graph = resume_workspace.get("graph")
                    if isinstance(saved_graph, dict):
                        if saved_graph.get("kind"):
                            st.session_state["composite_studio_graph"] = saved_graph["kind"]
                        if "wrapped_phase" in saved_graph:
                            restored_wrapped = bool(saved_graph["wrapped_phase"])
                            st.session_state["composite_studio_wrapped_phase"] = restored_wrapped
                            st.session_state["composite_studio_unwrapped_phase"] = not restored_wrapped
                            st.session_state[
                                "composite_studio_unwrapped_phase_widget"
                            ] = not restored_wrapped
                    st.session_state["_composite_resume_digest"] = resume_digest
                    st.rerun()
        st.markdown(ui_message('ui.ff35ad6419c9cd'))
        _render_system_library_header()
        common_iir_configs = _shared_iir_configs(mode_key, crossover_frequencies_hz)
        structural_sections = alignment_structure.automatic_sections(
            st.session_state.get("settings", {}), MODE_WAYS[mode_key], crossover_frequencies_hz,
            tuple(normalize_crossover_method(st.session_state.get(
                f"composite_studio_{mode_key}_crossover_method_{index}"))
                for index in range(len(crossover_frequencies_hz))),
        )
        st.markdown(ui_message('ui.e5f113fdc5fdf7'))
        layout_key = "composite_studio_output_layout"
        settings_payload = (
            current_settings
            if isinstance(current_settings, dict)
            else st.session_state.get("settings", {})
        )
        saved_layout = str(
            settings_payload.get("output_layout", "Mono")
            if isinstance(settings_payload, dict) else "Mono"
        )
        if st.session_state.get(layout_key) not in {"Mono", "Stereo"}:
            st.session_state[layout_key] = (
                saved_layout if saved_layout in {"Mono", "Stereo"} else "Mono"
            )

        def _commit_output_layout() -> None:
            value = str(st.session_state[layout_key])
            st.session_state[layout_key] = value
            stored = dict(st.session_state.get("settings", {}))
            stored["output_layout"] = value
            st.session_state["settings"] = stored
            if isinstance(current_settings, dict):
                current_settings["output_layout"] = value

        layout = shared_segmented_control(
            ui_message('ui.132f96868b537c'), ["Mono", "Stereo"],
            key=layout_key, width="stretch", required=True,
            on_change=_commit_output_layout,
        format_func=localized_formatter(str)) or "Mono"
        stereo = layout == "Stereo"
        shared_sub_key = "composite_studio_shared_sub"
        if shared_sub_key not in st.session_state:
            st.session_state[shared_sub_key] = bool(
                settings_payload.get("shared_sub", False)
                if isinstance(settings_payload, dict) else False
            )
        shared_sub = bool(st.session_state[shared_sub_key]) if stereo else False
        if stereo and "SUB" in bands:
            shared_sub = st.toggle(
                ui_message('ui.4cc08e9908770c'), key=shared_sub_key,
                help=ui_message('ui.6488afdeaec633'),
            )
        if isinstance(current_settings, dict):
            current_settings["output_layout"] = layout
            current_settings["shared_sub"] = bool(st.session_state[shared_sub_key])
        stored = dict(st.session_state.get("settings", {}))
        stored["output_layout"] = layout
        stored["shared_sub"] = bool(st.session_state[shared_sub_key])
        st.session_state["settings"] = stored
        st.markdown(ui_message('ui.a640673607e1d9'))
        st.caption(
            ui_message('ui.3e67cb1be0bed3')
        )
        package_options, package_by_id = _speaker_package_options()
        channel_rows: list[dict[str, object]] = []
        measurement_source_slots = []
        channel_specs = []
        for band in reversed(bands):
            groups = ("Main",) if not stereo else (("Sub",) if band == "SUB" and shared_sub else ("Left", "Right"))
            channel_specs.extend((band, group) for group in groups)
        for band, default_group in channel_specs:
            prefix = _channel_state_prefix(band, default_group, stereo=stereo)
            default_name = band if not stereo else f"{default_group} {band}"
            st.session_state.setdefault(f"{prefix}_name", default_name)
            st.session_state.setdefault(f"{prefix}_group", default_group)
            st.session_state.setdefault(f"{prefix}_gain", 0.0)
            st.session_state.setdefault(f"{prefix}_polarity", "Normal (+)")
            st.session_state.setdefault(f"{prefix}_delay", 0.0)
            st.session_state.setdefault(f"{prefix}_auto_alignment_delay", 0.0)
            st.session_state.setdefault(f"{prefix}_auto_alignment_allpass", [])
            if structural_sections is not None:
                next_sections = [
                    section.to_dict() for section in structural_sections[band]
                ]
                if st.session_state[f"{prefix}_auto_alignment_allpass"] != next_sections:
                    st.session_state[f"{prefix}_auto_alignment_allpass"] = next_sections
                    st.session_state["_alignment_refresh_requested"] = True
            st.session_state.setdefault(f"{prefix}_enabled", True)
            st.session_state.setdefault(f"{prefix}_speaker_package_id", "")
            missing_package_id = (
                str(st.session_state[f"{prefix}_speaker_package_id"]).strip()
                if st.session_state[f"{prefix}_speaker_package_id"] not in package_options
                else ""
            )
            if missing_package_id:
                st.session_state[f"{prefix}_speaker_package_id"] = ""
            with st.expander(ui_message('ui.ab230f2e6e2c40', p0=f'{default_name}'), expanded=False):
                if missing_package_id:
                    st.warning(ui_message('ui.a57c26aeed4f75', p0=f'{missing_package_id}'))
                enabled = st.checkbox(ui_message('ui.d0941e1ca405c3'), key=f"{prefix}_enabled")
                name = st.text_input(ui_message('ui.ce4683e7013a18'), key=f"{prefix}_name")
                group = st.text_input(ui_message('ui.aa513354c66d58'), key=f"{prefix}_group")
                from ui.multiway_dc_gain import render_channel_dc_gain
                dc_gain_normalize = render_channel_dc_gain(
                    prefix, fir_enabled=int((band_tap_lengths or {}).get(band, 0) or 0) > 0,
                )
                speaker_package_id = str(st.session_state.get(f"{prefix}_speaker_package_id", ""))
                measurement_source_slots.append(st.empty())
                gain_db = st.number_input(
                    ui_message('ui.c2c39b6c05e00a'), step=0.1, format="%.3f", key=f"{prefix}_gain",
                )
                polarity = shared_selectbox(
                    ui_message('ui.589bc4b7a26b66'), ["Normal (+)", "Invert (-)"], key=f"{prefix}_polarity",
                format_func=localized_formatter(str))
                delay = st.number_input(
                    ui_message('ui.34ce50a9edd47c'), step=0.1, format="%.3f", key=f"{prefix}_delay",
                )
                speaker_response = None
                iir_response = st.file_uploader(
                    ui_message('ui.20d4212f64ac83'), type=["wav", "frd", "csv", "txt"],
                    key=f"{prefix}_iir",
                    on_change=_capture_channel_upload,
                    args=(f"{prefix}_iir", f"{prefix}:iir"),
                    help=ui_message('ui.11919f80f9fca6'),
                )
                additional_fir = st.file_uploader(
                    ui_message('ui.4efd52d954b9c3'), type=["wav", "csv", "txt"],
                    key=f"{prefix}_additional_fir",
                    on_change=_capture_channel_upload,
                    args=(f"{prefix}_additional_fir", f"{prefix}:additional_fir"),
                )
                resolved_iir = common_iir_configs[band]
                st.caption(
                    ui_message('ui.c8c3b5ba41f77d', p0=f"{('LR' + str(resolved_iir.order) if resolved_iir.enabled else 'OFF')}", p1=f'{resolved_iir.highpass_hz:,.1f}', p2=f'{resolved_iir.lowpass_hz:,.1f}')
                )
            package_record = package_by_id.get(str(speaker_package_id))
            if package_record is not None:
                package_record = get_speaker_package(*_design_library_paths(), str(speaker_package_id))
            package_frd = _speaker_package_frd(package_record) if package_record is not None else None
            speaker_response = _channel_upload(
                speaker_response, f"{prefix}:speaker",
            )
            iir_response = _channel_upload(iir_response, f"{prefix}:iir")
            additional_fir = _channel_upload(
                additional_fir, f"{prefix}:additional_fir",
            )
            channel_rows.append({
                "band": band,
                "source": "generated_band",
                "dc_gain_normalize": bool(dc_gain_normalize),
                "phase_alignment_acoustic_target": bool(st.session_state.get("settings", {}).get(mode_key, {}).get("phase_alignment_acoustic_target", False)),
                "enabled": bool(enabled),
                "name": str(name).strip(),
                "way": band,
                "group": str(group).strip(),
                "gain_db": float(gain_db),
                "polarity": 1 if polarity == "Normal (+)" else -1,
                "delay_samples": float(delay),
                "auto_alignment_delay_samples": float(
                    st.session_state[f"{prefix}_auto_alignment_delay"]
                ),
                "auto_alignment_allpass": tuple(
                    AllPassSection(
                        float(item["frequency_hz"]), float(item.get("q", 0.0)),
                        order=int(item.get("order", 2)),
                        polarity=int(item.get("polarity", 1)),
                    )
                    for item in st.session_state[f"{prefix}_auto_alignment_allpass"]
                    if isinstance(item, dict)
                ),
                "speaker_response": speaker_response,
                "speaker_library_response": (
                    {"filename": f"speaker_package_{speaker_package_id}.frd", "data": package_frd}
                    if package_frd is not None else None
                ),
                "iir_response": iir_response,
                "additional_fir": additional_fir,
                "iir_crossover": common_iir_configs[band],
                "dsp_additional_delay_samples": 0.0,
                "channel_id": str(st.session_state.get(f"{prefix}_channel_id", "")),
                "speaker_package_id": str(speaker_package_id),
                "latest_assignment_id": str(st.session_state.get(f"{prefix}_latest_assignment_id", "")),
            })

        st.markdown(ui_message('ui.cac680f272cec5'))
        st.caption(ui_message('ui.9518853e34b063'))
        st.session_state.setdefault("composite_studio_extra_count", 0)
        count_column, add_column, remove_column = st.columns([1.5, 1, 1])
        with count_column:
            extra_count = st.number_input(
                ui_message('ui.cbecd3ed842460'), min_value=0, max_value=32, step=1,
                key="composite_studio_extra_count",
            )
        with add_column:
            st.button(
                ui_message('ui.06ec8903e807d0'), icon=":material/add:", width="stretch",
                key="composite_studio_add_extra_input",
                on_click=_adjust_additional_input_count, args=(1,),
            )
        with remove_column:
            st.button(
                ui_message('ui.4b66e955a52442'), icon=":material/remove:", width="stretch",
                key="composite_studio_remove_extra_input",
                disabled=int(extra_count) <= 0,
                on_click=_adjust_additional_input_count, args=(-1,),
                help=ui_message('ui.2095766f8d5b5c'),
            )
        if int(extra_count) > 0:
            st.caption(
                ui_message('ui.bab34832a3829f', p0=f'{_default_additional_input_group()}')
            )
        for index in range(int(extra_count)):
            prefix = f"composite_studio_extra_{index}"
            st.session_state.setdefault(f"{prefix}_name", f"Input {index + 1}")
            st.session_state.setdefault(f"{prefix}_way", "Fullrange")
            st.session_state.setdefault(f"{prefix}_group", "Main")
            st.session_state.setdefault(f"{prefix}_gain", 0.0)
            st.session_state.setdefault(f"{prefix}_polarity", "Normal (+)")
            st.session_state.setdefault(f"{prefix}_delay", 0.0)
            with st.expander(ui_message('ui.d4dbd6762febb7', p0=f'{index + 1}'), expanded=False):
                name = st.text_input(ui_message('ui.ce4683e7013a18'), key=f"{prefix}_name")
                way = st.text_input(ui_message('ui.bea4d7970552bd'), key=f"{prefix}_way")
                group = st.text_input(ui_message('ui.aa513354c66d58'), key=f"{prefix}_group")
                gain_db = st.number_input(
                    ui_message('ui.c2c39b6c05e00a'), step=0.1, format="%.3f", key=f"{prefix}_gain",
                )
                polarity = shared_selectbox(
                    ui_message('ui.589bc4b7a26b66'), ["Normal (+)", "Invert (-)"], key=f"{prefix}_polarity",
                format_func=localized_formatter(str))
                delay = st.number_input(
                    ui_message('ui.34ce50a9edd47c'), step=0.1, format="%.3f", key=f"{prefix}_delay",
                )
                upload = st.file_uploader(
                    ui_message('ui.d6c7b397d0754c'),
                    type=["wav", "frd", "csv", "txt"], key=f"{prefix}_upload",
                )
            channel_rows.append({
                "band": f"Additional {index + 1}", "source": "upload", "enabled": True,
                "name": str(name).strip(), "way": str(way).strip(),
                "group": str(group).strip(), "gain_db": float(gain_db),
                "polarity": 1 if polarity == "Normal (+)" else -1,
                "delay_samples": float(delay), "upload": upload,
                "channel_id": str(st.session_state.get(f"{prefix}_channel_id", "")),
                "speaker_package_id": str(st.session_state.get(f"{prefix}_speaker_package_id", "")),
                "latest_assignment_id": str(st.session_state.get(f"{prefix}_latest_assignment_id", "")),
            })

        current_library_system = _current_library_system()
        for row in channel_rows:
            if row.get("source") == "generated_band":
                _restore_fir_off_assignment_result(row, current_library_system)

        unmatched_exchange_names = []
        exchange_token = str(st.query_params.get("exchange", "")).strip()
        if not exchange_token:
            # Save and return adds the sample-rate query, but that query is not
            # a durable connection. Reconnect the last published FIRs for the
            # active sample rate after a plain reload or launcher navigation.
            exchange_token = str(int(sample_rate_hz))
        exchange_directory = _exchange_root() / exchange_token if exchange_token.isdigit() else None
        from utils.exchange_publication import has_published_exchange, published_exchange_package
        if exchange_directory is not None and has_published_exchange(_exchange_root(), int(sample_rate_hz)):
            try:
                exchange_package = published_exchange_package(_exchange_root(), int(sample_rate_hz))
            except (CompositeValidationError, ValueError, OSError) as exc:
                st.error(ui_message('ui.d72a6d4b65518c', p0=f'{exc}'))
                # Package loading precedes per-channel validation. Retain a
                # retry even when a missing FIR prevents reaching that loop.
                for row in channel_rows:
                    identifier = latest_published_phaseeq_assignment_id(
                        _exchange_root(), channel_id=_library_channel_id(row, current_library_system),
                    )
                    if identifier:
                        try:
                            failed_status = read_phaseeq_assignment_status(_exchange_root(), assignment_id=identifier)
                        except (OSError, ValueError):
                            continue
                        if failed_status is not None:
                            record_failure(st.session_state, identifier, failed_status.revision, exc)
            else:
                if exchange_package.manifest.sample_rate_hz != int(sample_rate_hz):
                    st.error(
                        ui_message('ui.5ce6d82032b5ec', p0=f'{exchange_package.manifest.sample_rate_hz:,}', p1=f'{int(sample_rate_hz):,}')
                    )
                else:
                    for index, channel in enumerate(exchange_package.manifest.channels):
                        source_name = channel.wav or channel.frd
                        assert source_name is not None
                        attached_row = next((
                            row for row in channel_rows
                            if row.get("source") == "generated_band"
                            and str(row.get("name", "")) == channel.name
                            and str(row.get("way", "")) == channel.way
                            and str(row.get("group", "")) == channel.group
                        ), None)
                        if attached_row is not None:
                            attached_channel_id = _library_channel_id(
                                attached_row, _current_library_system(),
                            )
                            attached_assignment_id = (
                                channel.assignment_id
                                or latest_phaseeq_assignment_id(
                                    _exchange_root(), channel_id=attached_channel_id,
                                )
                            )
                            attached_status = (
                                read_phaseeq_assignment_status(
                                    _exchange_root(), assignment_id=attached_assignment_id,
                                )
                                if attached_assignment_id else None
                            )
                            assignment_target, _target_application = _assignment_target_reference(
                                attached_assignment_id or "",
                            )
                            result_payload = (
                                attached_status.result
                                if attached_status is not None
                                and isinstance(attached_status.result, dict) else {}
                            )
                            if result_payload.get('asset_schema') == 1:
                                published_channel = result_payload.get('exchange_channel') or {}
                                if published_channel.get('wav') != channel.wav:
                                    continue
                            from composite_engine.assignment_snapshot import validated_speaker_asset, timing_for_asset
                            try:
                                validated_speaker = validated_speaker_asset(
                                    result_payload, _exchange_root(),
                                    attached_status.revision if attached_status is not None else 0,
                                )
                                from utils.composite_exchange import assignment_working_session_path
                                from utils.exchange_publication import working_session_from_result
                                working_path = (working_session_from_result(_exchange_root(), attached_assignment_id, result_payload)
                                    if result_payload.get('asset_schema') == 1 else assignment_working_session_path(_exchange_root(), attached_assignment_id))
                            except (OSError, ValueError) as exc:
                                record_failure(st.session_state, attached_assignment_id, attached_status.revision if attached_status else 0, exc)
                                st.warning(ui_message('ui.0cb0262cf0de6a', p0=f'{channel.name}', p1=f'{exc}'))
                                continue
                            if result_payload and not _assignment_result_provenance_valid(
                                attached_assignment_id or "", result_payload, _current_library_system(),
                            ):
                                st.warning(ui_message('ui.17ca8bf567ae4e', p0=f'{channel.name}'))
                                continue
                            recipe_payload = result_payload.get("fir_recipe")
                            recipe_generated = None
                            recipe_error = ""
                            if isinstance(recipe_payload, dict):
                                try:
                                    from fir_recipe_engine import verified_fir_recipe_result
                                    recipe_generated = verified_fir_recipe_result(
                                        recipe_payload,
                                        str(result_payload.get("fir_coefficient_hash", "")),
                                    )
                                except (TypeError, ValueError) as exc:
                                    recipe_error = str(exc)
                            if recipe_error:
                                st.warning(ui_message('ui.d43745d5e0c99a', p0=f'{channel.name}', p1=f'{recipe_error}'))
                                continue
                            exclude_unverified_target_fir = (
                                assignment_target is not None
                                and result_payload.get("target_applied_to_fir") is not False
                            )
                            fir_filename = source_name
                            fir_data = exchange_package.assets[source_name]
                            if recipe_generated is not None:
                                from scipy.io import wavfile
                                recipe_buffer = io.BytesIO()
                                wavfile.write(
                                    recipe_buffer, int(sample_rate_hz),
                                    recipe_generated.coefficients.astype(np.float64),
                                )
                                fir_filename = f"{channel.name}_composite_recipe.wav"
                                fir_data = recipe_buffer.getvalue()
                            fir_is_identity = bool(
                                recipe_generated.bypass
                                if recipe_generated is not None else
                                _fir_coefficients_are_identity(
                                    fir_filename, fir_data, int(sample_rate_hz),
                                )
                            )
                            # Rebuild the attachment decision from the current
                            # package/status.  Never let a cached FIR survive a
                            # bypass or Target-contract downgrade.
                            attached_row.pop("phaseeq_fir_response", None)
                            attached_row.pop("phaseeq_fir_target_excluded", None)
                            if (
                                not fir_is_identity
                                and not exclude_unverified_target_fir
                                and not recipe_error
                            ):
                                attached_row["phaseeq_fir_response"] = {
                                    "filename": fir_filename,
                                    "data": fir_data,
                                    "tap_count": int(channel.tap_count or 0),
                                    "target_dc_abs": (recipe_generated.target_dc_abs if recipe_generated is not None else None),
                                    "content_hash": hashlib.sha256(fir_data).hexdigest(),
                                    "generation": (
                                        "composite_fir_recipe"
                                        if recipe_generated is not None else "legacy_exchange_asset"
                                    ),
                                }
                            elif exclude_unverified_target_fir:
                                attached_row["phaseeq_fir_target_excluded"] = assignment_target
                            attached_row["phaseeq_iir"] = list(
                                result_payload.get("phaseeq_iir", [])
                            )
                            attached_row["phaseeq_iir_sos"] = tuple(
                                tuple(float(value) for value in section)
                                for section in result_payload.get("phaseeq_iir_sos", [])
                            )
                            if recipe_error:
                                st.warning(
                                    ui_message('ui.904a27015cdb6d', p0=f'{channel.name}', p1=f'{recipe_error}')
                                )
                            provenance_valid = (
                                not result_payload
                                or _assignment_result_provenance_valid(
                                    attached_assignment_id or "", result_payload,
                                    _current_library_system(),
                                )
                            )
                            if result_payload and not provenance_valid:
                                attached_row.pop("phaseeq_fir_response", None)
                                attached_row["phaseeq_iir"] = []
                                attached_row["phaseeq_iir_sos"] = ()
                                st.warning(
                                    ui_message('ui.e17de5d5aa6b2e', p0=f'{channel.name}')
                                )
                                result_payload = {}
                            if validated_speaker is not None:
                                attached_row["phaseeq_speaker_response"] = validated_speaker
                            alignment_target.receive(attached_row, result_payload)
                            attached_row["phaseeq_assignment_id"] = attached_assignment_id or ""
                            attached_row["phaseeq_assignment_revision"] = (
                                attached_status.revision if attached_status is not None else 0
                            )
                            if isinstance(result_payload.get("timing_provenance"), dict):
                                attached_row["timing_provenance"] = timing_for_asset(result_payload, validated_speaker)
                            else:
                                attached_row.pop("timing_provenance", None)
                            attached_row["phaseeq_status_state"] = (
                                attached_status.state if attached_status is not None else "Published"
                            )
                            attached_row["phaseeq_working_session"] = working_path
                            _cache_phaseeq_bundle(
                                attached_row, attached_assignment_id or "",
                            )
                            continue
                        # An Exchange return may update an existing output only.
                        # Old/unmatched results must never create implicit mix inputs.
                        unmatched_exchange_names.append(channel.name)

        current_system = _current_library_system()
        for row in channel_rows:
            row["channel_id"] = _library_channel_id(row, current_system)
            if current_system is not None:
                stored = next((
                    item for item in current_system.channels
                    if item.channel_id == row["channel_id"]
                ), None)
                if stored is not None:
                    row["speaker_package_id"] = stored.speaker_package_id
                    row["latest_assignment_id"] = stored.latest_assignment_id
            # Restore only after stable Channel and Assignment identity has
            # been resolved. A newer Exchange result above always wins.
            if "phaseeq_assignment_id" not in row:
                _restore_cached_phaseeq_bundle(row)

        for row in channel_rows:
            if row.get("source") == "generated_band":
                row["phase_alignment_recipe"] = band_split_from_studio(
                    mode_key, int(sample_rate_hz),
                    (current_settings or st.session_state.get("settings", {}))[mode_key], row["way"],
                    fir_enabled=int((band_tap_lengths or {}).get(row["band"], 0)) > 0,
                    phase_alignment=alignment_target.recipe_definition(row),
                )
            if alignment_target.pending(row):
                # A returned FIR may still contain the previous target phase.
                # Exclude stale correction FIRs. Target-only phase stages stay
                # absent from the direct response while waiting for a return.
                row.pop("phaseeq_fir_response", None)

        from utils.eq_link_exchange import invalidate_stale_returns
        try:
            _link_pending_names = invalidate_stale_returns(channel_rows, _exchange_root(),
                f"{current_system.id if current_system is not None else ''}:{int(sample_rate_hz)}")
            if _link_pending_names:
                st.warning(ui_message("ui.f1859f958520b5") + ", ".join(_link_pending_names))
        except (OSError, ValueError) as exc:
            st.error(f"Stereo Linkの返却設定を確認できません: {exc}")
            for row in channel_rows:
                row["phaseeq_link_pending"] = True

        st.markdown(ui_message('ui.42cbe1ed92e65b'))
        current_system = _render_system_library_save(
            channel_rows,
            mode_key=mode_key,
            sample_rate_hz=sample_rate_hz,
            current_settings=dict(
                current_settings
                if current_settings is not None
                else st.session_state.get("settings", {})
            ),
        )
        for row in channel_rows:
            source = row.get("phaseeq_fir_response")
            if (row.get("dc_gain_normalize") and isinstance(source, dict)
                    and source.get("target_dc_abs") is None):
                st.warning(ui_message('ui.7c549553053c3f', p0=str(row["name"])))
        st.session_state["composite_studio_channels"] = channel_rows
        try:
            editable_channels = _phaseeq_assignment_rows(channel_rows)
            write_multiway_workspace(
                _exchange_root(), system_id=current_system.id if current_system is not None else "",
                system_name=str(current_system.name if current_system is not None else st.session_state.get("composite_studio_system_name", "")),
                sample_rate_hz=int(sample_rate_hz),
                channel_ids=list(editable_channels),
                channels=[{key: str(row.get(key, '')) for key in ('channel_id', 'name', 'group', 'way')}
                          for row in editable_channels.values()],
                previous_channel_ids=[str(row["channel_id"]) for row in channel_rows if row.get("enabled", True)],
            )
        except (OSError, ValueError) as exc:
            st.error(ui_message('ui.43ef8f9eba56ca', p0=f'{exc}'))
            return

        from utils.multiway_measurements import apply_measurement_inputs
        from utils.composite_exchange import read_multiway_workspace
        try:
            applied_measurements = apply_measurement_inputs(
                _exchange_root(), read_multiway_workspace(_exchange_root()), channel_rows)
            for slot, row in zip(measurement_source_slots, channel_rows):
                source = resolve_speaker_source(row)
                slot.caption(display_text("現在の入力：") + (source.label if source is not None else display_text("未設定")))
            for applied in applied_measurements:
                st.caption(f"{applied['name']} ← {applied['measurement']}")
        except (ValueError, OSError) as exc:
            st.error(str(exc))

        for row in channel_rows:
            if str(row.get("channel_id")) in st.session_state.get("_history_pending_phaseeq", {}):
                for field in ("phaseeq_fir_response", "phaseeq_iir", "phaseeq_iir_sos", "phaseeq_working_session"):
                    row.pop(field, None)

        st.markdown(ui_message('ui.10b19b2b5f19ef'))
        st.caption(ui_message('ui.0a4e8ba66c4b72'))
        active_phaseeq_sessions = list_active_phaseeq_sessions(_exchange_root())
        assignment_screen_records = [
            session for session in active_phaseeq_sessions if session.mode == "Assignment"
        ]
        connection_view = _phaseeq_connection_view(active_phaseeq_sessions)
        getattr(st, connection_view.level)(display_notice(connection_view.message))
        if connection_view.incomplete_assignment_count:
            st.warning(
                ui_message('ui.567facd585102a')
            )
        assignment_rows = _phaseeq_assignment_rows(channel_rows)
        assignment_options = list(assignment_rows)
        from ui.studio_stereo_link import render_studio_link
        try:
            render_studio_link(st.session_state, _exchange_root(),
                current_system.id if current_system is not None else "",
                int(sample_rate_hz), assignment_rows)
        except (OSError, ValueError) as exc:
            st.error(f"Stereo Linkを読み込めません: {exc}")
        send_all_assignments = render_phaseeq_send_controls(assignment_rows)
        _render_phaseeq_workspace_link("PhaseEQで開く", os.environ.get("PHASEEQ_URL", "http://localhost:8501"))
        st.caption(ui_message('ui.457dd6bee3b351'))
        _render_phaseeq_workspace_link(display_text("測定データのチャンネル選択へ"), os.environ.get("PHASEEQ_URL", "http://localhost:8501").split("?")[0] + "?page=Project")
        if not assignment_options:
            st.caption(ui_message('ui.f4bdf73603e841'))
            return
        progress_rows = []
        assignment_reference_states: dict[str, dict[str, str]] = {}
        for channel_id, channel_row in assignment_rows.items():
            latest_id = latest_phaseeq_assignment_id(
                _exchange_root(), channel_id=channel_id,
            )
            latest_status = (
                read_phaseeq_assignment_status(
                    _exchange_root(), assignment_id=latest_id,
                )
                if latest_id else None
            )
            connected_id = _connected_assignment_id(channel_id, latest_id or "", assignment_screen_records)
            returned_id = str(channel_row.get("phaseeq_assignment_id", "")).strip()
            reference_state = _assignment_reference_state(connected_id, latest_id or "")
            screen_state = _phaseeq_screen_state(
                channel_id, latest_id or "", assignment_screen_records,
            )
            assignment_reference_states[channel_id] = {
                "state": reference_state,
                "connected_assignment_id": connected_id,
                "latest_assignment_id": latest_id or "",
            }
            progress_rows.append({
                "Channel": str(channel_row["name"]),
                "Assignment状態": _assignment_status_display(
                    latest_status.state if latest_status is not None
                    else "Assigned" if latest_id else ""
                ),
                "返却データ": ("最新を反映" if returned_id and returned_id == latest_id
                           else "前回結果を保持・最新版の返却待ち" if returned_id else "未返却"),
                **_phaseeq_return_summary(channel_row),
                "PhaseEQ画面": screen_state,
                "Way": str(channel_row["way"]),
                "Group": str(channel_row["group"]),
                "参照状態": reference_state,
                "rev": str(latest_status.revision) if latest_status is not None else "—",
                "接続ID": connected_id[:8] if connected_id else "—",
                "最新ID": latest_id[:8] if latest_id else "—",
            })
        st.session_state["composite_phaseeq_assignment_reference_states"] = (
            assignment_reference_states
        )
        st.dataframe(
            pd.DataFrame(progress_rows), hide_index=True, width="stretch",
            column_config={
                "Channel": st.column_config.TextColumn(width="medium"),
                "Assignment状態": st.column_config.TextColumn(width="medium"),
                "参照状態": st.column_config.TextColumn(width="medium"),
                "PhaseEQ画面": st.column_config.TextColumn(width="medium"),
            },
        )
        if unmatched_exchange_names:
            st.caption(ui_message('ui.2761c16d12b959',
                                  p0=", ".join(unmatched_exchange_names)))
        mismatched_names = [
            str(assignment_rows[channel_id]["name"])
            for channel_id, state in assignment_reference_states.items()
            if state["state"] == "別Assignmentを参照"
        ]
        if mismatched_names:
            st.warning(
                ui_message('ui.6ace5aa46d1bfc') + ", ".join(mismatched_names)
                + ui_message('ui.f2a940931e1d3e')
            )
        if send_all_assignments:
            completed_names = []
            try:
                from composite_engine.adapter.control import validate_phaseeq_assignment_request
                prepared_requests = []
                exchange_root = _exchange_root()
                for assignment_channel_id, assignment_row in assignment_rows.items():
                    assignment_band = str(assignment_row["way"])
                    assignment_taps = int((band_tap_lengths or {}).get(assignment_band, 0))
                    assignment_hpf_hz, assignment_lpf_hz = _way_band_limits(
                        assignment_band, mode_key, crossover_frequencies_hz,
                    )
                    assignment_target_definition = _selected_group_target_definition(
                        str(assignment_row["group"]), int(sample_rate_hz),
                    )
                    assignment_iir = assignment_row.get("iir_crossover", IIRCrossoverConfig())
                    assignment_iir_payload = {
                        **assignment_iir.to_dict(),
                        "type": "OFF" if not assignment_iir.enabled else "Mixed" if assignment_iir.highpass_order and assignment_iir.lowpass_order and assignment_iir.highpass_order != assignment_iir.lowpass_order else f"LR{int(assignment_iir.order)}",
                        "highpass_type": f"LR{assignment_iir.highpass_order}" if assignment_iir.highpass_order else "OFF",
                        "lowpass_type": f"LR{assignment_iir.lowpass_order}" if assignment_iir.lowpass_order else "OFF",
                        "application": "exclusive_crossover_stage",
                        "exclusive": True,
                        "coefficient_sample_rate_hz": int(sample_rate_hz),
                        "sos": iir_crossover_sos(assignment_iir, int(sample_rate_hz)).tolist(),
                        "definition_hash": assignment_iir.definition_hash(int(sample_rate_hz)),
                    }
                    composite_url = os.environ.get(
                        "COMPOSITE_ENGINE_URL",
                        f"http://localhost:{os.environ.get('COMPOSITE_ENGINE_PORT', '8502')}",
                    )
                    return_query = urlencode({
                        "exchange": str(int(sample_rate_hz)),
                        "group": str(assignment_row["group"]),
                        "channel": assignment_channel_id,
                    })
                    current_record = _current_library_record()
                    target_record = None
                    if assignment_target_definition is not None:
                        target_hash = str(assignment_target_definition.get("revision", ""))
                        target_record = find_target_revision_by_hash(
                            _multiway_system_db_path(), target_hash,
                        )
                        if target_record is None:
                            target_record = create_target(
                                _multiway_system_db_path(),
                                dict(assignment_target_definition),
                                lifecycle="temporary", source="composite_assignment",
                            )
                        if current_record is not None:
                            save_target_binding(
                                _multiway_system_db_path(),
                                TargetBinding(
                                    system_id=current_record.system.id,
                                    scope="group", group=str(assignment_row["group"]),
                                    target_id=target_record.ref.target_id, mode="latest",
                                ),
                            )
                    prepared_requests.append(PhaseEQAssignmentRequest(
                        sample_rate_hz=int(sample_rate_hz),
                        tap_count=int(assignment_taps),
                        channel_name=str(assignment_row["name"]),
                        way=str(assignment_row["way"]),
                        group=str(assignment_row["group"]),
                        channel_id=assignment_channel_id,
                        return_url=f"{composite_url}?{return_query}",
                        multiway_mode=str(mode_key),
                        highpass_hz=float(assignment_hpf_hz),
                        lowpass_hz=float(assignment_lpf_hz),
                        band_split_recipe=band_split_from_studio(
                            mode_key, int(sample_rate_hz), current_settings[mode_key], assignment_band,
                            fir_enabled=assignment_taps > 0,
                            phase_alignment=alignment_target.recipe_definition(assignment_row)),
                        linear_fir_filters=tuple(
                            (band_linear_fir_filters or {}).get(assignment_band, ())
                        ) if assignment_taps > 0 else (),
                        iir_crossover=assignment_iir_payload,
                        target_definition=(
                            dict(assignment_target_definition)
                            if assignment_target_definition is not None else None
                        ),
                        target_application="phaseeq_target",
                        system_id=current_record.system.id if current_record is not None else "",
                        management_no=current_record.management_no if current_record is not None else "",
                        system_revision_number=(
                            current_record.system.revision if current_record is not None else None
                        ),
                        system_content_hash=(
                            current_record.content_hash if current_record is not None else ""
                        ),
                        target_id=target_record.ref.target_id if target_record is not None else "",
                        target_revision_number=(
                            target_record.ref.revision_number if target_record is not None else None
                        ),
                        target_content_hash=(
                            target_record.ref.content_hash if target_record is not None else ""
                        ),
                    ))

                for request in prepared_requests:
                    validate_phaseeq_assignment_request(request)
                for request in prepared_requests:
                    assignment_path = write_phaseeq_assignment(exchange_root, request)
                    pending_history = st.session_state.get("_history_pending_phaseeq", {})
                    restored_payload = pending_history.get(request.channel_id)
                    if restored_payload is not None:
                        from utils.composite_exchange import write_assignment_working_session
                        buffer = io.BytesIO()
                        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                            archive.writestr("settings/phaseeq_settings.json", json.dumps(restored_payload, ensure_ascii=False))
                        write_assignment_working_session(exchange_root, assignment_path.parent.name, buffer.getvalue())
                        st.session_state.setdefault("_history_awaiting_return", {})[request.channel_id] = assignment_path.parent.name
                        pending_history.pop(request.channel_id, None)
                    completed_names.append(f"{request.group} / {request.channel_name}")
                    if current_system is not None:
                        update_multiway_channel_assignment(
                            _multiway_system_db_path(), system_id=current_system.id,
                            channel_id=request.channel_id, assignment_id=assignment_path.parent.name,
                        )
                    assignment_rows[request.channel_id]["latest_assignment_id"] = assignment_path.parent.name
            except (KeyError, OSError, UnicodeError, ValueError) as exc:
                st.error(ui_message('ui.c7d231c08217f3', p0=f'{exc}') + (", ".join(completed_names) or "なし"))
            else:
                st.success(ui_message('ui.865a95d452cb84', p0=f'{len(completed_names)}'))

        _render_studio_history(channel_rows)


def _way_band_limits(
    way: str,
    mode_key: str,
    crossover_frequencies_hz: tuple[float, ...],
) -> tuple[float, float]:
    frequencies = tuple(float(value) for value in crossover_frequencies_hz)
    normalized_way = str(way).strip()
    ordered_ways = MODE_WAYS.get(str(mode_key), ())
    if not ordered_ways or len(frequencies) != len(ordered_ways) - 1:
        return 20.0, 0.0
    if normalized_way not in ordered_ways:
        return 20.0, 0.0
    index = ordered_ways.index(normalized_way)
    highpass = 20.0 if index == 0 else frequencies[index - 1]
    lowpass = 0.0 if index == len(ordered_ways) - 1 else frequencies[index]
    return highpass, lowpass


def _way_iir_cutoffs(
    way: str,
    mode_key: str,
    crossover_frequencies_hz: tuple[float, ...],
) -> tuple[float, float]:
    frequencies = tuple(float(value) for value in crossover_frequencies_hz)
    ways = MODE_WAYS.get(str(mode_key), ())
    if str(way) not in ways or len(frequencies) != max(0, len(ways) - 1):
        return 0.0, 0.0
    index = ways.index(str(way))
    return (
        0.0 if index == 0 else frequencies[index - 1],
        0.0 if index == len(ways) - 1 else frequencies[index],
    )


def _phaseeq_channel_fir(row, sample_rate_hz):
    from composite_engine.dc_normalization import normalize_correction_fir
    source = row["phaseeq_fir_response"]
    values = _fir_coefficients(str(source["filename"]), bytes(source["data"]), int(sample_rate_hz))
    return normalize_correction_fir(
        values, enabled=bool(row.get("dc_gain_normalize", False)),
        target_dc_abs=source.get("target_dc_abs"),
    )


def _build_studio_pipelines(
    final_firs: dict[str, object], sample_rate_hz: int,
) -> MultichannelCompositeResult:
    configured = st.session_state.get("composite_studio_channels", [])
    band_fir_states = st.session_state.get("composite_studio_band_fir_states", {})
    pending = [str(row.get("name", "")) for row in configured
               if isinstance(row, dict) and row.get("enabled", True) and row.get("phaseeq_link_pending")]
    if pending:
        raise ValueError(ui_message("ui.1379b62f3433eb") + ", ".join(pending))
    inputs: list[ChannelPipelineInput] = []
    generated_rows = [
        row for row in configured
        if isinstance(row, dict) and row.get("source") == "generated_band"
        and str(row.get("band", "")) in final_firs
    ]
    if not generated_rows:
        generated_rows = [
            {
                "band": band, "name": band, "way": band, "group": "Main",
                "gain_db": 0.0, "polarity": 1, "delay_samples": 0.0,
            }
            for band in final_firs
        ]
    timing_projection = resolve_speaker_timing_projection(
        generated_rows, st.session_state.get("settings", {}),
        sample_rate_hz=int(sample_rate_hz),
    )
    for row in generated_rows:
        band = str(row["band"])
        coefficients = final_firs[band]
        fir_state = (
            band_fir_states.get(band, {})
            if isinstance(band_fir_states, dict) else {}
        )
        fir_enabled = bool(fir_state.get("enabled", False))
        if not bool(row.get("enabled", True)):
            continue
        from scipy.io import wavfile
        buffer = io.BytesIO()
        wavfile.write(buffer, int(sample_rate_hz), coefficients)
        stages: list[ResponseStage] = []
        for stage_name, upload_key in (("IIR", "iir_response"),):
            upload = row.get(upload_key)
            if upload is not None:
                stages.append(ResponseStage(
                    stage_name,
                    f"{band}_{stage_name.casefold()}_{Path(str(upload.name)).name}",
                    upload.getvalue(),
                ))
        speaker = resolve_speaker_source(row)
        if speaker is not None:
            stages.append(ResponseStage(
                speaker.label, speaker.filename, speaker.data,
            ))
        phaseeq_fir = row.get("phaseeq_fir_response")
        if fir_enabled and isinstance(phaseeq_fir, dict):
            normalized_buffer = io.BytesIO()
            wavfile.write(normalized_buffer, int(sample_rate_hz), _phaseeq_channel_fir(row, sample_rate_hz))
            stages.append(ResponseStage(
                "PhaseEQ FIR EQ",
                f"{band}_phaseeq_{Path(str(phaseeq_fir['filename'])).name}",
                normalized_buffer.getvalue(),
            ))
        if fir_enabled:
            stages.append(ResponseStage("Band FIR", f"studio_{band}.wav", buffer.getvalue()))
        additional_fir = row.get("additional_fir")
        if fir_enabled and additional_fir is not None:
            stages.append(_fir_stage_from_upload("Additional FIR", additional_fir, sample_rate_hz))
        speaker_relative_timing_samples = (
            speaker_timing_samples_for_row(timing_projection, row)
            if speaker is not None else 0.0
        )
        output_iir, output_polarity, output_allpass = alignment_target.output_settings(row)
        inputs.append(ChannelPipelineInput(
            name=str(row["name"]), way=str(row["way"]), group=str(row["group"]),
            stages=tuple(stages),
            speaker_relative_timing_samples=float(
                speaker_relative_timing_samples
            ),
            gain_db=float(row["gain_db"]), polarity=output_polarity,
            delay_samples=float(row["delay_samples"]),
            iir_crossover=output_iir,
            dsp_additional_delay_samples=float(row.get("dsp_additional_delay_samples", 0.0)),
            auto_alignment_delay_samples=float(row.get("auto_alignment_delay_samples", 0.0)),
            auto_alignment_allpass=output_allpass,
            iir_sos=(
                *tuple(row.get("phaseeq_iir_sos", ())),
                *tuple(row.get("baffle_iir_sos", ())),
            ),
        ))
    for index, row in enumerate(configured):
        if not isinstance(row, dict) or not bool(row.get("enabled", True)):
            continue
        source = row.get("source")
        if source == "upload":
            upload = row.get("upload")
            if upload is None:
                continue
            filename = f"input_{index}_{Path(str(upload.name)).name}"
            data = upload.getvalue()
        elif source == "exchange":
            filename = f"exchange_{index}_{Path(str(row['filename'])).name}"
            data = bytes(row["data"])
        else:
            continue
        inputs.append(ChannelPipelineInput(
            name=str(row["name"]), way=str(row["way"]), group=str(row["group"]),
            stages=(ResponseStage("Input", filename, data),), gain_db=float(row["gain_db"]),
            polarity=int(row["polarity"]), delay_samples=float(row["delay_samples"]),
        ))
    stereo_layout = (
        str(st.session_state.get("composite_studio_output_layout", "Mono")) == "Stereo"
    )
    expected_groups = ("Left", "Right") if stereo_layout else ("Main",)
    if inputs:
        multichannel = build_channel_pipelines(int(sample_rate_hz), inputs)
        groups = dict(multichannel.groups)
        sample_rate = int(multichannel.sample_rate_hz)
        reference = next(iter(groups.values()))
        fft_size = int(reference.fft_size)
        frequency = np.asarray(reference.frequency_hz, dtype=float)
    else:
        sample_rate = int(sample_rate_hz)
        fft_size = sample_rate if sample_rate % 2 else sample_rate + 1
        frequency = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        groups = {}
    for group in expected_groups:
        if group not in groups:
            groups[group] = result_from_responses(
                group=group,
                sample_rate_hz=sample_rate,
                fft_size=fft_size,
                frequency_hz=frequency,
                channel_responses={},
            )
    return MultichannelCompositeResult(sample_rate_hz=sample_rate, groups=groups)


def _render_phase_alignment_controls(
    result, group: str, *, host: object | None = None, compact: bool = False,
) -> None:
    configured = st.session_state.get("composite_studio_channels", [])
    settings = st.session_state.get("settings", {})
    mode = str(settings.get("last_mode", "")) if isinstance(settings, dict) else ""
    ways = MODE_WAYS.get(mode, ())
    mode_settings = settings.get(mode, {}) if isinstance(settings, dict) else {}
    crossovers = tuple(float(value) for value in mode_settings.get("cross_freqs", ()))
    method_prefix = f"composite_studio_{mode}"
    crossover_methods = tuple(
        normalize_crossover_method(
            st.session_state.get(f"{method_prefix}_crossover_method_{index}")
        )
        for index in range(len(crossovers))
    )
    rows = {
        str(row.get("way", "")): row for row in configured
        if isinstance(row, dict) and bool(row.get("enabled", True))
        and str(row.get("group", "")) == group
        and row.get("source") == "generated_band"
    }
    if not ways or any(way not in rows for way in ways) or len(crossovers) != len(ways) - 1:
        return
    response_names = tuple(str(rows[way]["name"]) for way in ways)
    if any(name not in result.channel_responses for name in response_names):
        return
    alignment_input_signature = studio_dsp_input_signature(
        [rows[way] for way in ways],
        sample_rate_hz=int(result.sample_rate_hz),
        crossover_frequencies_hz=crossovers,
        crossover_methods=crossover_methods,
    )
    studio_graphs_current = bool(
        st.session_state.get("_studio_graphs_current", False)
    )
    location = (
        host.container(border=True)
        if host is not None
        else st.container(border=not compact)
    )
    with location:
        if compact:
            st.markdown(ui_message('ui.061bd94945ee5b', p0=f'{group}'))
        else:
            st.markdown(ui_message('ui.e3b303ed848228'))
            st.caption(
                ui_message('ui.109c38fb46d62a')
            )
            st.caption(
                ui_message('ui.a34b91abc0a700')
            )
        speaker_inputs = {way: resolve_speaker_source(rows[way]) for way in ways}
        speaker_readiness = assess_speaker_alignment_readiness(
            [rows[way] for way in ways]
        )
        st.caption(
            ui_message('ui.d1cdc6e2a210fc') + " / ".join(
                (
                    f"{way}={source.label}"
                    + (f" (rev {source.revision})" if source.revision else "")
                ) if source is not None else f"{way}=Unity"
                for way, source in speaker_inputs.items()
            )
        )
        with st.container(horizontal=True):
            window = shared_segmented_control(
                ui_message('ui.6424a0ed1f8dc8'), [0.25, 0.5, 1.0], default=0.5,
                format_func=localized_formatter(lambda value: f'±{float(value):g} oct'),
                key=f"composite_alignment_window_{group}", required=True,
            ) or 0.5
            capability = shared_segmented_control(
                ui_message('ui.e7f485aab4ae1f'), ["使用不可", "理論配置"], default="理論配置",
                key=f"composite_alignment_ap_limit_{group}", required=True,
            format_func=localized_formatter(str)) or "理論配置"
        apply_requested = st.button(
            ui_message('ui.bcb4ec4d207493'), type="primary", icon=":material/check:",
            key=f"composite_alignment_apply_{group}", width="stretch",
            disabled=not studio_graphs_current or not speaker_readiness.ready
            or all(method == "Through" for method in crossover_methods),
        )
        if not speaker_readiness.ready:
            st.error(
                ui_message('ui.35f73725a56aee', p0=f"{' / '.join(speaker_readiness.unity_channels)}")
            )
        if crossover_methods and all(method == "Through" for method in crossover_methods):
            st.caption(ui_message('ui.61259b99a7dd04'))
        if apply_requested and any(method != "Through" for method in crossover_methods):
            previous_delays = {
                str(rows[way]["name"]): float(
                    rows[way].get("auto_alignment_delay_samples", 0.0)
                )
                for way in ways
            }
            previous_allpass = {
                str(rows[way]["name"]): tuple(
                    rows[way].get("auto_alignment_allpass", ())
                )
                for way in ways
            }
            disable_allpass = str(capability) == "使用不可"
            analysis_responses = {
                name: result.channel_responses[name].copy() for name in response_names
            }
            removed_allpass: dict[str, tuple[AllPassSection, ...]] = {}
            removed_delays: dict[str, float] = {}
            # Re-analysis is absolute and deterministic: remove only the
            # previously applied automatic alignment stages from the realized
            # response, then solve again from the same pre-alignment baseline.
            # Manual Delay, Speaker, PhaseEQ, Gain and Polarity remain active.
            for name in response_names:
                sections = previous_allpass.get(name, ())
                if sections:
                    analysis_responses[name] = analysis_responses[name] / allpass_response(
                        sections, result.frequency_hz, result.sample_rate_hz,
                    )
                    removed_allpass[name] = sections
                delay = float(previous_delays.get(name, 0.0))
                if delay:
                    analysis_responses[name] = analysis_responses[name] * np.exp(
                        1j * 2.0 * np.pi * result.frequency_hz
                        * delay / float(result.sample_rate_hz)
                    )
                    removed_delays[name] = delay
            residual_proposal = analyze_phase_alignment(
                result.frequency_hz,
                analysis_responses,
                response_names,
                crossovers,
                result.sample_rate_hz,
                window_oct=float(window),
                # The current response already contains applied All-pass. Do not
                # propose the same structural sections again on a residual run.
                allpass_limit=(
                    0 if disable_allpass else max(1, len(ways) - 2)
                ),
                crossover_methods=crossover_methods,
            )
            proposal = residual_proposal
            by_channel_name = {
                str(row["name"]): row for row in rows.values()
            }
            resource_warnings = []
            for item in proposal.channels:
                row = by_channel_name[item.channel]
                phaseeq_count = len([
                    value for value in row.get("phaseeq_iir", [])
                    if isinstance(value, dict)
                ])
                crossover_count = int(iir_crossover_sos(
                    row.get("iir_crossover", IIRCrossoverConfig()),
                    result.sample_rate_hz,
                ).shape[0])
                baffle_count = len(tuple(row.get("baffle_iir_sos", ())) or ())
                total = phaseeq_count + baffle_count + crossover_count + len(
                    item.allpass_sections
                )
                if total > 8:
                    resource_warnings.append(
                        f"{item.channel}: miniDSP想定の8 biquadを超えます "
                        f"({total}/8)"
                    )
            if resource_warnings:
                proposal = replace(
                    proposal,
                    warnings=tuple((*proposal.warnings, *resource_warnings)),
                )
            previous = {}
            by_name = {item.channel: item for item in proposal.channels}
            for way in ways:
                row = rows[way]
                prefix = _alignment_state_prefix(row)
                previous[prefix] = {
                    "delay": st.session_state.get(
                        f"{prefix}_auto_alignment_delay", 0.0,
                    ),
                    "allpass": st.session_state.get(
                        f"{prefix}_auto_alignment_allpass", [],
                    ),
                }
                item = by_name[str(row["name"])]
                st.session_state[f"{prefix}_auto_alignment_delay"] = (
                    item.dsp_delay_samples
                )
                st.session_state[f"{prefix}_auto_alignment_allpass"] = [
                    section.to_dict() for section in item.allpass_sections
                ]
            st.session_state[f"composite_alignment_undo_{group}"] = previous
            st.session_state[f"composite_alignment_undo_applied_{group}"] = (
                st.session_state.get(f"composite_alignment_applied_{group}")
            )
            st.session_state[f"composite_alignment_applied_{group}"] = proposal
            for suffix in (
                "preview", "residual", "removed_allpass", "removed_delays",
                "signature", "caution_confirmed",
            ):
                st.session_state.pop(f"composite_alignment_{suffix}_{group}", None)
            st.session_state["_alignment_refresh_requested"] = True
            st.rerun()
        if not studio_graphs_current:
            st.warning(
                ui_message('ui.9b5d026f2be20d')
            )
        # Discard previews left by the former Analyze -> Apply workflow. New
        # requests are analyzed and committed atomically by the Apply button.
        for suffix in (
            "preview", "residual", "removed_allpass", "removed_delays",
            "signature", "caution_confirmed",
        ):
            st.session_state.pop(f"composite_alignment_{suffix}_{group}", None)
        proposal = None
        if proposal is None:
            applied = st.session_state.get(f"composite_alignment_applied_{group}")
            if applied is not None:
                st.success(
                    ui_message('ui.7ae5411cb489a1', p0=f'{applied.reference_channel}', p1=f'{applied.confidence}')
                )
                with st.expander(ui_message('ui.36a75076c98897'), expanded=False):
                    st.caption(
                        ui_message('ui.09e5eee1d04765', p0=f'{format_for_display(applied.common_offset_samples, 2)}')
                    )
                    if applied.warnings:
                        st.warning("\n".join(applied.warnings))
                    st.dataframe(pd.DataFrame([
                        {
                            "Channel": item.channel,
                            "低域基準Delay [samples]": item.relative_delay_samples,
                            "DSP追加Delay [samples]": item.dsp_delay_samples,
                            "All-pass": " / ".join(
                                (
                                    f"{format_for_display(section.frequency_hz, 1)} Hz, 1st"
                                    f"{' + 180°' if section.polarity == -1 else ''}"
                                    if section.order == 1
                                    else (
                                        f"{format_for_display(section.frequency_hz, 1)} Hz, "
                                        f"Q {format_for_display(section.q, 2)}"
                                    )
                                )
                                for section in item.allpass_sections
                            ) or "未使用",
                        }
                        for item in applied.channels
                    ]), hide_index=True, width="stretch")
                    st.dataframe(pd.DataFrame([
                        {
                            "Boundary": f"{item.lower} / {item.upper}",
                            "Fc [Hz]": item.crossover_hz,
                            "解析範囲 [Hz]": (
                                f"{format_for_display(item.analysis_low_hz, 1)}–"
                                f"{format_for_display(item.analysis_high_hz, 1)}"
                            ),
                            "Lower傾斜 [ms]": item.lower_phase_slope_ms,
                            "Upper傾斜 [ms]": item.upper_phase_slope_ms,
                            "補正後群遅延差 [ms]": (
                                item.aligned_group_delay_difference_ms
                            ),
                            "All-pass判定": (
                                item.allpass_decision or "判定情報なし"
                            ),
                            "Status": item.caution or "良好",
                        }
                        for item in applied.boundaries
                    ]), hide_index=True, width="stretch")
            undo = st.session_state.get(f"composite_alignment_undo_{group}")
            if st.button(
                ui_message('ui.f94a6aadd5e4b4'), icon=":material/undo:",
                disabled=not isinstance(undo, dict),
                key=f"composite_alignment_undo_idle_{group}",
            ):
                for prefix, values in undo.items():
                    st.session_state[f"{prefix}_auto_alignment_delay"] = values["delay"]
                    st.session_state[f"{prefix}_auto_alignment_allpass"] = values["allpass"]
                st.session_state.pop(f"composite_alignment_undo_{group}", None)
                st.session_state.pop(f"composite_alignment_undo_applied_{group}", None)
                st.session_state.pop(f"composite_alignment_applied_{group}", None)
                st.session_state["_alignment_refresh_requested"] = True
                st.rerun()
            return
        st.caption(
            ui_message('ui.3b60f90b89fb02', p0=f'{proposal.reference_channel}', p1=f'{proposal.confidence}', p2=f'{format_for_display(proposal.common_offset_samples, 2)}')
        )
        preview_signature = str(
            st.session_state.get(f"composite_alignment_signature_{group}", "")
        )
        preview_is_current = bool(
            studio_graphs_current
            and preview_signature
            and preview_signature == alignment_input_signature
        )
        if not preview_is_current:
            st.error(
                ui_message('ui.bf881dce26b990')
            )
        st.dataframe(pd.DataFrame([
            {
                "Boundary": f"{item.lower} / {item.upper}",
                "Fc [Hz]": item.crossover_hz,
                "解析範囲 [Hz]": (
                    f"{format_for_display(item.analysis_low_hz, 1)}–"
                    f"{format_for_display(item.analysis_high_hz, 1)}"
                ),
                "Lower傾斜 [ms]": item.lower_phase_slope_ms,
                "Upper傾斜 [ms]": item.upper_phase_slope_ms,
                "Lower [dB]": item.lower_level_db,
                "Upper [dB]": item.upper_level_db,
                "Difference [dB]": item.level_difference_db,
                "Valid points [%]": item.valid_fraction * 100.0,
                "All-pass判定": item.allpass_decision or "判定情報なし",
                "Status": item.caution or "良好",
            }
            for item in proposal.boundaries
        ]), hide_index=True, width="stretch")
        by_channel_name = {str(row["name"]): row for row in rows.values()}
        resource_warnings = []
        biquad_usage = {}
        for item in proposal.channels:
            row = by_channel_name[item.channel]
            phaseeq_count = len([
                value for value in row.get("phaseeq_iir", []) if isinstance(value, dict)
            ])
            crossover_count = int(iir_crossover_sos(
                row.get("iir_crossover", IIRCrossoverConfig()), result.sample_rate_hz,
            ).shape[0])
            baffle_count = len(tuple(row.get("baffle_iir_sos", ())) or ())
            total = phaseeq_count + baffle_count + crossover_count + len(item.allpass_sections)
            biquad_usage[item.channel] = (total, phaseeq_count, crossover_count)
            if total > 8:
                resource_warnings.append(
                    f"{item.channel}: miniDSP想定の8 biquadを超えます ({total}/8)"
                )
        has_caution = bool(proposal.warnings or resource_warnings)
        if has_caution:
            st.warning("\n".join((*proposal.warnings, *resource_warnings)))
            confirmed = st.checkbox(
                ui_message('ui.7192074f9bd5f4'),
                key=f"composite_alignment_caution_confirmed_{group}",
            )
        else:
            confirmed = True
        st.dataframe(pd.DataFrame([
            {
                "Channel": item.channel,
                "低域基準Delay [samples]": item.relative_delay_samples,
                "DSP追加Delay [samples]": item.dsp_delay_samples,
                "All-pass": " / ".join(
                    (
                        f"{format_for_display(section.frequency_hz, 1)} Hz, 1st"
                        f"{' + 180°' if section.polarity == -1 else ''}"
                        if section.order == 1
                        else (
                            f"{format_for_display(section.frequency_hz, 1)} Hz, "
                            f"Q {format_for_display(section.q, 2)}"
                        )
                    )
                    for section in item.allpass_sections
                ) or "—",
                "Biquad": f"{biquad_usage[item.channel][0]} / 8",
            }
            for item in proposal.channels
        ]), hide_index=True, width="stretch")
        residual = st.session_state.get(
            f"composite_alignment_residual_{group}", proposal,
        )
        removed_allpass = st.session_state.get(
            f"composite_alignment_removed_allpass_{group}", {},
        )
        removed_delays = st.session_state.get(
            f"composite_alignment_removed_delays_{group}", {},
        )
        preview_responses = {}
        for item in residual.channels:
            response = result.channel_responses[item.channel]
            old_sections = tuple(removed_allpass.get(item.channel, ()))
            if old_sections:
                response = response / allpass_response(
                    old_sections, result.frequency_hz, result.sample_rate_hz,
                )
            old_delay = float(removed_delays.get(item.channel, 0.0))
            if old_delay:
                response = response * np.exp(
                    1j * 2.0 * np.pi * result.frequency_hz
                    * old_delay / float(result.sample_rate_hz)
                )
            response = response * allpass_response(
                item.allpass_sections, result.frequency_hz, result.sample_rate_hz,
            )
            response = response * np.exp(
                -1j * 2.0 * np.pi * result.frequency_hz
                * item.dsp_delay_samples / float(result.sample_rate_hz)
            )
            preview_responses[item.channel] = response
        with st.container(horizontal=True):
            if st.button(
                ui_message('ui.52ac6fb621acbc'), type="primary", icon=":material/check:",
                disabled=not bool(confirmed) or not preview_is_current,
                key=f"composite_alignment_apply_{group}",
            ):
                previous = {}
                by_name = {item.channel: item for item in proposal.channels}
                for way in ways:
                    row = rows[way]
                    prefix = _alignment_state_prefix(row)
                    previous[prefix] = {
                        "delay": st.session_state.get(f"{prefix}_auto_alignment_delay", 0.0),
                        "allpass": st.session_state.get(f"{prefix}_auto_alignment_allpass", []),
                    }
                    item = by_name[str(row["name"])]
                    st.session_state[f"{prefix}_auto_alignment_delay"] = item.dsp_delay_samples
                    st.session_state[f"{prefix}_auto_alignment_allpass"] = [
                        section.to_dict() for section in item.allpass_sections
                    ]
                st.session_state[f"composite_alignment_undo_{group}"] = previous
                st.session_state[f"composite_alignment_undo_applied_{group}"] = (
                    st.session_state.get(f"composite_alignment_applied_{group}")
                )
                st.session_state[f"composite_alignment_applied_{group}"] = proposal
                st.session_state.pop(f"composite_alignment_preview_{group}", None)
                st.session_state.pop(f"composite_alignment_residual_{group}", None)
                st.session_state.pop(f"composite_alignment_removed_allpass_{group}", None)
                st.session_state.pop(f"composite_alignment_removed_delays_{group}", None)
                st.session_state.pop(f"composite_alignment_signature_{group}", None)
                st.session_state["_alignment_refresh_requested"] = True
                st.rerun()
            if st.button(
                ui_message('ui.bca84ea5c65fee'), icon=":material/close:",
                key=f"composite_alignment_cancel_{group}",
            ):
                st.session_state.pop(f"composite_alignment_preview_{group}", None)
                st.session_state.pop(f"composite_alignment_residual_{group}", None)
                st.session_state.pop(f"composite_alignment_removed_allpass_{group}", None)
                st.session_state.pop(f"composite_alignment_removed_delays_{group}", None)
                st.session_state.pop(f"composite_alignment_signature_{group}", None)
                st.session_state["_alignment_refresh_requested"] = True
                st.rerun()
            undo = st.session_state.get(f"composite_alignment_undo_{group}")
            if st.button(
                ui_message('ui.dc65bd3603e822'), icon=":material/undo:", disabled=not isinstance(undo, dict),
                key=f"composite_alignment_undo_button_{group}",
            ):
                for prefix, values in undo.items():
                    st.session_state[f"{prefix}_auto_alignment_delay"] = values["delay"]
                    st.session_state[f"{prefix}_auto_alignment_allpass"] = values["allpass"]
                st.session_state.pop(f"composite_alignment_undo_{group}", None)
                st.session_state.pop(f"composite_alignment_undo_applied_{group}", None)
                st.session_state.pop(f"composite_alignment_applied_{group}", None)
                st.session_state["_alignment_refresh_requested"] = True
                st.rerun()


def _system_alignment_context(
    multichannel: MultichannelCompositeResult,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[AlignmentBranch, ...],
    dict[str, np.ndarray],
    np.ndarray,
    int,
] | None:
    configured = st.session_state.get("composite_studio_channels", [])
    settings = st.session_state.get("settings", {})
    mode = str(settings.get("last_mode", "")) if isinstance(settings, dict) else ""
    ways = MODE_WAYS.get(mode, ())
    if len(ways) < 2:
        return None
    generated = [
        row for row in configured
        if isinstance(row, dict)
        and bool(row.get("enabled", True))
        and row.get("source") == "generated_band"
    ]
    if not generated:
        return None
    groups = {str(row.get("group", "")) for row in generated}
    branch_groups = (
        tuple(group for group in ("Left", "Right") if group in groups)
        if groups.intersection({"Left", "Right"})
        else ("Main",) if "Main" in groups else ()
    )
    if not branch_groups:
        return None
    rows_by_group_way = {
        (str(row.get("group", "")), str(row.get("way", ""))): row
        for row in generated
    }
    responses: dict[str, np.ndarray] = {}
    physical_rows: dict[str, dict[str, object]] = {}
    branches: list[AlignmentBranch] = []
    frequency: np.ndarray | None = None
    sample_rate = 0
    for group in branch_groups:
        if group not in multichannel.groups:
            return None
        result = _result_with_shared_sub(multichannel, group)
        frequency = result.frequency_hz
        sample_rate = int(result.sample_rate_hz)
        ordered_ids: list[str] = []
        for way in ways:
            row = rows_by_group_way.get((group, way))
            if row is None and way == "SUB":
                row = rows_by_group_way.get(("Sub", "SUB"))
            if row is None:
                return None
            name = str(row.get("name", ""))
            if name not in result.channel_responses:
                return None
            channel_id = str(row.get("channel_id", "")).strip() or _alignment_state_prefix(row)
            if channel_id in responses:
                if not np.array_equal(responses[channel_id], result.channel_responses[name]):
                    raise ValueError(f"{name}: shared Channel response is inconsistent")
            else:
                responses[channel_id] = result.channel_responses[name].copy()
                physical_rows[channel_id] = row
            ordered_ids.append(channel_id)
        branches.append(AlignmentBranch(group, tuple(ordered_ids)))
    if frequency is None:
        return None
    return (
        tuple(physical_rows[channel_id] for channel_id in physical_rows),
        tuple(branches), responses, frequency, sample_rate,
    )


def _render_alignment_target_control(mode):
    settings = st.session_state.get("settings", {})
    automatic_key = "composite_alignment_auto_structure"
    st.session_state[automatic_key] = bool(settings.get("phase_alignment_auto_structure", False))
    def automatic_changed():
        current = dict(st.session_state.get("settings", {}))
        current["phase_alignment_auto_structure"] = bool(st.session_state[automatic_key])
        st.session_state["settings"] = current
        st.session_state["_alignment_refresh_requested"] = True
    st.checkbox(display_text("位相構造補償を自動適用"), key=automatic_key,
                on_change=automatic_changed,
                help=display_text("ONではWay・分割方式・分割周波数に合わせて理論配置を更新します。適用ボタンは時間整合だけを実行します。"))
    selected = bool(settings.get(mode, {}).get("phase_alignment_acoustic_target", False))
    key = f"composite_alignment_target_{mode}"
    st.session_state[key] = selected
    def changed():
        current = dict(st.session_state.get("settings", {}))
        current[mode] = dict(current.get(mode, {}), phase_alignment_acoustic_target=bool(st.session_state[key]))
        if current[mode]["phase_alignment_acoustic_target"]:
            current[mode]["fir_output_enabled"] = True
            st.session_state[f"fir_output_enabled_{mode}"] = True
        st.session_state["settings"] = current
    st.checkbox(ui_message('ui.f0531283f6c414'), key=key, on_change=changed,
                help=ui_message('ui.051c94b46a54d8'))


def _render_system_phase_alignment_controls(
    multichannel: MultichannelCompositeResult,
    *,
    host: object | None = None,
) -> None:
    settings = st.session_state.get("settings", {})
    mode = str(settings.get("last_mode", "")) if isinstance(settings, dict) else ""
    ways = MODE_WAYS.get(mode, ())
    # Fullrange is a deliberate bypass: one Channel has no relative boundary.
    if len(ways) < 2:
        return
    location = host.container(border=True) if host is not None else st.container(border=True)
    try:
        context = _system_alignment_context(multichannel)
    except ValueError as exc:
        with location:
            st.markdown(ui_message('ui.e3b303ed848228'))
            _render_alignment_target_control(mode)
            st.error(str(exc))
        return
    with location:
        st.markdown(ui_message('ui.e3b303ed848228'))
        _render_alignment_target_control(mode)
        if context is None:
            st.warning(ui_message('ui.360d62475abddc'))
            return
        rows, branches, responses, frequency, sample_rate = context
        automatic_structure = bool(settings.get("phase_alignment_auto_structure", False))
        mode_settings = settings.get(mode, {}) if isinstance(settings, dict) else {}
        crossovers = tuple(float(value) for value in mode_settings.get("cross_freqs", ()))
        if len(crossovers) != len(ways) - 1:
            st.warning(ui_message('ui.dacda49e1b312d'))
            return
        method_prefix = f"composite_studio_{mode}"
        crossover_methods = tuple(
            normalize_crossover_method(
                st.session_state.get(f"{method_prefix}_crossover_method_{index}")
            )
            for index in range(len(crossovers))
        )
        current_signature = studio_dsp_input_signature(
            list(rows), sample_rate_hz=sample_rate,
            crossover_frequencies_hz=crossovers,
            crossover_methods=crossover_methods,
        )
        shared_sub = any(
            sum(channel_id in branch.ordered_channels for branch in branches) > 1
            for channel_id in responses
        )
        st.caption(
            ui_message('ui.1774241f740248', p0=f'{mode}', p1=f"{' / '.join((branch.name for branch in branches))}", p2=f'{len(rows)}')
            + (ui_message('ui.002fc429374d4c') if shared_sub else "")
        )
        st.caption(
            ui_message('ui.e83e0e331be332')
        )
        st.caption(
            ui_message('ui.d1cdc6e2a210fc') + " / ".join(
                f"{row.get('name', row.get('way', 'Channel'))}="
                + (
                    source.label + (f" (rev {source.revision})" if source.revision else "")
                    if (source := resolve_speaker_source(row)) is not None else "Unity"
                )
                for row in rows
            )
        )
        speaker_readiness = assess_speaker_alignment_readiness(list(rows))
        if not speaker_readiness.ready:
            st.error(
                ui_message('ui.dcace7d9a8cda7', p0=f"{' / '.join(speaker_readiness.unity_channels)}")
            )
        elif speaker_readiness.mode == "filter_only":
            st.info(
                ui_message('ui.94eb04e4f240fb')
            )
        distance_config = (
            settings.get("external_distance_timing", {})
            if isinstance(settings, dict) else {}
        )
        if not isinstance(distance_config, dict):
            distance_config = {}
        timing_source_key = "composite_alignment_system_timing_source"
        st.session_state.setdefault(
            timing_source_key,
            "外部測定距離"
            if bool(distance_config.get("enabled", False)) else "自動選択",
        )
        timing_source = shared_segmented_control(
            ui_message('ui.2564d9c619587f'), ["自動選択", "外部測定距離"],
            key=timing_source_key, width="stretch", required=True,
        format_func=localized_formatter(str)) or "自動選択"
        stored_distance_rows = distance_config.get("channels", [])
        if not isinstance(stored_distance_rows, list):
            stored_distance_rows = []
        distance_by_id = {
            str(item.get("channel_id", "")): float(item.get("distance_m", 0.0))
            for item in stored_distance_rows if isinstance(item, dict)
        }
        distance_by_identity = {
            (
                str(item.get("name", "")), str(item.get("way", "")),
                str(item.get("group", "")),
            ): float(item.get("distance_m", 0.0))
            for item in stored_distance_rows if isinstance(item, dict)
        }
        external_distances: dict[str, float] = {}
        missing_distance_names: list[str] = []
        distance_rows_to_save: list[dict[str, object]] = []
        if timing_source == "外部測定距離":
            with st.expander(ui_message('ui.440fec6154300e'), expanded=True):
                st.caption(
                    ui_message('ui.51e534b0b75298', p0=f'{NOMINAL_SOUND_SPEED_M_S:.2f}')
                )
                way_rank = {
                    "SUB": 0, "Low": 1, "Low-Mid": 2, "Mid": 2,
                    "High-Mid": 3, "High": 4, "Fullrange": 4,
                }
                ordered_distance_rows = sorted(
                    rows,
                    key=lambda item: way_rank.get(str(item.get("way", "")), 0),
                    reverse=True,
                )
                ordered_distance_ids = tuple(
                    str(row.get("channel_id", "")).strip()
                    or _alignment_state_prefix(row)
                    for row in ordered_distance_rows
                )
                with st.container(border=True):
                    st.markdown(ui_message('ui.031cf450988234'))
                    st.caption(
                        ui_message('ui.1133ebf5a7ced6')
                        + " / ".join(
                            str(row.get("name", row.get("way", "Channel")))
                            for row in ordered_distance_rows
                        )
                    )
                    st.text_input(
                        ui_message('ui.f882c2906007e1'), key="composite_alignment_external_distance_bulk",
                        placeholder=ui_message('ui.a1e8878b4a8422'),
                    )
                    st.button(
                        ui_message('ui.74f6f5e6807c13'), icon=":material/input:", width="stretch",
                        key="composite_alignment_external_distance_bulk_apply",
                        on_click=_apply_external_distance_bulk,
                        args=(ordered_distance_ids,),
                    )
                    bulk_error = st.session_state.get(
                        "composite_alignment_external_distance_bulk_error"
                    )
                    if bulk_error:
                        st.error(str(bulk_error))
                for row in ordered_distance_rows:
                    channel_id = (
                        str(row.get("channel_id", "")).strip()
                        or _alignment_state_prefix(row)
                    )
                    identity = (
                        str(row.get("name", "")), str(row.get("way", "")),
                        str(row.get("group", "")),
                    )
                    initial_distance = distance_by_id.get(
                        channel_id, distance_by_identity.get(identity, 0.0)
                    )
                    distance_key = f"composite_alignment_external_distance_{channel_id}"
                    st.session_state.setdefault(distance_key, initial_distance * 1_000.0)
                    distance_mm = st.number_input(
                        ui_message('ui.5a4d26afc1e1a2', p0=f"{row.get('name', row.get('way', 'Channel'))}"),
                        min_value=0.0, step=1.0, format="%.1f", key=distance_key,
                    )
                    distance_m = float(distance_mm) / 1_000.0
                    if distance_m <= 0.0:
                        missing_distance_names.append(
                            str(row.get("name", row.get("way", "Channel")))
                        )
                    external_distances[channel_id] = distance_m
                    distance_rows_to_save.append({
                        "channel_id": channel_id,
                        "name": identity[0], "way": identity[1], "group": identity[2],
                        "distance_m": distance_m,
                    })
                    row["external_distance_m"] = distance_m
                if missing_distance_names:
                    st.warning(
                        ui_message('ui.869e5a82888917') + " / ".join(missing_distance_names)
                    )
                else:
                    preview = external_distance_timing(
                        external_distances, sample_rate_hz=sample_rate,
                    )
                    reference_name = next(
                        (
                            str(item["name"]) for item in distance_rows_to_save
                            if str(item["channel_id"]) == preview.reference_channel
                        ),
                        preview.reference_channel,
                    )
                    st.caption(
                        ui_message('ui.3622ef91d270ea', p0=f'{reference_name}')
                    )
                    st.dataframe(pd.DataFrame([
                        {
                            "Channel": row_by_id_preview,
                            "距離 [mm]": preview.distances_m[channel_id] * 1_000.0,
                            "追加Delay [samples]": preview.delay_samples[channel_id],
                            "追加Delay [ms]": (
                                preview.delay_samples[channel_id] * 1_000.0 / sample_rate
                            ),
                        }
                        for channel_id, row_by_id_preview in (
                            (
                                str(item["channel_id"]),
                                str(item["name"]),
                            ) for item in distance_rows_to_save
                        )
                    ]), hide_index=True, width="stretch")
        updated_settings = dict(settings) if isinstance(settings, dict) else {}
        updated_settings["external_distance_timing"] = {
            "enabled": timing_source == "外部測定距離",
            "channels": distance_rows_to_save or stored_distance_rows,
        }
        st.session_state["settings"] = updated_settings
        current_signature = hashlib.sha256(
            b"\0".join((
                current_signature.encode("ascii"),
                str(timing_source).encode("utf-8"),
                json.dumps(
                    external_distances if timing_source == "外部測定距離" else {},
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8"),
            ))
        ).hexdigest()
        st.session_state["composite_alignment_system_current_signature"] = current_signature
        if (
            st.session_state.pop("composite_alignment_system_signature_pending", False)
            and st.session_state.get("composite_alignment_system_applied") is not None
        ):
            st.session_state["composite_alignment_system_signature"] = current_signature
        with st.container(horizontal=True):
            window = shared_segmented_control(
                ui_message('ui.6424a0ed1f8dc8'), [0.25, 0.5, 1.0], default=0.5,
                format_func=localized_formatter(lambda value: f'±{float(value):g} oct'),
                key="composite_alignment_system_window", required=True,
            ) or 0.5
            capability = shared_segmented_control(
                ui_message('ui.e7f485aab4ae1f'), ["使用不可", "理論配置"], default="理論配置",
                key="composite_alignment_system_ap_limit", required=True,
                disabled=automatic_structure,
            format_func=localized_formatter(str)) or "理論配置"
        if automatic_structure:
            st.caption(display_text("位相構造補償は理論配置で自動更新中です。適用ボタンでは時間整合とDelayだけを更新します。"))
        studio_graphs_current = bool(st.session_state.get("_studio_graphs_current", False))
        apply_requested = st.button(
            ui_message('ui.016b94cde34d1b'),
            type="primary", icon=":material/check:",
            key="composite_alignment_apply_System", width="stretch",
            disabled=(
                (not automatic_structure and all(method == "Through" for method in crossover_methods))
                or not studio_graphs_current
                or not speaker_readiness.ready
                or (timing_source == "外部測定距離" and bool(missing_distance_names))
            ),
        )
        if not studio_graphs_current:
            st.warning(
                ui_message('ui.9b5d026f2be20d')
            )
        row_by_id = {
            (str(row.get("channel_id", "")).strip() or _alignment_state_prefix(row)): row
            for row in rows
        }
        if not automatic_structure and crossover_methods and all(method == "Through" for method in crossover_methods):
            st.caption(ui_message('ui.61259b99a7dd04'))
        if apply_requested and (automatic_structure or any(method != "Through" for method in crossover_methods)):
            previous: dict[str, dict[str, object]] = {}
            analysis_responses = {
                channel_id: response.copy() for channel_id, response in responses.items()
            }
            for channel_id, row in row_by_id.items():
                prefix = _alignment_state_prefix(row)
                previous[prefix] = alignment_structure.previous_values(
                    st.session_state, prefix, row, time_only=automatic_structure,
                )
                analysis_responses[channel_id] = alignment_structure.analysis_response(
                    row, responses[channel_id], frequency, sample_rate,
                    time_only=automatic_structure,
                )
            try:
                proposal = analyze_system_phase_alignment(
                    frequency,
                    analysis_responses,
                    branches,
                    crossovers,
                    sample_rate,
                    window_oct=float(window),
                    allpass_limit=(
                        0 if automatic_structure or str(capability) == "使用不可" else max(1, len(ways) - 2)
                    ),
                    crossover_methods=crossover_methods,
                    timing_provenance_by_channel={
                        channel_id: row.get("timing_provenance")
                        for channel_id, row in row_by_id.items()
                    },
                    external_distances_m_by_channel=(
                        external_distances
                        if timing_source == "外部測定距離" else None
                    ),
                )
                if automatic_structure:
                    proposal = replace(proposal,
                        channels=tuple(replace(item, allpass_sections=tuple(
                            row_by_id[item.channel].get("auto_alignment_allpass", ())))
                            for item in proposal.channels),
                        boundaries=tuple(replace(item, allpass_decision=display_text(
                            "自動理論配置を維持（時間整合のみ更新）")) for item in proposal.boundaries))
                resource_warnings = []
                for item in proposal.channels:
                    row = row_by_id[item.channel]
                    phaseeq_count = len([
                        value for value in row.get("phaseeq_iir", [])
                        if isinstance(value, dict)
                    ])
                    crossover_count = int(iir_crossover_sos(
                        row.get("iir_crossover", IIRCrossoverConfig()), sample_rate,
                    ).shape[0])
                    baffle_count = len(tuple(row.get("baffle_iir_sos", ())) or ())
                    total = phaseeq_count + baffle_count + crossover_count + len(item.allpass_sections)
                    if total > 8:
                        resource_warnings.append(
                            f"{row.get('name', item.channel)}: miniDSP想定の8 biquadを超えます "
                            f"({total}/8)"
                        )
                if resource_warnings:
                    proposal = replace(
                        proposal,
                        warnings=tuple((*proposal.warnings, *resource_warnings)),
                    )
            except (KeyError, ValueError) as exc:
                st.error(ui_message('ui.0aabd9e0c4c6a8', p0=f'{exc}'))
            else:
                by_id = {item.channel: item for item in proposal.channels}
                for channel_id, row in row_by_id.items():
                    prefix = _alignment_state_prefix(row)
                    item = by_id[channel_id]
                    alignment_structure.apply_channel(
                        st.session_state, prefix, item, time_only=automatic_structure,
                    )
                st.session_state["composite_alignment_system_undo"] = previous
                st.session_state["composite_alignment_system_applied"] = proposal
                st.session_state["composite_alignment_system_signature_pending"] = True
                # Remove obsolete per-Group UI state after the atomic commit.
                for key in list(st.session_state):
                    if key.startswith("composite_alignment_applied_") or key.startswith(
                        "composite_alignment_undo_"
                    ):
                        st.session_state.pop(key, None)
                st.session_state["_alignment_refresh_requested"] = True
                st.rerun()
        applied = st.session_state.get("composite_alignment_system_applied")
        if applied is not None:
            applied_is_current = (
                st.session_state.get("composite_alignment_system_signature")
                == current_signature
            )
            if applied_is_current:
                st.success(
                    ui_message('ui.bc8a52438177fd', p0=f"{' / '.join(applied.branches)}", p1=f'{applied.confidence}')
                    + (
                        _alignment_timing_source_label(applied.timing_source)
                    )
                )
            else:
                st.warning(
                    ui_message('ui.12975524f9bd7f')
                )
            with st.expander(ui_message('ui.36a75076c98897'), expanded=False):
                st.caption(
                    ui_message('ui.09e5eee1d04765', p0=f'{format_for_display(applied.common_offset_samples, 2)}')
                )
                if applied.warnings:
                    st.warning("\n".join(applied.warnings))
                st.dataframe(pd.DataFrame([
                    {
                        "Channel": row_by_id[item.channel].get("name", item.channel),
                        "低域基準Delay [samples]": item.relative_delay_samples,
                        "DSP追加Delay [samples]": item.dsp_delay_samples,
                        "時間根拠": (
                            _alignment_timing_source_label(applied.timing_source)
                        ),
                        "All-pass": " / ".join(
                            f"{format_for_display(section.frequency_hz, 1)} Hz"
                            for section in item.allpass_sections
                        ) or "未使用",
                    }
                    for item in applied.channels
                    if item.channel in row_by_id
                ]), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame([
                    {
                        "Boundary": (
                            f"{row_by_id.get(item.lower, {}).get('name', item.lower)} / "
                            f"{row_by_id.get(item.upper, {}).get('name', item.upper)}"
                        ),
                        "Fc [Hz]": item.crossover_hz,
                        "補正後群遅延差 [ms]": item.aligned_group_delay_difference_ms,
                        "All-pass判定": item.allpass_decision or "判定情報なし",
                        "Status": item.caution or "良好",
                    }
                    for item in applied.boundaries
                ]), hide_index=True, width="stretch")
        undo = st.session_state.get("composite_alignment_system_undo")
        if st.button(
            ui_message('ui.4497e6c60590f9'), icon=":material/undo:",
            disabled=not isinstance(undo, dict),
            key="composite_alignment_undo_System",
        ):
            for prefix, values in undo.items():
                alignment_structure.undo_channel(
                    st.session_state, prefix, values, automatic=automatic_structure,
                )
            st.session_state.pop("composite_alignment_system_undo", None)
            st.session_state.pop("composite_alignment_system_applied", None)
            st.session_state.pop("composite_alignment_system_signature", None)
            st.session_state.pop("composite_alignment_system_signature_pending", None)
            st.session_state["_alignment_refresh_requested"] = True
            st.rerun()


def _result_with_shared_sub(
    multichannel: MultichannelCompositeResult, group: str,
):
    base = multichannel.groups[group]
    sub = multichannel.groups.get("Sub")
    if sub is None:
        return base
    responses = dict(base.channel_responses)
    responses.update(sub.channel_responses)
    unwrapped = dict(base.channel_unwrapped_phase_deg or {})
    unwrapped.update(sub.channel_unwrapped_phase_deg or {})
    return result_from_responses(
        group=group,
        sample_rate_hz=base.sample_rate_hz,
        fft_size=base.fft_size,
        frequency_hz=base.frequency_hz,
        channel_responses=responses,
        target_response=base.target_response,
        target_name=base.target_name,
        target_phase_available=base.target_phase_available,
        channel_unwrapped_phase_deg=unwrapped or None,
    )


def _physical_group_for_display(displayed_channel: object, *, stereo: bool) -> str:
    display = normalize_display_value(displayed_channel, stereo=stereo)
    if not stereo:
        return "Main"
    return "Right" if display == DISPLAY_RIGHT else "Left"


def _group_target_scope(group: object) -> str:
    """Map physical stereo Groups to their single shared Target scope."""
    normalized = str(group).strip()
    return "Stereo" if normalized in {"Left", "Right", "Stereo"} else "Main"


def _target_edit_session_matches_scope(session_group: object, scope: object) -> bool:
    return _group_target_scope(session_group) == _group_target_scope(scope)


def _migrate_legacy_stereo_group_target_state() -> None:
    """Seed shared Stereo state without deleting recoverable legacy L/R values."""
    for prefix in (
        "composite_studio_target_source_",
        "composite_studio_group_target_preset_",
        "_composite_resumed_group_target_",
        "_composite_group_target_edit_session_",
        "composite_studio_show_group_target_",
        "composite_studio_group_target_phase_enabled_",
        "composite_studio_group_target_gain_db_",
    ):
        shared_key = f"{prefix}Stereo"
        if shared_key in st.session_state:
            continue
        for legacy_group in ("Left", "Right"):
            legacy_key = f"{prefix}{legacy_group}"
            if legacy_key in st.session_state:
                st.session_state[shared_key] = st.session_state[legacy_key]
                break


def _sync_displayed_channel_widget(*, stereo: bool) -> None:
    """Publish the widget value before the next top-to-bottom Streamlit rerun."""
    st.session_state["composite_studio_result_group"] = normalize_display_value(
        st.session_state.get("composite_studio_result_group_widget"), stereo=stereo,
    )


def render_composite_results(
    *, alignment_host: object | None = None, output_timing_host: object | None = None,
    band_tap_count_hosts: dict | None = None,
) -> None:
    """Build Composite Groups only from the Studio's realized final FIRs."""
    if st.session_state.get("_history_pending_phaseeq") or st.session_state.get("_history_awaiting_return"):
        st.caption(display_text("復元したEQ設定をPhaseEQへ送り、再生成したフィルターを返却してください。"))
        return
    failures = st.session_state.get('_exchange_receive_failures', {})
    if failures:
        st.warning(display_text('一部の出力を読み込めません。前回結果を保持しています。'))
        if st.button(display_text('出力の受信を再試行'), key='retry_phaseeq_receive'):
            reset_retries(st.session_state)
            st.rerun()
    if st.session_state.pop("_phaseeq_result_auto_received", False):
        st.toast(
            display_text("PhaseEQの新しい出力を検出しました。検証に失敗した出力は適用しません。"),
            icon=":material/sync:",
        )
    final_firs = st.session_state.get("final_firs")
    sample_rate_hz = st.session_state.get("result_fs")
    if not isinstance(final_firs, dict) or not final_firs or not sample_rate_hz:
        return
    st.markdown("---")
    st.header(ui_message('ui.037aa0f825229c'))
    timing_status = resolve_speaker_timing_projection(
        st.session_state.get("composite_studio_channels", []),
        st.session_state.get("settings", {}), sample_rate_hz=int(sample_rate_hz),
    )
    if timing_status.source == "phaseeq_tweeter_reference" and timing_status.complete:
        st.caption(ui_message('ui.bc43359ff85fb8'))
    elif timing_status.source == "external_distance":
        st.caption(ui_message('ui.223bba3646c328') if timing_status.complete else timing_status.warning)
    elif not timing_status.complete:
        st.caption(ui_message('ui.7b4c77986c2dea'))
    st.caption(
        ui_message('ui.9090da477e73c4')
    )
    try:
        multichannel = _build_studio_pipelines(final_firs, int(sample_rate_hz))
    except (CompositeValidationError, ValueError, OSError) as exc:
        st.error(ui_message('ui.572d7b3cffb4c9', p0=f'{exc}'))
        return

    group_names = list(multichannel.group_names)
    stereo_groups = (
        str(st.session_state.get("composite_studio_output_layout", "Mono")) == "Stereo"
    )
    display_groups = [DISPLAY_LEFT, DISPLAY_RIGHT, DISPLAY_STEREO] if stereo_groups else [
        DISPLAY_MAIN
    ]
    requested_group = normalize_display_value(
        st.query_params.get("group", ""), stereo=stereo_groups,
    )
    group_state_key = "composite_studio_result_group"
    group_widget_key = "composite_studio_result_group_widget"
    state_has_selection = bool(str(st.session_state.get(group_state_key, "") or "").strip())
    selected_group = normalize_display_value(
        st.session_state.get(group_state_key, requested_group), stereo=stereo_groups,
    )
    if not state_has_selection:
        selected_group = requested_group
    st.session_state[group_state_key] = selected_group
    widget_display = normalize_display_value(
        st.session_state.get(group_widget_key, selected_group), stereo=stereo_groups,
    )
    if widget_display not in display_groups:
        st.session_state[group_widget_key] = selected_group
    else:
        st.session_state[group_widget_key] = widget_display
    display_group = shared_segmented_control(
        ui_message('ui.b980e52ee54c43'), display_groups,
        key=group_widget_key, width="stretch", required=True,
        on_change=_sync_displayed_channel_widget,
        kwargs={"stereo": stereo_groups},
    ) or selected_group
    display_group = normalize_display_value(display_group, stereo=stereo_groups)
    st.session_state[group_state_key] = str(display_group)
    group = _physical_group_for_display(display_group, stereo=stereo_groups)
    target_scope = _group_target_scope(group)
    _migrate_legacy_stereo_group_target_state()
    with st.sidebar, st.container(border=True):
        st.markdown(ui_message('ui.fcd588b91276fc'))
        st.caption(
            ui_message('ui.fe54eaac31af43')
        )
        if stereo_groups:
            st.info(ui_message('ui.8363137b914696'))
        target_source_key = f"composite_studio_target_source_{target_scope}"
        group_target_key = f"composite_studio_group_target_preset_{target_scope}"
        target_catalog = _group_target_catalog(target_scope)
        target_ids = [str(item["id"]) for item in target_catalog]
        if st.session_state.get(target_source_key, "PhaseEQ preset") not in {"PhaseEQ preset", "Global Target", "Built-in"}:
            # Keep the old file/source recoverable, but never silently apply it
            # as a registered preset under the new selection-only workflow.
            st.session_state[f"_legacy_target_source_{target_scope}"] = st.session_state[target_source_key]
            st.session_state[group_target_key] = ""
        st.session_state[target_source_key] = "PhaseEQ preset"
        target_source = "PhaseEQ preset"
        target_upload = None
        if group_target_key not in st.session_state:
            st.session_state[group_target_key] = st.session_state.get("settings", {}).get(
                "group_target_presets", {}).get(target_scope, target_ids[0] if target_ids else "")
        missing_target = st.session_state[group_target_key]
        if missing_target and missing_target not in target_ids:
            target_ids.append(missing_target)
            st.warning(ui_message('ui.f33815a5fbf552'))
        selected_group_target_id = _group_target_choice(
            "Targetプリセット", ["", *target_ids], group_target_key, target_scope,
            format_func=lambda value: next(
                (f"{item['name']} · {item.get('source', 'User')}" for item in target_catalog if str(item["id"]) == value),
                "登録が見つからないTarget" if value else "登録済みTargetを選択してください"),
        )
        st.caption(ui_message('ui.d6809c2e7fb373'))
        from ui.phaseeq_target_navigation import render_target_menu_button
        render_target_menu_button(_exchange_root())
        if st.button(ui_message('ui.4a7f36166f1363'), key=f"refresh_target_presets_{target_scope}"):
            st.rerun()
        if not selected_group_target_id:
            st.info(ui_message('ui.d003c571fe7514'))
        show_target = st.toggle(
            ui_message('ui.8da80afa22605f'),
            value=True,
            key=f"composite_studio_show_group_target_{target_scope}",
        )

    # Use the same registered definition for preview and Assignment publication.
    try:
        group_target_definition = _selected_group_target_definition(group, int(sample_rate_hz))
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        group_target_definition = None
        st.error(ui_message('ui.308d9b51b092fe', p0=f'{exc}'))
    from target_engine.realization import realized_target_frd_bytes
    from ui.target_preview import render_target_preview
    render_target_preview(group_target_definition, int(sample_rate_hz), key=f"group_target_preview_{target_scope}")

    target_input = None
    if group_target_definition is not None:
        target_input = GroupTargetInput(
            group=group,
            name=f"{group} Target · {group_target_definition.get('name', 'Target')}",
            filename=f"{group}_target_definition.frd",
            data=realized_target_frd_bytes(group_target_definition, int(sample_rate_hz)),
            minimum_phase=False,
            edit_payload={},
        )
        st.caption(
            ui_message('ui.b6c98d169091cd', p0=f"{group_target_definition.get('name', 'Target')}", p1=f"{str(group_target_definition.get('revision', ''))[:12]}")
        )
    configured = st.session_state.get("composite_studio_channels", [])
    target_inputs = [target_input] if target_input is not None else []
    target_definitions_by_group = (
        {group: dict(group_target_definition)}
        if isinstance(group_target_definition, dict) else {}
    )
    required_target_groups = (
        ("Left", "Right") if display_group == DISPLAY_STEREO else (group,)
    )
    for other_group in required_target_groups:
        if other_group == group:
            continue
        try:
            other_definition = _selected_group_target_definition(
                other_group, int(sample_rate_hz),
            )
        except (KeyError, OSError, UnicodeError, ValueError) as exc:
            st.warning(ui_message('ui.73ffd221b17907', p0=f'{other_group}', p1=f'{exc}'))
            continue
        if other_definition is None:
            continue
        target_definitions_by_group[other_group] = dict(other_definition)
        target_inputs.append(GroupTargetInput(
            group=other_group,
            name=f"{other_group} Target · {other_definition.get('name', 'Target')}",
            filename=f"{other_group}_target_definition.frd",
            data=realized_target_frd_bytes(other_definition, int(sample_rate_hz)),
            minimum_phase=False,
            edit_payload={},
        ))
    if target_inputs:
        try:
            multichannel = attach_group_targets(multichannel, tuple(target_inputs))
        except (CompositeValidationError, ValueError, OSError) as exc:
            st.error(ui_message('ui.224357c813a4f5', p0=f'{exc}'))
            return
    try:
        display_projection = resolve_display_projection(
            multichannel, configured, display_group,
        )
    except (CompositeValidationError, ValueError) as exc:
        st.error(ui_message('ui.b68e863ed37b07', p0=f'{exc}'))
        return
    result = display_projection.primary_result
    st.session_state["composite_studio_display_projection_signature"] = (
        display_projection.signature
    )
    with st.container(horizontal=True):
        st.metric(ui_message('ui.4711310cc2f6bf'), len(group_names), border=True)
        st.metric(ui_message('ui.4c8906cf76f574'), len(display_projection.channel_sources), border=True)
        st.metric(ui_message('ui.218e0d43275e43'), f"{result.sample_rate_hz:,} Hz", border=True)
        st.metric(ui_message('ui.94fa3fe96ddeef'), f"{result.fft_size:,}", border=True)

    _render_system_phase_alignment_controls(
        multichannel, host=alignment_host,
    )

    # Keep the selected view outside the widget-owned key. The alignment Apply
    # action reruns before this lower section is rendered; Streamlit otherwise
    # cleans up the absent widget and restores Magnitude on the next run.
    graph_state_key = "composite_studio_graph"
    graph_widget_key = "composite_studio_graph_widget"
    graph_choices = [
        "Magnitude", "Phase", "Group Delay", "Impulse", "Step", "Wavelet",
    ]
    saved_graph_kind = str(st.session_state.get(graph_state_key, "Magnitude"))
    if saved_graph_kind not in graph_choices:
        saved_graph_kind = "Magnitude"
    st.session_state.setdefault(graph_widget_key, saved_graph_kind)
    graph_kind = shared_segmented_control(
        ui_message('ui.32ab018fd38a7b'), graph_choices, key=graph_widget_key, width="stretch", required=True,
    ) or "Magnitude"
    st.session_state[graph_state_key] = graph_kind
    wrapped_phase = True
    if graph_kind == "Phase":
        wrapped_state_key = "composite_studio_wrapped_phase"
        unwrapped_state_key = "composite_studio_unwrapped_phase"
        unwrapped_widget_key = "composite_studio_unwrapped_phase_widget"
        legacy_wrapped = bool(st.session_state.get(wrapped_state_key, True))
        st.session_state.setdefault(
            unwrapped_widget_key,
            bool(st.session_state.get(unwrapped_state_key, not legacy_wrapped)),
        )
        unwrapped_phase = st.toggle(
            ui_message('ui.30cf8b4c8efa7a'), key=unwrapped_widget_key,
            help=(
                ui_message('ui.f94980c6076700')
            ),
        )
        wrapped_phase = not bool(unwrapped_phase)
        st.session_state[wrapped_state_key] = bool(wrapped_phase)
        st.session_state[unwrapped_state_key] = bool(unwrapped_phase)
    configured = st.session_state.get("composite_studio_channels", [])
    group_rows = [
        row for row in configured
        if isinstance(row, dict)
        and bool(row.get("enabled", True))
        and str(row.get("group", "")) in {
            *display_projection.physical_groups,
            *(('Sub',) if "Sub" in multichannel.groups else ()),
        }
    ]
    phaseeq_connected = any(
        isinstance(row.get("phaseeq_fir_response"), dict) for row in group_rows
    )
    phaseeq_iir_pending = any(
        _enabled_phaseeq_iir_definitions(row) and not row.get("phaseeq_iir_sos")
        for row in group_rows
    )
    if not phaseeq_connected:
        st.info(
            ui_message('ui.85fc8fde4a4a01')
        )
    elif phaseeq_iir_pending:
        st.warning(
            ui_message('ui.4d6b1c56e24f54')
        )
    else:
        st.caption(
            ui_message('ui.f2229162588229')
        )
    if graph_kind == "Magnitude" and any(
        isinstance(row, dict)
        and bool(row.get("enabled", True))
        and str(row.get("group", "")) in {
            *display_projection.physical_groups,
            *(('Sub',) if "Sub" in multichannel.groups else ()),
        }
        and int(row.get("polarity", 1)) == -1
        for row in configured
    ):
        st.caption(
            ui_message('ui.63d1649e9c1ac0')
        )
    if graph_kind == "Wavelet":
        from composite_engine.wavelet.view import render_composite_wavelet
        wavelet_columns = st.columns(len(display_projection.wavelet_sources))
        for column, source in zip(wavelet_columns, display_projection.wavelet_sources):
            with column:
                st.markdown(ui_message('ui.e27684c7d76531', p0=f'{source.display_name}'))
                render_composite_wavelet(
                    source.source_ir,
                    source.target_ir if bool(show_target) else None,
                    source.sample_rate_hz,
                    key_prefix=f"composite_wavelet_{source.physical_group}",
                )
        series = ()
    else:
        series = build_display_graph_series(
            display_projection,
            graph_kind,
            include_target=bool(show_target),
            wrapped_phase=bool(wrapped_phase),
        )
    phase_gain_mask_db = float(
        st.session_state.get("phase_gain_mask_db", DEFAULT_PHASE_GAIN_MASK_DB)
    )
    if series:
        frames = []
        for item in series:
            x_values = np.asarray(item.x, dtype=float)
            y_values = np.asarray(item.y, dtype=float)
            y_values = phase_gain_masked_values(
                graph_kind, y_values, item.mask_response, phase_gain_mask_db,
            )
            frames.append(pd.DataFrame({
                "x": x_values,
                "value": y_values,
                "series": item.name,
                "role": item.role,
            }))
        frame = pd.concat(frames, ignore_index=True)
        frame = frame[
            np.isfinite(frame["x"].to_numpy(dtype=float))
        ].reset_index(drop=True)
        is_frequency = graph_kind not in {"Impulse", "Step"}
        if is_frequency:
            frame = frame[
                (frame["x"] >= 10.0)
                & (frame["x"] <= float(result.sample_rate_hz) / 2.0)
            ].reset_index(drop=True)
            frame = _downsample_graph_frame(frame,
                max_points_per_series=int(st.session_state.get("graph_max_points", DEFAULT_GRAPH_MAX_POINTS)),
                wrapped_phase=(graph_kind == "Phase" and bool(wrapped_phase)))
        else:
            # Crop before rendering and retain one point per sample. Thinning
            # the full FFT first leaves only a few points in the visible
            # -2..+10 ms window and makes the time axis look low-resolution.
            frame = _time_response_display_frame(frame)
        if frame.empty:
            st.warning(ui_message('ui.9b0db04dabda75'))
        else:
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(11, 5.2))
            for (series_name, role), series_frame in frame.groupby(
                ["series", "role"], sort=False, dropna=False,
            ):
                axis.plot(
                    series_frame["x"].to_numpy(dtype=float),
                    series_frame["value"].to_numpy(dtype=float),
                    label=str(series_name),
                    linestyle="--" if role == "Target" else "-",
                    linewidth=1.8 if role == "Target" else 1.2,
                )
            if is_frequency:
                axis.set_xscale("log")
                axis.set_xlim(10.0, float(result.sample_rate_hz) / 2.0)
            else:
                set_time_response_xlim(axis)
            if graph_kind == "Magnitude":
                axis.set_ylim(
                    float(st.session_state.get("gain_y_min_db", DEFAULT_GAIN_Y_MIN_DB)),
                    float(st.session_state.get("gain_y_max_db", DEFAULT_GAIN_Y_MAX_DB)),
                )
            elif graph_kind == "Group Delay":
                set_group_delay_ylim(axis)
            else:
                axis.set_ylim(*_finite_axis_domain(frame["value"]))
            axis.set_xlabel("Frequency [Hz]" if is_frequency else "Time [ms]")
            axis.set_ylabel(graph_kind)
            axis.grid(True, which="both", alpha=0.2)
            axis.legend(loc="best")
            figure.tight_layout()
            from composite_engine.multiway_studio.components.visualization import studio_figure_revision
            figure_revision = studio_figure_revision((
                'composite-analysis-v1', graph_kind, frame,
                tuple(axis.get_xlim()), tuple(axis.get_ylim()),
                tuple(figure.get_size_inches()),
            ))
            render_studio_figure(
                figure,
                cache_key=figure_revision,
                mode=str(st.session_state.get("graph_mode", DEFAULT_GRAPH_MODE)),
                interaction="x" if graph_kind in {"Impulse", "Step"} else "both",
                clear_figure=True,
                max_points=int(st.session_state.get(
                    "graph_max_points", DEFAULT_GRAPH_MAX_POINTS,
                )),
            )
            plt.close(figure)
    if (
        show_target and result.target_response is not None
        and not result.target_phase_available and graph_kind != "Magnitude"
    ):
        st.info(ui_message('ui.74de891b4157e3'))

    st.dataframe(pd.DataFrame([
        {
            "Channel": row["name"], "Way": row["way"], "Group": row["group"],
            "Source": row["source"], "Gain [dB]": row["gain_db"],
            "Polarity": row["polarity"], "Delay [samples]": row["delay_samples"],
            "Auto alignment [samples]": row.get("auto_alignment_delay_samples", 0.0),
            "Time basis": _alignment_timing_source_label(
                getattr(
                    st.session_state.get("composite_alignment_system_applied"),
                    "timing_source", "",
                )
            ) if st.session_state.get("composite_alignment_system_applied") else "未適用",
            "External distance [mm]": (
                float(row.get("external_distance_m", 0.0)) * 1_000.0
                if float(row.get("external_distance_m", 0.0) or 0.0) > 0.0 else "—"
            ),
            "PhaseEQ": _phaseeq_status_label(row),
            "PhaseEQ FIR": "Connected" if isinstance(row.get("phaseeq_fir_response"), dict) else "—",
            "Speaker stage": (
                resolve_speaker_source(row).label
                if resolve_speaker_source(row) is not None else "Unity"
            ),
            "Speaker revision": (
                str(resolve_speaker_source(row).revision)
                if resolve_speaker_source(row) is not None else "—"
            ),
            "IIR stage": (
                f"PhaseEQ ({len(_enabled_phaseeq_iir_definitions(row))})"
                if row.get("phaseeq_iir_sos") else
                f"PhaseEQ ({len(_enabled_phaseeq_iir_definitions(row))}) · 再保存"
                if _enabled_phaseeq_iir_definitions(row) else
                "File" if row.get("iir_response") is not None else "—"
            ),
            "Baffle compensation": row.get("baffle_correction_mode", "OFF"),
        }
        for row in configured
        if isinstance(row, dict) and bool(row.get("enabled", True))
    ]), hide_index=True, width="stretch")

    if any(alignment_target.pending(row) for row in configured
           if row.get("source") == "generated_band" and row.get("enabled", True)):
        st.warning(ui_message('ui.54ebb6913ed3a6'))
        return

    st.markdown(ui_message('ui.8b1e5296a1e9f3'))
    st.caption(
        ui_message('ui.0674a0a5f5baab')
    )
    with st.container(border=True):
        st.markdown(ui_message('ui.c9f90144b7c873'))
        st.caption(
            ui_message('ui.5030f55faa4563')
        )
        st.download_button(
            ui_message('ui.be36fc8da8ee9e'), deferred_call(_build_multichannel_export_zip, multichannel),
            "composite_multichannel.zip", "application/zip", width="stretch",
            icon=":material/download:", key="composite_studio_download", on_click="ignore",
        )
    from composite_engine.export import DSPChannelExportInput
    dsp_channels: list[DSPChannelExportInput] = []
    restore_output_fir_settings(st.session_state.get("settings", {}), overwrite=False)
    final_fir_panel = st.container(border=True)
    with final_fir_panel:
        st.markdown(ui_message('ui.f302de343537a0'))
        st.caption(ui_message('ui.ac662edb6c68df'))
        output_remove_nyquist = st.checkbox(
            ui_message('ui.13ec7c644f445c'), key="composite_output_remove_nyquist",
            on_change=_commit_output_taper,
            help=ui_message('ui.4af9483c20a26d'),
        )
        output_nyquist_strength = st.slider(
            ui_message('ui.9004bca80c9432'), min_value=0.0, max_value=1.0, step=0.05,
            key="composite_output_nyquist_strength", on_change=_commit_output_taper,
            disabled=not output_remove_nyquist,
            help=ui_message('ui.1f204c955f24b0'),
        )
        output_taper = st.checkbox(
            ui_message('ui.78c3bbd1fa1439'),
            key="composite_output_cosine_taper",
            on_change=_commit_output_taper,
            help=ui_message('ui.06fa323b93f168'),
        )
        with st.expander(display_text("最終FIRグラフ"), expanded=False):
            show_final_response = st.checkbox(ui_message('ui.1e3e6c4ad3bd1e'), value=True,
                                              key="composite_show_final_fir_response")
            show_final_coefficients = st.checkbox(ui_message('ui.d6af320bcc686e'), key="composite_show_fir_cofs")
            final_fir_graphs = st.container()
    _commit_output_taper()
    kaiser_split_firs = st.session_state.get("kaiser_split_firs", {})
    studio_additional_firs = st.session_state.get("studio_additional_firs", {})
    band_tap_lengths = st.session_state.get("composite_studio_band_tap_lengths", {})
    band_fir_states = st.session_state.get("composite_studio_band_fir_states", {})
    generated_bands = {
        str(row.get("band", ""))
        for row in configured
        if isinstance(row, dict)
        and row.get("source") == "generated_band"
        and bool(row.get("enabled", True))
        and str(row.get("band", "")) in final_firs
    }
    result_settings = st.session_state.get("result_settings", {})
    align_output_taps = bool(
        result_settings.get("align_output_taps", False)
        if isinstance(result_settings, dict) else False
    )
    assigned_taps = {
        band: (
            int(band_fir_states[band].get("tap_count") or band_tap_lengths.get(band, 0))
            if band_fir_states.get(band, {}).get("enabled", False) else None
        )
        for band in generated_bands
    }
    output_timing = resolve_output_timing(assigned_taps, align_output_taps=align_output_taps)
    for row in configured:
        if not isinstance(row, dict) or row.get("source") != "generated_band" or not bool(row.get("enabled", True)):
            continue
        band = str(row.get("band", ""))
        if band not in final_firs or band not in kaiser_split_firs:
            continue
        fir_state = (
            band_fir_states.get(band, {})
            if isinstance(band_fir_states, dict) else {}
        )
        fir_enabled = bool(fir_state.get("enabled", False))
        kaiser_coefficients = np.asarray(kaiser_split_firs[band], dtype=float)
        stages: list[tuple[str, np.ndarray]] = []
        if fir_enabled and not (
            kaiser_coefficients.size == 1 and float(kaiser_coefficients[0]) == 1.0
        ):
            stages.append(("kaiser_fir_crossover", kaiser_coefficients))
        phaseeq_fir = row.get("phaseeq_fir_response")
        if fir_enabled and isinstance(phaseeq_fir, dict):
            stages.append((
                "phaseeq_fir_eq",
                _phaseeq_channel_fir(row, sample_rate_hz),
            ))
        studio_additional = studio_additional_firs.get(band)
        if fir_enabled and studio_additional is not None and not (
            len(studio_additional) == 1 and float(studio_additional[0]) == 1.0
        ):
            stages.append(("additional_fir_studio", np.asarray(studio_additional, dtype=float)))
        additional_fir = row.get("additional_fir")
        if fir_enabled and additional_fir is not None:
            stages.append((
                "additional_fir",
                _fir_coefficients(str(additional_fir.name), additional_fir.getvalue(), int(sample_rate_hz)),
            ))
        normalization = st.session_state.get("normalization_info", {})
        if fir_enabled and not stages:
            stages.append(("manual_fir_frame", np.asarray([1.0], dtype=float)))
        normalization_gain_db = (
            float(normalization.get("gain_db", 0.0))
            if isinstance(normalization, dict) and bool(normalization.get("enabled", False))
            else 0.0
        )
        working_path = row.get("phaseeq_working_session")
        working_bytes = (
            Path(working_path).read_bytes()
            if working_path is not None and Path(working_path).is_file() else None
        )
        output_iir, output_polarity, output_allpass = alignment_target.output_settings(row)
        dsp_channels.append(DSPChannelExportInput(
            channel_id=str(row.get("channel_id") or phaseeq_channel_id(
                str(row["name"]), str(row["way"]), str(row["group"]),
            )),
            name=str(row["name"]), way=str(row["way"]), group=str(row["group"]),
            sample_rate_hz=int(sample_rate_hz),
            tap_count=output_timing[band].tap_count,
            pre_alignment_tap_count=assigned_taps[band] if align_output_taps else None,
            fir_stages=tuple(stages),
            cosine_taper_enabled=bool(output_taper),
            remove_nyquist_enabled=bool(output_remove_nyquist),
            remove_nyquist_strength=float(output_nyquist_strength),
            phaseeq_iir=tuple(
                dict(item) for item in row.get("phaseeq_iir", []) if isinstance(item, dict)
            ),
            phaseeq_iir_sos=tuple(
                tuple(float(value) for value in section)
                for section in row.get("phaseeq_iir_sos", ())
            ),
            baffle_iir_parameters=(
                dict(row["baffle_iir_parameters"])
                if isinstance(row.get("baffle_iir_parameters"), dict) else None
            ),
            baffle_iir_sos=tuple(
                tuple(float(value) for value in section)
                for section in row.get("baffle_iir_sos", ())
            ),
            iir_crossover=output_iir,
            gain_db=float(row["gain_db"]) + normalization_gain_db,
            polarity=output_polarity,
            dsp_additional_delay_samples=output_timing[band].additional_delay_samples,
            channel_relative_delay_samples=float(row["delay_samples"]),
            auto_alignment_delay_samples=float(row.get("auto_alignment_delay_samples", 0.0)),
            auto_alignment_allpass=output_allpass,
            timing_provenance=_timing_export_metadata(row),
            working_session_zip=working_bytes,
        ))
    if dsp_channels:
        current_system = _current_library_system()
        current_record = _current_library_record()
        workspace = {
            "format_version": 2,
            "multiway_system": (
                {
                    "id": current_system.id,
                    "name": current_system.name,
                    "revision": current_system.revision,
                    "management_no": (
                        current_record.management_no if current_record is not None else ""
                    ),
                    "content_hash": (
                        current_record.content_hash if current_record is not None else ""
                    ),
                }
                if current_system is not None else None
            ),
            "multiway_settings": {
                **dict(st.session_state.get("settings", {})),
                "selected_display": str(display_group),
            },
            "selected_display": str(display_group),
            # Backward-compatible physical Group pointer. L+R remains a
            # display-only value and is never serialized as a physical Group.
            "selected_group": _physical_group_for_display(
                display_group, stereo=stereo_groups,
            ),
            "target": {
                "group": target_scope,
                "source": target_source,
                "preset_id": str(selected_group_target_id),
                "preset_definition": (
                    dict(group_target_definition)
                    if isinstance(group_target_definition, dict) else None
                ),
                "file": (
                    {
                        "filename": Path(str(target_upload.name)).name,
                        "data_base64": base64.b64encode(target_upload.getvalue()).decode("ascii"),
                    }
                    if target_upload is not None else None
                ),
            },
            "targets": {
                target_group: {
                    "group": target_group,
                    "source": str(st.session_state.get(
                        f"composite_studio_target_source_{_group_target_scope(target_group)}",
                        "PhaseEQ preset",
                    )),
                    "preset_id": str(st.session_state.get(
                        f"composite_studio_group_target_preset_{_group_target_scope(target_group)}", "",
                    )),
                    "preset_definition": dict(definition),
                }
                for target_group, definition in target_definitions_by_group.items()
            },
            "graph": {"kind": graph_kind, "wrapped_phase": bool(wrapped_phase)},
            "channels": [
                {
                    "channel_id": str(row.get("channel_id", "")),
                    "source": str(row.get("source", "")),
                    "band": str(row.get("band", "")), "name": str(row.get("name", "")),
                    "way": str(row.get("way", "")), "group": str(row.get("group", "")),
                    "enabled": bool(row.get("enabled", True)),
                    "speaker_package_id": str(row.get("speaker_package_id", "")),
                    "latest_assignment_id": str(row.get("latest_assignment_id", "")),
                    "gain_db": float(row.get("gain_db", 0.0)),
                    "polarity": int(row.get("polarity", 1)),
                    "dc_gain_normalize": bool(row.get("dc_gain_normalize", False)),
                    "delay_samples": float(row.get("delay_samples", 0.0)),
                    "auto_alignment_delay_samples": float(row.get("auto_alignment_delay_samples", 0.0)),
                    "auto_alignment_allpass": [
                        section.to_dict() for section in row.get("auto_alignment_allpass", ())
                    ],
                    "baffle_correction_mode": str(
                        row.get("baffle_correction_mode", "OFF")
                    ),
                    "baffle_iir_parameters": (
                        dict(row["baffle_iir_parameters"])
                        if isinstance(row.get("baffle_iir_parameters"), dict) else None
                    ),
                    "iir_xover_enabled": bool(row.get("iir_crossover", IIRCrossoverConfig()).enabled),
                    "iir_xover_family": str(row.get("iir_crossover", IIRCrossoverConfig()).family),
                    "iir_xover_order": int(row.get("iir_crossover", IIRCrossoverConfig()).order),
                    "iir_xover_hp": float(row.get("iir_crossover", IIRCrossoverConfig()).highpass_hz),
                    "iir_xover_lp": float(row.get("iir_crossover", IIRCrossoverConfig()).lowpass_hz),
                }
                for row in configured if isinstance(row, dict)
            ],
        }
        with st.container(border=True):
            st.markdown(ui_message('ui.44408a3f833557'))
            st.caption(
                ui_message('ui.0c3a49c6282dd6')
            )
            resume_download_host = st.container()

        with st.container(border=True):
            st.markdown(ui_message('ui.63a38ef6e96997'))
            st.caption(
                ui_message('ui.0bac1b243b81ce')
            )
            from composite_engine.dsp_export import (
                adapter_by_id,
                available_adapters,
                canonical_from_export_inputs,
                package_filename,
            )
            adapters = available_adapters()
            adapter_labels = {adapter.adapter_id: adapter.display_name for adapter in adapters}
            selected_adapter_id = shared_selectbox(
                ui_message('ui.5928e6ccdd119b'), tuple(adapter_labels),
                format_func=lambda value: adapter_labels[value],
                key="composite_studio_dsp_export_adapter",
            )
            selected_adapter = adapter_by_id(selected_adapter_id)
            profiles = selected_adapter.profiles()
            profile_labels = {profile.profile_id: profile.display_name for profile in profiles}
            selected_profile_id = shared_selectbox(
                ui_message('ui.1e80a85cbd7c37'), tuple(profile_labels),
                format_func=lambda value: profile_labels[value],
                key=f"composite_studio_dsp_export_profile_{selected_adapter_id}",
            )
            selected_profile = next(
                profile for profile in profiles if profile.profile_id == selected_profile_id
            )
            system_name = (
                current_system.name if current_system is not None else
                f"{st.session_state.get('result_mode_key', 'Multiway')} System"
            )
            design_generated_at = str(st.session_state.get("result_generated_at", ""))
            export_workspace = dict(workspace)
            export_workspace["design_generated_at"] = design_generated_at
            canonical_package = canonical_from_export_inputs(
                tuple(dsp_channels), system_name=system_name,
                generated_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                mode=str(st.session_state.get("result_mode_key", "Multiway")),
                source_signature=str(st.session_state.get("_studio_dsp_input_signature", "")),
                workspace=export_workspace,
            )
            if band_tap_count_hosts and st.session_state.get("_studio_graphs_current", False):
                from composite_engine.multiway_studio.ui.output_timing import render_band_tap_counts
                render_band_tap_counts(canonical_package, band_tap_count_hosts)
            if output_timing_host is not None:
                from composite_engine.multiway_studio.ui.output_timing import render_output_timing
                with output_timing_host:
                    render_output_timing(
                        canonical_package, assigned_taps=assigned_taps,
                        split_lengths={band: len(values) for band, values in kaiser_split_firs.items()
                                       if band_fir_states.get(band, {}).get("split_fir_required", False)},
                        intermediate_lengths={band: len(values) for band, values in final_firs.items()},
                    )
            compatibility = selected_adapter.validate(canonical_package, selected_profile)
            with resume_download_host:
                st.download_button(
                    ui_message('ui.17cb888870169b'),
                    deferred_call(_build_dsp_resume_zip, tuple(dsp_channels),
                                  workspace=workspace, composite=multichannel,
                                  final_fir_artifacts={c.channel_id: c.final_fir_artifact
                                                       for c in canonical_package.channels
                                                       if c.final_fir_artifact is not None}),
                    "multiway_dsp_package.zip", "application/zip", width="stretch",
                    icon=":material/archive:", key="composite_studio_dsp_package_download",
                    on_click="ignore",
                )
            if show_final_response or show_final_coefficients:
                from response_display.fir_coefficients import coefficient_figure
                from response_display.plotly_charts import PLOTLY_CONFIG
                from utils.ui_semantic_colors import semantic_chart_colors
                chart_theme = semantic_chart_colors(str(st.context.theme.type or "light"))
                coefficient_colors = {
                    "text": chart_theme["text"], "chart_grid": chart_theme["grid"],
                    "series_delta": chart_theme["delta"],
                    "series_result": chart_theme["result"],
                    "series_gain_error": chart_theme["gain_error"],
                }
                with final_fir_graphs:
                    st.caption(
                        ui_message('ui.6160584f4f190d')
                    )
                    for channel in canonical_package.channels:
                        if channel.final_fir is None:
                            st.caption(ui_message('ui.a7f2612885ea5b', p0=f'{channel.group}', p1=f'{channel.name}'))
                            continue
                        st.caption(ui_message('ui.acecce7bb9d51c', p0=f'{channel.group}', p1=f'{channel.name}', p2=f'{len(channel.final_fir):,}', p3=f"{('ON' if output_remove_nyquist else 'OFF')}", p4=f"{('ON' if output_taper else 'OFF')}"))
                        if show_final_response:
                            from response_display.fir_response import final_fir_response_figure
                            st.plotly_chart(final_fir_response_figure(
                                channel.final_fir, channel.sample_rate_hz, colors=chart_theme,
                                max_points=int(st.session_state.get("graph_max_points", DEFAULT_GRAPH_MAX_POINTS)),
                                phase_mask_db=float(st.session_state.get("phase_gain_mask_db", -80)),
                                gain_range=(float(st.session_state.get("gain_y_min_db", DEFAULT_GAIN_Y_MIN_DB)),
                                            float(st.session_state.get("gain_y_max_db", DEFAULT_GAIN_Y_MAX_DB))),
                            ), key=f"composite_final_fir_response_{channel.channel_id}",
                                theme=None, width="stretch", config=PLOTLY_CONFIG)
                        if show_final_coefficients:
                            st.plotly_chart(coefficient_figure(channel.final_fir, window_enabled=output_taper, colors=coefficient_colors),
                                            key=f"composite_fir_cofs_{channel.channel_id}",
                                            theme=None, width="stretch", config=PLOTLY_CONFIG)
            st.dataframe(pd.DataFrame(compatibility.channel_rows), hide_index=True, width="stretch")
            if compatibility.issues:
                for issue in compatibility.issues:
                    message = f"{issue.channel_id + ': ' if issue.channel_id else ''}{issue.message}"
                    if issue.severity == "error":
                        st.error(message)
                    elif issue.severity == "warning":
                        st.warning(message)
                    else:
                        st.info(message)
            else:
                st.success(ui_message('ui.76f48a8304b221'))
            st.download_button(
                ui_message('ui.0f0028b1a0ab35'),
                data=deferred_call(first_result, _build_dsp_export_zip,
                                   canonical_package, selected_adapter, selected_profile),
                file_name=package_filename(canonical_package, selected_adapter),
                mime="application/zip", width="stretch",
                disabled=not compatibility.compatible,
                icon=":material/download:", key="composite_studio_download_dsp_export_v3",
                on_click="ignore",
            )
