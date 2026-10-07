from utils.ui_localization import ui_message, display_text, localized_formatter, display_notice
import streamlit as st
from utils.list_menu_ui import selectbox as shared_selectbox, radio as shared_radio, segmented_control as shared_segmented_control
from utils.ui_semantic_colors import semantic_button_css
import hashlib
import html
import json
import os
import io
import re
import time
import unicodedata
import zipfile
import numpy as np
import datetime
import copy
import pandas as pd
from scipy.signal import freqz
# 旧互換APIは削除。新ui_index.pyに合わせてインポートを整理
from composite_engine.multiway_studio.utils.ui_index import radio_indexed, selectbox_indexed, number_input_stateful_safe
from composite_engine.kaiser_help import KAISER_SLOPE_GUIDE
from composite_engine.multiway_studio.utils.kaiser import (
    kaiser_attenuation_usage_hint,
    kaiser_beta_to_attenuation_db,
)
from composite_engine.multiway_studio.utils.auto_crop import plan_auto_crop_candidates
from composite_engine.multiway_studio.utils.display_rounding import round_for_display
from composite_engine.multiway_studio.processing.filter import (
    generate_2way_filters, generate_3way_filters, generate_4way_filters,
    combine_and_crop_fir_conv,
    make_baffle_step_fir, measure_fir_group_response_peaks,
    normalize_fir_group_response, odd_number, kaiser_overlap_edges_hz,
    generate_exclusive_kaiser_firs, generate_residual_multiband_filters,
)
from composite_engine.multiway_studio.processing.design_response import (
    build_crossover_design_responses,
    build_speaker_phaseeq_responses,
    centered_impulses_from_responses,
    sum_crossover_comparison_response,
    ir_length_for_frequency_resolution,
    response_points_for_frequency_resolution,
    select_crossover_design_inputs,
)
from composite_engine.multiway_studio.processing.dsp_input import (
    studio_dsp_input_signature,
)
from composite_engine.multiway_studio.processing.baffle_compensation import (
    BaffleCompensationPlan,
    baffle_plan,
    common_baffle_response,
)
from utils.band_split_capability import convert_studio_methods
from composite_engine.multiway_studio.processing.fir_state import (
    DEFAULT_MANUAL_FIR_TAPS,
    bands_requiring_split_fir,
    inferred_fir_output_enabled,
    initialize_manual_taps_for_enable,
    plan_acoustic_target_taps,
    resolve_band_fir_states,
)
from composite_engine.multiway_studio.utils.io import load_fir_file
import composite_engine.multiway_studio.components.visualization as visualization
from composite_engine.multiway_studio.components.visualization import (
    center_fir_lengths,
)
from composite_engine.multiway_studio.components.studio_charts import (
    build_studio_chart_bundle,
    render_studio_chart,
)
from composite_engine.multiway_studio.extension import _shared_iir_configs
from composite_engine.multiway_studio.extension import (
    _watch_phaseeq_result_inbox,
    normalize_crossover_method,
    render_crossover_method_control,
    render_composite_results,
    render_composite_sidebar,
    render_shared_iir_crossover_ui,
)
from composite_engine.display_projection import normalize_display_value
from composite_engine.speaker_timing import resolve_speaker_timing_projection


def _studio_chart_source_revision(
    dsp_revision, display, graph_max_points, db_min, db_max, phase_gain_mask_db,
):
    """Identify display projection inputs without coupling them to a renderer."""
    payload = json.dumps(
        {
            "dsp": str(dsp_revision),
            "display": str(display),
            "points": int(graph_max_points),
            "db_min": float(db_min),
            "db_max": float(db_max),
            "phase_mask": float(phase_gain_mask_db),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _project_studio_charts(source, *, source_revision, graph_max_points,
                           db_min, db_max, phase_gain_mask_db):
    """Commit one validated six-chart bundle from a renderer-neutral source."""
    return build_studio_chart_bundle(
        source["impulses"], source["frequency_hz"], source["responses"],
        sample_rate_hz=source["sample_rate_hz"],
        sum_groups=source["sum_groups"], way_sums=source["way_sums"],
        db_min=db_min, db_max=db_max,
        phase_gain_mask_db=phase_gain_mask_db,
        phaseeq_responses=source["phaseeq_responses"],
        speaker_responses=source["speaker_responses"],
        baffle_response=source["baffle_response"],
        baffle_label=source["baffle_label"],
        max_points=graph_max_points, source_revision=source_revision,
    )


def _render_studio_chart(bundle, chart_id, *, graph_mode, plot_theme):
    render_studio_chart(
        bundle.chart(chart_id), mode=graph_mode, theme=plot_theme,
    )


_STUDIO_GRAPH_LAYOUT = (
    ("band_impulse", "ui.e0d37186efff20"),
    ("band_step", "ui.ceb9132f645569"),
    ("relative_group_delay", "ui.59017b5299f697"),
    ("system_impulse", "ui.bda762804bef52"),
    ("system_gain_phase", "ui.7eeec0195e2788"),
)


def _create_studio_graph_layout():
    """Create stable graph positions before any fragment redraw begins."""
    chart_hosts = []
    for chart_id, title_key in _STUDIO_GRAPH_LAYOUT:
        st.subheader(ui_message(title_key))
        host = st.container()
        host.empty()
        chart_hosts.append((chart_id, host))
    metrics_host = st.container()
    st.subheader(ui_message('ui.bbb9e421006621'))
    crossover_host = st.container()
    crossover_host.empty()
    chart_hosts.append(("crossover_design", crossover_host))
    return tuple(chart_hosts), metrics_host


def _persist_graph_mode(selected_mode):
    """Commit renderer state without invalidating DSP or projection state."""
    selected_mode = str(selected_mode)
    st.session_state[KEY_GRAPH_MODE] = selected_mode
    stored = normalize_settings(st.session_state.get("settings", {}))
    if str(stored.get("graph_mode")) == selected_mode:
        return
    updated = copy.deepcopy(stored)
    updated["graph_mode"] = selected_mode
    # Renderer-only switches are frequent and already visible in the widget.
    # Persist the preference without growing the user-facing operation log.
    save_settings(updated, log_change=False)
    st.session_state["settings"] = updated


@st.fragment
def _render_studio_graph_fragment(
    bundle, *, plot_theme, chart_hosts,
):
    """Redraw only the fixed graph slots when renderer mode changes."""
    selected_mode = shared_selectbox(
        ui_message('ui.5e23ec6a300dc6'),
        visualization.GRAPH_MODE_OPTIONS,
        key="composite_studio_graph_mode_widget",
        help=ui_message('ui.fb8d9182596a2d'),
    )
    _persist_graph_mode(selected_mode)
    for chart_id, host in chart_hosts:
        with host:
            _render_studio_chart(
                bundle, chart_id, graph_mode=selected_mode,
                plot_theme=plot_theme,
            )


def _studio_display_graph_bundle(
    final_firs,
    graph_rows,
    displayed_channel,
    fir_states,
    *,
    mode_key,
    crossover_frequencies_hz,
    sample_rate_hz,
    response_points,
    graph_max_points,
    dsp_revision,
    db_min,
    db_max,
    phase_gain_mask_db,
):
    """Rebuild display-only responses and one atomic chart bundle."""
    stereo = (
        str(st.session_state.get("composite_studio_output_layout", "Mono")) == "Stereo"
    )
    display = normalize_display_value(displayed_channel, stereo=stereo)
    groups = (
        ("Left", "Right") if display == "L+R"
        else ("Left",) if display == "L"
        else ("Right",) if display == "R"
        else ("Main",)
    )
    iir_configs = _shared_iir_configs(mode_key, tuple(crossover_frequencies_hz))
    responses = {}
    settings = {}
    phaseeq_responses = {}
    speaker_responses = {}
    sum_groups = {}
    way_sums = {}
    frequency = np.array([], dtype=float)
    primary_settings = {}
    timing_projection = resolve_speaker_timing_projection(
        graph_rows, st.session_state.get("settings", {}),
        sample_rate_hz=int(sample_rate_hz),
    )
    for physical_group in groups:
        side_firs, side_settings = select_crossover_design_inputs(
            final_firs, graph_rows, physical_group,
        )
        side_fir_enabled = {
            way: bool(fir_states.get(way, {}).get("enabled", False))
            for way in side_firs
        }
        upstream = build_speaker_phaseeq_responses(
            graph_rows, tuple(side_firs), physical_group, sample_rate_hz,
            points=response_points, fir_enabled_by_way=side_fir_enabled,
            speaker_timing_projection=timing_projection,
        )
        phaseeq = build_speaker_phaseeq_responses(
            graph_rows, tuple(side_firs), physical_group, sample_rate_hz,
            points=response_points, phaseeq_only=True,
            fir_enabled_by_way=side_fir_enabled,
            speaker_timing_projection=timing_projection,
        )
        speaker = build_speaker_phaseeq_responses(
            graph_rows, tuple(side_firs), physical_group, sample_rate_hz,
            points=response_points, speaker_only=True,
            fir_enabled_by_way=side_fir_enabled,
            speaker_timing_projection=timing_projection,
        )
        for way, value in upstream.items():
            if way in side_settings:
                side_settings[way]["speaker_phaseeq_response"] = value
        frequency, side_responses, side_sum = build_crossover_design_responses(
            side_firs, iir_configs, sample_rate_hz,
            points=response_points, way_settings=side_settings,
            fir_enabled_by_way=side_fir_enabled,
        )
        prefix = "L" if physical_group == "Left" else "R" if physical_group == "Right" else ""
        def renamed(way):
            if way == "SUB" and side_settings.get(way, {}).get("group") == "Sub":
                return "SUB"
            return f"{prefix} {way}" if len(groups) > 1 else way
        keys = tuple(renamed(way) for way in side_responses)
        for target, source in (
            (responses, side_responses),
            (settings, side_settings),
            (phaseeq_responses, phaseeq),
            (speaker_responses, speaker),
        ):
            for way, value in source.items():
                key = renamed(way)
                if key not in target:
                    target[key] = value
        sum_label = f"{prefix} System Sum" if prefix else "Main System Sum"
        sum_groups[sum_label] = keys
        way_sums[sum_label] = side_sum
        if physical_group == groups[0]:
            primary_settings = side_settings
    impulses = centered_impulses_from_responses(responses)
    cached_baffle_plan = st.session_state.get("composite_studio_baffle_plan")
    baffle_response = (
        common_baffle_response(
            cached_baffle_plan, frequency, sample_rate_hz,
            fir=st.session_state.get("composite_studio_baffle_fir"),
        )
        if isinstance(cached_baffle_plan, BaffleCompensationPlan) and frequency.size
        else None
    )
    source = {
        "sample_rate_hz": int(sample_rate_hz),
        "frequency_hz": frequency,
        "responses": responses,
        "impulses": impulses,
        "sum_groups": sum_groups,
        "way_sums": way_sums,
        "phaseeq_responses": phaseeq_responses,
        "speaker_responses": speaker_responses,
        "baffle_response": baffle_response,
        "baffle_label": (
            f"Baffle compensation · {cached_baffle_plan.display_mode}"
            if isinstance(cached_baffle_plan, BaffleCompensationPlan)
            and cached_baffle_plan.mode != "off" else None
        ),
    }
    source_revision = _studio_chart_source_revision(
        dsp_revision, display, graph_max_points, db_min, db_max,
        phase_gain_mask_db,
    )
    chart_bundle = _project_studio_charts(
        source, source_revision=source_revision,
        graph_max_points=graph_max_points, db_min=db_min, db_max=db_max,
        phase_gain_mask_db=phase_gain_mask_db,
    )
    return {
        "display": display,
        "settings": settings,
        "primary_settings": primary_settings,
        "responses": responses,
        "chart_source": source,
        "chart_bundle": chart_bundle,
    }


def _assignment_band_fir_states(mode_key, conf, split_firs, *, auto_crop=False):
    """Resolve Assignment FIR state from the exact generated split bank."""
    bands = MODE_BANDS[mode_key]
    methods = tuple(conf.get("crossover_methods", ()))
    acoustic_targets = tuple(conf.get("boundary_acoustic_targets", ()))
    configured_taps = conf.get("crop_lens", {})
    automatic_taps = plan_acoustic_target_taps(
        bands,
        methods,
        acoustic_targets,
        conf.get("cross_freqs", ()),
        configured_taps,
        sample_rate_hz=int(conf.get("fs", 96_000)),
    )
    return resolve_band_fir_states(
        bands,
        ["Through" if i < len(acoustic_targets) and acoustic_targets[i] else method for i, method in enumerate(methods)],
        configured_taps,
        fir_output_enabled=bool(conf.get("fir_output_enabled", False)),
        split_firs=split_firs,
        automatic_taps=automatic_taps,
        auto_crop=auto_crop,
    )


def _assignment_band_linear_fir_filters(mode_key, conf):
    """Project copied Studio crossover boundaries into PhaseEQ Linear FIR filters."""
    cross = [float(value) for value in conf.get("cross_freqs", [])]
    cycles = float(conf.get("cycles", DEFAULT_FIR_CYCLES))
    beta = float(conf.get("beta", DEFAULT_FIR_BETA))
    methods = tuple(conf.get("crossover_methods", ["Kaiser FIR"] * len(cross)))

    def item(mode, fc, item_cycles, item_beta, method):
        return {
            "enabled": True,
            "mode": mode,
            "response": {"Linear-phase LR2 FIR": "linear_phase_lr2", "Linear-phase LR4 FIR": "linear_phase_lr4"}.get(method, "kaiser"),
            "fc": float(fc),
            "cycles": float(item_cycles),
            "beta": float(item_beta),
        }

    if mode_key == "Fullrange":
        return {"Fullrange": ()}
    if mode_key == "Fullrange+SUB":
        overlaps = conf.get("boundary_overlaps_oct", [0.0])
        lp_edge, hp_edge = kaiser_overlap_edges_hz(cross[0], overlaps[0])
        sub = item("lp", lp_edge, cycles, beta, methods[0])
        fullrange = item("hp", hp_edge, cycles, beta, methods[0])
        is_fir = methods[0] in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
        return {
            "SUB": (sub,) if is_fir else (),
            "Fullrange": (fullrange,) if is_fir else (),
        }

    if mode_key == "2Way":
        lp_edge, hp_edge = kaiser_overlap_edges_hz(cross[0], conf.get("kaiser_overlap_oct", 0.0))
        low = item("lp", lp_edge, cycles, beta, methods[0])
        high = item("hp", hp_edge, cycles, beta, methods[0])
        is_fir = methods[0] in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
        return {"Low": (low,) if is_fir else (), "High": (high,) if is_fir else ()}
    if mode_key == "3Way":
        low_boundary = (
            cross[0], float(conf.get("cycles_low", cycles)),
            float(conf.get("beta_low", beta)),
        )
        high_boundary = (
            cross[1], float(conf.get("cycles_high", cycles)),
            float(conf.get("beta_high", beta)),
        )
        low_lp, low_hp = kaiser_overlap_edges_hz(low_boundary[0], conf.get("kaiser_overlap_low_mid_oct", 0.0))
        high_lp, high_hp = kaiser_overlap_edges_hz(high_boundary[0], conf.get("kaiser_overlap_mid_high_oct", 0.0))
        low = item("lp", low_lp, *low_boundary[1:], methods[0])
        mid_hp = item("hp", low_hp, *low_boundary[1:], methods[0])
        mid_lp = item("lp", high_lp, *high_boundary[1:], methods[1])
        high = item("hp", high_hp, *high_boundary[1:], methods[1])
        fir_methods = {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
        return {
            "Low": (low,) if methods[0] in fir_methods else (),
            "Mid": tuple(value for value, index in ((mid_hp, 0), (mid_lp, 1)) if methods[index] in fir_methods),
            "High": (high,) if methods[1] in fir_methods else (),
        }
    if mode_key == "3Way+SUB":
        boundaries = (
            (cross[0], float(conf.get("cycles_sub_low", cycles)), float(conf.get("beta_sub_low", beta))),
            (cross[1], float(conf.get("cycles_low_mid", cycles)), float(conf.get("beta_low_mid", beta))),
            (cross[2], float(conf.get("cycles_mid_high", cycles)), float(conf.get("beta_mid_high", beta))),
        )
        overlap_values = (
            conf.get("kaiser_overlap_sub_low_oct", 0.0),
            conf.get("kaiser_overlap_low_mid_oct", 0.0),
            conf.get("kaiser_overlap_mid_high_oct", 0.0),
        )
        edges = [kaiser_overlap_edges_hz(boundary[0], overlap) for boundary, overlap in zip(boundaries, overlap_values, strict=True)]
        return {
            "SUB": (item("lp", edges[0][0], *boundaries[0][1:], methods[0]),) if methods[0] in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"} else (),
            "Low": tuple(value for value, index in ((item("hp", edges[0][1], *boundaries[0][1:], methods[0]), 0), (item("lp", edges[1][0], *boundaries[1][1:], methods[1]), 1)) if methods[index] in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}),
            "Mid": tuple(value for value, index in ((item("hp", edges[1][1], *boundaries[1][1:], methods[1]), 1), (item("lp", edges[2][0], *boundaries[2][1:], methods[2]), 2)) if methods[index] in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}),
            "High": (item("hp", edges[2][1], *boundaries[2][1:], methods[2]),) if methods[2] in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"} else (),
        }
    boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(mode_key, conf)
    edges = [
        kaiser_overlap_edges_hz(frequency, boundary_overlaps[index])
        for index, frequency in enumerate(cross)
    ]
    fir_methods = {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
    result = {}
    for band_index, band in enumerate(MODE_BANDS[mode_key]):
        filters = []
        if band_index > 0 and methods[band_index - 1] in fir_methods:
            filters.append(item(
                "hp", edges[band_index - 1][1], boundary_cycles[band_index - 1],
                boundary_betas[band_index - 1], methods[band_index - 1],
            ))
        if band_index < len(cross) and methods[band_index] in fir_methods:
            filters.append(item(
                "lp", edges[band_index][0], boundary_cycles[band_index],
                boundary_betas[band_index], methods[band_index],
            ))
        result[band] = tuple(filters)
    return result


def _generate_split_filter_bank(mode_key, fs, conf, crossover_methods):
    from crossover_engine.bank import generate_split_filter_bank
    return generate_split_filter_bank(mode_key, fs, conf, crossover_methods)

# ==== key一元管理 ====
KEY_MODE = "mode"
KEY_FS = "fs"
KEY_CYCLES = "cycles"
KEY_BETA = "beta"
KEY_CYCLES_LOW = "cycles_low"
KEY_BETA_LOW = "beta_low"
KEY_CYCLES_HIGH = "cycles_high"
KEY_BETA_HIGH = "beta_high"
KEY_CYCLES_SUB_LOW = "cycles_sub_low"
KEY_BETA_SUB_LOW = "beta_sub_low"
KEY_CYCLES_LOW_MID = "cycles_low_mid"
KEY_BETA_LOW_MID = "beta_low_mid"
KEY_CYCLES_MID_HIGH = "cycles_mid_high"
KEY_BETA_MID_HIGH = "beta_mid_high"
KEY_DBMIN = "db_min"
KEY_GAIN_Y_MIN = "gain_y_min_db"
KEY_GAIN_Y_MAX = "gain_y_max_db"
KEY_PHASE_GAIN_MASK = "phase_gain_mask_db"
KEY_GRAPH_MODE = "graph_mode"
KEY_GRAPH_MAX_POINTS = "graph_max_points"
KEY_BAFFLE_ON = "baffle_on"
KEY_BAFFLE_ATTEN = "baffle_atten"
KEY_BAFFLE_F1 = "baffle_f1_hz"      # ← 追加：ハイ側コーナ f1
KEY_GEN_BUTTON = "generate_button"
KEY_EXPORT_BUTTON = "snapshot_button"
KEY_RESUME_SELECT = "resume_select"
KEY_RESUME_BTN = "resume_btn"
KEY_AUTO_UPDATE = "auto_update"
KEY_MANUAL_REFRESH = "manual_refresh"
KEY_OUTPUT_FORMAT = "output_format"
KEY_OUTPUT_NORMALIZE = "output_normalize"
KEY_ALIGN_OUTPUT_TAPS = "align_output_taps"
KEY_FIR_OUTPUT_PREFIX = "fir_output_enabled_"
KEY_AUTO_CROP = "auto_crop"
KEY_AUTO_CROP_PROFILE = "auto_crop_profile"
KEY_AUTO_CROP_PASS_DB = "auto_crop_pass_db"
KEY_AUTO_CROP_CROSS_DB = "auto_crop_cross_db"
KEY_SETTINGS_UPLOAD = "settings_upload"
KEY_SETTINGS_APPLY = "settings_apply"

STUDIO_DATA_DIR = os.path.join(
    os.environ.get("PHASEEQ_DATA_DIR", "data"), "composite_engine", "multiway_studio"
)
SETTINGS_FILE = os.path.join(STUDIO_DATA_DIR, "settings.json")
EQ_SAVE_DIR = os.path.join(STUDIO_DATA_DIR, "uploaded_eq")
CONFIG_EXPORT_DIR = os.path.join(STUDIO_DATA_DIR, "config_export")
os.makedirs(STUDIO_DATA_DIR, exist_ok=True)
os.makedirs(EQ_SAVE_DIR, exist_ok=True)
os.makedirs(CONFIG_EXPORT_DIR, exist_ok=True)

AUTO_SKIP_RENDER_SECONDS = 4.0
SLOW_RENDER_NOTICE_SECONDS = 2.0
VERY_SLOW_RENDER_SECONDS = 8.0

AUTO_CROP_PROFILES = {
    "high_precision": {"label": "高精度", "pass_db": 0.05, "cross_db": 0.03},
    "standard": {"label": "標準", "pass_db": 0.10, "cross_db": 0.05},
    "lightweight": {"label": "軽量化・試聴", "pass_db": 0.20, "cross_db": 0.08},
}
DEFAULT_AUTO_CROP_PROFILE = "high_precision"
DEFAULT_AUTO_CROP_PASS_DB = AUTO_CROP_PROFILES[DEFAULT_AUTO_CROP_PROFILE]["pass_db"]
DEFAULT_AUTO_CROP_CROSS_DB = AUTO_CROP_PROFILES[DEFAULT_AUTO_CROP_PROFILE]["cross_db"]
OUTPUT_MAX_BAND_GAIN_DB = -0.3
SUPPORTED_SAMPLE_RATES = [48000, 96000, 192000]
DEFAULT_FIR_CYCLES = 3.5
DEFAULT_FIR_BETA = 12.0
FIR_CYCLES_MIN, FIR_CYCLES_MAX = 2.7, 15.0
FIR_BETA_MIN, FIR_BETA_MAX = 7.0, 14.0

MODE_BANDS = {
    "Fullrange": ["Fullrange"],
    "Fullrange+SUB": ["SUB", "Fullrange"],
    "2Way": ["Low", "High"],
    "2Way+SUB": ["SUB", "Low", "High"],
    "3Way": ["Low", "Mid", "High"],
    "3Way+SUB": ["SUB", "Low", "Mid", "High"],
    "4Way": ["Low", "Low-Mid", "High-Mid", "High"],
    "4Way+SUB": ["SUB", "Low", "Low-Mid", "High-Mid", "High"],
}

MODE_LABELS = {
    mode: f"{mode} ({'/'.join(reversed(bands))})"
    for mode, bands in MODE_BANDS.items()
}


def _mode_boundary_values(mode_key, conf):
    """Return ordered cycles, beta and overlap values for every boundary."""
    count = len(MODE_BANDS[mode_key]) - 1
    cycles = list(conf.get("boundary_cycles", ()))
    betas = list(conf.get("boundary_betas", ()))
    overlaps = list(conf.get("boundary_overlaps_oct", ()))
    fallback_cycles = float(conf.get("cycles", DEFAULT_FIR_CYCLES))
    fallback_beta = float(conf.get("beta", DEFAULT_FIR_BETA))
    cycles = (cycles + [fallback_cycles] * count)[:count]
    betas = (betas + [fallback_beta] * count)[:count]
    overlaps = (overlaps + [0.0] * count)[:count]
    return tuple(map(float, cycles)), tuple(map(float, betas)), tuple(map(float, overlaps))


def _load_app_version():
    try:
        version_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
        with open(version_path, "r", encoding="utf-8") as file:
            version = file.read().strip()
        return version or "unknown"
    except OSError:
        return "unknown"


def _load_phaseeq_integration_version():
    try:
        version_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "VERSION",
        )
        with open(version_path, "r", encoding="utf-8") as file:
            version = file.read().strip()
        return version or "unknown"
    except OSError:
        return "unknown"


APP_VERSION = _load_app_version()
PHASEEQ_INTEGRATION_VERSION = _load_phaseeq_integration_version()

default_settings = {
    "schema_version": 5,
    "2Way": {
        "fs": 96000,
        "cycles": 5.3,
        "beta": 12.0,
        "cross_freqs": [2500],
        "crop_lens": {"Low": 0, "High": 0},
        "eq_files": {"Low": ["", "", ""], "High": ["", "", ""]},
        "baffle_on": False,
        "baffle_atten": 6.0,
        "baffle_f1_hz": 1000,
        "kaiser_overlap_oct": 0.0,
        "crossover_methods": ["Kaiser FIR"],
        "iir_lr2_auto_polarity": True,
    },
    "3Way": {
        "fs": 96000,
        "cycles": DEFAULT_FIR_CYCLES,
        "beta": DEFAULT_FIR_BETA,
        "cycles_low": DEFAULT_FIR_CYCLES,
        "beta_low": DEFAULT_FIR_BETA,
        "cycles_high": 5.3,
        "beta_high": 12.0,
        "cross_freqs": [500, 2500],
        "crop_lens": {"Low": 0, "Mid": 0, "High": 0},
        "eq_files": {"Low": ["", "", ""], "Mid": ["", "", ""], "High": ["", "", ""]},
        "baffle_on": False,
        "baffle_atten": 6.0,
        "baffle_f1_hz": 1000,
        "kaiser_overlap_low_mid_oct": 0.0,
        "kaiser_overlap_mid_high_oct": 0.0,
        "crossover_methods": ["Kaiser FIR", "Kaiser FIR"],
        "iir_lr2_auto_polarity": True,
    },
    "4Way": {
        "fs": 96000,
        "cycles": DEFAULT_FIR_CYCLES,
        "beta": DEFAULT_FIR_BETA,
        "cycles_sub_low": DEFAULT_FIR_CYCLES,
        "beta_sub_low": DEFAULT_FIR_BETA,
        "cycles_low_mid": 4.5,
        "beta_low_mid": DEFAULT_FIR_BETA,
        "cycles_mid_high": 5.3,
        "beta_mid_high": 12.0,
        "cross_freqs": [120, 500, 2500],
        "crop_lens": {"SUB": 0, "Low": 0, "Mid": 0, "High": 0},
        "eq_files": {
            "SUB": ["", "", ""],
            "Low": ["", "", ""],
            "Mid": ["", "", ""],
            "High": ["", "", ""],
        },
        "baffle_on": False,
        "baffle_atten": 6.0,
        "baffle_f1_hz": 1000,
        "kaiser_overlap_sub_low_oct": 0.0,
        "kaiser_overlap_low_mid_oct": 0.0,
        "kaiser_overlap_mid_high_oct": 0.0,
        "crossover_methods": ["Kaiser FIR", "Kaiser FIR", "Kaiser FIR"],
        "iir_lr2_auto_polarity": True,
    },
    "last_mode": "2Way",
    "db_min": -30,
    "gain_y_min_db": -20,
    "gain_y_max_db": 20,
    "phase_gain_mask_db": -60,
    "graph_mode": "Light",
    "graph_max_points": 1024,
    "auto_update": True,
    "output_format": "bin_float32",
    "output_normalize": False,
    "output_layout": "Mono",
    "shared_sub": False,
    "align_output_taps": False,
    "auto_crop": False,
    "auto_crop_profile": DEFAULT_AUTO_CROP_PROFILE,
    "auto_crop_pass_db": DEFAULT_AUTO_CROP_PASS_DB,
    "auto_crop_cross_db": DEFAULT_AUTO_CROP_CROSS_DB
}

default_settings["Fullrange"] = {
    **copy.deepcopy(default_settings["2Way"]),
    "cross_freqs": [],
    "crop_lens": {"Fullrange": 0},
    "eq_files": {"Fullrange": ["", "", ""]},
    "crossover_methods": [],
    "boundary_cycles": [],
    "boundary_betas": [],
    "boundary_overlaps_oct": [],
}
default_settings["Fullrange+SUB"] = {
    **copy.deepcopy(default_settings["2Way"]),
    "cross_freqs": [80],
    "crop_lens": {"SUB": 0, "Fullrange": 0},
    "eq_files": {"SUB": ["", "", ""], "Fullrange": ["", "", ""]},
    "crossover_methods": ["Kaiser FIR"],
    "boundary_cycles": [3.5],
    "boundary_betas": [DEFAULT_FIR_BETA],
    "boundary_overlaps_oct": [0.0],
}

# Preserve the former 4Way (SUB/Low/Mid/High) settings as 3Way+SUB.
default_settings["3Way+SUB"] = copy.deepcopy(default_settings["4Way"])
default_settings["2Way+SUB"] = {
    **copy.deepcopy(default_settings["3Way"]),
    "cross_freqs": [120, 2500],
    "crop_lens": {"SUB": 0, "Low": 0, "High": 0},
    "eq_files": {band: ["", "", ""] for band in MODE_BANDS["2Way+SUB"]},
    "boundary_cycles": [3.5, 5.3],
    "boundary_betas": [DEFAULT_FIR_BETA, DEFAULT_FIR_BETA],
    "boundary_overlaps_oct": [0.0, 0.0],
}
default_settings["4Way"] = {
    **copy.deepcopy(default_settings["4Way"]),
    "cross_freqs": [250, 1000, 4000],
    "crop_lens": {band: 0 for band in MODE_BANDS["4Way"]},
    "eq_files": {band: ["", "", ""] for band in MODE_BANDS["4Way"]},
    "boundary_cycles": [3.5, 4.5, 5.3],
    "boundary_betas": [DEFAULT_FIR_BETA, DEFAULT_FIR_BETA, DEFAULT_FIR_BETA],
    "boundary_overlaps_oct": [0.0, 0.0, 0.0],
}
default_settings["4Way+SUB"] = {
    **copy.deepcopy(default_settings["4Way"]),
    "cross_freqs": [80, 250, 1000, 4000],
    "crop_lens": {band: 0 for band in MODE_BANDS["4Way+SUB"]},
    "eq_files": {band: ["", "", ""] for band in MODE_BANDS["4Way+SUB"]},
    "boundary_cycles": [3.5, 3.5, 4.5, 5.3],
    "boundary_betas": [DEFAULT_FIR_BETA, DEFAULT_FIR_BETA, DEFAULT_FIR_BETA, DEFAULT_FIR_BETA],
    "boundary_overlaps_oct": [0.0, 0.0, 0.0, 0.0],
}
for _default_mode, _default_bands in MODE_BANDS.items():
    _default_conf = default_settings[_default_mode]
    _default_methods = list(_default_conf.get("crossover_methods", ()))
    _default_methods = (
        _default_methods + ["Kaiser FIR"] * max(0, len(_default_bands) - 1)
    )[:max(0, len(_default_bands) - 1)]
    _default_conf["crossover_methods"] = _default_methods
    _default_conf["fir_output_enabled"] = bool(bands_requiring_split_fir(
        _default_bands, _default_methods,
    ))

# ===== ログ出力用 =====
def append_log(msg):
    if "log_msgs" not in st.session_state:
        st.session_state["log_msgs"] = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["log_msgs"].insert(0, f"`[{now}]` {msg}")

def clear_results():
    st.session_state.pop('last_conf_signature', None)
    st.session_state['results_stale'] = True


def _current_theme_mode():
    try:
        theme_type = str(st.context.theme.type or "").lower()
    except (AttributeError, TypeError):
        theme_type = ""
    if theme_type in {"dark", "light"}:
        st.session_state["_composite_studio_ui_theme"] = theme_type
        return theme_type
    return str(st.session_state.get("_composite_studio_ui_theme", "light"))


def _table_theme_colors():
    if _current_theme_mode() == "dark":
        return {
            "background": "#121719",
            "alternate": "#1a2123",
            "header": "#283438",
            "border": "#3b494c",
            "text": "#e8ece9",
            "header_text": "#f7f8f6",
            "muted": "#aeb9b6",
            "good": "#72d39a",
            "warn": "#dfc273",
            "bad": "#f28c83",
        }
    return {
        "background": "#f7f7f5",
        "alternate": "#ecefed",
        "header": "#253238",
        "border": "#c8cfcc",
        "text": "#182126",
        "header_text": "#ffffff",
        "muted": "#536066",
        "good": "#28734a",
        "warn": "#806229",
        "bad": "#b53a32",
    }


def _apply_app_theme():
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"],
        [data-testid="stMain"] .block-container {
            width: 100%;
            max-width: 1240px;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"],
            [data-testid="stMain"] .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li {
            font-size: 16px;
            line-height: 1.58;
        }
        [data-testid="stCaptionContainer"] p {
            font-size: 14px;
            line-height: 1.5;
            opacity: 0.82;
        }
        [data-testid="stWidgetLabel"] p {
            font-size: 15px;
            line-height: 1.4;
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-weight: 650;
        }
        [data-testid="stSidebar"] .sidebar-workflow {
            margin: 0.15rem 0 0.7rem;
            padding: 0.65rem 0.7rem;
            border: 1px solid color-mix(in srgb, var(--primary-color) 35%, transparent);
            border-radius: 0.45rem;
            background: color-mix(in srgb, var(--secondary-background-color) 88%, transparent);
        }
        [data-testid="stSidebar"] .sidebar-workflow-title {
            margin-bottom: 0.4rem;
            font-size: 0.82rem;
            font-weight: 750;
            letter-spacing: 0.04em;
        }
        [data-testid="stSidebar"] .sidebar-workflow-steps {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.25rem 0.4rem;
            font-size: 0.76rem;
            line-height: 1.35;
            opacity: 0.88;
        }
        [data-testid="stSidebar"] .sidebar-section-tag {
            margin-left: auto;
            padding: 0.12rem 0.42rem;
            border: 1px solid color-mix(in srgb, currentColor 28%, transparent);
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 650;
            opacity: 0.8;
            white-space: nowrap;
        }
        .stButton button,
        .stDownloadButton button {
            font-size: 15px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(semantic_button_css(_current_theme_mode()), unsafe_allow_html=True)
    if _current_theme_mode() == "dark":
        st.markdown(
            """
            <style>
            :root {
                --background-color: #0f1416;
                --secondary-background-color: #182023;
                --text-color: #e8ece9;
                --primary-color: #62a5a2;
            }
            html, body, .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"] {
                background: #0f1416;
                color: #e8ece9;
            }
            [data-testid="stHeader"] {
                background: rgba(15, 20, 22, 0.95);
                border-bottom: 1px solid #2c373a;
            }
            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"] {
                background: #182023;
            }
            [data-testid="stSidebar"] hr,
            [data-testid="stMain"] hr {
                border-color: #354245;
            }
            [data-baseweb="input"],
            [data-baseweb="select"] > div,
            [data-baseweb="textarea"],
            [data-testid="stFileUploaderDropzone"] {
                background: #20292c;
                border-color: #435155;
                color: #edf0ee;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035);
            }
            [data-baseweb="popover"],
            [data-baseweb="menu"] {
                background: #20292c;
            }
            [data-testid="stExpander"] {
                background: #151c1e;
                border-color: #354245;
            }
            [data-testid="stNotification"] {
                border-color: #435155;
            }
            [data-testid="stSidebar"] .sidebar-section-header {
                display: flex;
                align-items: center;
                gap: 0.55rem;
                margin: 1.15rem 0 0.35rem;
                padding: 0.55rem 0.65rem;
                background: #20292c;
                border-left: 4px solid #72aaa7;
                border-radius: 0.3rem;
                color: #eef1ee;
                font-weight: 700;
            }
            [data-testid="stSidebar"] .sidebar-section-number {
                color: #9bc5c1;
                font-size: 0.85rem;
            }
            [data-testid="stSidebar"] .crossover-header {
                margin: 0.85rem 0 0.35rem;
                padding: 0.4rem 0.55rem;
                background: #192123;
                border: 1px solid #354245;
                border-radius: 0.3rem;
                color: #eef1ee;
                font-weight: 650;
            }
            .stButton > button[kind="primary"],
            .stDownloadButton > button[kind="primary"] {
                background: #3e6b59;
                border-color: #79a991;
                color: #ffffff;
            }
            .stButton > button[kind="primary"]:hover,
            .stDownloadButton > button[kind="primary"]:hover {
                background: #365e4d;
                border-color: #79a991;
            }
            h1, h2, h3 {
                color: #eef1ee;
            }
            h1 {
                border-bottom: 1px solid #8f7b51;
                padding-bottom: 0.28rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        """
        <style>
        :root {
            --background-color: #f2f3f1;
            --secondary-background-color: #e6eae7;
            --text-color: #182126;
            --primary-color: #2f6265;
        }
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #f2f3f1;
            color: #182126;
        }
        [data-testid="stHeader"] {
            background: rgba(242, 243, 241, 0.95);
            border-bottom: 1px solid #d6dbd8;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background: #e6eae7;
        }
        [data-testid="stSidebar"] hr,
        [data-testid="stMain"] hr {
            border-color: #c8cfcc;
        }
        [data-baseweb="input"],
        [data-baseweb="select"] > div,
        [data-baseweb="textarea"],
        [data-testid="stFileUploaderDropzone"] {
            background: #fafaf8;
            border-color: #bdc6c2;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }
        [data-baseweb="popover"],
        [data-baseweb="menu"] {
            background: #fafaf8;
        }
        [data-testid="stExpander"] {
            background: #f7f7f5;
            border-color: #c8cfcc;
        }
        [data-testid="stNotification"] {
            border-color: #bdc6c2;
        }
        [data-testid="stSidebar"] .sidebar-section-header {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 1.15rem 0 0.35rem;
            padding: 0.55rem 0.65rem;
            background: #d9e0dc;
            border-left: 4px solid #2f6265;
            border-radius: 0.3rem;
            color: #253238;
            font-weight: 700;
        }
        [data-testid="stSidebar"] .sidebar-section-number {
            color: #2f6265;
            font-size: 0.85rem;
        }
        [data-testid="stSidebar"] .crossover-header {
            margin: 0.85rem 0 0.35rem;
            padding: 0.4rem 0.55rem;
            background: #f7f7f5;
            border: 1px solid #c8cfcc;
            border-radius: 0.3rem;
            color: #253238;
            font-weight: 650;
        }
        .stButton > button[kind="primary"],
        .stDownloadButton > button[kind="primary"] {
            background: #3e6b59;
            border-color: #3e6b59;
            color: #ffffff;
        }
        .stButton > button[kind="primary"]:hover,
        .stDownloadButton > button[kind="primary"]:hover {
            background: #365e4d;
            border-color: #3e6b59;
        }
        h1, h2, h3 {
            color: #253238;
        }
        h1 {
            border-bottom: 1px solid #b6a071;
            padding-bottom: 0.28rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if (
    st.session_state.get("_group_delay_display_revision")
    != visualization.GROUP_DELAY_DISPLAY_REVISION
):
    st.session_state.pop("studio_chart_bundle", None)
    st.session_state.pop("studio_chart_source", None)
    st.session_state.pop("last_conf_signature", None)
    st.session_state["_group_delay_display_revision"] = (
        visualization.GROUP_DELAY_DISPLAY_REVISION
    )


def _number_input_with_session_default(label, *, key, value, **kwargs):
    if key not in st.session_state:
        kwargs["value"] = value
    return st.number_input(label, key=key, **kwargs)


def _show_fir_design_guide(beta, *, method, sample_rate_hz, crossover_hz, cycles, mode_key, index):
    if method not in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}:
        return
    taps = odd_number(float(sample_rate_hz) / max(float(crossover_hz), 1.0) * float(cycles))
    if method == "Kaiser FIR":
        attenuation_db = kaiser_beta_to_attenuation_db(beta)
        usage_hint = display_text(kaiser_attenuation_usage_hint(attenuation_db))
        text = ui_message('ui.6fbeca11e519ab', p0=f'{attenuation_db:.0f}', p1=usage_hint, p2=f'{taps:,}')
    else:
        from ui.band_split_guide import render_band_split_guide
        from composite_engine.multiway_studio.extension import _boundary_overlap_from_state
        render_band_split_guide(method=method, sample_rate=sample_rate_hz,
            crossover_hz=crossover_hz, cycles=cycles, beta=beta,
            overlap_oct=_boundary_overlap_from_state(mode_key, index), current_auto_taps=True)
        return
    st.caption(
        text,
        help=ui_message('ui.ca1db9e3e7b836'),
    )
    with st.popover(ui_message('ui.cecdffbccd548d')):
        st.markdown(display_text(KAISER_SLOPE_GUIDE))


def _sidebar_section_header(number, title, caption=None, *, tag=None):
    tag_html = (
        f'<span class="sidebar-section-tag">{html.escape(str(tag))}</span>'
        if tag else ""
    )
    st.markdown(
        (
            '<div class="sidebar-section-header">'
            f'<span class="sidebar-section-number">{html.escape(str(number))}</span>'
            f"<span>{html.escape(title)}</span>"
            f"{tag_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(display_text(caption))


def _crossover_header(title):
    st.markdown(
        f'<div class="crossover-header">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def _checkbox_with_session_default(label, *, key, value, **kwargs):
    if key not in st.session_state:
        kwargs["value"] = value
    return st.checkbox(label, key=key, **kwargs)


def _auto_crop_profile_from_limits(pass_db, cross_db):
    pass_db = float(pass_db)
    cross_db = float(cross_db)
    return min(
        AUTO_CROP_PROFILES,
        key=lambda profile_key: (
            abs(pass_db - AUTO_CROP_PROFILES[profile_key]["pass_db"])
            + abs(cross_db - AUTO_CROP_PROFILES[profile_key]["cross_db"])
        ),
    )


def _sum_response_db(firs_by_band, fs, worN=16384):
    items = list(firs_by_band.items())
    if not items:
        return np.array([]), np.array([])
    names = [name for name, _ in items]
    aligned = center_fir_lengths([h for _, h in items])
    w = None
    H_sum = None
    for _, h in zip(names, aligned):
        w_i, H_i = freqz(h, worN=worN, fs=fs)
        if w is None:
            w = w_i
            H_sum = np.array(H_i, dtype=np.complex128)
        else:
            H_sum += H_i
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H_sum), 1e-12))
    return w, mag_db


def _crossover_dip_metrics(freqs_hz, mag_db, fc_hz):
    if len(freqs_hz) == 0:
        return {"fc_hz": fc_hz, "dip_db": np.nan, "fc_level_db": np.nan, "local_max_db": np.nan, "local_min_db": np.nan}
    f_lo = max(20.0, fc_hz * (2 ** (-1/6)))
    f_hi = min(float(freqs_hz[-1]), fc_hz * (2 ** (1/6)))
    mask = (freqs_hz >= f_lo) & (freqs_hz <= f_hi)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(freqs_hz - fc_hz)))
        level = float(mag_db[idx])
        return {"fc_hz": fc_hz, "dip_db": 0.0, "fc_level_db": level, "local_max_db": level, "local_min_db": level}
    local = mag_db[mask]
    local_max = float(np.max(local))
    local_min = float(np.min(local))
    idx_fc = int(np.argmin(np.abs(freqs_hz - fc_hz)))
    fc_level = float(mag_db[idx_fc])
    return {
        "fc_hz": fc_hz,
        "dip_db": max(0.0, local_max - local_min),
        "fc_level_db": fc_level,
        "local_max_db": local_max,
        "local_min_db": local_min,
    }


def _render_dip_metrics_table(before_metrics, after_metrics):
    rows = []
    for label in before_metrics.keys():
        b = before_metrics[label]
        a = after_metrics[label]
        rows.append({
            "クロス": label,
            "基準周波数 [Hz]": int(round_for_display(a["fc_hz"], 0)),
            "補正前 変動量 [dB]": round_for_display(b["dip_db"], 3),
            "補正後 変動量 [dB]": round_for_display(a["dip_db"], 3),
            "改善量 [dB]": round_for_display(b["dip_db"] - a["dip_db"], 3),
            "補正前 fcレベル [dB]": round_for_display(b["fc_level_db"], 3),
            "補正後 fcレベル [dB]": round_for_display(a["fc_level_db"], 3),
        })
    return rows


def _dip_quality_label(after_dip_db, improvement_db):
    if after_dip_db <= 0.5 and improvement_db >= -0.05:
        return '良'
    if after_dip_db <= 1.5 and improvement_db >= -0.2:
        return '注意'
    return '要調整'


def _style_dip_dataframe(df: pd.DataFrame):
    def row_style(row):
        label = row['評価']
        if label == '良':
            color = '#e8f7e8'
        elif label == '注意':
            color = '#fff6d9'
        else:
            color = '#fdeaea'
        return [f'background-color: {color}'] * len(row)

    def val_color(val):
        try:
            v = float(val)
        except Exception:
            return ''
        if v >= 0.3:
            return 'color: #17803d; font-weight: 700'
        if v < 0:
            return 'color: #b42318; font-weight: 700'
        return ''

    return (df.style
        .apply(row_style, axis=1)
        .map(val_color, subset=['改善量 [dB]'])
        .format({
            '補正前 変動量 [dB]': '{:.3f}',
            '補正後 変動量 [dB]': '{:.3f}',
            '改善量 [dB]': '{:.3f}',
            '補正前 fcレベル [dB]': '{:.3f}',
            '補正後 fcレベル [dB]': '{:.3f}',
        }))


def _ordered_dip_dataframe(rows):
    cols = [
        "クロス",
        "基準周波数 [Hz]",
        "補正前 変動量 [dB]",
        "補正後 変動量 [dB]",
        "改善量 [dB]",
        "補正前 fcレベル [dB]",
        "補正後 fcレベル [dB]",
        "評価",
    ]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def _render_dip_metrics_html(rows):
    df = _ordered_dip_dataframe(rows)
    colors = _table_theme_colors()

    def fmt(v, integer=False):
        try:
            if v == "":
                return ""
            if integer:
                return f"{int(round(float(v)))}"
            return f"{float(v):.3f}"
        except Exception:
            return str(v)

    headers = list(df.columns)
    html = []
    html.append(f"""
    <style>
    .dip-table-wrap {{ overflow-x: auto; margin: 0.25rem 0 1rem 0; }}
        table.dip-table {{
        border-collapse: collapse;
        width: 100%;
        min-width: 980px;
        font-size: 15px;
        background: {colors["background"]};
        color: {colors["text"]};
    }}
    .dip-table th, .dip-table td {{
        border: 1px solid {colors["border"]};
        padding: 0.55rem 0.68rem;
        text-align: right;
        white-space: nowrap;
        color: {colors["text"]};
    }}
    .dip-table th:first-child, .dip-table td:first-child,
    .dip-table th:last-child, .dip-table td:last-child {{
        text-align: center;
    }}
    .dip-table thead th {{
        background: {colors["header"]};
        font-weight: 700;
        color: {colors["header_text"]};
    }}
    .dip-table td.metric {{
        font-weight: 600;
    }}
    .dip-table td.improvement-good {{
        color: {colors["good"]};
        font-weight: 700;
    }}
    .dip-table td.improvement-bad {{
        color: {colors["bad"]};
        font-weight: 700;
    }}
    .dip-table td.eval-good {{
        color: {colors["good"]};
        font-weight: 700;
    }}
    .dip-table td.eval-warn {{
        color: {colors["warn"]};
        font-weight: 700;
    }}
    .dip-table td.eval-bad {{
        color: {colors["bad"]};
        font-weight: 700;
    }}
    .dip-table .dip-row-even {{
        background: {colors["alternate"]};
    }}
    .dip-table .dip-row-odd {{
        background: {colors["background"]};
    }}
    </style>
    <div class="dip-table-wrap">
    <table class="dip-table">
      <thead><tr>
    """)
    html.append("".join(f"<th>{display_text(h)}</th>" for h in headers))
    html.append("</tr></thead><tbody>")

    for row_index, (_, row) in enumerate(df.iterrows()):
        label = row["評価"]
        if label == "良":
            eval_txt = "🟢 " + display_text("良")
        elif label == "注意":
            eval_txt = "🟡 " + display_text("注意")
        else:
            eval_txt = "🔴 " + display_text("要調整")
        row_class = "dip-row-even" if row_index % 2 == 0 else "dip-row-odd"

        imp_raw = row["改善量 [dB]"]
        try:
            imp = float(imp_raw)
            if imp >= 0.3:
                imp_class = 'improvement-good'
            elif imp < 0:
                imp_class = 'improvement-bad'
            else:
                imp_class = 'metric'
            imp_html = f'<td class="{imp_class}">{imp:.3f}</td>'
        except Exception:
            imp_html = f'<td class="metric">{str(imp_raw)}</td>'

        eval_class = 'eval-good' if label == '良' else ('eval-warn' if label == '注意' else 'eval-bad')
        cells_html = [
            f'<td>{str(row["クロス"])}</td>',
            f'<td class="metric">{fmt(row["基準周波数 [Hz]"], integer=True)}</td>',
            f'<td class="metric">{fmt(row["補正前 変動量 [dB]"])}</td>',
            f'<td class="metric">{fmt(row["補正後 変動量 [dB]"])}</td>',
            imp_html,
            f'<td class="metric">{fmt(row["補正前 fcレベル [dB]"])}</td>',
            f'<td class="metric">{fmt(row["補正後 fcレベル [dB]"])}</td>',
            f'<td class="{eval_class}">{eval_txt}</td>',
        ]
        html.append(f'<tr class="{row_class}">' + "".join(cells_html) + "</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_styled_table(data, min_width_px=720):
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if df.empty:
        return
    colors = _table_theme_colors()
    df = df.copy()
    for column in ("タップ取得元", "適用済みAll-pass", "Speaker応答", "PhaseEQ FIR", "基準", "ゲイン判定"):
        if column in df:
            df[column] = df[column].map(display_text)
    if "対象" in df:
        df["対象"] = df["対象"].map(lambda value: display_text(value) if value == "Sum（全帯域合成）" else value)

    headers = "".join(f"<th>{html.escape(display_text(column))}</th>" for column in df.columns)
    body_rows = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        row_class = "appy-row-even" if row_index % 2 == 0 else "appy-row-odd"
        cells = "".join(
            f"<td>{html.escape('' if pd.isna(value) else str(value))}</td>"
            for value in row
        )
        body_rows.append(f'<tr class="{row_class}">{cells}</tr>')

    st.markdown(
        f"""
        <style>
        .appy-table-wrap {{
            overflow-x: auto;
            margin: 0.25rem 0 1rem 0;
        }}
        table.appy-table {{
            border-collapse: collapse;
            width: 100%;
            min-width: {int(min_width_px)}px;
            font-size: 15px;
            background: {colors["background"]};
            color: {colors["text"]};
        }}
        .appy-table th, .appy-table td {{
            border: 1px solid {colors["border"]};
            padding: 0.55rem 0.68rem;
            text-align: right;
            white-space: nowrap;
            color: {colors["text"]};
        }}
        .appy-table th:first-child, .appy-table td:first-child {{
            text-align: center;
        }}
        .appy-table thead th {{
            background: {colors["header"]};
            color: {colors["header_text"]};
            font-weight: 700;
        }}
        .appy-table .appy-row-even {{
            background: {colors["alternate"]};
        }}
        .appy-table .appy-row-odd {{
            background: {colors["background"]};
        }}
        </style>
        <div class="appy-table-wrap">
          <table class="appy-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_output_gain_table(normalization_info, *, way_settings=None, responses=None):
    if not normalization_info.get("enabled") and not responses:
        return
    rows = []
    for band, response in (responses or {}).items():
        settings = (way_settings or {}).get(band, {})
        rows.append({
            "対象": band,
            "FIR共通調整 [dB]": round_for_display(float(normalization_info.get("gain_db", 0.0)), 4),
            "Channel Gain [dB]": round_for_display(float(settings.get("gain_db", 0.0)), 4),
            "適用後ピーク [dB]": round_for_display(
                20.0 * np.log10(max(float(np.max(np.abs(response))), 1e-15)), 4
            ),
            "ゲイン判定": "対象",
        })
    summed = np.sum(tuple((responses or {}).values()), axis=0) if responses else np.array([0.0])
    rows.append({
        "対象": "Sum（全帯域合成）",
        "FIR共通調整 [dB]": round_for_display(float(normalization_info.get("gain_db", 0.0)), 4),
        "Channel Gain [dB]": "—",
        "適用後ピーク [dB]": round_for_display(
            20.0 * np.log10(max(float(np.max(np.abs(summed))), 1e-15)), 4
        ),
        "ゲイン判定": "表示のみ",
    })
    st.subheader(ui_message('ui.5b0ce063e8d4fd'))
    st.caption(
        ui_message('ui.4cbd5af56995d1')
    )
    _render_styled_table(rows)


def _auto_crop_masks(freqs, mag_db, band, mode_key, cross_freqs, fs):
    valid = mag_db > -80.0
    ordered_bands = MODE_BANDS[mode_key]
    band_index = ordered_bands.index(band)
    boundaries = tuple(map(float, cross_freqs))
    if not boundaries:
        return (freqs >= 20.0) & (freqs <= fs * 0.45) & valid, []
    if band_index == 0:
        pass_mask = (freqs >= 20.0) & (freqs <= boundaries[0] * 0.8) & valid
        adjacent = (boundaries[0],)
    elif band_index == len(ordered_bands) - 1:
        pass_mask = (freqs >= boundaries[-1] * 1.2) & (freqs <= fs * 0.45) & valid
        adjacent = (boundaries[-1],)
    else:
        lower, upper = boundaries[band_index - 1], boundaries[band_index]
        pass_mask = (freqs >= lower * 1.2) & (freqs <= upper * 0.8) & valid
        adjacent = (lower, upper)
    cross_masks = [
        (freqs >= fc * 0.7) & (freqs <= fc * 1.3) & valid
        for fc in adjacent
    ]
    return pass_mask, cross_masks


def _auto_crop_pass_error(reference, candidate, band, mode_key, cross_freqs, fs):
    freqs, ref_h = freqz(reference, worN=32768, fs=fs)
    _, candidate_h = freqz(candidate, worN=32768, fs=fs)
    ref_db = 20.0 * np.log10(np.maximum(np.abs(ref_h), 1e-12))
    candidate_db = 20.0 * np.log10(np.maximum(np.abs(candidate_h), 1e-12))
    diff_db = candidate_db - ref_db
    pass_mask, _ = _auto_crop_masks(
        freqs, ref_db, band, mode_key, cross_freqs, fs
    )
    if np.any(pass_mask):
        return float(np.max(np.abs(diff_db[pass_mask])))
    if len(reference) == len(candidate) and np.array_equal(reference, candidate):
        return 0.0
    return np.inf


def _combine_band_fir(split_fir, baffle_fir, eq_fir, crop_len):
    fir = combine_and_crop_fir_conv(split_fir, baffle_fir, crop_len)
    return combine_and_crop_fir_conv(fir, eq_fir, crop_len)


def _auto_crop_cross_pairs(mode_key, cross_freqs):
    bands = MODE_BANDS[mode_key]
    return [
        (
            f"{bands[index]}/{bands[index + 1]}",
            (bands[index], bands[index + 1]),
            float(frequency),
        )
        for index, frequency in enumerate(cross_freqs)
    ]


def _auto_crop_sum_error(reference_firs, candidate_firs, pair_bands, fc, fs):
    reference_pair = {band: reference_firs[band] for band in pair_bands}
    candidate_pair = {band: candidate_firs[band] for band in pair_bands}
    freqs, reference_db = _sum_response_db(reference_pair, fs, worN=32768)
    _, candidate_db = _sum_response_db(candidate_pair, fs, worN=32768)
    mask = (freqs >= fc * 0.7) & (freqs <= fc * 1.3) & (reference_db > -80.0)
    if not np.any(mask):
        return np.inf, np.inf, -np.inf
    diff_db = candidate_db[mask] - reference_db[mask]
    peak_db = max(0.0, float(np.max(diff_db)))
    dip_db = min(0.0, float(np.min(diff_db)))
    return float(np.max(np.abs(diff_db))), peak_db, dip_db


def _auto_crop_all_bands(
    split_firs,
    baffle_fir,
    eq_firs,
    mode_key,
    cross_freqs,
    fs,
    pass_limit_db=0.1,
    cross_limit_db=0.05,
    align_output_taps=False,
):
    bands = list(split_firs)
    references = {
        band: _combine_band_fir(split_firs[band], baffle_fir, eq_firs.get(band), 0)
        for band in bands
    }
    minimum_taps = {band: len(split_firs[band]) for band in bands}
    reference_lengths = {
        band: len(references[band])
        for band in bands
    }
    search_bands, candidate_taps = plan_auto_crop_candidates(
        reference_lengths,
        minimum_taps,
        align_output_taps=align_output_taps,
    )
    steps = {}
    for band in bands:
        step = max(2, round(minimum_taps[band] * 0.05))
        steps[band] = step + (step % 2)
    candidates = {}
    pass_errors = {}
    cross_results = []

    while True:
        candidates = {
            band: _combine_band_fir(
                split_firs[band], baffle_fir, eq_firs.get(band), candidate_taps[band]
            )
            for band in bands
        }
        pass_errors = {
            band: _auto_crop_pass_error(
                references[band], candidates[band], band, mode_key, cross_freqs, fs
            )
            for band in bands
        }
        cross_results = []
        cross_raw_errors = []
        for label, pair_bands, fc in _auto_crop_cross_pairs(mode_key, cross_freqs):
            error_db, peak_db, dip_db = _auto_crop_sum_error(
                references, candidates, pair_bands, fc, fs
            )
            cross_raw_errors.append(error_db)
            cross_results.append({
                "クロス": label,
                "基準周波数 [Hz]": int(round_for_display(fc, 0)),
                "合成波 最大誤差 [dB]": round_for_display(error_db, 4),
                "合成波 ピーク [dB]": round_for_display(peak_db, 4),
                "合成波 ディップ [dB]": round_for_display(dip_db, 4),
            })

        grow_bands = {
            band for band, error_db in pass_errors.items()
            if band in search_bands and error_db > pass_limit_db
        }
        for error_db, (_, pair_bands, _) in zip(
            cross_raw_errors, _auto_crop_cross_pairs(mode_key, cross_freqs)
        ):
            if error_db > cross_limit_db:
                grow_bands.update(
                    band for band in pair_bands
                    if band in search_bands
                )
        if not grow_bands:
            break

        changed = False
        for band in grow_bands:
            current = candidate_taps[band]
            maximum = len(references[band])
            if current >= maximum:
                continue
            next_taps = min(maximum, current + steps[band])
            if next_taps < maximum and next_taps % 2 == 0:
                next_taps += 1
            candidate_taps[band] = min(next_taps, maximum)
            changed = changed or candidate_taps[band] != current
        if not changed:
            break

    rows = []
    for band in bands:
        reduction = len(references[band]) - len(candidates[band])
        adjacent_errors = [
            row["合成波 最大誤差 [dB]"]
            for row, (_, pair_bands, _) in zip(
                cross_results, _auto_crop_cross_pairs(mode_key, cross_freqs)
            )
            if band in pair_bands
        ]
        adjacent_error = (
            round_for_display(max(adjacent_errors), 4)
            if adjacent_errors
            else "—"
        )
        rows.append({
            "帯域": band,
            "探索": (
                "○"
                if band in search_bands
                else "－"
            ),
            "帯域分割時 [taps]": int(minimum_taps[band]),
            "合成後 [taps]": int(len(references[band])),
            "自動クロップ後 [taps]": int(len(candidates[band])),
            "削減 [taps]": int(reduction),
            "削減率 [%]": round_for_display(reduction / len(references[band]) * 100.0, 1),
            "探索刻み [taps]": int(steps[band]),
            "通過帯誤差 [dB]": round_for_display(pass_errors[band], 4),
            "隣接クロス合成波 最大誤差 [dB]": adjacent_error,
        })
    return candidates, rows, cross_results


def _render_auto_crop_table(
    rows, cross_rows=None, pass_limit_db=None, cross_limit_db=None
):
    if not rows:
        return
    st.subheader(ui_message('ui.b2d525093e0d6e'))
    limits = ""
    if pass_limit_db is not None and cross_limit_db is not None:
        limits = (
            f" 合格条件: 通過帯 {pass_limit_db:.2f}dB以下 / "
            f"合成波クロス近傍 {cross_limit_db:.2f}dB以下。"
        )
    st.caption(
        ui_message('ui.4d587393cacd05', p0=f'{limits}')
    )
    if any(row.get("探索", "○") == "－" for row in rows):
        st.caption(
            ui_message('ui.df5a3fb7e0e150')
        )
    if any(row.get("隣接クロス合成波 最大誤差 [dB]") == "—" for row in rows):
        st.caption(
            ui_message('ui.ab00bc5209c64e')
        )
    _render_styled_table(rows, min_width_px=1040)
    if cross_rows:
        st.caption(ui_message('ui.ec79335034b561'))
        _render_styled_table(cross_rows)


# ===== 設定管理 =====
def _deep_merge_dict(base, override):
    merged = {}
    for k, v in base.items():
        if isinstance(v, dict):
            ov = override.get(k, {}) if isinstance(override.get(k, {}), dict) else {}
            merged[k] = _deep_merge_dict(v, ov)
        else:
            merged[k] = override.get(k, v)
    for k, v in override.items():
        if k not in merged:
            merged[k] = v
    return merged


def _coerce_float(value, default, min_value=None, max_value=None):
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), out)
    if max_value is not None:
        out = min(float(max_value), out)
    return out


def _coerce_int(value, default, min_value=None, max_value=None):
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), out)
    if max_value is not None:
        out = min(int(max_value), out)
    return out


def _normalize_eq_files(eq_files, bands):
    out = {}
    eq_files = eq_files if isinstance(eq_files, dict) else {}
    for band in bands:
        vals = eq_files.get(band, ["", "", ""])
        if not isinstance(vals, list):
            vals = ["", "", ""]
        vals = [str(v) if v else "" for v in vals[:3]]
        vals += [""] * (3 - len(vals))
        out[band] = vals
    return out


def _normalize_crop_lens(crop_lens, bands):
    out = {}
    crop_lens = crop_lens if isinstance(crop_lens, dict) else {}
    for band in bands:
        out[band] = _coerce_int(crop_lens.get(band, 0), 0, min_value=0)
    return out


def normalize_settings(raw_settings):
    raw_settings = copy.deepcopy(raw_settings) if isinstance(raw_settings, dict) else {}
    if _coerce_int(raw_settings.get("schema_version", 3), 3) <= 3:
        legacy_4way = raw_settings.get("4Way")
        if isinstance(legacy_4way, dict) and "3Way+SUB" not in raw_settings:
            raw_settings["3Way+SUB"] = copy.deepcopy(legacy_4way)
            if raw_settings.get("last_mode") == "4Way":
                raw_settings["last_mode"] = "3Way+SUB"
    settings = _deep_merge_dict(default_settings, raw_settings)
    settings["schema_version"] = 5
    settings["phase_alignment_auto_structure"] = bool(settings.get("phase_alignment_auto_structure", False))
    settings["output_layout"] = (
        str(settings.get("output_layout", "Mono"))
        if str(settings.get("output_layout", "Mono")) in {"Mono", "Stereo"}
        else "Mono"
    )
    settings["shared_sub"] = bool(settings.get("shared_sub", False))

    for mode_key, bands in MODE_BANDS.items():
        mode_conf = settings.get(mode_key, {})
        raw_mode_conf = raw_settings.get(mode_key, {}) if isinstance(raw_settings.get(mode_key, {}), dict) else {}
        defaults = default_settings[mode_key]
        mode_conf["phase_alignment_acoustic_target"] = bool(mode_conf.get("phase_alignment_acoustic_target", False))

        mode_conf["fs"] = _coerce_int(mode_conf.get("fs", defaults["fs"]), defaults["fs"], min_value=8000)
        mode_conf["cycles"] = _coerce_float(mode_conf.get("cycles", defaults["cycles"]), defaults["cycles"], min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX)
        mode_conf["beta"] = _coerce_float(mode_conf.get("beta", defaults["beta"]), defaults["beta"], min_value=FIR_BETA_MIN, max_value=FIR_BETA_MAX)

        cross = mode_conf.get("cross_freqs", defaults["cross_freqs"])
        if not isinstance(cross, list):
            cross = defaults["cross_freqs"]
        needed = len(bands) - 1
        cross = (cross + defaults["cross_freqs"])[:needed]
        mode_conf["cross_freqs"] = [_coerce_int(v, defaults["cross_freqs"][i], min_value=10) for i, v in enumerate(cross)]
        raw_methods = mode_conf.get("crossover_methods", defaults["crossover_methods"])
        if not isinstance(raw_methods, list):
            raw_methods = defaults["crossover_methods"]
        mode_conf["crossover_methods"] = [
            normalize_crossover_method(
                raw_methods[index] if index < len(raw_methods) else "Kaiser FIR"
            )
            for index in range(needed)
        ]
        target_flags = mode_conf.get("boundary_acoustic_targets", [])
        mode_conf["boundary_acoustic_targets"] = [bool(target_flags[i]) and method not in {"Kaiser FIR", "Through"} if i < len(target_flags) else False for i, method in enumerate(mode_conf["crossover_methods"])]
        mode_conf["iir_lr2_auto_polarity"] = bool(
            mode_conf.get("iir_lr2_auto_polarity", True)
        )

        mode_conf["crop_lens"] = _normalize_crop_lens(mode_conf.get("crop_lens", {}), bands)
        mode_conf["eq_files"] = _normalize_eq_files(mode_conf.get("eq_files", {}), bands)
        mode_conf["baffle_on"] = bool(mode_conf.get("baffle_on", defaults["baffle_on"]))
        mode_conf["baffle_atten"] = _coerce_float(mode_conf.get("baffle_atten", defaults["baffle_atten"]), defaults["baffle_atten"], min_value=0.0, max_value=12.0)
        mode_conf["baffle_f1_hz"] = _coerce_int(mode_conf.get("baffle_f1_hz", defaults["baffle_f1_hz"]), defaults["baffle_f1_hz"], min_value=100)
        if "fir_output_enabled" in raw_mode_conf:
            mode_conf["fir_output_enabled"] = bool(raw_mode_conf["fir_output_enabled"])
        else:
            mode_conf["fir_output_enabled"] = inferred_fir_output_enabled(
                bands,
                mode_conf["crossover_methods"],
                mode_conf["crop_lens"],
                baffle_enabled=mode_conf["baffle_on"],
                eq_files=mode_conf["eq_files"],
            )

        mode_conf["crossover_methods"] = convert_studio_methods(
            mode_conf["crossover_methods"], fir_enabled=mode_conf["fir_output_enabled"],
        )

        if mode_key == "2Way":
            mode_conf["kaiser_overlap_oct"] = _coerce_float(mode_conf.get("kaiser_overlap_oct", defaults["kaiser_overlap_oct"]), defaults["kaiser_overlap_oct"], min_value=-1.0, max_value=1.0)
        elif mode_key == "3Way":
            mode_conf["kaiser_overlap_low_mid_oct"] = _coerce_float(mode_conf.get("kaiser_overlap_low_mid_oct", defaults["kaiser_overlap_low_mid_oct"]), defaults["kaiser_overlap_low_mid_oct"], min_value=-1.0, max_value=1.0)
            mode_conf["kaiser_overlap_mid_high_oct"] = _coerce_float(mode_conf.get("kaiser_overlap_mid_high_oct", defaults["kaiser_overlap_mid_high_oct"]), defaults["kaiser_overlap_mid_high_oct"], min_value=-1.0, max_value=1.0)
            legacy_cycles = mode_conf["cycles"] if "cycles" in raw_mode_conf else None
            legacy_beta = mode_conf["beta"] if "beta" in raw_mode_conf else None
            mode_conf["cycles_low"] = _coerce_float(
                raw_mode_conf.get("cycles_low", legacy_cycles if legacy_cycles is not None else defaults["cycles_low"]),
                defaults["cycles_low"],
                min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX,
            )
            mode_conf["beta_low"] = _coerce_float(
                raw_mode_conf.get("beta_low", legacy_beta if legacy_beta is not None else defaults["beta_low"]),
                defaults["beta_low"],
                min_value=FIR_BETA_MIN,
                max_value=FIR_BETA_MAX,
            )
            mode_conf["cycles_high"] = _coerce_float(
                raw_mode_conf.get("cycles_high", legacy_cycles if legacy_cycles is not None else defaults["cycles_high"]),
                defaults["cycles_high"],
                min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX,
            )
            mode_conf["beta_high"] = _coerce_float(
                raw_mode_conf.get("beta_high", legacy_beta if legacy_beta is not None else defaults["beta_high"]),
                defaults["beta_high"],
                min_value=FIR_BETA_MIN,
                max_value=FIR_BETA_MAX,
            )
            # 旧形式との互換用。3Wayの計算では個別値を優先する。
            mode_conf["cycles"] = mode_conf.get("cycles", mode_conf["cycles_low"])
            mode_conf["beta"] = mode_conf.get("beta", mode_conf["beta_low"])
        elif mode_key == "3Way+SUB":
            for overlap_key in (
                "kaiser_overlap_sub_low_oct",
                "kaiser_overlap_low_mid_oct",
                "kaiser_overlap_mid_high_oct",
            ):
                mode_conf[overlap_key] = _coerce_float(
                    mode_conf.get(overlap_key, defaults[overlap_key]),
                    defaults[overlap_key],
                    min_value=-1.0,
                    max_value=1.0,
                )
            for cycles_key, beta_key in (
                ("cycles_sub_low", "beta_sub_low"),
                ("cycles_low_mid", "beta_low_mid"),
                ("cycles_mid_high", "beta_mid_high"),
            ):
                legacy_cycles = mode_conf["cycles"] if "cycles" in raw_mode_conf else None
                legacy_beta = mode_conf["beta"] if "beta" in raw_mode_conf else None
                mode_conf[cycles_key] = _coerce_float(
                    raw_mode_conf.get(
                        cycles_key,
                        legacy_cycles if legacy_cycles is not None else defaults[cycles_key],
                    ),
                    defaults[cycles_key],
                    min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX,
                )
                mode_conf[beta_key] = _coerce_float(
                    raw_mode_conf.get(
                        beta_key,
                        legacy_beta if legacy_beta is not None else defaults[beta_key],
                    ),
                    defaults[beta_key],
                    min_value=FIR_BETA_MIN,
                    max_value=FIR_BETA_MAX,
                )
        else:
            default_cycles, default_betas, default_overlaps = _mode_boundary_values(
                mode_key, defaults,
            )
            raw_cycles = mode_conf.get("boundary_cycles", default_cycles)
            raw_betas = mode_conf.get("boundary_betas", default_betas)
            raw_overlaps = mode_conf.get("boundary_overlaps_oct", default_overlaps)
            if not isinstance(raw_cycles, (list, tuple)):
                raw_cycles = default_cycles
            if not isinstance(raw_betas, (list, tuple)):
                raw_betas = default_betas
            if not isinstance(raw_overlaps, (list, tuple)):
                raw_overlaps = default_overlaps
            mode_conf["boundary_cycles"] = [
                _coerce_float(
                    raw_cycles[index] if index < len(raw_cycles) else default_cycles[index],
                    default_cycles[index], min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX,
                )
                for index in range(needed)
            ]
            mode_conf["boundary_betas"] = [
                _coerce_float(
                    raw_betas[index] if index < len(raw_betas) else default_betas[index],
                    default_betas[index], min_value=FIR_BETA_MIN, max_value=FIR_BETA_MAX,
                )
                for index in range(needed)
            ]
            mode_conf["boundary_overlaps_oct"] = [
                _coerce_float(
                    raw_overlaps[index] if index < len(raw_overlaps) else default_overlaps[index],
                    default_overlaps[index], min_value=-1.0, max_value=1.0,
                )
                for index in range(needed)
            ]

        settings[mode_key] = mode_conf

    if settings.get("last_mode") not in MODE_BANDS:
        settings["last_mode"] = "2Way"
    legacy_db_min = _coerce_int(settings.get("db_min", -30), -30)
    settings["gain_y_min_db"] = min(
        visualization.GAIN_Y_MIN_OPTIONS,
        key=lambda option: abs(option - _coerce_int(
            settings.get("gain_y_min_db", legacy_db_min), legacy_db_min,
        )),
    )
    settings["gain_y_max_db"] = min(
        visualization.GAIN_Y_MAX_OPTIONS,
        key=lambda option: abs(option - _coerce_int(
            settings.get("gain_y_max_db", 20), 20,
        )),
    )
    settings["phase_gain_mask_db"] = min(
        visualization.PHASE_GAIN_MASK_OPTIONS_DB,
        key=lambda option: abs(option - _coerce_int(
            settings.get("phase_gain_mask_db", -60), -60,
        )),
    )
    if settings.get("graph_mode") not in visualization.GRAPH_MODE_OPTIONS:
        settings["graph_mode"] = visualization.DEFAULT_GRAPH_MODE
    settings["graph_max_points"] = min(
        visualization.GRAPH_MAX_POINTS_OPTIONS,
        key=lambda option: abs(option - _coerce_int(
            settings.get("graph_max_points", visualization.DEFAULT_GRAPH_MAX_POINTS),
            visualization.DEFAULT_GRAPH_MAX_POINTS,
        )),
    )
    settings["db_min"] = int(settings["gain_y_min_db"])
    settings.pop("group_delay_range_samples", None)
    settings["auto_update"] = bool(settings.get("auto_update", True))
    if settings.get("output_format") not in ("bin_float32", "txt_float", "csv_float", "wav_float32"):
        settings["output_format"] = "bin_float32"
    settings["output_normalize"] = bool(settings.get("output_normalize", False))
    settings["align_output_taps"] = bool(settings.get("align_output_taps", False))
    active_mode = settings["last_mode"]
    settings["auto_crop"] = bool(settings.get("auto_crop", False)) and bool(
        bands_requiring_split_fir(MODE_BANDS[active_mode], ["Through" if flag else method for method, flag in zip(settings[active_mode]["crossover_methods"], settings[active_mode]["boundary_acoustic_targets"])])
    )
    legacy_pass_db = _coerce_float(
        settings.get("auto_crop_pass_db", DEFAULT_AUTO_CROP_PASS_DB),
        DEFAULT_AUTO_CROP_PASS_DB,
        min_value=0.01,
        max_value=0.50,
    )
    legacy_cross_db = _coerce_float(
        settings.get("auto_crop_cross_db", DEFAULT_AUTO_CROP_CROSS_DB),
        DEFAULT_AUTO_CROP_CROSS_DB,
        min_value=0.01,
        max_value=0.30,
    )
    profile_key = raw_settings.get("auto_crop_profile")
    if profile_key not in AUTO_CROP_PROFILES:
        profile_key = _auto_crop_profile_from_limits(legacy_pass_db, legacy_cross_db)
    profile = AUTO_CROP_PROFILES[profile_key]
    settings["auto_crop_profile"] = profile_key
    settings["auto_crop_pass_db"] = profile["pass_db"]
    settings["auto_crop_cross_db"] = profile["cross_db"]
    external_distance_config = settings.get("external_distance_timing")
    if isinstance(external_distance_config, dict):
        # Older settings stored a room-temperature value.  Delay conversion is
        # now intentionally based on one nominal speed and ignores that field.
        external_distance_config.pop("temperature_c", None)
    return settings


def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            return normalize_settings(loaded)
    except Exception as e:
        append_log(f"設定ファイル読み込み失敗: {e}")
    return normalize_settings({})



def _serialize_fir_for_download(arr: np.ndarray, output_format: str, fs=None):
    arr = np.asarray(arr, dtype=np.float64)
    if output_format == "bin_float32":
        return arr.astype("<f4").tobytes(), "application/octet-stream", ".bin", "float32 little-endian binary"
    if output_format == "txt_float":
        lines = "\n".join(f"{v:.9f}" for v in arr) + "\n"
        return lines.encode("utf-8"), "text/plain", ".txt", "text/plain"
    if output_format == "csv_float":
        lines = "index,value\n" + "\n".join(f"{i},{v:.9f}" for i, v in enumerate(arr)) + "\n"
        return lines.encode("utf-8"), "text/csv", ".csv", "text/csv"
    if output_format == "wav_float32":
        # WAV support is an export-only dependency.  Keep it outside the
        # startup path used to calculate and display the first graphs.
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, arr.astype(np.float32), int(fs or 48000), format="WAV", subtype="FLOAT")
        return buf.getvalue(), "audio/wav", ".wav", "WAV float32"
    # fallback
    return arr.astype("<f4").tobytes(), "application/octet-stream", ".bin", "float32 little-endian binary"


def _output_format_label(fmt: str) -> str:
    return {
        "bin_float32": "Binary float32 (.bin)",
        "txt_float": "Float text (.txt)",
        "csv_float": "CSV float (.csv)",
        "wav_float32": "WAV float32 (.wav)",
    }.get(fmt, fmt)



def _fmt_cycles_for_filename(cycles: float) -> str:
    txt = f"{float(cycles):.2f}"
    return txt.rstrip("0").rstrip(".")


def _build_output_filename(mode_key: str, band: str, settings_conf: dict, arr_len: int, ext: str) -> str:
    fs_k = int(round(float(settings_conf["fs"]) / 1000.0))
    date_str = datetime.datetime.now().strftime("%y%m%d")

    if mode_key == "2Way":
        cycles_txt = _fmt_cycles_for_filename(settings_conf["cycles"])
        fc = int(settings_conf["cross_freqs"][0])
        ov = float(settings_conf.get("kaiser_overlap_oct", 0.0))
        core = f"2W_{band}_{fc}Hz_ov{ov:+.2f}oct_{cycles_txt}cy_{fs_k}k_{int(arr_len)}tap_{date_str}"
    elif mode_key == "3Way":
        fc1 = int(settings_conf["cross_freqs"][0])
        fc2 = int(settings_conf["cross_freqs"][1])
        ov1 = float(settings_conf.get("kaiser_overlap_low_mid_oct", 0.0))
        ov2 = float(settings_conf.get("kaiser_overlap_mid_high_oct", 0.0))
        lcy = _fmt_cycles_for_filename(settings_conf.get("cycles_low", settings_conf["cycles"]))
        hcy = _fmt_cycles_for_filename(settings_conf.get("cycles_high", settings_conf["cycles"]))
        lb = f"{float(settings_conf.get('beta_low', settings_conf['beta'])):.1f}".rstrip("0").rstrip(".")
        hb = f"{float(settings_conf.get('beta_high', settings_conf['beta'])):.1f}".rstrip("0").rstrip(".")
        core = f"3W_{band}_{fc1}-{fc2}Hz_ov{ov1}-{ov2}_L{lcy}-B{lb}_H{hcy}-B{hb}_{fs_k}k_{int(arr_len)}tap_{date_str}"
    elif mode_key == "3Way+SUB":
        crosses = "-".join(str(int(v)) for v in settings_conf["cross_freqs"])
        overlaps = "-".join(f"{float(settings_conf.get(key, 0.0)):+.2f}" for key in (
            "kaiser_overlap_sub_low_oct", "kaiser_overlap_low_mid_oct", "kaiser_overlap_mid_high_oct"
        ))
        params = []
        for label, cycles_key, beta_key in (
            ("SL", "cycles_sub_low", "beta_sub_low"),
            ("LM", "cycles_low_mid", "beta_low_mid"),
            ("MH", "cycles_mid_high", "beta_mid_high"),
        ):
            cycles_txt = _fmt_cycles_for_filename(settings_conf[cycles_key])
            beta_txt = f"{float(settings_conf[beta_key]):.1f}".rstrip("0").rstrip(".")
            params.append(f"{label}{cycles_txt}-B{beta_txt}")
        core = (
            f"3W+SUB_{band}_{crosses}Hz_ov{overlaps}_{'_'.join(params)}_"
            f"{fs_k}k_{int(arr_len)}tap_{date_str}"
        )
    else:
        cycles_values, beta_values, overlap_values = _mode_boundary_values(
            mode_key, settings_conf,
        )
        crosses = "-".join(str(int(value)) for value in settings_conf["cross_freqs"])
        overlaps = "-".join(f"{value:+.2f}" for value in overlap_values)
        params = "-".join(
            f"{_fmt_cycles_for_filename(cycle)}cy-B{beta:g}"
            for cycle, beta in zip(cycles_values, beta_values, strict=True)
        )
        core = (
            f"{mode_key}_{band}_{crosses}Hz_ov{overlaps}_{params}_"
            f"{fs_k}k_{int(arr_len)}tap_{date_str}"
        )

    return f"{core}{ext}"


def _prepare_output_firs(firs_by_band, align_to_longest):
    if not firs_by_band:
        return {}, []
    max_len = max(len(arr) for arr in firs_by_band.values())
    output_firs = {}
    rows = []
    for band, arr in firs_by_band.items():
        arr = np.asarray(arr)
        original_taps = len(arr)
        pad_total = max_len - original_taps if align_to_longest else 0
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        output_arr = np.pad(arr, (pad_before, pad_after)) if pad_total else arr.copy()
        target_center = (max_len - 1) / 2.0
        actual_center = pad_before + (original_taps - 1) / 2.0
        dsp_delay_samples = (
            0.0 if align_to_longest else (max_len - original_taps) / 2.0
        )
        output_firs[band] = output_arr
        rows.append({
            "band": band,
            "original_taps": int(original_taps),
            "output_taps": int(len(output_arr)),
            "pad_before_samples": int(pad_before),
            "pad_after_samples": int(pad_after),
            "alignment_delay_samples": int(pad_before),
            "dsp_delay_samples": round_for_display(dsp_delay_samples, 3),
            "center_alignment_error_samples": round_for_display(actual_center - target_center, 3),
        })
    return output_firs, rows


def _build_filter_specification(
    mode_key, settings, output_rows, generated_at,
    auto_crop_rows=None, auto_crop_cross_rows=None
):
    conf = settings[mode_key]
    fs = int(conf["fs"])
    fmt = _output_format_label(settings["output_format"])
    align_output_taps = bool(settings.get("align_output_taps", False))
    lines = [
        "APPY Multiway FIR Studio",
        "フィルター仕様書",
        "",
        "■ 基本情報",
        f"PhaseEQ Composite統合版: {PHASEEQ_INTEGRATION_VERSION}",
        f"Multiwayベース版: {APP_VERSION}",
        f"生成日時: {generated_at}",
        f"分割方式: {mode_key}",
        f"サンプリング周波数: {fs} Hz",
        f"出力形式: {fmt}",
        (
            f"出力ゲイン調整: ON（各出力帯域の最大周波数応答 "
            f"{OUTPUT_MAX_BAND_GAIN_DB:.1f} dB）"
            if settings.get("output_normalize")
            else "出力ゲイン調整: OFF"
        ),
        f"周波数特性による自動クロップ: {'有効' if settings.get('auto_crop') else '無効'}",
        f"自動クロップ設定: {AUTO_CROP_PROFILES[settings.get('auto_crop_profile', DEFAULT_AUTO_CROP_PROFILE)]['label']}"
        f"{'' if settings.get('auto_crop') else '（無効時は未使用）'}",
        f"自動クロップ 通過帯許容差: {float(settings.get('auto_crop_pass_db', DEFAULT_AUTO_CROP_PASS_DB)):.2f} dB",
        f"自動クロップ 合成波クロス近傍許容差: {float(settings.get('auto_crop_cross_db', DEFAULT_AUTO_CROP_CROSS_DB)):.2f} dB",
        f"出力タップ長を最長フィルターに合わせる: {'ON' if align_output_taps else 'OFF'}",
        "",
    ]
    if mode_key == "2Way":
        lines.extend([
            "■ クロスオーバー",
            f"周波数: {int(conf['cross_freqs'][0])} Hz",
            f"FIR周期数: {float(conf['cycles']):g}",
            f"カイザー窓β値: {float(conf['beta']):g}",
            f"Kaiser overlap: {float(conf.get('kaiser_overlap_oct', 0.0)):+.2f} oct",
            "",
        ])
    elif mode_key == "3Way":
        lines.extend([
            "■ 上側クロス High/Mid",
            f"クロスオーバー: {int(conf['cross_freqs'][1])} Hz",
            f"FIR周期数: {float(conf['cycles_high']):g}",
            f"カイザー窓β値: {float(conf['beta_high']):g}",
            f"Kaiser overlap: {float(conf.get('kaiser_overlap_mid_high_oct', 0.0)):+.2f} oct",
            "",
            "■ 下側クロス Mid/Low",
            f"クロスオーバー: {int(conf['cross_freqs'][0])} Hz",
            f"FIR周期数: {float(conf['cycles_low']):g}",
            f"カイザー窓β値: {float(conf['beta_low']):g}",
            f"Kaiser overlap: {float(conf.get('kaiser_overlap_low_mid_oct', 0.0)):+.2f} oct",
            "",
        ])
    elif mode_key == "3Way+SUB":
        crossover_specs = (
            ("上側クロス High/Mid", 2, "cycles_mid_high", "beta_mid_high", "kaiser_overlap_mid_high_oct"),
            ("中央クロス Mid/Low", 1, "cycles_low_mid", "beta_low_mid", "kaiser_overlap_low_mid_oct"),
            ("下側クロス Low/SUB", 0, "cycles_sub_low", "beta_sub_low", "kaiser_overlap_sub_low_oct"),
        )
        for label, index, cycles_key, beta_key, overlap_key in crossover_specs:
            lines.extend([
                f"■ {label}",
                f"クロスオーバー: {int(conf['cross_freqs'][index])} Hz",
                f"FIR周期数: {float(conf[cycles_key]):g}",
                f"カイザー窓β値: {float(conf[beta_key]):g}",
                f"Kaiser overlap: {float(conf.get(overlap_key, 0.0)):+.2f} oct",
                "",
            ])
    else:
        boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(
            mode_key, conf,
        )
        ordered_bands = MODE_BANDS[mode_key]
        for index in reversed(range(len(conf["cross_freqs"]))):
            label = f"{ordered_bands[index]} / {ordered_bands[index + 1]}"
            lines.extend([
                f"■ クロス {label}",
                f"クロスオーバー: {int(conf['cross_freqs'][index])} Hz",
                f"FIR周期数: {boundary_cycles[index]:g}",
                f"カイザー窓β値: {boundary_betas[index]:g}",
                f"Kaiser overlap: {boundary_overlaps[index]:+.2f} oct",
                "",
            ])

    lines.extend([
        "■ DSPディレイ設定",
        (
            "出力タップ統一: ON。短い帯域は前後へのゼロ追加で最長タップへ"
            "整列済みです。DSP側の追加ディレイは全帯域0にしてください。"
            if align_output_taps
            else
            "出力タップ統一: OFF。最長フィルターをディレイ0の基準とし、"
            "各帯域へ下記のDSP追加ディレイを設定してください。"
        ),
        "",
        "■ 出力フィルター",
    ])
    for row in output_rows:
        dsp_delay_ms = row["dsp_delay_samples"] / fs * 1000.0
        lines.append(
            f"{row['band']}: 生成 {row['original_taps']} taps / 出力 {row['output_taps']} taps / "
            f"前ゼロ {row['pad_before_samples']} / 後ゼロ {row['pad_after_samples']} samples / "
            f"DSP追加ディレイ {row['dsp_delay_samples']:g} samples "
            f"({dsp_delay_ms:.4f} ms) / "
            f"出力中心差 {row['center_alignment_error_samples']:+g} sample"
        )

    if auto_crop_rows:
        lines.extend(["", "■ 自動クロップ結果"])
        for row in auto_crop_rows:
            adjacent_error = row["隣接クロス合成波 最大誤差 [dB]"]
            adjacent_error_text = (
                f"{adjacent_error:.4f} dB"
                if isinstance(adjacent_error, (int, float, np.number))
                else "対象なし"
            )
            lines.append(
                f"{row['帯域']}: {row['合成後 [taps]']} -> "
                f"{row['自動クロップ後 [taps]']} taps "
                f"({row['削減率 [%]']:.1f}%削減), "
                f"探索 {row.get('探索', '○')}, "
                f"通過帯誤差 {row['通過帯誤差 [dB]']:.4f} dB, "
                f"隣接クロス合成波 最大誤差 "
                f"{adjacent_error_text}"
            )
    if auto_crop_cross_rows:
        lines.extend(["", "■ 合成波クロス近傍誤差"])
        for row in auto_crop_cross_rows:
            lines.append(
                f"{row['クロス']} ({row['基準周波数 [Hz]']} Hz): "
                f"最大 {row['合成波 最大誤差 [dB]']:.4f} dB / "
                f"ピーク {row['合成波 ピーク [dB]']:.4f} dB / "
                f"ディップ {row['合成波 ディップ [dB]']:.4f} dB"
            )

    lines.extend([
        "",
        "■ バッフル補正",
        f"有効: {'はい' if conf.get('baffle_on') else 'いいえ'}",
        f"上限周波数: {int(conf.get('baffle_f1_hz', 1000))} Hz",
        f"補正量: {float(conf.get('baffle_atten', 0.0)):g} dB",
        "",
        "■ EQ合成",
    ])
    if not conf.get("fir_output_enabled", False):
        lines.append("FIR OFF: EQ合成は未適用（登録ファイルは保持）")
    for band, filenames in conf.get("eq_files", {}).items():
        active = [name for name in filenames if name]
        lines.append(f"{band}: {' -> '.join(active) if active else 'なし'}")

    lines.extend([
        "",
        "■ 注意",
        (
            "出力ゲイン調整はクロップ後の全帯域へ同一ゲインを適用しています。"
            "帯域間の相対レベル、タップ数、ピーク位置、係数形状は変更しません。"
            if settings.get("output_normalize")
            else "FIR係数は設計値のレベルを変更せず保存しています。"
        ),
        "最長タップへの整列は、自動クロップなどの生成処理後、ダウンロード出力へだけ適用します。",
        "整列は前後へのゼロ追加で行うため振幅特性は変わりませんが、出力長は最長タップへ揃います。",
        "DSP追加ディレイは最長フィルターを0基準とした値です。ON時は整列済みのため追加設定しないでください。",
        "DSPがゼロ係数も演算する場合、短縮した帯域の処理負荷削減効果は失われます。",
        "元タップ数の偶数・奇数が混在する場合、整数サンプルのゼロ追加では中心に最大0.5 sampleの差が残ります。",
    ])
    return "\n".join(lines) + "\n"


def _build_filter_package(
    firs_by_band, mode_key, settings, fs, generated_at,
    auto_crop_rows=None, auto_crop_cross_rows=None
):
    align = bool(settings.get("align_output_taps", False))
    output_firs, output_rows = _prepare_output_firs(firs_by_band, align)
    timestamp = datetime.datetime.fromisoformat(generated_at).strftime("%Y%m%d_%H%M%S")
    fs_k = int(round(float(fs) / 1000.0))
    zip_name = f"APPY_FIR_{mode_key}_{fs_k}k_{timestamp}.zip"
    manifest = {
        "format_version": 1,
        "generated_at": generated_at,
        "application": "APPY Multiway FIR Studio",
        "application_version": APP_VERSION,
        "mode": mode_key,
        "sample_rate_hz": int(fs),
        "coefficient_format": settings["output_format"],
        "normalized": bool(settings.get("output_normalize", False)),
        "normalization_mode": "common_independent_band_response_gain",
        "normalization_target_max_gain_db": OUTPUT_MAX_BAND_GAIN_DB,
        "align_output_taps": align,
        "auto_crop": bool(settings.get("auto_crop", False)),
        "auto_crop_profile": settings.get(
            "auto_crop_profile", DEFAULT_AUTO_CROP_PROFILE
        ),
        "auto_crop_pass_db": float(
            settings.get("auto_crop_pass_db", DEFAULT_AUTO_CROP_PASS_DB)
        ),
        "auto_crop_cross_db": float(
            settings.get("auto_crop_cross_db", DEFAULT_AUTO_CROP_CROSS_DB)
        ),
        "auto_crop_results": auto_crop_rows or [],
        "auto_crop_cross_results": auto_crop_cross_rows or [],
        "files": [],
    }

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        from phase_fir_designer.output_terms import write_output_terms
        write_output_terms(zf)
        for row in output_rows:
            band = row["band"]
            data, _, ext, fmt_label = _serialize_fir_for_download(
                output_firs[band], settings["output_format"], fs=fs
            )
            path = f"filters/{band}{ext}"
            zf.writestr(path, data)
            manifest["files"].append({
                **row,
                "path": path,
                "format": fmt_label,
                "sha256": hashlib.sha256(data).hexdigest(),
            })

        specification = _build_filter_specification(
            mode_key, settings, output_rows, generated_at,
            auto_crop_rows, auto_crop_cross_rows
        )
        zf.writestr("filter_specification.txt", specification.encode("utf-8-sig"))
        zf.writestr(
            "settings.json",
            json.dumps(normalize_settings(settings), ensure_ascii=False, indent=2).encode("utf-8"),
        )
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    return zip_buffer.getvalue(), zip_name, output_firs


def _render_filter_downloads(
    firs_by_band,
    mode_key,
    settings,
    fs,
    generated_at,
    key_suffix,
    auto_crop_rows=None,
    auto_crop_cross_rows=None,
    fir_states=None,
):
    active_firs = {
        band: values for band, values in firs_by_band.items()
        if bool((fir_states or {}).get(band, {}).get("enabled", len(values) > 1))
    }
    with st.expander(ui_message('ui.96acdaf7b4f091'), expanded=False):
        st.caption(
            ui_message('ui.7ef69576f39670')
        )
        if not active_firs:
            st.caption(ui_message('ui.795bf002abf122'))
            return
        zip_data, zip_name, output_firs = _build_filter_package(
            active_firs, mode_key, settings, fs, generated_at,
            auto_crop_rows, auto_crop_cross_rows
        )
        st.caption(
            ui_message('ui.96028ebbcd7e1c', p0=f"{_output_format_label(settings['output_format'])}", p1=f"{('ON' if settings.get('align_output_taps') else 'OFF')}")
        )
        st.download_button(
            ui_message('ui.6562de302b569f'),
            zip_data,
            file_name=zip_name,
            mime="application/zip",
            key=f"dl_zip_{key_suffix}",
        )
        st.caption(ui_message('ui.8e2f8b48555d50'))

        with st.expander(ui_message('ui.54ad464131e00a'), expanded=False):
            for band, arr in output_firs.items():
                data, mime, ext, fmt_label = _serialize_fir_for_download(
                    arr, settings["output_format"], fs=fs
                )
                fname = _build_output_filename(mode_key, band, settings[mode_key], len(arr), ext)
                st.download_button(
                    ui_message('ui.8e4cadeb595862', p0=f'{band}', p1=f'{len(arr)}', p2=f'{fmt_label}'),
                    data,
                    file_name=fname,
                    mime=mime,
                    key=f"dl_{key_suffix}_{band}_{len(arr)}_{settings['output_format']}",
                )


def save_settings(settings, *, log_change=True):
    """設定を差分がある時だけ保存（無駄な書き込みを抑制）"""
    try:
        settings = normalize_settings(settings)
        new_txt = json.dumps(settings, ensure_ascii=False, sort_keys=True, indent=2)
        cur_txt = ''
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as rf:
                    cur_txt = rf.read()
            except Exception:
                cur_txt = ''
        if new_txt != cur_txt:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as wf:
                wf.write(new_txt)
            if log_change:
                append_log("設定を保存しました（差分あり）")
    except Exception as e:
        append_log(f"設定保存に失敗: {e}")


def _safe_uploaded_filename(filename):
    name = unicodedata.normalize("NFC", str(filename or ""))
    name = name.replace("\\", "/").split("/")[-1]
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name)
    name = name.strip().rstrip(". ")
    if not name:
        name = "uploaded_fir.bin"
    stem, ext = os.path.splitext(name)
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if stem.upper() in reserved:
        stem = f"_{stem}"
    max_stem_length = max(1, 180 - len(ext))
    return f"{stem[:max_stem_length]}{ext[:20]}"


def save_uploaded_eq(band, idx, uploaded_file):
    load_fir_file(uploaded_file)
    stored_name = _safe_uploaded_filename(uploaded_file.name)
    filename = f"{band}_eq{idx+1}_{stored_name}"
    save_path = os.path.join(EQ_SAVE_DIR, filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    append_log(f"{band}帯域 EQ{idx+1}ファイルをアップロード: {stored_name}")
    return save_path, stored_name

def export_config(settings):
    settings = normalize_settings(settings)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"settings_{now}.json"
    filepath = os.path.join(CONFIG_EXPORT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    append_log(f"スナップショットを書き出し: {filename}")
    return filepath


def validate_config_roundtrip(settings, filepath):
    """スナップショット保存後に、読み戻しで設定が欠けないか確認する。"""
    try:
        expected = normalize_settings(settings)
        with open(filepath, "r", encoding="utf-8") as f:
            restored = normalize_settings(json.load(f))
        return expected == restored, ""
    except Exception as e:
        return False, str(e)


def _read_settings_upload(filename, payload):
    filename_lower = filename.lower()
    if filename_lower.endswith(".json"):
        if len(payload) > 1024 * 1024:
            raise ValueError("設定JSONのサイズが大きすぎます。")
        imported = json.loads(payload.decode("utf-8-sig"))
    elif filename_lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            candidates = [
                info for info in zf.infolist()
                if not info.is_dir() and info.filename.split("/")[-1] == "settings.json"
            ]
            if not candidates:
                raise ValueError("ZIP内に settings.json がありません。")
            if len(candidates) > 1:
                raise ValueError("ZIP内に settings.json が複数あります。")
            if candidates[0].file_size > 1024 * 1024:
                raise ValueError("settings.json のサイズが大きすぎます。")
            imported = json.loads(zf.read(candidates[0]).decode("utf-8-sig"))
    else:
        raise ValueError("対応形式は .json または .zip です。")

    if not isinstance(imported, dict):
        raise ValueError("設定ファイルのルートはJSONオブジェクトである必要があります。")
    return normalize_settings(imported)


def _clear_missing_eq_references(imported):
    imported = normalize_settings(imported)
    missing = []
    for mode_key, bands in MODE_BANDS.items():
        eq_files = imported[mode_key].get("eq_files", {})
        for band in bands:
            filenames = list(eq_files.get(band, ["", "", ""]))
            for idx, filename in enumerate(filenames):
                if not filename:
                    continue
                expected = os.path.join(EQ_SAVE_DIR, f"{band}_eq{idx + 1}_{filename}")
                if not os.path.exists(expected):
                    missing.append(f"{mode_key} {band} EQ{idx + 1}: {filename}")
                    filenames[idx] = ""
            eq_files[band] = filenames
        imported[mode_key]["eq_files"] = eq_files
    return imported, missing


def _settings_summary(imported):
    mode = imported.get("last_mode", "2Way")
    conf = imported[mode]
    cross = " / ".join(f"{int(v)} Hz" for v in conf.get("cross_freqs", []))
    return {
        "分割方式": mode,
        "サンプリング周波数": f"{int(conf.get('fs', 96000))} Hz",
        "クロスオーバー": cross,
        "DSP FIR出力": "ON" if conf.get("fir_output_enabled") else "OFF",
        "出力形式": _output_format_label(imported.get("output_format", "bin_float32")),
        "出力ゲイン調整": "ON" if imported.get("output_normalize") else "OFF",
        "最長タップへ整列": "ON" if imported.get("align_output_taps") else "OFF",
        "自動クロップ": "ON" if imported.get("auto_crop") else "OFF",
        "自動クロップ設定": AUTO_CROP_PROFILES[
            imported.get("auto_crop_profile", DEFAULT_AUTO_CROP_PROFILE)
        ]["label"],
        "通過帯許容差": f"{float(imported.get('auto_crop_pass_db', DEFAULT_AUTO_CROP_PASS_DB)):.2f} dB",
        "合成波クロス近傍許容差": f"{float(imported.get('auto_crop_cross_db', DEFAULT_AUTO_CROP_CROSS_DB)):.2f} dB",
    }


def _set_indexed_widget_state(prefix, settings_key, value):
    st.session_state[settings_key] = value
    st.session_state[f"{prefix}{settings_key}"] = value


def _apply_imported_config(imported, source_label, resume_filename=None):
    """レジューム内容をセッションへ反映する。
    注意: widget生成前にだけ呼ぶこと。
    """
    imported = normalize_settings(imported)
    for scope in ("Main", "Stereo"):
        key = f"composite_studio_group_target_preset_{scope}"
        saved_targets = imported.get("group_target_presets", {})
        if scope in saved_targets:
            st.session_state[key] = str(saved_targets[scope])
    # Widget state survives reruns independently of the settings payload.  Clear
    # distance-entry widgets before importing so a previous System cannot
    # override the restored external-distance configuration.
    for state_key in tuple(st.session_state):
        if str(state_key).startswith("composite_alignment_external_distance_"):
            st.session_state.pop(state_key, None)
    imported_distance_config = imported.get("external_distance_timing", {})
    if not isinstance(imported_distance_config, dict):
        imported_distance_config = {}
    st.session_state["composite_alignment_system_timing_source"] = (
        "外部測定距離"
        if bool(imported_distance_config.get("enabled", False))
        else "自動選択"
    )
    st.session_state.pop("composite_alignment_external_temperature_c", None)
    st.session_state["settings"] = imported
    st.session_state["filter_type"] = imported.get("last_mode", "2Way")

    mode = imported.get("last_mode", "2Way")
    mode_conf = imported.get(mode, {})

    # keyed widget state は widget 生成前に同期する
    st.session_state[KEY_MODE] = MODE_LABELS[mode]
    st.session_state[f"{KEY_FIR_OUTPUT_PREFIX}{mode}"] = bool(
        mode_conf.get("fir_output_enabled", False)
    )
    _set_indexed_widget_state("__sel_val_", KEY_FS, int(mode_conf.get("fs", 96000)))
    st.session_state[KEY_CYCLES] = float(mode_conf.get("cycles", DEFAULT_FIR_CYCLES))
    st.session_state[KEY_BETA] = float(mode_conf.get("beta", DEFAULT_FIR_BETA))
    st.session_state[KEY_CYCLES_LOW] = float(mode_conf.get("cycles_low", mode_conf.get("cycles", DEFAULT_FIR_CYCLES)))
    st.session_state[KEY_BETA_LOW] = float(mode_conf.get("beta_low", mode_conf.get("beta", DEFAULT_FIR_BETA)))
    st.session_state[KEY_CYCLES_HIGH] = float(mode_conf.get("cycles_high", mode_conf.get("cycles", DEFAULT_FIR_CYCLES)))
    st.session_state[KEY_BETA_HIGH] = float(mode_conf.get("beta_high", mode_conf.get("beta", DEFAULT_FIR_BETA)))
    st.session_state[KEY_CYCLES_SUB_LOW] = float(mode_conf.get("cycles_sub_low", mode_conf.get("cycles", DEFAULT_FIR_CYCLES)))
    st.session_state[KEY_BETA_SUB_LOW] = float(mode_conf.get("beta_sub_low", mode_conf.get("beta", DEFAULT_FIR_BETA)))
    st.session_state[KEY_CYCLES_LOW_MID] = float(mode_conf.get("cycles_low_mid", mode_conf.get("cycles", DEFAULT_FIR_CYCLES)))
    st.session_state[KEY_BETA_LOW_MID] = float(mode_conf.get("beta_low_mid", mode_conf.get("beta", DEFAULT_FIR_BETA)))
    st.session_state[KEY_CYCLES_MID_HIGH] = float(mode_conf.get("cycles_mid_high", mode_conf.get("cycles", DEFAULT_FIR_CYCLES)))
    st.session_state[KEY_BETA_MID_HIGH] = float(mode_conf.get("beta_mid_high", mode_conf.get("beta", DEFAULT_FIR_BETA)))
    st.session_state[KEY_GAIN_Y_MIN] = int(imported.get("gain_y_min_db", -20))
    st.session_state[KEY_GAIN_Y_MAX] = int(imported.get("gain_y_max_db", 20))
    st.session_state[KEY_PHASE_GAIN_MASK] = int(imported.get("phase_gain_mask_db", -60))
    st.session_state["composite_studio_gain_y_min_widget"] = st.session_state[KEY_GAIN_Y_MIN]
    st.session_state["composite_studio_gain_y_max_widget"] = st.session_state[KEY_GAIN_Y_MAX]
    st.session_state["composite_studio_phase_gain_mask_widget"] = st.session_state[KEY_PHASE_GAIN_MASK]
    st.session_state[KEY_GRAPH_MODE] = str(imported.get("graph_mode", "Light"))
    st.session_state["composite_studio_graph_mode_widget"] = st.session_state[KEY_GRAPH_MODE]
    st.session_state[KEY_GRAPH_MAX_POINTS] = int(imported.get("graph_max_points", 1024))
    st.session_state["composite_studio_graph_max_points_widget"] = st.session_state[KEY_GRAPH_MAX_POINTS]
    _set_indexed_widget_state("__rad_val_", KEY_DBMIN, int(imported.get("db_min", -30)))
    st.session_state[KEY_BAFFLE_ON] = bool(mode_conf.get("baffle_on", False))
    st.session_state[KEY_BAFFLE_ATTEN] = float(mode_conf.get("baffle_atten", 6.0))
    _set_indexed_widget_state(
        "__num_", KEY_BAFFLE_F1, int(mode_conf.get("baffle_f1_hz", 1000))
    )
    st.session_state[KEY_AUTO_UPDATE] = bool(imported.get("auto_update", True))
    _set_indexed_widget_state(
        "__sel_val_", KEY_OUTPUT_FORMAT, imported.get("output_format", "bin_float32")
    )
    st.session_state[KEY_OUTPUT_NORMALIZE] = bool(imported.get("output_normalize", False))
    st.session_state[KEY_ALIGN_OUTPUT_TAPS] = bool(imported.get("align_output_taps", False))
    st.session_state[KEY_AUTO_CROP] = bool(imported.get("auto_crop", False))
    st.session_state[KEY_AUTO_CROP_PROFILE] = imported.get(
        "auto_crop_profile", DEFAULT_AUTO_CROP_PROFILE
    )
    st.session_state[KEY_AUTO_CROP_PASS_DB] = float(
        imported.get("auto_crop_pass_db", DEFAULT_AUTO_CROP_PASS_DB)
    )
    st.session_state[KEY_AUTO_CROP_CROSS_DB] = float(
        imported.get("auto_crop_cross_db", DEFAULT_AUTO_CROP_CROSS_DB)
    )
    if resume_filename is not None:
        st.session_state[KEY_RESUME_SELECT] = resume_filename

    if mode == "2Way":
        _set_indexed_widget_state(
            "__num_", "fc", int(mode_conf.get("cross_freqs", [2500])[0])
        )
        st.session_state["kaiser_overlap_oct"] = float(mode_conf.get("kaiser_overlap_oct", 0.0))
        st.session_state["Low_crop"] = int(mode_conf.get("crop_lens", {}).get("Low", 0))
        st.session_state["High_crop"] = int(mode_conf.get("crop_lens", {}).get("High", 0))
    elif mode == "3Way":
        cross = mode_conf.get("cross_freqs", [500, 2500])
        _set_indexed_widget_state("__num_", "fc1", int(cross[0]))
        _set_indexed_widget_state("__num_", "fc2", int(cross[1]))
        st.session_state["kaiser_overlap_low_mid_oct"] = float(mode_conf.get("kaiser_overlap_low_mid_oct", 0.0))
        st.session_state["kaiser_overlap_mid_high_oct"] = float(mode_conf.get("kaiser_overlap_mid_high_oct", 0.0))
        st.session_state["Low_crop"] = int(mode_conf.get("crop_lens", {}).get("Low", 0))
        st.session_state["Mid_crop"] = int(mode_conf.get("crop_lens", {}).get("Mid", 0))
        st.session_state["High_crop"] = int(mode_conf.get("crop_lens", {}).get("High", 0))
    elif mode == "3Way+SUB":
        cross = mode_conf.get("cross_freqs", [120, 500, 2500])
        _set_indexed_widget_state("__num_", "fc4_sub_low", int(cross[0]))
        _set_indexed_widget_state("__num_", "fc4_low_mid", int(cross[1]))
        _set_indexed_widget_state("__num_", "fc4_mid_high", int(cross[2]))
        st.session_state["kaiser_overlap_sub_low_oct"] = float(mode_conf.get("kaiser_overlap_sub_low_oct", 0.0))
        st.session_state["kaiser_overlap_low_mid_oct"] = float(mode_conf.get("kaiser_overlap_low_mid_oct", 0.0))
        st.session_state["kaiser_overlap_mid_high_oct"] = float(mode_conf.get("kaiser_overlap_mid_high_oct", 0.0))
        for band in MODE_BANDS["3Way+SUB"]:
            st.session_state[f"{band}_crop"] = int(
                mode_conf.get("crop_lens", {}).get(band, 0)
            )
    else:
        cross = mode_conf.get("cross_freqs", ())
        boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(
            mode, mode_conf,
        )
        for index, frequency in enumerate(cross):
            _set_indexed_widget_state(
                "__num_", f"boundary_fc_{mode}_{index}", int(frequency),
            )
            st.session_state[f"boundary_cycles_{mode}_{index}"] = boundary_cycles[index]
            st.session_state[f"boundary_beta_{mode}_{index}"] = boundary_betas[index]
            st.session_state[f"boundary_overlap_{mode}_{index}"] = boundary_overlaps[index]
        for band in MODE_BANDS[mode]:
            st.session_state[f"{band}_crop"] = int(
                mode_conf.get("crop_lens", {}).get(band, 0)
            )

    iir_prefix = f"composite_studio_{mode}"
    st.session_state[f"{iir_prefix}_lr2_auto_polarity"] = bool(mode_conf.get("iir_lr2_auto_polarity", True))
    for index, method in enumerate(mode_conf.get("crossover_methods", [])):
        st.session_state[f"{iir_prefix}_crossover_method_{index}"] = str(method)
        st.session_state[f"{iir_prefix}_acoustic_target_{index}"] = bool(mode_conf.get("boundary_acoustic_targets", [False]*len(mode_conf["crossover_methods"]))[index])

    save_settings(imported)
    append_log(f"設定を読み込み: {source_label}")
    clear_results()


def import_config(filename):
    """widget生成後のクリックイベントからは直接 session_state を書き換えず、
    次の rerun 冒頭で適用する。
    """
    st.session_state["_pending_resume_file"] = filename
    st.rerun()

# ------------------------ UI・説明 ------------------------

st.set_page_config(
    page_title=(
        f"Composite Engine v{PHASEEQ_INTEGRATION_VERSION} · "
        f"Multiway base v{APP_VERSION}"
    ),
    layout="wide",
)
# Fragment IDs include their layout path. Keep the timer before conditional
# graphs and keep registering it even when the FIR configuration is invalid.
from utils.ui_language import initialize_language, render_language_setting
_watch_phaseeq_result_inbox()
initialize_language()
from utils.ui_help import initialize_help, render_help_setting
initialize_help()
_apply_app_theme()

if visualization.jp_font is None:
    st.warning(
        ui_message('ui.78dd8f858a0ebb')
    )

st.markdown(
    ui_message('ui.e4ce44c3205188', p0=PHASEEQ_INTEGRATION_VERSION, p1=APP_VERSION)
)

with st.expander(ui_message('ui.8a794eb49f96f4'), expanded=False):
    st.markdown(
        ui_message('ui.036e81f58e864a')
    )

# ------------------------ セッション初期化 ------------------------

if "settings" not in st.session_state:
    settings = load_settings()
    st.session_state["settings"] = settings
else:
    settings = normalize_settings(st.session_state["settings"])
    st.session_state["settings"] = settings

if "filter_type" not in st.session_state:
    st.session_state["filter_type"] = settings.get("last_mode", "2Way")

# widget生成前に pending resume を適用
if st.session_state.get("_pending_resume_file"):
    _resume_filename = st.session_state.pop("_pending_resume_file")
    _resume_path = os.path.join(CONFIG_EXPORT_DIR, _resume_filename)
    with open(_resume_path, "r", encoding="utf-8") as _rf:
        _imported = normalize_settings(json.load(_rf))
    _apply_imported_config(_imported, _resume_filename, resume_filename=_resume_filename)
    st.success(ui_message('ui.4ed43f6a263b49', p0=f'{_resume_filename}'))

if st.session_state.get("_history_apply_system") is not None:
    from composite_engine.multiway_studio.extension import _apply_library_system
    _apply_library_system(st.session_state.pop("_history_apply_system"))

if st.session_state.get("_pending_uploaded_settings"):
    _uploaded_settings = st.session_state.pop("_pending_uploaded_settings")
    from composite_engine.multiway_studio.extension import restore_output_fir_settings
    restore_output_fir_settings(_uploaded_settings)
    _uploaded_name = st.session_state.pop("_pending_uploaded_name", "settings.json")
    _uploaded_missing_eq = st.session_state.pop("_pending_uploaded_missing_eq", [])
    _uploaded_kind = st.session_state.pop("_pending_uploaded_kind", "settings")
    st.session_state.pop(KEY_SETTINGS_UPLOAD, None)
    _apply_imported_config(_uploaded_settings, _uploaded_name)
    if _uploaded_kind == "dsp_package":
        st.success(ui_message('ui.0a4d9bfd9b301e', p0=f'{_uploaded_name}'))
    else:
        st.success(ui_message('ui.a6e30aa3e2680b', p0=f'{_uploaded_name}'))
    if _uploaded_missing_eq:
        st.warning(
            ui_message('ui.90025dac526136')
        )

# ------------------------ サイドバーUI ------------------------
dsp_resume_upload = None
with st.sidebar:
    preferences_host = st.expander(display_text("設定"))
    with preferences_host:
        render_language_setting(key="multiway_ui_language")
        render_help_setting(key="multiway_help_delay")
    st.markdown(
        f"""
        <div class="sidebar-workflow">
          <div class="sidebar-workflow-title">{html.escape(display_text('設計の流れ'))}</div>
          <div class="sidebar-workflow-steps">
            <span>{html.escape(display_text('1 基本設計'))}</span><b>›</b><span>{html.escape(display_text('3 FIR設定'))}</span><b>›</b>
            <span>{html.escape(display_text('5 確認'))}</span><b>›</b><span>{html.escape(display_text('6–8 保存・連携'))}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _sidebar_section_header(
        1,
        ui_message('ui.492c8e3fc1fb77'),
        "方式とサンプリング周波数を選び、クロスごとに設定します。",
        tag=ui_message('ui.cd843e170534b0'),
    )
    mode_list = list(MODE_BANDS)
    mode_label = [MODE_LABELS[mode] for mode in mode_list]
    mode_key_map = dict(zip(mode_label, mode_list))
    legacy_mode_labels = {
        "2Way (Low/High)": "2Way (High/Low)",
        "3Way (Low/Mid/High)": "3Way (High/Mid/Low)",
        "4Way (SUB/Low/Mid/High)": MODE_LABELS["3Way+SUB"],
        "4Way (High/Mid/Low/SUB)": MODE_LABELS["3Way+SUB"],
    }
    if st.session_state.get(KEY_MODE) in legacy_mode_labels:
        st.session_state[KEY_MODE] = legacy_mode_labels[st.session_state[KEY_MODE]]
    elif KEY_MODE in st.session_state and st.session_state[KEY_MODE] not in mode_label:
        st.session_state.pop(KEY_MODE, None)
    last_mode = settings.get("last_mode", "2Way")
    mode_idx = mode_list.index(last_mode)

    def on_mode_change():
        selected_label = st.session_state[KEY_MODE]
        selected_key = mode_key_map[selected_label]
        selected_conf = settings[selected_key]
        settings["last_mode"] = selected_key
        st.session_state["filter_type"] = selected_key
        st.session_state[f"{KEY_FIR_OUTPUT_PREFIX}{selected_key}"] = bool(
            selected_conf.get("fir_output_enabled", False)
        )
        _set_indexed_widget_state(
            "__sel_val_", KEY_FS, int(selected_conf.get("fs", 96000))
        )
        st.session_state[KEY_BAFFLE_ON] = bool(selected_conf.get("baffle_on", False))
        st.session_state[KEY_BAFFLE_ATTEN] = float(
            selected_conf.get("baffle_atten", 6.0)
        )
        _set_indexed_widget_state(
            "__num_", KEY_BAFFLE_F1, int(selected_conf.get("baffle_f1_hz", 1000))
        )
        # Band names such as High/Low are shared by several modes, while their
        # tap settings are mode-owned.  Re-seed the widgets from the selected
        # mode so values cannot leak across mode switches.
        selected_crop_lens = selected_conf.get("crop_lens", {})
        for selected_band in MODE_BANDS[selected_key]:
            st.session_state[f"{selected_band}_crop"] = int(
                selected_crop_lens.get(selected_band, 0)
            )
        if selected_key == "2Way":
            st.session_state[KEY_CYCLES] = float(selected_conf["cycles"])
            st.session_state[KEY_BETA] = float(selected_conf["beta"])
            st.session_state["kaiser_overlap_oct"] = float(
                selected_conf.get("kaiser_overlap_oct", 0.0)
            )
        elif selected_key == "3Way":
            st.session_state[KEY_CYCLES_LOW] = float(selected_conf["cycles_low"])
            st.session_state[KEY_BETA_LOW] = float(selected_conf["beta_low"])
            st.session_state[KEY_CYCLES_HIGH] = float(selected_conf["cycles_high"])
            st.session_state[KEY_BETA_HIGH] = float(selected_conf["beta_high"])
            st.session_state["kaiser_overlap_low_mid_oct"] = float(
                selected_conf.get("kaiser_overlap_low_mid_oct", 0.0)
            )
            st.session_state["kaiser_overlap_mid_high_oct"] = float(
                selected_conf.get("kaiser_overlap_mid_high_oct", 0.0)
            )
        elif selected_key == "3Way+SUB":
            st.session_state[KEY_CYCLES_SUB_LOW] = float(selected_conf["cycles_sub_low"])
            st.session_state[KEY_BETA_SUB_LOW] = float(selected_conf["beta_sub_low"])
            st.session_state[KEY_CYCLES_LOW_MID] = float(selected_conf["cycles_low_mid"])
            st.session_state[KEY_BETA_LOW_MID] = float(selected_conf["beta_low_mid"])
            st.session_state[KEY_CYCLES_MID_HIGH] = float(selected_conf["cycles_mid_high"])
            st.session_state[KEY_BETA_MID_HIGH] = float(selected_conf["beta_mid_high"])
            st.session_state["kaiser_overlap_sub_low_oct"] = float(
                selected_conf.get("kaiser_overlap_sub_low_oct", 0.0)
            )
            st.session_state["kaiser_overlap_low_mid_oct"] = float(
                selected_conf.get("kaiser_overlap_low_mid_oct", 0.0)
            )
            st.session_state["kaiser_overlap_mid_high_oct"] = float(
                selected_conf.get("kaiser_overlap_mid_high_oct", 0.0)
            )
        else:
            boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(
                selected_key, selected_conf,
            )
            for index, (cycle_value, beta_value, overlap_value) in enumerate(zip(
                boundary_cycles, boundary_betas, boundary_overlaps,
            )):
                st.session_state[f"boundary_cycles_{selected_key}_{index}"] = cycle_value
                st.session_state[f"boundary_beta_{selected_key}_{index}"] = beta_value
                st.session_state[f"boundary_overlap_{selected_key}_{index}"] = overlap_value
        save_settings(settings)
        st.session_state["settings"] = settings
        clear_results()

    mode_radio_kwargs = {
        "key": KEY_MODE,
        "on_change": on_mode_change,
    }
    if KEY_MODE not in st.session_state:
        mode_radio_kwargs["index"] = mode_idx
    current_mode_label = shared_radio(ui_message('ui.d223c7a1df7b05'), mode_label, **mode_radio_kwargs, format_func=localized_formatter(str))
    mode_key = mode_key_map[current_mode_label]
    mode_conf = settings[mode_key]

    def reset_selected_way_settings():
        from crossover_engine.reset import reset_mode_boundaries, boundary_frequency_keys
        from composite_engine.multiway_studio.extension import _clear_applied_phase_alignment
        updated = reset_mode_boundaries(settings, default_settings, mode_key)
        settings[mode_key] = updated[mode_key]
        conf = settings[mode_key]
        for index, frequency_key in enumerate(boundary_frequency_keys(mode_key, len(conf["cross_freqs"]))):
            _set_indexed_widget_state("__num_", frequency_key, int(conf["cross_freqs"][index]))
            st.session_state[f"composite_studio_{mode_key}_crossover_method_{index}"] = conf["crossover_methods"][index]
            st.session_state[f"composite_studio_{mode_key}_acoustic_target_{index}"] = False
        st.session_state[f"composite_studio_{mode_key}_lr2_auto_polarity"] = bool(conf.get("iir_lr2_auto_polarity", True))
        _clear_applied_phase_alignment()
        on_mode_change()

    st.button(ui_message('ui.6d363dc1e29dd7'), key="reset_selected_way_settings",
              on_click=reset_selected_way_settings,
              disabled=not mode_conf.get("cross_freqs"), help=ui_message('ui.e9334b739965cd'))

    # --- 初期値のシード ---
    if KEY_FS not in st.session_state:
        st.session_state[KEY_FS] = mode_conf.get('fs', 96000)
    if KEY_BAFFLE_ON not in st.session_state:
        st.session_state[KEY_BAFFLE_ON] = bool(mode_conf.get('baffle_on', False))

    bands = MODE_BANDS[mode_key]
    ui_bands = list(reversed(bands))
    fir_output_state_key = f"{KEY_FIR_OUTPUT_PREFIX}{mode_key}"
    if fir_output_state_key not in st.session_state:
        st.session_state[fir_output_state_key] = bool(
            mode_conf.get("fir_output_enabled", False)
        )
    fir_output_enabled = bool(st.session_state[fir_output_state_key])
    # fs は「値」で返る（新ui_index）
    with preferences_host:
        st.markdown(display_text("**サンプリング周波数**"))
        fs = selectbox_indexed(
            ui_message('ui.22e86c38e31428'),
            values=SUPPORTED_SAMPLE_RATES,
            settings_key="fs",
            default=mode_conf.get('fs', 96000),
            key=KEY_FS,
            on_change=clear_results
        )

    def _crossover_settings_snapshot():
        method_count = max(0, len(bands) - 1)
        updated = json.loads(json.dumps(st.session_state.get("settings", settings)))
        methods = [
            normalize_crossover_method(
                st.session_state.get(
                    f"composite_studio_{mode_key}_crossover_method_{index}"
                )
            )
            for index in range(method_count)
        ]
        updated_conf = updated.setdefault(mode_key, {})
        updated_conf["crossover_methods"] = methods
        updated_conf["boundary_acoustic_targets"] = [bool(st.session_state.get(f"composite_studio_{mode_key}_acoustic_target_{i}", False)) and method not in {"Kaiser FIR", "Through"} for i, method in enumerate(methods)]
        return updated, updated_conf, methods

    def _commit_crossover_settings(updated) -> None:
        updated = normalize_settings(updated)
        st.session_state["settings"] = updated
        save_settings(updated)
        clear_results()

    def _persist_crossover_methods() -> None:
        updated, _updated_conf, _methods = _crossover_settings_snapshot()
        _commit_crossover_settings(updated)

    def _persist_acoustic_targets() -> None:
        updated, _updated_conf, _methods = _crossover_settings_snapshot()
        _commit_crossover_settings(updated)

    initial_crossover_methods = tuple(mode_conf.get("crossover_methods", ()))

    def _boundary_method(index: int, label: str) -> str:
        return render_crossover_method_control(
            mode_key, index, label,
            initial_method=(
                initial_crossover_methods[index]
                if index < len(initial_crossover_methods) else "Kaiser FIR"
            ),
            on_method_change=_persist_crossover_methods,
            on_acoustic_target_change=_persist_acoustic_targets,
            fir_enabled=fir_output_enabled,
        )

    if mode_key == "Fullrange":
        cross_freqs = []
        boundary_cycles = boundary_betas = boundary_overlaps = ()
        cycles_low = cycles_high = cycles = float(mode_conf.get("cycles", DEFAULT_FIR_CYCLES))
        beta_low = beta_high = beta = float(mode_conf.get("beta", DEFAULT_FIR_BETA))
        st.caption(ui_message('ui.08f68b6d9a95b0'))
    elif mode_key == "2Way":
        _crossover_header(ui_message('ui.d6d1148df4b85d'))
        boundary_method = _boundary_method(0, "High / Low")
        boundary_uses_fir = boundary_method in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
        fc = number_input_stateful_safe(
            ui_message('ui.2951ffd237d74a'),
            settings_key="fc",
            min_value=10,
            max_value=max(10, int(fs//2) - 1),
            default=int(mode_conf.get('cross_freqs', [2500])[0]),
            step=1,
            key="fc",
            on_change=clear_results
        )
        cycles = _number_input_with_session_default(
            ui_message('ui.2c548f58e181f3'), min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX, step=0.1,
            value=float(mode_conf.get('cycles', DEFAULT_FIR_CYCLES)), format="%.4f",
            key=KEY_CYCLES,
            on_change=clear_results,
            disabled=boundary_method != "Kaiser FIR",
            help=ui_message('ui.1b911b58f117e7')
        )
        beta = _number_input_with_session_default(
            ui_message('ui.b9adc18c131e99'), min_value=FIR_BETA_MIN, max_value=FIR_BETA_MAX,
            value=float(mode_conf.get('beta', DEFAULT_FIR_BETA)),
            key=KEY_BETA,
            on_change=clear_results,
            disabled=boundary_method != "Kaiser FIR",
            help=ui_message('ui.ad1c8cc1adefe3')
        )
        if boundary_uses_fir:
            _show_fir_design_guide(beta, method=boundary_method, sample_rate_hz=fs, crossover_hz=fc, cycles=cycles, mode_key=mode_key, index=0)
        else:
            st.caption(ui_message('ui.1bec2fb1ad1703'))
        with st.expander(ui_message('ui.5b86f014610e90'), expanded=False):
            kaiser_overlap_oct = _number_input_with_session_default(
                ui_message('ui.d24b90a5d17922'),
                min_value=-1.0,
                max_value=1.0,
                value=float(mode_conf.get("kaiser_overlap_oct", 0.0)),
                step=0.01,
                format="%.2f",
                key="kaiser_overlap_oct",
                on_change=clear_results,
                disabled=not boundary_uses_fir,
                help=ui_message('ui.8dfa81f6d8b349')
            )
            _lp_edge = float(fc) * 2.0 ** (float(kaiser_overlap_oct) / 2.0)
            _hp_edge = float(fc) * 2.0 ** (-float(kaiser_overlap_oct) / 2.0)
            if boundary_uses_fir:
                st.caption(ui_message('ui.5633cb53228733', p0=f'{_lp_edge:.1f}', p1=f'{_hp_edge:.1f}'))
        cross_freqs = [int(fc)]
        cycles_low = cycles_high = float(cycles)
        beta_low = beta_high = float(beta)
    elif mode_key == "3Way":
        default_cross = mode_conf.get('cross_freqs', [500, 2500])

        _crossover_header(ui_message('ui.e871c13d9a91b1'))
        high_method = _boundary_method(1, "High / Mid")
        high_uses_fir = high_method in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
        fc2 = number_input_stateful_safe(
            ui_message('ui.96d5b48500eb95'),
            settings_key="fc2",
            min_value=10,
            max_value=max(10, int(fs//2) - 1),
            default=int(default_cross[1]),
            step=1,
            key="fc2",
            on_change=clear_results
        )
        cycles_high = _number_input_with_session_default(
            ui_message('ui.b9a07498d1a75f'), min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX, step=0.1,
            value=float(mode_conf.get('cycles_high', mode_conf.get('cycles', DEFAULT_FIR_CYCLES))), format="%.4f",
            key=KEY_CYCLES_HIGH,
            on_change=clear_results,
            disabled=high_method != "Kaiser FIR",
            help=ui_message('ui.a87567900297c7')
        )
        beta_high = _number_input_with_session_default(
            ui_message('ui.e529de8f4249e0'), min_value=FIR_BETA_MIN, max_value=FIR_BETA_MAX,
            value=float(mode_conf.get('beta_high', mode_conf.get('beta', DEFAULT_FIR_BETA))),
            key=KEY_BETA_HIGH,
            on_change=clear_results,
            disabled=high_method != "Kaiser FIR",
            help=ui_message('ui.b8f3795a990fc5')
        )
        if high_uses_fir:
            _show_fir_design_guide(beta_high, method=high_method, sample_rate_hz=fs, crossover_hz=fc2, cycles=cycles_high, mode_key=mode_key, index=1)
        else:
            st.caption(ui_message('ui.1bec2fb1ad1703'))
        with st.expander(ui_message('ui.936528495a6dff'), expanded=False):
            kaiser_overlap_mid_high_oct = _number_input_with_session_default(
                ui_message('ui.141877b6676157'), min_value=-1.0, max_value=1.0,
                value=float(mode_conf.get("kaiser_overlap_mid_high_oct", 0.0)), step=0.01,
                format="%.2f", key="kaiser_overlap_mid_high_oct",
                on_change=clear_results, disabled=not high_uses_fir,
                help=ui_message('ui.b9780e8846794c')
            )
            if high_uses_fir:
                st.caption(ui_message('ui.5633cb53228733', p0=f'{float(fc2) * 2.0 ** (float(kaiser_overlap_mid_high_oct) / 2.0):.1f}', p1=f'{float(fc2) * 2.0 ** (-float(kaiser_overlap_mid_high_oct) / 2.0):.1f}'))

        _crossover_header(ui_message('ui.5a068956447334'))
        low_method = _boundary_method(0, "Mid / Low")
        low_uses_fir = low_method in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"}
        fc1 = number_input_stateful_safe(
            ui_message('ui.a495f504fd63fa'),
            settings_key="fc1",
            min_value=10,
            max_value=max(10, int(fs//2) - 1),
            default=int(default_cross[0]),
            step=1,
            key="fc1",
            on_change=clear_results
        )
        cycles_low = _number_input_with_session_default(
            ui_message('ui.b669fae1e0eb8f'), min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX, step=0.1,
            value=float(mode_conf.get('cycles_low', mode_conf.get('cycles', DEFAULT_FIR_CYCLES))), format="%.4f",
            key=KEY_CYCLES_LOW,
            on_change=clear_results,
            disabled=low_method != "Kaiser FIR",
            help=ui_message('ui.91a847851ed4fc')
        )
        beta_low = _number_input_with_session_default(
            ui_message('ui.09cac93772da46'), min_value=FIR_BETA_MIN, max_value=FIR_BETA_MAX,
            value=float(mode_conf.get('beta_low', mode_conf.get('beta', DEFAULT_FIR_BETA))),
            key=KEY_BETA_LOW,
            on_change=clear_results,
            disabled=low_method != "Kaiser FIR",
            help=ui_message('ui.035d9dc490b60f')
        )
        if low_uses_fir:
            _show_fir_design_guide(beta_low, method=low_method, sample_rate_hz=fs, crossover_hz=fc1, cycles=cycles_low, mode_key=mode_key, index=0)
        else:
            st.caption(ui_message('ui.1bec2fb1ad1703'))
        with st.expander(ui_message('ui.96fdfe6111d439'), expanded=False):
            kaiser_overlap_low_mid_oct = _number_input_with_session_default(
                ui_message('ui.7c1b2557dbe95d'), min_value=-1.0, max_value=1.0,
                value=float(mode_conf.get("kaiser_overlap_low_mid_oct", 0.0)), step=0.01,
                format="%.2f", key="kaiser_overlap_low_mid_oct",
                on_change=clear_results, disabled=not low_uses_fir,
                help=ui_message('ui.d4aa9d177d761c')
            )
            if low_uses_fir:
                st.caption(ui_message('ui.5633cb53228733', p0=f'{float(fc1) * 2.0 ** (float(kaiser_overlap_low_mid_oct) / 2.0):.1f}', p1=f'{float(fc1) * 2.0 ** (-float(kaiser_overlap_low_mid_oct) / 2.0):.1f}'))
        if high_method == low_method == "Kaiser FIR":
            st.caption(ui_message('ui.4ce0420ff6f4a9'))
        cross_freqs = [int(fc1), int(fc2)]
        cycles = float(cycles_low)
        beta = float(beta_low)
    elif mode_key == "3Way+SUB":
        default_cross = mode_conf.get("cross_freqs", [120, 500, 2500])

        def render_4way_crossover(
            title, label, index, frequency_key,
            cycles_key, beta_key, overlap_key, help_prefix
        ):
            _crossover_header(title)
            crossover_method = _boundary_method(index, label)
            crossover_uses_fir = crossover_method in {
                "Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR",
            }
            crossover = number_input_stateful_safe(
                ui_message('ui.31492e471afeef', p0=f'{label}'),
                settings_key=frequency_key,
                min_value=10,
                max_value=max(10, int(fs // 2) - 1),
                default=int(default_cross[index]),
                step=1,
                key=frequency_key,
                on_change=clear_results,
            )
            crossover_cycles = _number_input_with_session_default(
                ui_message('ui.1b94e0a619b19f', p0=f'{label}'),
                min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX,
                step=0.1,
                value=float(mode_conf.get(cycles_key, mode_conf.get("cycles", DEFAULT_FIR_CYCLES))),
                format="%.4f",
                key=cycles_key,
                on_change=clear_results,
                disabled=crossover_method != "Kaiser FIR",
                help=ui_message('ui.d3ca313680c291', p0=f'{help_prefix}'),
            )
            crossover_beta = _number_input_with_session_default(
                ui_message('ui.b7051290cc81b5', p0=f'{label}'),
                min_value=FIR_BETA_MIN,
                max_value=FIR_BETA_MAX,
                value=float(mode_conf.get(beta_key, mode_conf.get("beta", DEFAULT_FIR_BETA))),
                key=beta_key,
                on_change=clear_results,
                disabled=crossover_method != "Kaiser FIR",
                help=ui_message('ui.ba03f30e780c3d', p0=f'{help_prefix}'),
            )
            if crossover_uses_fir:
                _show_fir_design_guide(crossover_beta, method=crossover_method, sample_rate_hz=fs, crossover_hz=crossover, cycles=crossover_cycles, mode_key=mode_key, index=index)
            else:
                st.caption(ui_message('ui.1bec2fb1ad1703'))
            with st.expander(ui_message('ui.e4c82789812778', p0=f'{label}'), expanded=False):
                crossover_overlap = _number_input_with_session_default(
                    ui_message('ui.a85d25253081f7', p0=f'{label}'), min_value=-1.0, max_value=1.0,
                    value=float(mode_conf.get(overlap_key, 0.0)), step=0.01, format="%.2f",
                    key=overlap_key,
                    on_change=clear_results,
                    disabled=not crossover_uses_fir,
                    help=ui_message('ui.c15125b3adc33f'),
                )
                st.caption(
                    ui_message('ui.2ead3de0457365', p0=f'{max(1, round(crossover * 0.01))}')
                )
            return (
                int(crossover),
                float(crossover_cycles),
                float(crossover_beta),
                float(crossover_overlap),
            )

        fc3, cycles_mid_high, beta_mid_high, kaiser_overlap_mid_high_oct = render_4way_crossover(
            "上側クロス｜High / Mid",
            "High / Mid",
            2,
            "fc4_mid_high",
            KEY_CYCLES_MID_HIGH,
            KEY_BETA_MID_HIGH,
            "kaiser_overlap_mid_high_oct",
            "HighとMid上側",
        )
        fc2, cycles_low_mid, beta_low_mid, kaiser_overlap_low_mid_oct = render_4way_crossover(
            "中央クロス｜Mid / Low",
            "Mid / Low",
            1,
            "fc4_low_mid",
            KEY_CYCLES_LOW_MID,
            KEY_BETA_LOW_MID,
            "kaiser_overlap_low_mid_oct",
            "Low上側とMid下側",
        )
        fc1, cycles_sub_low, beta_sub_low, kaiser_overlap_sub_low_oct = render_4way_crossover(
            "下側クロス｜Low / SUB",
            "Low / SUB",
            0,
            "fc4_sub_low",
            KEY_CYCLES_SUB_LOW,
            KEY_BETA_SUB_LOW,
            "kaiser_overlap_sub_low_oct",
            "SUBとLow下側",
        )
        if all(st.session_state.get(f"composite_studio_{mode_key}_crossover_method_{i}") == "Kaiser FIR" for i in range(3)):
            st.caption(ui_message('ui.f18c855684f002'))
        cross_freqs = [fc1, fc2, fc3]
        cycles = float(cycles_sub_low)
        beta = float(beta_sub_low)
    else:
        default_cross = list(mode_conf.get("cross_freqs", ()))
        boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(
            mode_key, mode_conf,
        )
        cross_freqs = [0] * (len(bands) - 1)
        edited_cycles = list(boundary_cycles)
        edited_betas = list(boundary_betas)
        edited_overlaps = list(boundary_overlaps)
        for index in reversed(range(len(cross_freqs))):
            lower_band, upper_band = bands[index], bands[index + 1]
            boundary_label = f"{upper_band} / {lower_band}"
            _crossover_header(ui_message('ui.6b14fc2dc35def', p0=f'{boundary_label}'))
            boundary_method = _boundary_method(index, boundary_label)
            boundary_uses_fir = boundary_method in {
                "Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR",
            }
            cross_freqs[index] = int(number_input_stateful_safe(
                ui_message('ui.31492e471afeef', p0=f'{boundary_label}'),
                settings_key=f"boundary_fc_{mode_key}_{index}",
                min_value=10,
                max_value=max(10, int(fs // 2) - 1),
                default=int(default_cross[index]),
                step=1,
                key=f"boundary_fc_{mode_key}_{index}",
                on_change=clear_results,
            ))
            edited_cycles[index] = float(_number_input_with_session_default(
                ui_message('ui.36ca1e5162ff27', p0=f'{boundary_label}'), min_value=FIR_CYCLES_MIN, max_value=FIR_CYCLES_MAX, step=0.1,
                value=float(boundary_cycles[index]), format="%.4f",
                key=f"boundary_cycles_{mode_key}_{index}", on_change=clear_results,
                disabled=boundary_method != "Kaiser FIR",
            ))
            edited_betas[index] = float(_number_input_with_session_default(
                ui_message('ui.a6f17efb2e363d', p0=f'{boundary_label}'), min_value=FIR_BETA_MIN, max_value=FIR_BETA_MAX,
                value=float(boundary_betas[index]),
                key=f"boundary_beta_{mode_key}_{index}", on_change=clear_results,
                disabled=boundary_method != "Kaiser FIR",
            ))
            if boundary_uses_fir:
                _show_fir_design_guide(edited_betas[index], method=boundary_method, sample_rate_hz=fs, crossover_hz=cross_freqs[index], cycles=edited_cycles[index], mode_key=mode_key, index=index)
            else:
                st.caption(ui_message('ui.1bec2fb1ad1703'))
            with st.expander(ui_message('ui.e4c82789812778', p0=f'{boundary_label}'), expanded=False):
                edited_overlaps[index] = float(_number_input_with_session_default(
                    ui_message('ui.a85d25253081f7', p0=f'{boundary_label}'),
                    min_value=-1.0, max_value=1.0, step=0.01, format="%.2f",
                    value=float(boundary_overlaps[index]),
                    key=f"boundary_overlap_{mode_key}_{index}", on_change=clear_results,
                    disabled=not boundary_uses_fir,
                ))
        boundary_cycles = tuple(edited_cycles)
        boundary_betas = tuple(edited_betas)
        boundary_overlaps = tuple(edited_overlaps)
        cycles = float(boundary_cycles[0])
        beta = float(boundary_betas[0])
        cycles_low = cycles_high = cycles
        beta_low = beta_high = beta

    phase_alignment_host = render_shared_iir_crossover_ui(
        mode_key,
        tuple(float(value) for value in cross_freqs),
        initial_methods=tuple(mode_conf.get("crossover_methods", ())),
        initial_lr2_auto_polarity=bool(mode_conf.get("iir_lr2_auto_polarity", True)),
        on_method_change=_persist_crossover_methods,
        on_acoustic_target_change=_persist_acoustic_targets,
        render_methods=False,
    )

    _sidebar_section_header(2, ui_message('ui.7aa3c607f4688b'), tag=ui_message('ui.812293fce89648'))
    baffle_on = st.checkbox(
        ui_message('ui.8b1ddefa78498c'),
        key=KEY_BAFFLE_ON,
        on_change=clear_results
    )
    baffle_f1_hz = number_input_stateful_safe(
        ui_message('ui.e146e5d7d37bfd'),
        settings_key=KEY_BAFFLE_F1,
        min_value=100,
        max_value=max(100, int(fs//2) - 1),
        default=int(mode_conf.get("baffle_f1_hz", 1000)),
        step=50,
        key=KEY_BAFFLE_F1,
        on_change=clear_results
    )
    baffle_atten = _number_input_with_session_default(
        ui_message('ui.468001c6d03fef'), min_value=0.0, max_value=12.0,
        value=mode_conf.get("baffle_atten", 6.0), step=0.1, format="%.1f",
        key=KEY_BAFFLE_ATTEN,
        on_change=clear_results
    )
    st.caption(ui_message('ui.f990b6f1785d00'))
    # Filled after FIR state resolution; retain its height during reruns.
    baffle_mode_status = st.container()

    _sidebar_section_header(
        3,
        ui_message('ui.83c835a46db247'),
        "FIRの使用有無を決めてから、自動クロップまたは帯域ごとのタップ長を選びます。",
        tag=ui_message('ui.cd843e170534b0'),
    )

    def _on_fir_output_change() -> None:
        enabled = bool(st.session_state.get(fir_output_state_key, False))
        updated = json.loads(json.dumps(st.session_state.get("settings", settings)))
        updated_conf = updated.setdefault(mode_key, {})
        updated_conf["fir_output_enabled"] = enabled
        if enabled:
            methods = [
                normalize_crossover_method(
                    st.session_state.get(
                        f"composite_studio_{mode_key}_crossover_method_{index}"
                    )
                )
                for index in range(len(cross_freqs))
            ]
            current_taps = {
                band: int(st.session_state.get(f"{band}_crop", 0) or 0)
                for band in bands
            }
            initialized = initialize_manual_taps_for_enable(
                bands, methods, current_taps,
            )
            for band, taps in initialized.items():
                st.session_state[f"{band}_crop"] = int(taps)
            updated_conf["crop_lens"] = initialized
        updated = normalize_settings(updated)
        for index, method in enumerate(updated[mode_key]["crossover_methods"]):
            st.session_state[f"composite_studio_{mode_key}_crossover_method_{index}"] = method
        st.session_state["settings"] = updated
        save_settings(updated)
        clear_results()

    fir_output_enabled = st.checkbox(
        ui_message('ui.a0828a6876abd0'),
        key=fir_output_state_key,
        on_change=_on_fir_output_change,
        help=(
            ui_message('ui.f0108b5de31092')
        ),
    )
    _has_fir_split = bool(bands_requiring_split_fir(bands, [
        "Through" if st.session_state.get(f"composite_studio_{mode_key}_acoustic_target_{index}", False) else normalize_crossover_method(st.session_state.get(
            f"composite_studio_{mode_key}_crossover_method_{index}",
            mode_conf["crossover_methods"][index],
        )) for index in range(len(cross_freqs))
    ]))
    if fir_output_enabled:
        st.success(ui_message('ui.70a9713177531b'))
    else:
        st.info(ui_message('ui.3ad0e487dc68bb'))
    if not _has_fir_split:
        # A restored ON value must not lock manual Way lengths when all
        # crossovers are IIR/Through (including a Fullrange with no boundary).
        st.session_state[KEY_AUTO_CROP] = False
    elif KEY_AUTO_CROP not in st.session_state:
        st.session_state[KEY_AUTO_CROP] = bool(settings.get("auto_crop", False))
    auto_crop = st.checkbox(
        ui_message('ui.2af8931ca87631'),
        key=KEY_AUTO_CROP,
        on_change=clear_results,
        disabled=not fir_output_enabled or not _has_fir_split,
        help=ui_message('ui.02b8b70e6386ef')
    )
    if _has_fir_split:
        st.caption(ui_message('ui.341070169568d5'))
    else:
        st.caption(ui_message('ui.d9d9113191c2a6'))
    if KEY_AUTO_CROP_PROFILE not in st.session_state:
        st.session_state[KEY_AUTO_CROP_PROFILE] = settings.get(
            "auto_crop_profile", DEFAULT_AUTO_CROP_PROFILE
        )
    auto_crop_profile = shared_selectbox(
        ui_message('ui.fc6074b8b4c69d'),
        options=list(AUTO_CROP_PROFILES),
        format_func=lambda key: AUTO_CROP_PROFILES[key]["label"],
        key=KEY_AUTO_CROP_PROFILE,
        on_change=clear_results,
        disabled=not fir_output_enabled or not auto_crop,
        help=ui_message('ui.a8628852850a69'),
    )
    auto_crop_pass_db = AUTO_CROP_PROFILES[auto_crop_profile]["pass_db"]
    auto_crop_cross_db = AUTO_CROP_PROFILES[auto_crop_profile]["cross_db"]
    st.caption(
        ui_message('ui.6e2643aa9c367d', p0=f'{auto_crop_pass_db:.2f}', p1=f'{auto_crop_cross_db:.2f}')
    )
    lr2_bands = set()
    for boundary_index in range(len(bands)-1):
        if st.session_state.get(f"composite_studio_{mode_key}_crossover_method_{boundary_index}") == "Linear-phase LR2 FIR":
            lr2_bands.update(bands[boundary_index:boundary_index+2])
    if lr2_bands:
        st.caption(ui_message('ui.b7d4d80bb3eaef'))
    user_crop_lengths = {}
    band_tap_count_hosts = {}
    crop_lens_conf = mode_conf.get("crop_lens", {})
    application_max_fir_taps = max(ir_length_for_frequency_resolution(int(fs), 2.0), 262143)
    st.caption(
        ui_message('ui.222f6bb28785b1', p0=f'{DEFAULT_MANUAL_FIR_TAPS}')
    )
    for band in ui_bands:
        crop_key = f"{band}_crop"
        user_crop_lengths[band] = _number_input_with_session_default(
            ui_message('ui.9429f84acd694d', p0=f'{band}'),
            min_value=0, max_value=application_max_fir_taps, step=1,
            value=int(crop_lens_conf.get(band, 0)),
            key=crop_key,
            on_change=clear_results,
            disabled=not fir_output_enabled or auto_crop,
            help=(
                ui_message('ui.12730b4003aa7e')
            ),
        )
        band_tap_count_hosts[band] = st.empty()
        band_tap_count_hosts[band].caption(display_text("現在の最終FIRタップ数：再計算待ち"))
    st.caption(
        ui_message('ui.6d2408f7090298', p0=f'{application_max_fir_taps:,}')
    )

    if KEY_ALIGN_OUTPUT_TAPS not in st.session_state:
        st.session_state[KEY_ALIGN_OUTPUT_TAPS] = bool(settings.get("align_output_taps", False))
    align_output_taps = st.checkbox(
        ui_message('ui.ce0c46eace41fa'),
        key=KEY_ALIGN_OUTPUT_TAPS,
        on_change=clear_results,
        disabled=not fir_output_enabled,
        help=ui_message('ui.6fa299baadb738')
    )
    st.caption(
        ui_message('ui.e45f0cbbe6748f')
    )

    from ui.multiway_eq_files import render_eq_files
    eq_files_conf = _normalize_eq_files(mode_conf.get("eq_files", {}), bands)

    def _persist_eq_files(files):
        settings[mode_key]["eq_files"] = files
        save_settings(settings)
        clear_results()

    render_eq_files(
        ui_bands, eq_files_conf, enabled=fir_output_enabled,
        save_upload=save_uploaded_eq, persist=_persist_eq_files,
    )

    _sidebar_section_header(5, ui_message('ui.a8695cacb554c1'), tag=ui_message('ui.dce52f3e11196e'))
    st.markdown(ui_message('ui.116819f72bc69e'))
    gain_min_widget_key = "composite_studio_gain_y_min_widget"
    gain_max_widget_key = "composite_studio_gain_y_max_widget"
    gain_min_default = int(settings.get("gain_y_min_db", -20))
    gain_max_default = int(settings.get("gain_y_max_db", 20))
    if st.session_state.get(gain_min_widget_key) not in visualization.GAIN_Y_MIN_OPTIONS:
        st.session_state[gain_min_widget_key] = gain_min_default
    if st.session_state.get(gain_max_widget_key) not in visualization.GAIN_Y_MAX_OPTIONS:
        st.session_state[gain_max_widget_key] = gain_max_default
    db_min = shared_segmented_control(
        ui_message('ui.646134c1d76907'), visualization.GAIN_Y_MIN_OPTIONS,
        key=gain_min_widget_key,
        format_func=lambda value: f"{int(value):+d}",
        required=True,
        on_change=clear_results,
        width="stretch",
    ) or gain_min_default
    db_max = shared_segmented_control(
        ui_message('ui.e5e6d18d33b5cc'), visualization.GAIN_Y_MAX_OPTIONS,
        key=gain_max_widget_key,
        format_func=lambda value: f"{int(value):+d}",
        required=True,
        on_change=clear_results,
        width="stretch",
    ) or gain_max_default
    phase_mask_widget_key = "composite_studio_phase_gain_mask_widget"
    phase_mask_default = int(settings.get("phase_gain_mask_db", -60))
    if st.session_state.get(phase_mask_widget_key) not in visualization.PHASE_GAIN_MASK_OPTIONS_DB:
        st.session_state[phase_mask_widget_key] = phase_mask_default
    phase_gain_mask_db = shared_segmented_control(
        ui_message('ui.de64fe76ec95c5'), visualization.PHASE_GAIN_MASK_OPTIONS_DB,
        key=phase_mask_widget_key,
        format_func=lambda value: f"Gain < {int(value)} dB",
        required=True,
        on_change=clear_results,
        width="stretch",
    ) or phase_mask_default
    st.session_state[KEY_GAIN_Y_MIN] = int(db_min)
    st.session_state[KEY_GAIN_Y_MAX] = int(db_max)
    st.session_state[KEY_PHASE_GAIN_MASK] = int(phase_gain_mask_db)
    st.caption(
        ui_message('ui.30562270dcbdb0')
    )
    current_conf = {
        'phase_alignment_acoustic_target': bool(st.session_state.get('settings', {}).get(mode_key, {}).get('phase_alignment_acoustic_target', False)),
        'lr2_auto_taps': True,
        'boundary_acoustic_targets': [bool(st.session_state.get(f"composite_studio_{mode_key}_acoustic_target_{i}", False)) for i in range(len(cross_freqs))],
        'fs': int(fs),
        'cycles': float(cycles),
        'beta': float(beta),
        'cross_freqs': [int(v) for v in (cross_freqs or [])],
        'crop_lens': {band: int(user_crop_lengths[band]) for band in bands},
        'eq_files': eq_files_conf,
        'baffle_on': bool(baffle_on),
        'baffle_atten': float(baffle_atten),
        'baffle_f1_hz': int(baffle_f1_hz),
        'kaiser_overlap_oct': float(kaiser_overlap_oct) if mode_key == '2Way' else 0.0,
        'kaiser_overlap_sub_low_oct': float(kaiser_overlap_sub_low_oct) if mode_key == '3Way+SUB' else 0.0,
        'kaiser_overlap_low_mid_oct': float(kaiser_overlap_low_mid_oct) if mode_key in ('3Way', '3Way+SUB') else 0.0,
        'kaiser_overlap_mid_high_oct': float(kaiser_overlap_mid_high_oct) if mode_key in ('3Way', '3Way+SUB') else 0.0,
        'iir_lr2_auto_polarity': bool(st.session_state.get(f"composite_studio_{mode_key}_lr2_auto_polarity", True)),
        'crossover_methods': [
            normalize_crossover_method(
                st.session_state.get(f"composite_studio_{mode_key}_crossover_method_{index}")
            )
            for index in range(len(cross_freqs))
        ],
    }
    if mode_key == "3Way":
        current_conf.update({
            'cycles_low': float(cycles_low),
            'beta_low': float(beta_low),
            'cycles_high': float(cycles_high),
            'beta_high': float(beta_high),
        })
    elif mode_key == "3Way+SUB":
        current_conf.update({
            'cycles_sub_low': float(cycles_sub_low),
            'beta_sub_low': float(beta_sub_low),
            'cycles_low_mid': float(cycles_low_mid),
            'beta_low_mid': float(beta_low_mid),
            'cycles_mid_high': float(cycles_mid_high),
            'beta_mid_high': float(beta_mid_high),
        })
    elif mode_key not in {"Fullrange", "2Way", "3Way", "3Way+SUB"}:
        current_conf.update({
            'boundary_cycles': list(map(float, boundary_cycles)),
            'boundary_betas': list(map(float, boundary_betas)),
            'boundary_overlaps_oct': list(map(float, boundary_overlaps)),
        })
    current_settings = json.loads(json.dumps(settings))
    current_settings[mode_key] = current_conf
    current_settings["db_min"] = int(db_min)
    current_settings["gain_y_min_db"] = int(db_min)
    current_settings["gain_y_max_db"] = int(db_max)
    current_settings["phase_gain_mask_db"] = int(phase_gain_mask_db)
    current_settings["last_mode"] = mode_key
    current_settings["auto_crop"] = bool(auto_crop)
    current_settings["auto_crop_profile"] = str(auto_crop_profile)
    current_settings["auto_crop_pass_db"] = float(auto_crop_pass_db)
    current_settings["auto_crop_cross_db"] = float(auto_crop_cross_db)

    with preferences_host:
        st.markdown(display_text("**FIR出力設定**"))
        output_format = selectbox_indexed(
            ui_message('ui.da6f2f6bb6b734'),
            values=["bin_float32", "wav_float32", "txt_float", "csv_float"],
            labels=["Binary float32 (.bin)", "WAV float32 (.wav)", "Float text (.txt)", "CSV float (.csv)"],
            settings_key="output_format",
            default=str(settings.get("output_format", st.session_state.get(KEY_OUTPUT_FORMAT, "bin_float32"))),
            key=KEY_OUTPUT_FORMAT,
            on_change=clear_results,
            disabled=not fir_output_enabled,
        format_func=localized_formatter(str))
        st.caption(ui_message('ui.b67843b0d4e50e'))

        output_normalize = st.checkbox(
            ui_message('ui.53ca5c1d4607c3'),
            value=bool(settings.get("output_normalize", False)),
            key=KEY_OUTPUT_NORMALIZE,
            on_change=clear_results,
            disabled=not fir_output_enabled,
            help=(
                ui_message('ui.5da6832c933d66')
            ),
        )
        st.caption(ui_message('ui.ee44027a83160d'))

    current_settings["output_format"] = str(output_format)
    current_settings["output_normalize"] = bool(output_normalize)
    current_settings["align_output_taps"] = bool(align_output_taps)
    current_conf["fir_output_enabled"] = bool(fir_output_enabled)
    current_settings[mode_key] = current_conf
    current_settings = normalize_settings(current_settings)

    _sidebar_section_header(7, ui_message('ui.d44762dd26e749'), tag=ui_message('ui.a3030bf8f16dc6'))
    st.caption(ui_message('ui.f9962ae5344a6a'))
    if st.button(ui_message('ui.4b0f3c3779a191'), key=KEY_EXPORT_BUTTON):
        path = export_config(current_settings)
        ok, err = validate_config_roundtrip(current_settings, path)
        if ok:
            st.success(ui_message('ui.7a4df6a29f21d2', p0=f'{os.path.basename(path)}'))
        else:
            st.warning(ui_message('ui.8988a20e83ca47', p0=f'{err}'))

    st.markdown(ui_message('ui.e9e4eb14645b08'))
    dsp_resume_feedback = st.empty()
    uploaded_settings = st.file_uploader(
        ui_message('ui.45b2d0e5d3c2d8'),
        type=["json", "zip"],
        key=KEY_SETTINGS_UPLOAD,
        help=(
            ui_message('ui.34d33aeed8502b')
        ),
    )
    if uploaded_settings is not None:
        try:
            imported_preview = _read_settings_upload(
                uploaded_settings.name,
                uploaded_settings.getvalue(),
            )
            imported_preview, missing_eq = _clear_missing_eq_references(imported_preview)
            summary = _settings_summary(imported_preview)
            _render_styled_table(pd.DataFrame([summary]), min_width_px=980)
            if missing_eq:
                st.warning(ui_message('ui.170b34166e3ee6', p0=f'{len(missing_eq)}'))
                with st.expander(ui_message('ui.151bc85d7214ca'), expanded=False):
                    for item in missing_eq:
                        st.write(ui_message('ui.9a4a817443e20e', p0=f'{item}'))
            if st.button(ui_message('ui.f6e00058e8eefe'), key=KEY_SETTINGS_APPLY, type="primary"):
                st.session_state["_pending_uploaded_settings"] = imported_preview
                st.session_state["_pending_uploaded_name"] = uploaded_settings.name
                st.session_state["_pending_uploaded_missing_eq"] = missing_eq
                st.rerun()
        except ValueError as e:
            if (
                uploaded_settings.name.lower().endswith(".zip")
                and str(e) == "ZIP内に settings.json がありません。"
            ):
                dsp_resume_upload = uploaded_settings
                st.caption(ui_message('ui.9cd93242133f83'))
            else:
                st.error(ui_message('ui.d4357aefc7014b', p0=f'{e}'))
        except (json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as e:
            st.error(ui_message('ui.d4357aefc7014b', p0=f'{e}'))

    st.markdown(ui_message('ui.6509a8d8c066e2'))
    config_files = sorted([f for f in os.listdir(CONFIG_EXPORT_DIR) if f.endswith(".json")], reverse=True)
    resume_file = shared_selectbox(ui_message('ui.225f8a03123b78'), [""] + config_files, key=KEY_RESUME_SELECT)
    if st.button(ui_message('ui.4c5bfd29a190e7'), key=KEY_RESUME_BTN):
        if resume_file:
            import_config(resume_file)

    st.divider()
    st.caption(f"PhaseEQ Composite Engine v{PHASEEQ_INTEGRATION_VERSION} · Multiway v{APP_VERSION} · Tomii323")
    sidebar_crossover_methods = tuple(current_conf.get("crossover_methods", ()))
    _sidebar_split_before, sidebar_split_firs = _generate_split_filter_bank(
        mode_key, int(fs), current_conf, sidebar_crossover_methods,
    )
    sidebar_fir_states = _assignment_band_fir_states(
        mode_key, current_conf, sidebar_split_firs,
        auto_crop=bool(current_settings.get("auto_crop", False)),
    )
    st.session_state["composite_studio_pending_band_fir_states"] = {
        band: state.to_dict() for band, state in sidebar_fir_states.items()
    }
    st.session_state["composite_studio_band_tap_lengths"] = {
        band: int(state.tap_count or 0) for band, state in sidebar_fir_states.items()
    }
    render_composite_sidebar(
        bands,
        int(fs),
        mode_key=mode_key,
        crossover_frequencies_hz=tuple(
            float(value) for value in current_conf["cross_freqs"]
        ),
        band_tap_lengths=st.session_state["composite_studio_band_tap_lengths"],
        band_linear_fir_filters=_assignment_band_linear_fir_filters(mode_key, current_conf),
        current_settings=current_settings,
        resume_upload=dsp_resume_upload,
        resume_feedback=dsp_resume_feedback,
    )
    active_generated_bands = tuple(
        str(row.get("band", ""))
        for row in st.session_state.get("composite_studio_channels", [])
        if isinstance(row, dict)
        and row.get("source") == "generated_band"
        and bool(row.get("enabled", True))
    )
    baffle_compensation_plan = baffle_plan(
        int(fs), float(baffle_f1_hz), float(baffle_atten),
        active_generated_bands or tuple(bands),
        {band: state.to_dict() for band, state in sidebar_fir_states.items()},
        enabled=bool(baffle_on),
    )
    configured_channels = st.session_state.get("composite_studio_channels", [])
    for row in configured_channels:
        if not isinstance(row, dict) or row.get("source") != "generated_band":
            continue
        row["baffle_correction_mode"] = baffle_compensation_plan.display_mode
        row["baffle_iir_sos"] = (
            baffle_compensation_plan.sos
            if baffle_compensation_plan.mode == "iir" else ()
        )
        row["baffle_iir_parameters"] = (
            {
                "type": "high_shelf",
                "frequency_hz": baffle_compensation_plan.frequency_hz,
                "gain_db": baffle_compensation_plan.gain_db,
                "q": baffle_compensation_plan.q,
                "fit_rms_db": baffle_compensation_plan.fit_rms_db,
                "fit_max_error_db": baffle_compensation_plan.fit_max_error_db,
            }
            if baffle_compensation_plan.mode == "iir" else None
        )
    st.session_state["composite_studio_channels"] = configured_channels
    st.session_state["composite_studio_baffle_plan"] = baffle_compensation_plan
    if baffle_compensation_plan.mode == "iir":
        baffle_mode_status.info(
            ui_message('ui.f1b12661286d05', p0=f'{baffle_compensation_plan.frequency_hz:,.1f}', p1=f'{baffle_compensation_plan.gain_db:+.2f}', p2=f'{baffle_compensation_plan.q:.2f}')
        )
    elif baffle_compensation_plan.mode == "fir":
        baffle_mode_status.info(
            ui_message('ui.4a12599a01782c')
        )
    else:
        baffle_mode_status.caption(ui_message('ui.d2c161f2dd9f5f'))

# Render infrequent controls into the shared settings host without changing evaluation order.
with preferences_host:
    st.markdown(display_text("**グラフ表示設定**"))
    auto_update = _checkbox_with_session_default(
        ui_message('ui.0286249762f7c9'),
        value=bool(settings.get("auto_update", st.session_state.get(KEY_AUTO_UPDATE, True))),
        key=KEY_AUTO_UPDATE,
        help=ui_message('ui.22eb883c0635de'),
    )
    graph_mode_widget_key = "composite_studio_graph_mode_widget"
    graph_mode_default = str(settings.get("graph_mode", visualization.DEFAULT_GRAPH_MODE))
    if st.session_state.get(graph_mode_widget_key) not in visualization.GRAPH_MODE_OPTIONS:
        st.session_state[graph_mode_widget_key] = graph_mode_default
    # The graph fragment appends the mode selector to this already populated
    # settings expander. Mode changes therefore do not rerun DSP or the page.
    graph_mode_host = preferences_host
    graph_mode = str(st.session_state[graph_mode_widget_key])
    graph_points_widget_key = "composite_studio_graph_max_points_widget"
    graph_points_default = int(settings.get(
        "graph_max_points", visualization.DEFAULT_GRAPH_MAX_POINTS,
    ))
    if st.session_state.get(graph_points_widget_key) not in visualization.GRAPH_MAX_POINTS_OPTIONS:
        st.session_state[graph_points_widget_key] = graph_points_default
    graph_max_points = int(shared_selectbox(
        ui_message('ui.77deabe238de2f'),
        visualization.GRAPH_MAX_POINTS_OPTIONS,
        key=graph_points_widget_key,
        format_func=lambda value: f"{value} points / series",
        help=(
            ui_message('ui.1d7b766a6ba310')
        ),
    ))
    st.caption(ui_message('ui.973d8879816a81'))

st.markdown(ui_message('ui.1d83bc28cb9047'))
manual_refresh = st.button(
    ui_message('ui.071b9758577759'),
    key=KEY_MANUAL_REFRESH,
    type="secondary",
    width="content",
    help=ui_message('ui.6a76cc2937c968'),
)
# Apply/undo must refresh the plotted impulses even when Auto is disabled.
alignment_refresh = bool(st.session_state.pop("_alignment_refresh_requested", False))
manual_refresh = manual_refresh or alignment_refresh
st.session_state[KEY_GRAPH_MODE] = str(graph_mode)
st.session_state[KEY_GRAPH_MAX_POINTS] = int(graph_max_points)
current_settings["auto_update"] = bool(auto_update)
current_settings["graph_mode"] = str(graph_mode)
current_settings["graph_max_points"] = int(graph_max_points)

# ------------------------ フィルター生成 ------------------------
conf = current_conf
settings = current_settings
save_settings(settings)
st.session_state["settings"] = settings
plot_theme = _current_theme_mode()

signature_channel_rows = [
    row for row in st.session_state.get("composite_studio_channels", [])
    if isinstance(row, dict)
]
signature_crossover_methods = tuple(
    normalize_crossover_method(
        st.session_state.get(
            f"composite_studio_{mode_key}_crossover_method_{index}"
        )
    )
    for index in range(len(conf.get("cross_freqs", [])))
)
studio_dsp_signature = studio_dsp_input_signature(
    signature_channel_rows,
    sample_rate_hz=int(fs),
    crossover_frequencies_hz=tuple(float(value) for value in conf.get("cross_freqs", ())),
    crossover_methods=signature_crossover_methods,
    speaker_timing_settings=current_settings,
)

conf_signature = json.dumps({
    'group_delay_display_revision': visualization.GROUP_DELAY_DISPLAY_REVISION,
    'mode': mode_key,
    'conf': conf,
    'output_normalize': bool(current_settings.get('output_normalize', False)),
    'fir_output_enabled': bool(conf.get('fir_output_enabled', False)),
    'auto_crop': bool(current_settings.get('auto_crop', False)),
    'auto_crop_profile': current_settings.get(
        'auto_crop_profile', DEFAULT_AUTO_CROP_PROFILE
    ),
    'align_output_taps': bool(current_settings.get('align_output_taps', False)),
    'studio_design_channels': [
        {
            'way': str(row.get('way', '')),
            'group': str(row.get('group', '')),
            'enabled': bool(row.get('enabled', True)),
            'gain_db': float(row.get('gain_db', 0.0)),
            'polarity': int(row.get('polarity', 1)),
            'delay_samples': float(row.get('delay_samples', 0.0)),
            'alignment_delay_samples': float(row.get('auto_alignment_delay_samples', 0.0)),
            'alignment_allpass': [section.to_dict() for section in row.get('auto_alignment_allpass', ())],
        }
        for row in st.session_state.get('composite_studio_channels', [])
        if isinstance(row, dict) and row.get('source') == 'generated_band'
    ],
    'studio_dsp_input_signature': studio_dsp_signature,
    'auto_crop_pass_db': float(
        current_settings.get('auto_crop_pass_db', DEFAULT_AUTO_CROP_PASS_DB)
    ),
    'auto_crop_cross_db': float(
        current_settings.get('auto_crop_cross_db', DEFAULT_AUTO_CROP_CROSS_DB)
    ),
}, ensure_ascii=False, sort_keys=True)
if mode_key == "Fullrange":
    estimated_taps = 1
elif mode_key == "3Way":
    estimated_taps = int(max(
        fs / max(conf['cross_freqs'][0], 1) * conf.get('cycles_low', conf['cycles']),
        fs / max(conf['cross_freqs'][1], 1) * conf.get('cycles_high', conf['cycles']),
        1,
    ))
elif mode_key == "4Way":
    estimated_taps = int(max(
        fs / max(conf['cross_freqs'][0], 1) * conf.get('cycles_sub_low', conf['cycles']),
        fs / max(conf['cross_freqs'][1], 1) * conf.get('cycles_low_mid', conf['cycles']),
        fs / max(conf['cross_freqs'][2], 1) * conf.get('cycles_mid_high', conf['cycles']),
        1,
    ))
else:
    estimated_taps = int(max(1, fs / max(min(conf['cross_freqs']), 1) * conf['cycles']))
if estimated_taps > 1800:
    st.warning(ui_message('ui.376d2f2901a29a'))

last_conf_signature = st.session_state.get('last_conf_signature')
signature_changed = last_conf_signature != conf_signature
theme_changed = st.session_state.get("last_plot_theme") != plot_theme
last_render_seconds = float(st.session_state.get('last_render_seconds', 0.0) or 0.0)
required_graph_cache_keys = (
    "studio_chart_bundle", "studio_chart_source",
    "final_firs", "studio_graph_settings", "studio_graph_responses",
    "result_settings", "result_mode_key", "result_fs",
)
has_cached_result = all(
    st.session_state.get(key) is not None for key in required_graph_cache_keys
)
cross_order_valid = (
    mode_key == "2Way"
    or all(
        int(left) < int(right)
        for left, right in zip(conf["cross_freqs"], conf["cross_freqs"][1:])
    )
)
fir_cross_required = bool(bands_requiring_split_fir(
    bands, signature_crossover_methods,
))
fir_assets_required = any(
    isinstance(row.get("phaseeq_fir_response"), dict) or row.get("additional_fir") is not None
    for row in signature_channel_rows
)
fir_configuration_valid = bool(conf.get("fir_output_enabled", False)) or not (
    fir_cross_required or fir_assets_required
)
cross_order_valid = cross_order_valid and fir_configuration_valid
if not fir_configuration_valid:
    st.error(
        ui_message('ui.5531f87fbc78dd')
    )
if not cross_order_valid:
    if not fir_configuration_valid:
        pass
    elif mode_key == "3Way":
        st.error(
            ui_message('ui.f5e9cd6d939754')
        )
    else:
        st.error(ui_message('ui.39270e49fc04da'))
auto_skip_due_to_render = (
    auto_update
    and cross_order_valid
    and signature_changed
    and not theme_changed
    and not manual_refresh
    and has_cached_result
    and last_render_seconds >= AUTO_SKIP_RENDER_SECONDS
)

should_generate = cross_order_valid and (
    not has_cached_result
    or manual_refresh
    or (auto_update and signature_changed and not auto_skip_due_to_render)
)

if auto_skip_due_to_render:
    skipped_signature = st.session_state.get('last_auto_skip_signature')
    if skipped_signature != conf_signature:
        append_log(f"前回更新が重いため自動更新をスキップ: {last_render_seconds:.1f}秒")
        st.session_state['last_auto_skip_signature'] = conf_signature

tap_alignment_host = None

if should_generate:
    render_start_time = time.perf_counter()
    result_generated_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    crossover_methods = tuple(
        normalize_crossover_method(st.session_state.get(f"composite_studio_{mode_key}_crossover_method_{index}"))
        for index in range(len(conf.get("cross_freqs", [])))
    )
    all_kaiser = all(method == "Kaiser FIR" for method in crossover_methods)
    # フィルター生成
    if mode_key == "Fullrange":
        split_firs_before = {"Fullrange": np.asarray([1.0], dtype=float)}
        split_firs = {"Fullrange": np.asarray([1.0], dtype=float)}
    elif mode_key == "Fullrange+SUB":
        boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(
            mode_key, conf,
        )
        base_before = generate_exclusive_kaiser_firs(
            fs, ("SUB", "Fullrange"), tuple(conf["cross_freqs"]), crossover_methods,
            (0.0,), boundary_cycles, boundary_betas,
        )
        base = generate_exclusive_kaiser_firs(
            fs, ("SUB", "Fullrange"), tuple(conf["cross_freqs"]), crossover_methods,
            boundary_overlaps, boundary_cycles, boundary_betas,
        )
        split_firs_before = {
            "SUB": base_before["SUB"],
            "Fullrange": base_before["Fullrange"],
        }
        split_firs = {
            "SUB": base["SUB"],
            "Fullrange": base["Fullrange"],
        }
    elif mode_key == "2Way":
        fc = conf['cross_freqs'][0]
        split_firs_before = generate_2way_filters(fs, fc, cycles, beta, overlap_hz=0)
        split_firs = generate_2way_filters(fs, fc, cycles, beta, overlap_oct=conf.get('kaiser_overlap_oct', 0.0)) if all_kaiser else generate_exclusive_kaiser_firs(
            fs, ("Low", "High"), (fc,), crossover_methods,
            (conf.get('kaiser_overlap_oct', 0.0),), (cycles,), (beta,),
        )
        if not all_kaiser:
            split_firs_before = generate_exclusive_kaiser_firs(fs, ("Low", "High"), (fc,), crossover_methods, (0.0,), (cycles,), (beta,))
    elif mode_key == "3Way":
        fc1, fc2 = conf['cross_freqs']
        split_firs_before = generate_3way_filters(
            fs, fc1, fc2, cycles, beta,
            overlap1_hz=0,
            overlap2_hz=0,
            cycles_low=conf.get('cycles_low', cycles),
            beta_low=conf.get('beta_low', beta),
            cycles_high=conf.get('cycles_high', cycles),
            beta_high=conf.get('beta_high', beta),
        )
        split_firs = generate_3way_filters(
            fs, fc1, fc2, cycles, beta,
            overlap1_oct=conf.get('kaiser_overlap_low_mid_oct', 0.0),
            overlap2_oct=conf.get('kaiser_overlap_mid_high_oct', 0.0),
            cycles_low=conf.get('cycles_low', cycles),
            beta_low=conf.get('beta_low', beta),
            cycles_high=conf.get('cycles_high', cycles),
            beta_high=conf.get('beta_high', beta),
        ) if all_kaiser else generate_exclusive_kaiser_firs(
            fs, ("Low", "Mid", "High"), (fc1, fc2), crossover_methods,
            (conf.get('kaiser_overlap_low_mid_oct', 0.0), conf.get('kaiser_overlap_mid_high_oct', 0.0)),
            (conf.get('cycles_low', cycles), conf.get('cycles_high', cycles)),
            (conf.get('beta_low', beta), conf.get('beta_high', beta)),
        )
        if not all_kaiser:
            split_firs_before = generate_exclusive_kaiser_firs(
                fs, ("Low", "Mid", "High"), (fc1, fc2), crossover_methods, (0.0, 0.0),
                (conf.get('cycles_low', cycles), conf.get('cycles_high', cycles)),
                (conf.get('beta_low', beta), conf.get('beta_high', beta)),
            )
    elif mode_key == "3Way+SUB":
        fc1, fc2, fc3 = conf['cross_freqs']
        common_4way_args = {
            "cycles_sub_low": conf.get("cycles_sub_low", cycles),
            "beta_sub_low": conf.get("beta_sub_low", beta),
            "cycles_low_mid": conf.get("cycles_low_mid", cycles),
            "beta_low_mid": conf.get("beta_low_mid", beta),
            "cycles_mid_high": conf.get("cycles_mid_high", cycles),
            "beta_mid_high": conf.get("beta_mid_high", beta),
        }
        split_firs_before = generate_4way_filters(
            fs, fc1, fc2, fc3,
            overlap1_hz=0,
            overlap2_hz=0,
            overlap3_hz=0,
            **common_4way_args,
        )
        split_firs = generate_4way_filters(
            fs, fc1, fc2, fc3,
            overlap1_oct=conf.get("kaiser_overlap_sub_low_oct", 0.0),
            overlap2_oct=conf.get("kaiser_overlap_low_mid_oct", 0.0),
            overlap3_oct=conf.get("kaiser_overlap_mid_high_oct", 0.0),
            **common_4way_args,
        ) if all_kaiser else generate_exclusive_kaiser_firs(
            fs, ("SUB", "Low", "Mid", "High"), (fc1, fc2, fc3), crossover_methods,
            (conf.get("kaiser_overlap_sub_low_oct", 0.0), conf.get("kaiser_overlap_low_mid_oct", 0.0), conf.get("kaiser_overlap_mid_high_oct", 0.0)),
            (common_4way_args["cycles_sub_low"], common_4way_args["cycles_low_mid"], common_4way_args["cycles_mid_high"]),
            (common_4way_args["beta_sub_low"], common_4way_args["beta_low_mid"], common_4way_args["beta_mid_high"]),
        )
        if not all_kaiser:
            split_firs_before = generate_exclusive_kaiser_firs(
                fs, ("SUB", "Low", "Mid", "High"), (fc1, fc2, fc3), crossover_methods,
                (0.0, 0.0, 0.0),
                (common_4way_args["cycles_sub_low"], common_4way_args["cycles_low_mid"], common_4way_args["cycles_mid_high"]),
                (common_4way_args["beta_sub_low"], common_4way_args["beta_low_mid"], common_4way_args["beta_mid_high"]),
            )
    else:
        boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(
            mode_key, conf,
        )
        split_firs_before = generate_residual_multiband_filters(
            fs, tuple(bands), tuple(conf["cross_freqs"]),
            boundary_cycles, boundary_betas, tuple(0.0 for _ in boundary_overlaps),
        ) if all_kaiser else generate_exclusive_kaiser_firs(
            fs, tuple(bands), tuple(conf["cross_freqs"]), crossover_methods,
            tuple(0.0 for _ in boundary_overlaps), boundary_cycles, boundary_betas,
        )
        split_firs = generate_residual_multiband_filters(
            fs, tuple(bands), tuple(conf["cross_freqs"]),
            boundary_cycles, boundary_betas, boundary_overlaps,
        ) if all_kaiser else generate_exclusive_kaiser_firs(
            fs, tuple(bands), tuple(conf["cross_freqs"]), crossover_methods,
            boundary_overlaps, boundary_cycles, boundary_betas,
        )
    # Sidebar、Assignment、グラフ、Exportで同じ実生成Split FIRを使用する。
    split_firs_before = {
        band: np.asarray(values, dtype=float).copy()
        for band, values in _sidebar_split_before.items()
    }
    split_firs = {
        band: np.asarray(values, dtype=float).copy()
        for band, values in sidebar_split_firs.items()
    }

    from composite_engine.multiway_studio.processing.eq_files import load_eq_firs
    eq_firs = load_eq_firs(
        bands, eq_files_conf, EQ_SAVE_DIR,
        enabled=bool(conf.get("fir_output_enabled", False)),
    )

    # 合成＆クロップ適用
    final_firs = {}
    auto_crop_rows = []
    auto_crop_cross_rows = []
    baffle_fir = (
        make_baffle_step_fir(
            fs,
            f0=170,
            f1=conf['baffle_f1_hz'],
            max_atten_db=baffle_atten,
        )
        if baffle_compensation_plan.mode == "fir"
        else None
    )
    st.session_state["composite_studio_baffle_fir"] = baffle_fir
    if bool(conf.get("fir_output_enabled", False)) and current_settings.get("auto_crop", False):
        final_firs, auto_crop_rows, auto_crop_cross_rows = _auto_crop_all_bands(
            split_firs,
            baffle_fir,
            eq_firs,
            mode_key,
            conf["cross_freqs"],
            fs,
            pass_limit_db=float(
                current_settings.get("auto_crop_pass_db", DEFAULT_AUTO_CROP_PASS_DB)
            ),
            cross_limit_db=float(
                current_settings.get("auto_crop_cross_db", DEFAULT_AUTO_CROP_CROSS_DB)
            ),
            align_output_taps=bool(
                current_settings.get("align_output_taps", False)
            ),
        )
    else:
        for band in bands:
            fir = split_firs[band]
            state = sidebar_fir_states[band]
            crop_len = (
                int(state.tap_count or 0)
                if state.tap_source in {"manual", "auto_target"}
                else 0
            )
            # 手動指定は従来通り、バッフル合成後とEQ合成後にクロップする。
            if baffle_fir is not None:
                fir = combine_and_crop_fir_conv(fir, baffle_fir, crop_len)
            fir = combine_and_crop_fir_conv(fir, eq_firs[band], crop_len)
            final_firs[band] = fir

    # クロップ後、グラフ表示と出力の前に全帯域へ同一ゲインを適用する。
    normalization_info = {
        "enabled": False,
        "gain_linear": 1.0,
        "gain_db": 0.0,
        "reference_response": None,
        "response_peaks_before": {},
    }
    if bool(conf.get("fir_output_enabled", False)) and bool(current_settings.get("output_normalize", False)):
        final_firs, gain, reference_response, response_peaks_before = normalize_fir_group_response(
            final_firs,
            OUTPUT_MAX_BAND_GAIN_DB,
        )
        normalization_info = {
            "enabled": True,
            "gain_linear": gain,
            "gain_db": float(20.0 * np.log10(gain)) if gain > 0.0 else 0.0,
            "reference_response": reference_response,
            "response_peaks_before": response_peaks_before,
        }
        band_peaks_after, sum_peak_after = measure_fir_group_response_peaks(
            final_firs
        )
        normalization_info["band_response_peaks_after"] = band_peaks_after
        normalization_info["sum_response_peak_after"] = sum_peak_after
        append_log(
            f"各出力帯域を共通ゲイン {normalization_info['gain_db']:+.3f} dB で調整"
            f"（最大応答: {reference_response or 'なし'}、上限: {OUTPUT_MAX_BAND_GAIN_DB:.1f} dB）"
        )

    # 中央揃え（可視化用）
    fir_list = [final_firs[band] for band in bands]
    fir_centered = center_fir_lengths(fir_list)
    plot_firs = dict(zip(bands, fir_centered))
    st.session_state['final_firs'] = final_firs
    st.session_state["composite_studio_band_fir_states"] = {
        band: state.to_dict() for band, state in sidebar_fir_states.items()
    }
    # DSP package export keeps the declared stages separate.  The pure Kaiser
    # crossover must not be replaced by the Studio preview FIR, which may also
    # contain baffle/EQ processing and cropping.
    st.session_state['kaiser_split_firs'] = {
        band: np.asarray(coefficients, dtype=float)
        for band, coefficients in split_firs.items()
    }
    studio_additional_firs = {}
    for band in bands:
        additional = np.asarray([1.0], dtype=float)
        if baffle_fir is not None:
            additional = np.convolve(additional, np.asarray(baffle_fir, dtype=float))
        if eq_firs.get(band) is not None:
            additional = np.convolve(additional, np.asarray(eq_firs[band], dtype=float))
        studio_additional_firs[band] = additional
    st.session_state['studio_additional_firs'] = studio_additional_firs
    st.session_state['plot_firs'] = plot_firs
    st.session_state['normalization_info'] = normalization_info

    # Studio tables and graphs share the same realized response path.
    graph_rows = [
        row for row in st.session_state.get("composite_studio_channels", [])
        if isinstance(row, dict)
    ]
    stereo_graph_available = (
        str(st.session_state.get("composite_studio_output_layout", "Mono")) == "Stereo"
    )
    displayed_graph_group = normalize_display_value(
        st.session_state.get("composite_studio_result_group", ""),
        stereo=stereo_graph_available,
    )
    st.session_state["composite_studio_result_group"] = displayed_graph_group
    graph_groups = (
        ("Left", "Right") if displayed_graph_group == "L+R"
        else ("Left",) if displayed_graph_group == "L"
        else ("Right",) if displayed_graph_group == "R"
        else ("Main",)
    )
    graph_group = graph_groups[0]
    graph_iir_configs = _shared_iir_configs(mode_key, tuple(conf["cross_freqs"]))
    graph_ir_length = ir_length_for_frequency_resolution(fs, 2.0)
    graph_response_points = response_points_for_frequency_resolution(fs, 2.0)
    graph_responses = {}
    graph_settings = {}
    phaseeq_graph_responses = {}
    speaker_graph_responses = {}
    graph_sum_groups = {}
    graph_way_sums = {}
    primary_graph_firs = {}
    primary_graph_settings = {}
    primary_graph_responses = {}
    primary_speaker_phaseeq_responses = {}
    graph_frequency_hz = np.array([], dtype=float)
    graph_timing_projection = resolve_speaker_timing_projection(
        graph_rows, st.session_state.get("settings", {}), sample_rate_hz=int(fs),
    )
    for physical_group in graph_groups:
        side_firs, side_settings = select_crossover_design_inputs(
            final_firs, graph_rows, physical_group,
        )
        side_fir_enabled = {
            way: sidebar_fir_states[way].enabled for way in side_firs
        }
        side_speaker_phaseeq = build_speaker_phaseeq_responses(
            graph_rows, tuple(side_firs), physical_group, fs,
            points=graph_response_points, fir_enabled_by_way=side_fir_enabled,
            speaker_timing_projection=graph_timing_projection,
        )
        side_phaseeq = build_speaker_phaseeq_responses(
            graph_rows, tuple(side_firs), physical_group, fs,
            points=graph_response_points, phaseeq_only=True,
            fir_enabled_by_way=side_fir_enabled,
            speaker_timing_projection=graph_timing_projection,
        )
        side_speaker = build_speaker_phaseeq_responses(
            graph_rows, tuple(side_firs), physical_group, fs,
            points=graph_response_points, speaker_only=True,
            fir_enabled_by_way=side_fir_enabled,
            speaker_timing_projection=graph_timing_projection,
        )
        for way, response in side_speaker_phaseeq.items():
            if way in side_settings:
                side_settings[way]["speaker_phaseeq_response"] = response
        side_frequency, side_responses, side_sum = build_crossover_design_responses(
            side_firs, graph_iir_configs, fs, points=graph_response_points,
            way_settings=side_settings,
            fir_enabled_by_way=side_fir_enabled,
        )
        prefix = (
            "L" if physical_group == "Left" else "R" if physical_group == "Right" else ""
        )
        def rename(way):
            if way == "SUB" and side_settings.get(way, {}).get("group") == "Sub":
                return "SUB"
            return f"{prefix} {way}" if len(graph_groups) > 1 else way
        renamed_keys = tuple(rename(way) for way in side_responses)
        for target, source in (
            (graph_responses, side_responses),
            (graph_settings, side_settings),
            (phaseeq_graph_responses, side_phaseeq),
            (speaker_graph_responses, side_speaker),
        ):
            for way, value in source.items():
                key = rename(way)
                if key not in target:
                    target[key] = value
        sum_label = f"{prefix} System Sum" if prefix else "Main System Sum"
        graph_sum_groups[sum_label] = renamed_keys
        graph_way_sums[sum_label] = side_sum
        graph_frequency_hz = side_frequency
        if physical_group == graph_group:
            primary_graph_firs = side_firs
            primary_graph_settings = side_settings
            primary_graph_responses = side_responses
            primary_speaker_phaseeq_responses = side_speaker_phaseeq
    graph_firs = primary_graph_firs
    speaker_phaseeq_responses = primary_speaker_phaseeq_responses
    realized_plot_firs = centered_impulses_from_responses(graph_responses)
    baffle_graph_response = (
        common_baffle_response(
            baffle_compensation_plan, graph_frequency_hz, fs, fir=baffle_fir,
        )
        if graph_frequency_hz.size and baffle_compensation_plan.mode != "off"
        else None
    )
    st.session_state["studio_graph_settings"] = graph_settings
    st.session_state["studio_graph_primary_settings"] = primary_graph_settings
    st.session_state["studio_graph_responses"] = graph_responses

    st.session_state['auto_crop_rows'] = auto_crop_rows
    st.session_state['auto_crop_cross_rows'] = auto_crop_cross_rows

    # Numeric projection is completed and validated before any chart is drawn.
    # Session State receives one bundle revision atomically; renderer artifacts
    # are transient and never become the source of truth.
    chart_source = {
        "sample_rate_hz": int(fs),
        "frequency_hz": graph_frequency_hz,
        "responses": graph_responses,
        "impulses": realized_plot_firs,
        "sum_groups": graph_sum_groups,
        "way_sums": graph_way_sums,
        "phaseeq_responses": phaseeq_graph_responses,
        "speaker_responses": speaker_graph_responses,
        "baffle_response": baffle_graph_response,
        "baffle_label": (
            f"Baffle compensation · {baffle_compensation_plan.display_mode}"
            if baffle_graph_response is not None else None
        ),
    }
    chart_revision = _studio_chart_source_revision(
        conf_signature, displayed_graph_group, graph_max_points,
        db_min, db_max, phase_gain_mask_db,
    )
    chart_bundle = _project_studio_charts(
        chart_source, source_revision=chart_revision,
        graph_max_points=graph_max_points, db_min=db_min, db_max=db_max,
        phase_gain_mask_db=phase_gain_mask_db,
    )
    st.session_state["studio_chart_source"] = chart_source
    st.session_state["studio_chart_bundle"] = chart_bundle

    tap_alignment_host = st.container()
    if st.session_state.get('auto_crop_rows'):
        with st.expander(display_text("自動クロップの明細"), expanded=False):
            _render_auto_crop_table(
                auto_crop_rows,
                auto_crop_cross_rows,
                float(current_settings.get("auto_crop_pass_db", DEFAULT_AUTO_CROP_PASS_DB)),
                float(current_settings.get("auto_crop_cross_db", DEFAULT_AUTO_CROP_CROSS_DB)),
            )
    _render_output_gain_table(
        normalization_info, way_settings=graph_settings, responses=graph_responses,
    )

    # Reserve every graph position before fragment rendering. The metrics host
    # remains between the fifth chart and crossover chart on every rerun.
    chart_hosts, dip_metrics_host = _create_studio_graph_layout()
    with graph_mode_host:
        _render_studio_graph_fragment(
            chart_bundle, plot_theme=plot_theme, chart_hosts=chart_hosts,
        )

    # クロス近傍は、IIRを含む設計基準と全Channel変換適用後を比較する。
    before_metrics = {}
    after_metrics = {}
    baseline_firs, _baseline_settings = select_crossover_design_inputs(
        split_firs_before, graph_rows, graph_group,
    )
    baseline_frequency_hz, baseline_responses, _baseline_sum = (
        build_crossover_design_responses(
            baseline_firs, graph_iir_configs, fs, points=graph_response_points,
            way_settings={
                way: {"speaker_phaseeq_response": response}
                for way, response in speaker_phaseeq_responses.items()
                if way in baseline_firs
            },
            fir_enabled_by_way={
                way: sidebar_fir_states[way].enabled for way in baseline_firs
            },
        )
    )
    for label, pair_bands, crossover in _auto_crop_cross_pairs(
        mode_key, conf["cross_freqs"]
    ):
        before_pair = [baseline_responses[band] for band in pair_bands if band in baseline_responses]
        after_pair = {
            band: primary_graph_responses[band]
            for band in pair_bands if band in primary_graph_responses
        }
        if len(before_pair) != len(pair_bands) or len(after_pair) != len(pair_bands):
            continue
        before_freqs = baseline_frequency_hz
        after_freqs = graph_frequency_hz
        before_mag = 20.0 * np.log10(np.maximum(np.abs(np.sum(before_pair, axis=0)), 1e-12))
        after_sum = sum_crossover_comparison_response(
            after_pair, output_gain_linear=float(normalization_info.get("gain_linear", 1.0)),
            fir_enabled_by_way={way: sidebar_fir_states[way].enabled for way in after_pair},
        )
        after_mag = 20.0 * np.log10(np.maximum(np.abs(after_sum), 1e-12))
        before_metrics[label] = _crossover_dip_metrics(
            before_freqs, before_mag, crossover
        )
        after_metrics[label] = _crossover_dip_metrics(
            after_freqs, after_mag, crossover
        )

    with dip_metrics_host:
        st.subheader(ui_message('ui.39fdd8ed24e9df'))
        rows = _render_dip_metrics_table(before_metrics, after_metrics)
        for row in rows:
            row['評価'] = _dip_quality_label(row['補正後 変動量 [dB]'], row['改善量 [dB]'])
        st.caption(ui_message('ui.11b674eba5e837'))
        st.caption(display_text("補正前後ともFIR出力合わせの共通ゲインを除いた基準で比較します。手動Gain・極性・Delay・位相整合は比較に含みます。出力合わせ後の絶対レベルは出力ゲイン調整結果で確認してください。"))
        st.caption(
            ui_message('ui.a7efcdb00ce12e')
        )
        _render_dip_metrics_html(rows)
    st.session_state['dip_rows'] = rows
    st.session_state["studio_graph_display"] = displayed_graph_group

    _render_filter_downloads(
        final_firs,
        mode_key,
        current_settings,
        fs,
        result_generated_at,
        key_suffix=f"new_{mode_key}_{int(fs)}",
        auto_crop_rows=auto_crop_rows,
        auto_crop_cross_rows=auto_crop_cross_rows,
        fir_states=st.session_state.get("composite_studio_band_fir_states", {}),
    )
    st.session_state['last_conf_signature'] = conf_signature
    st.session_state['last_plot_theme'] = plot_theme
    st.session_state['result_settings'] = json.loads(json.dumps(current_settings))
    st.session_state['result_mode_key'] = mode_key
    st.session_state['result_fs'] = int(fs)
    st.session_state['result_generated_at'] = result_generated_at
    st.session_state['results_stale'] = False
    elapsed = time.perf_counter() - render_start_time
    st.session_state['last_render_seconds'] = elapsed
    if elapsed >= VERY_SLOW_RENDER_SECONDS:
        st.warning(ui_message('ui.779b592a82d008', p0=f'{elapsed:.1f}'))
    elif elapsed >= SLOW_RENDER_NOTICE_SECONDS:
        st.caption(ui_message('ui.b814997ab8707d', p0=f'{elapsed:.1f}'))
    append_log(f"FIRフィルター生成・グラフ描画・ダウンロード枠表示完了（{elapsed:.1f}秒）")


if (not should_generate) and st.session_state.get('studio_chart_bundle') is not None:
    cached_settings = st.session_state.get('result_settings', current_settings)
    cached_mode_key = st.session_state.get('result_mode_key', mode_key)
    cached_fs = int(st.session_state.get('result_fs', fs))
    cached_generated_at = st.session_state.get(
        'result_generated_at',
        datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    download_settings = json.loads(json.dumps(cached_settings))
    download_settings["output_format"] = current_settings.get("output_format", download_settings.get("output_format", "bin_float32"))
    download_settings["align_output_taps"] = current_settings.get(
        "align_output_taps", download_settings.get("align_output_taps", False)
    )
    if auto_skip_due_to_render:
        st.warning(
            ui_message('ui.3450d8e3934d18', p0=f'{last_render_seconds:.1f}')
        )
    elif signature_changed:
        st.info(ui_message('ui.622e8562ef702c'))
    cached_final_firs = st.session_state.get("final_firs", {})
    cached_graph_rows = [
        row for row in st.session_state.get("composite_studio_channels", [])
        if isinstance(row, dict)
    ]
    current_graph_display = normalize_display_value(
        st.session_state.get("composite_studio_result_group", ""),
        stereo=(
            str(st.session_state.get("composite_studio_output_layout", "Mono"))
            == "Stereo"
        ),
    )
    expected_chart_revision = _studio_chart_source_revision(
        st.session_state.get("last_conf_signature", ""),
        current_graph_display, graph_max_points,
        db_min, db_max, phase_gain_mask_db,
    )
    if (
        not signature_changed
        and (
            st.session_state.get("studio_graph_display") != current_graph_display
            or st.session_state["studio_chart_bundle"].source_revision
            != expected_chart_revision
        )
        and isinstance(cached_final_firs, dict)
        and cached_final_firs
    ):
        if st.session_state.get("studio_graph_display") != current_graph_display:
            rebuilt = _studio_display_graph_bundle(
                cached_final_firs,
                cached_graph_rows,
                current_graph_display,
                st.session_state.get("composite_studio_band_fir_states", {}),
                mode_key=cached_mode_key,
                crossover_frequencies_hz=tuple(
                    cached_settings[cached_mode_key]["cross_freqs"]
                ),
                sample_rate_hz=cached_fs,
                response_points=response_points_for_frequency_resolution(cached_fs, 2.0),
                graph_max_points=graph_max_points,
                dsp_revision=st.session_state.get("last_conf_signature", ""),
                db_min=db_min,
                db_max=db_max,
                phase_gain_mask_db=phase_gain_mask_db,
            )
            st.session_state["studio_graph_settings"] = rebuilt["settings"]
            st.session_state["studio_graph_primary_settings"] = rebuilt["primary_settings"]
            st.session_state["studio_graph_responses"] = rebuilt["responses"]
            st.session_state["studio_graph_display"] = rebuilt["display"]
            st.session_state["studio_chart_source"] = rebuilt["chart_source"]
            st.session_state["studio_chart_bundle"] = rebuilt["chart_bundle"]
        else:
            # Axis/mask/point changes reuse the full renderer-neutral source;
            # FIR and realized response calculation are not repeated.
            projected = _project_studio_charts(
                st.session_state["studio_chart_source"],
                source_revision=expected_chart_revision,
                graph_max_points=graph_max_points,
                db_min=db_min, db_max=db_max,
                phase_gain_mask_db=phase_gain_mask_db,
            )
            st.session_state["studio_chart_bundle"] = projected
    chart_bundle = st.session_state["studio_chart_bundle"]
    tap_alignment_host = st.container()
    if st.session_state.get('auto_crop_rows'):
        with st.expander(display_text("自動クロップの明細"), expanded=False):
            _render_auto_crop_table(
                st.session_state.get('auto_crop_rows', []),
                st.session_state.get('auto_crop_cross_rows', []),
                float(cached_settings.get("auto_crop_pass_db", DEFAULT_AUTO_CROP_PASS_DB)),
                float(cached_settings.get("auto_crop_cross_db", DEFAULT_AUTO_CROP_CROSS_DB)),
            )
    _render_output_gain_table(
        st.session_state.get("normalization_info", {"enabled": False}),
        way_settings=st.session_state.get("studio_graph_settings", {}),
        responses=st.session_state.get("studio_graph_responses", {}),
    )
    chart_hosts, dip_metrics_host = _create_studio_graph_layout()
    with graph_mode_host:
        _render_studio_graph_fragment(
            chart_bundle, plot_theme=plot_theme, chart_hosts=chart_hosts,
        )
    if st.session_state.get('dip_rows'):
        with dip_metrics_host:
            st.subheader(ui_message('ui.39fdd8ed24e9df'))
            st.caption(ui_message('ui.11b674eba5e837'))
            st.caption(display_text("補正前後ともFIR出力合わせの共通ゲインを除いた基準で比較します。手動Gain・極性・Delay・位相整合は比較に含みます。出力合わせ後の絶対レベルは出力ゲイン調整結果で確認してください。"))
            st.caption(ui_message('ui.475d24e3dcb7b1'))
            _render_dip_metrics_html(st.session_state['dip_rows'])
    if st.session_state.get('final_firs'):
        _render_filter_downloads(
            st.session_state['final_firs'],
            cached_mode_key,
            download_settings,
            cached_fs,
            cached_generated_at,
            key_suffix=f"cached_{cached_mode_key}_{cached_fs}",
            auto_crop_rows=st.session_state.get('auto_crop_rows', []),
            auto_crop_cross_rows=st.session_state.get('auto_crop_cross_rows', []),
            fir_states=st.session_state.get("composite_studio_band_fir_states", {}),
        )


if (not should_generate) and st.session_state.get('last_conf_signature') == conf_signature:
    st.info(ui_message('ui.913b040dd32b5a') if auto_update else ui_message('ui.1b03cc4c60b28d'))

st.session_state["_studio_graphs_current"] = bool(
    st.session_state.get("last_conf_signature") == conf_signature
)
st.session_state["_studio_dsp_input_signature"] = studio_dsp_signature
if fir_configuration_valid:
    render_composite_results(
        alignment_host=phase_alignment_host, output_timing_host=tap_alignment_host,
        band_tap_count_hosts=band_tap_count_hosts,
    )

# ---- 実行ログ欄（メイン画面下） ----
st.markdown("---")
with st.expander(ui_message('ui.a98bc1b2d6ff62'), expanded=True):
    for msg in st.session_state.get("log_msgs", []):
        st.markdown(ui_message('ui.9a4a817443e20e', p0=(msg.partition(']` ')[0] + ']` ' + display_notice(msg.partition(']` ')[2]) if ']` ' in msg else display_notice(msg))))
