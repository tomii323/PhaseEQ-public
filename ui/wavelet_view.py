from __future__ import annotations
from utils.ui_localization import ui_message, display_text

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "phaseeq_matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from phase_fir_designer.analysis.alignment import AlignmentResult, align_impulse_to_reference
from phase_fir_designer.cache_coordinator import CacheCoordinator, CacheDomain, CacheEvent, content_digest
from wavelet_analysis import (
    WAVELET_BANDWIDTH_OPTIONS,
    WAVELET_MODE_OPTIONS,
    WAVELET_PRESETS,
    WaveletMap,
    WaveletSettings,
    bandwidth_label_to_oct,
    bandwidth_oct_to_label,
    complex_morlet_scalogram,
    settings_from_preset,
    with_wavelet_source_metadata,
    wavelet_difference_map,
)
from utils.wavelet_plot_style import (
    WAVELET_ABSOLUTE_COLORS,
    WAVELET_CENTROID_TRACE_COLOR,
    WAVELET_PEAK_TRACE_COLOR,
    WAVELET_WINDOW_TRACE_COLOR,
    plotly_absolute_colorscale,
    wavelet_color_domain,
)
from utils.ui_localization import localized_button, localized_segmented_control

WAVELET_GRAPH_POINTS_FREQUENCY_BINS = {
    1024: 512,
    2048: 768,
    4096: 1024,
}
WAVELET_GRAPH_POINTS_TIME_BINS = {
    1024: 1024,
    2048: 1448,
    4096: 2048,
}
WAVELET_CHART_HEIGHT_PX = 720
WAVELET_MODE_LABELS = {
    "Source": "Source",
    "Target": "Target",
    "Source - Target": "Delta",
}
WAVELET_STAGE_SOURCE_LABELS = {
    "Speaker/Input": "Speaker/Input",
    "Target Response": "Target Response",
    "IIR EQ": "IIR Result",
    "Gain EQ": "Speaker + Realized",
    "Phase EQ": "Speaker + Realized",
    "Auto Gain EQ": "Speaker + Realized",
    "Auto Phase EQ": "Speaker + Realized",
    "Export FIR": "Speaker + Realized",
    "Correction Filter": "Correction Filter",
    "Correction Filter Realized": "Correction Filter Realized",
    "Speaker + Realized": "Speaker + Realized",
}
WAVELET_RENDER_ALGORITHM_VERSION = "2026-07-24-wavelet-local-window-lazy-maps-v2"
LEGACY_WAVELET_PRESET_MAP = {
    "High frequency": "Hi",
    "Mid / crossover": "Mid",
    "Low frequency": "Low",
}


def wavelet_source_label_for_stage(stage: str) -> str:
    return WAVELET_STAGE_SOURCE_LABELS.get(str(stage), "Speaker + Realized")


CARD_STATE_PREFIX = "__card_open__"
CARD_TOGGLE_PREFIX = "__card_toggle__"


@dataclass(frozen=True)
class WaveletSourceBundle:
    stage: str
    source_label: str
    source_ir: np.ndarray | None
    target_ir: np.ndarray | None
    system_ir: np.ndarray | None
    sample_rate: int
    fingerprint: str
    status_note: str = ""
    source_type: str = "reconstructed_from_gain_phase"
    reconstruction_confidence: float = 0.90
    phase_available: bool = True
    smoothing_detected: bool = False
    estimated_smoothing_oct: float | None = None
    smoothing_likelihood: float = 0.0
    source_warning: str | None = None


@dataclass(frozen=True)
class WaveletRenderResult:
    map: WaveletMap
    source_alignment: AlignmentResult | None
    system_alignment: AlignmentResult | None
    stale: bool


def render_wavelet_view(
    bundle: WaveletSourceBundle,
    *,
    auto_update: bool,
    manual_refresh: bool,
    graph_mode: str,
    graph_max_points: int,
) -> None:
    _normalize_wavelet_toolbar_state()
    toolbar = st.columns([0.38, 0.14, 0.18, 0.15, 0.15])
    with toolbar[0]:
        mode = _wavelet_segmented_widget(
            ui_message('ui.a8da7ac27d774c'),
            WAVELET_MODE_OPTIONS,
            key="wavelet_mode",
            default="Source",
            format_func=lambda value: WAVELET_MODE_LABELS.get(str(value), str(value)),
        )
    with toolbar[1]:
        window_preview = _wavelet_checkbox_widget(ui_message('ui.4ca970fba9029f'), key="wavelet_window_preview", default=False)
    with toolbar[2]:
        preset = _wavelet_selectbox_widget(ui_message('ui.7252e7ce008587'), list(WAVELET_PRESETS.keys()), key="wavelet_preset", default="Overview")
    base_settings = settings_from_preset(str(preset))
    with toolbar[3]:
        bandwidth_label = _wavelet_selectbox_widget(
            ui_message('ui.bec749e6fb9e59'),
            WAVELET_BANDWIDTH_OPTIONS,
            key="wavelet_bandwidth",
            default=bandwidth_oct_to_label(base_settings.bandwidth_oct),
        )
    with toolbar[4]:
        dynamic_range = _wavelet_selectbox_widget(
            ui_message('ui.5de74a81b19230'),
            [20, 40, 60],
            key="wavelet_dynamic_range",
            default=40,
            format_func=lambda value: f"{value} dB",
        )
    auto_align = True

    frequency_bins, time_bins = _wavelet_bins_for_graph_points(int(graph_max_points))
    settings = WaveletSettings(
        f_min=float(base_settings.f_min),
        f_max=float(base_settings.f_max),
        t_min_ms=float(base_settings.t_min_ms),
        t_max_ms=float(base_settings.t_max_ms),
        bandwidth_oct=bandwidth_label_to_oct(str(bandwidth_label)),
        dynamic_range_db=float(dynamic_range),
        frequency_bins=frequency_bins,
        time_bins=time_bins,
    )
    analysis_settings = _wavelet_analysis_settings(settings)
    settings_fingerprint = _settings_fingerprint(mode, analysis_settings, auto_align)
    fingerprint = CacheCoordinator(st.session_state).key(
        CacheDomain.WAVELET,
        algorithm_version=WAVELET_RENDER_ALGORITHM_VERSION,
        source_digest=content_digest(bundle.fingerprint),
        settings_digest=content_digest(settings_fingerprint),
        stage=bundle.stage,
    ).digest
    cached = st.session_state.get("_wavelet_last_result")
    cache_algorithm_changed = (
        isinstance(cached, dict)
        and cached.get("algorithm_version") != WAVELET_RENDER_ALGORITHM_VERSION
    )
    if cache_algorithm_changed:
        CacheCoordinator(st.session_state).notify(CacheEvent.ALGORITHM_CHANGED)
        cached = None
    stale = not isinstance(cached, dict) or cached.get("fingerprint") != fingerprint

    resolution_note = f"Wavelet resolution follows Graph points: {frequency_bins} frequency bins x {time_bins} time bins"

    should_compute = not isinstance(cached, dict) or manual_refresh or (auto_update and stale)
    if should_compute:
        try:
            render_result = _compute_wavelet_result(
                bundle,
                mode=str(mode),
                settings=analysis_settings,
                auto_align=bool(auto_align),
            )
            cached = {
                "fingerprint": fingerprint,
                "algorithm_version": WAVELET_RENDER_ALGORITHM_VERSION,
                "result": render_result,
            }
            st.session_state["_wavelet_last_result"] = cached
            stale = False
        except Exception as exc:
            st.error(ui_message('ui.585810bbde74ae', p0=f'{exc}'))
            return

    render_result = cached.get("result") if isinstance(cached, dict) else None
    if not isinstance(render_result, WaveletRenderResult):
        st.info(ui_message('ui.108d0267a8702c'))
        return
    if stale and not auto_update:
        st.warning(ui_message('ui.c5173a3180b706'))
    status_text = "Stale" if stale else "Ready"
    status_note = (
        f"Source: {bundle.source_label} / Target: Target Response / "
        f"Stage: {bundle.stage} / {status_text} / Graph mode: {graph_mode} / "
        f"Window preview: {'ON' if window_preview else 'OFF'}"
    )
    _draw_wavelet_map(
        render_result.map,
        settings,
        str(mode),
        str(bandwidth_label),
        graph_mode=str(graph_mode),
        window_preview=bool(window_preview),
    )
    _render_wavelet_details(
        render_result,
        settings,
        bool(auto_align),
        status_note=status_note,
        resolution_note=resolution_note,
        bundle_note=bundle.status_note,
    )


def _normalize_wavelet_toolbar_state() -> None:
    if st.session_state.get("wavelet_mode") not in WAVELET_MODE_OPTIONS:
        st.session_state["wavelet_mode"] = "Source"
    if str(st.session_state.get("wavelet_preset")) in LEGACY_WAVELET_PRESET_MAP:
        st.session_state["wavelet_preset"] = LEGACY_WAVELET_PRESET_MAP[str(st.session_state.get("wavelet_preset"))]
    if st.session_state.get("wavelet_preset") not in WAVELET_PRESETS:
        st.session_state["wavelet_preset"] = "Overview"
    if st.session_state.get("wavelet_bandwidth") not in WAVELET_BANDWIDTH_OPTIONS:
        st.session_state["wavelet_bandwidth"] = "1/3 oct"
    dynamic_range = st.session_state.get("wavelet_dynamic_range", 40)
    if dynamic_range not in {20, 40, 60}:
        try:
            dynamic_range_value = float(dynamic_range)
        except (TypeError, ValueError):
            dynamic_range_value = 40.0
        st.session_state["wavelet_dynamic_range"] = min((20, 40, 60), key=lambda value: abs(value - dynamic_range_value))
    st.session_state["wavelet_window_preview"] = bool(st.session_state.get("wavelet_window_preview", False))


def _wavelet_stored_option(key: str, options: list[Any] | tuple[Any, ...], default: Any) -> Any:
    option_list = list(options)
    value = st.session_state.get(key, default)
    if key == "wavelet_preset":
        value = LEGACY_WAVELET_PRESET_MAP.get(str(value), value)
    if value not in option_list:
        value = default if default in option_list else (option_list[0] if option_list else default)
    st.session_state[key] = value
    return value


def _commit_wavelet_widget_value(settings_key: str, widget_key: str) -> None:
    """Commit a temporary Wavelet widget value before conditional removal."""

    if widget_key in st.session_state:
        st.session_state[settings_key] = st.session_state[widget_key]


def _wavelet_selectbox_widget(
    label: str,
    options: list[Any] | tuple[Any, ...],
    *,
    key: str,
    default: Any,
    format_func: Any | None = None,
) -> Any:
    current = _wavelet_stored_option(key, options, default)
    widget_key = f"{key}_widget"
    # Detail view conditionally removes these widgets. Always reconstruct the
    # temporary widget mirror from the canonical value when it is shown again.
    st.session_state[widget_key] = current
    value = st.selectbox(
        label,
        options,
        key=widget_key,
        format_func=format_func if format_func is not None else str,
        on_change=_commit_wavelet_widget_value,
        args=(key, widget_key),
    )
    st.session_state[key] = value
    return value


def _wavelet_segmented_widget(
    label: str,
    options: list[Any] | tuple[Any, ...],
    *,
    key: str,
    default: Any,
    format_func: Any | None = None,
) -> Any:
    current = _wavelet_stored_option(key, options, default)
    widget_key = f"{key}_widget"
    st.session_state[widget_key] = current
    value = localized_segmented_control(
        label,
        options,
        key=widget_key,
        format_func=format_func if format_func is not None else str,
        width="stretch",
        on_change=_commit_wavelet_widget_value,
        args=(key, widget_key),
    )
    if value is None:
        value = current
    st.session_state[key] = value
    return value


def _wavelet_checkbox_widget(label: str, *, key: str, default: bool = False) -> bool:
    current = bool(st.session_state.get(key, default))
    st.session_state[key] = current
    widget_key = f"{key}_widget"
    st.session_state[widget_key] = current
    value = st.checkbox(
        label,
        key=widget_key,
        on_change=_commit_wavelet_widget_value,
        args=(key, widget_key),
    )
    st.session_state[key] = bool(value)
    return bool(value)


def _compute_wavelet_result(
    bundle: WaveletSourceBundle,
    *,
    mode: str,
    settings: WaveletSettings,
    auto_align: bool,
) -> WaveletRenderResult:
    required_maps = _wavelet_maps_for_mode(mode)
    target = (_require_ir(bundle.target_ir, "Target")
              if "Target" in required_maps else bundle.target_ir)
    source = _require_ir(bundle.source_ir, "Source") if "Source" in required_maps else None
    system = (
        _require_ir(bundle.system_ir if bundle.system_ir is not None else bundle.source_ir, "System")
        if "Source" in required_maps
        else None
    )

    source_alignment = None
    system_alignment = None
    if auto_align and source is not None and target is not None:
        source_alignment = align_impulse_to_reference(source, target, bundle.sample_rate)
        source = source_alignment.aligned
        system_alignment = align_impulse_to_reference(system, target, bundle.sample_rate) if system is not None else None
        if system_alignment is not None:
            system = system_alignment.aligned

    source_cache_key = CacheCoordinator(st.session_state).key(
        CacheDomain.WAVELET,
        algorithm_version=WAVELET_RENDER_ALGORITHM_VERSION,
        source_digest=content_digest(bundle.fingerprint, "Source"),
        settings_digest=content_digest(settings),
        stage=bundle.stage,
    ).digest
    target_cache_key = CacheCoordinator(st.session_state).key(
        CacheDomain.WAVELET,
        algorithm_version=WAVELET_RENDER_ALGORITHM_VERSION,
        source_digest=content_digest(bundle.fingerprint, "Target"),
        settings_digest=content_digest(settings),
        stage=bundle.stage,
    ).digest
    source_map = None
    if "Source" in required_maps and source is not None:
        source_map = _cached_scalogram(source, bundle.sample_rate, settings, "Source", source_cache_key)
        source_map = with_wavelet_source_metadata(
            source_map,
            source_type=bundle.source_type,
            reconstruction_confidence=bundle.reconstruction_confidence,
            phase_available=bundle.phase_available,
            smoothing_detected=bundle.smoothing_detected,
            estimated_smoothing_oct=bundle.estimated_smoothing_oct,
            smoothing_likelihood=bundle.smoothing_likelihood,
            source_warning=bundle.source_warning,
        )
    target_map = (
        _cached_scalogram(target, bundle.sample_rate, settings, "Target", target_cache_key)
        if "Target" in required_maps
        else None
    )

    if mode == "Target":
        if target_map is None:
            raise ValueError("Target Wavelet map is unavailable.")
        selected = target_map
    elif mode == "Source - Target":
        if source_map is None or target_map is None:
            raise ValueError("Source and Target Wavelet maps are required for Delta mode.")
        selected = wavelet_difference_map(source_map, target_map)
    else:
        if source_map is None:
            raise ValueError("Source Wavelet map is unavailable.")
        selected = source_map

    return WaveletRenderResult(
        map=selected,
        source_alignment=source_alignment,
        system_alignment=system_alignment,
        stale=False,
    )


@st.cache_data(max_entries=8, ttl=1800, show_spinner=False)
def _cached_scalogram(
    impulse: np.ndarray,
    sample_rate: int,
    settings: WaveletSettings,
    label: str,
    cache_key: str,
) -> WaveletMap:
    _ = cache_key
    return complex_morlet_scalogram(np.asarray(impulse, dtype=float), int(sample_rate), settings, label=label)


def _wavelet_maps_for_mode(mode: str) -> frozenset[str]:
    if str(mode) == "Target":
        return frozenset({"Target"})
    if str(mode) == "Source - Target":
        return frozenset({"Source", "Target"})
    return frozenset({"Source"})


def _wavelet_analysis_settings(settings: WaveletSettings) -> WaveletSettings:
    """Return analysis-only settings independent from display color range."""

    analysis_range_db = max(60.0, abs(float(settings.mask_floor_db)))
    return replace(settings, dynamic_range_db=analysis_range_db)


def _draw_wavelet_map(
    wavelet_map: WaveletMap,
    settings: WaveletSettings,
    mode: str,
    bandwidth_label: str,
    *,
    graph_mode: str,
    window_preview: bool = False,
) -> None:
    if str(graph_mode) == "Interactive":
        _draw_wavelet_map_plotly(wavelet_map, settings, mode, bandwidth_label, window_preview=window_preview)
        return
    _draw_wavelet_map_matplotlib(wavelet_map, settings, mode, bandwidth_label, window_preview=window_preview)


def draw_measurement_wavelet_map(
    wavelet_map: WaveletMap,
    *,
    graph_mode: str,
) -> None:
    """Draw a microphone measurement map with the Wavelet Mode renderer."""
    frequency = np.asarray(wavelet_map.frequency_hz, dtype=float)
    time_ms = np.asarray(wavelet_map.time_ms, dtype=float)
    if frequency.size == 0 or time_ms.size == 0:
        st.info(ui_message('ui.e037d5d15b74e2'))
        return
    settings = WaveletSettings(
        f_min=float(np.nanmin(frequency)),
        f_max=float(np.nanmax(frequency)),
        t_min_ms=float(np.nanmin(time_ms)),
        t_max_ms=float(np.nanmax(time_ms)),
        bandwidth_oct=1.0 / 3.0,
        dynamic_range_db=40.0,
        frequency_bins=int(frequency.size),
        time_bins=int(time_ms.size),
    )
    _draw_wavelet_map(
        wavelet_map,
        settings,
        "Source",
        "1/3 oct",
        graph_mode=str(graph_mode),
        window_preview=False,
    )


def _wavelet_color_domain(settings: WaveletSettings, mode: str) -> tuple[float, float]:
    return wavelet_color_domain(settings.dynamic_range_db, mode)


def _draw_wavelet_map_matplotlib(
    wavelet_map: WaveletMap,
    settings: WaveletSettings,
    mode: str,
    bandwidth_label: str,
    *,
    window_preview: bool = False,
) -> None:
    dark_bg = "#10161a"
    fig, ax = plt.subplots(figsize=(16.0, 9.0), dpi=120)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor("#071019")
    cmap = _matplotlib_absolute_cmap()
    vmin, vmax = _wavelet_color_domain(settings, mode)
    mesh = ax.pcolormesh(
        wavelet_map.time_ms,
        wavelet_map.frequency_hz,
        wavelet_map.level_db,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_yscale("log")
    ax.set_xlim(float(settings.t_min_ms), float(settings.t_max_ms))
    ax.set_ylim(max(float(settings.f_min), 1.0), float(settings.f_max))
    ax.axvline(0.0, color="#f2f2f2", linewidth=0.8, linestyle="--", alpha=0.7)
    peak_time, peak_freq = _wavelet_peak_trace(wavelet_map, mode)
    if peak_time.size >= 2:
        ax.plot(
            peak_time,
            peak_freq,
            color=WAVELET_PEAK_TRACE_COLOR,
            label="Peak time",
            linewidth=1.15,
            alpha=0.92,
            solid_capstyle="round",
            path_effects=[],
        )
    centroid_time, centroid_freq = _wavelet_centroid_trace(wavelet_map)
    if centroid_time.size >= 2:
        ax.plot(
            centroid_time,
            centroid_freq,
            color=WAVELET_CENTROID_TRACE_COLOR,
            label="Centroid time",
            linewidth=0.95,
            alpha=0.78,
            linestyle="-",
            solid_capstyle="round",
        )
    window_start, window_end, window_freq, window_confidence = _wavelet_window_preview_traces(wavelet_map)
    if window_preview and window_start.size >= 2:
        _plot_window_preview_matplotlib(ax, window_start, window_freq, window_confidence, label="Window start")
        _plot_window_preview_matplotlib(ax, window_end, window_freq, window_confidence, label="Window end")
    if peak_time.size >= 2 or centroid_time.size >= 2 or (window_preview and window_start.size >= 2):
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.985, 0.89),
            frameon=True,
            facecolor="#10161a",
            edgecolor="#69727a",
            framealpha=0.82,
            fontsize=9,
        )
        for text in legend.get_texts():
            text.set_color("#f0f3f5")
    ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
    visible_ticks = [tick for tick in ticks if settings.f_min <= tick <= settings.f_max]
    ax.set_yticks(visible_ticks)
    ax.set_yticklabels([_format_frequency_tick(tick) for tick in visible_ticks])
    ax.set_xlabel(ui_message('ui.48693e0569ffab'), color="#d8dde2")
    ax.set_ylabel(ui_message('ui.5f9597e24414b2'), color="#d8dde2")
    ax.tick_params(colors="#d8dde2")
    ax.grid(which="major", color="#ffffff", alpha=0.18, linewidth=0.6)
    ax.grid(which="minor", color="#ffffff", alpha=0.08, linewidth=0.4)
    ax.set_title(wavelet_map.label, color="#f0f3f5", loc="left", pad=12)
    ax.text(
        0.985,
        0.97,
        bandwidth_label,
        transform=ax.transAxes,
        color="#f0f3f5",
        ha="right",
        va="top",
        bbox={"facecolor": "#10161a", "edgecolor": "#69727a", "alpha": 0.85, "pad": 4},
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    colorbar.ax.tick_params(colors="#d8dde2")
    colorbar.set_label("dB", color="#d8dde2")
    for spine in ax.spines.values():
        spine.set_color("#56616a")
    from matplotlib.text import Text
    from utils.plot_fonts import get_japanese_font
    font = get_japanese_font()
    if font is not None:
        # Apply to this figure, including Japanese measurement names, without
        # changing other sessions' global Matplotlib defaults or text sizes.
        for text in fig.findobj(Text):
            properties = text.get_fontproperties().copy()
            properties.set_file(font.get_file())
            text.set_fontproperties(properties)
    st.pyplot(fig, clear_figure=True)


def _draw_wavelet_map_plotly(
    wavelet_map: WaveletMap,
    settings: WaveletSettings,
    mode: str,
    bandwidth_label: str,
    *,
    window_preview: bool = False,
) -> None:
    vmin, vmax = _wavelet_color_domain(settings, mode)
    time = np.asarray(wavelet_map.time_ms, dtype=float)
    freq = np.asarray(wavelet_map.frequency_hz, dtype=float)
    level = np.asarray(wavelet_map.level_db, dtype=float)
    if time.size == 0 or freq.size == 0 or level.size == 0:
        st.info(ui_message('ui.e037d5d15b74e2'))
        return
    colorscale = _plotly_absolute_colorscale()
    fig = go.Figure(
        data=go.Heatmap(
            x=time,
            y=freq,
            z=level,
            zmin=vmin,
            zmax=vmax,
            colorscale=colorscale,
            colorbar={
                "title": {"text": "dB", "font": {"color": "#d8dde2"}},
                "tickfont": {"color": "#d8dde2", "size": 13},
            },
            hovertemplate="Time: %{x:.2f} ms<br>Frequency: %{y:.0f} Hz<br>Level: %{z:.2f} dB<extra></extra>",
        )
    )
    fig.add_vline(x=0.0, line_color="#f2f2f2", line_width=1, line_dash="dash", opacity=0.7)
    peak_time, peak_freq = _wavelet_peak_trace(wavelet_map, mode)
    if peak_time.size >= 2:
        fig.add_trace(
            go.Scatter(
                x=peak_time,
                y=peak_freq,
                mode="lines",
                name="Peak trace",
                showlegend=True,
                line={"color": WAVELET_PEAK_TRACE_COLOR, "width": 2.0},
                hovertemplate="Peak time: %{x:.2f} ms<br>Frequency: %{y:.0f} Hz<extra></extra>",
            )
        )
    centroid_time, centroid_freq = _wavelet_centroid_trace(wavelet_map)
    if centroid_time.size >= 2:
        fig.add_trace(
            go.Scatter(
                x=centroid_time,
                y=centroid_freq,
                mode="lines",
                name="Centroid",
                showlegend=True,
                line={"color": WAVELET_CENTROID_TRACE_COLOR, "width": 1.6},
                hovertemplate="Centroid: %{x:.2f} ms<br>Frequency: %{y:.0f} Hz<extra></extra>",
            )
        )
    window_start, window_end, window_freq, window_confidence = _wavelet_window_preview_traces(wavelet_map)
    if window_preview and window_start.size >= 2:
        _add_window_preview_plotly(fig, window_start, window_freq, window_confidence, "Window start")
        _add_window_preview_plotly(fig, window_end, window_freq, window_confidence, "Window end")
    fig.add_annotation(
        x=0.985,
        y=0.97,
        xref="paper",
        yref="paper",
        text=bandwidth_label,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"color": "#f0f3f5", "size": 13},
        bgcolor="rgba(16,22,26,0.85)",
        bordercolor="#69727a",
        borderwidth=1,
    )
    fig.update_layout(
        title={"text": wavelet_map.label, "x": 0.0, "font": {"color": "#f0f3f5", "size": 18}},
        height=WAVELET_CHART_HEIGHT_PX,
        margin={"l": 62, "r": 28, "t": 52, "b": 56},
        paper_bgcolor="#10161a",
        plot_bgcolor="#071019",
        font={"color": "#d8dde2", "size": 14},
        hovermode="closest",
        dragmode="zoom",
        showlegend=True,
        legend={
            "x": 0.985,
            "y": 0.90,
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(16,22,26,0.82)",
            "bordercolor": "#69727a",
            "borderwidth": 1,
            "font": {"color": "#f0f3f5", "size": 12},
        },
    )
    fig.update_xaxes(
        title="Time (ms)",
        range=[float(settings.t_min_ms), float(settings.t_max_ms)],
        gridcolor="rgba(255,255,255,0.16)",
        zeroline=False,
        color="#d8dde2",
        title_font={"size": 15},
        tickfont={"size": 13},
    )
    fig.update_yaxes(
        title="Frequency (Hz)",
        type="log",
        range=[np.log10(max(float(settings.f_min), 1.0)), np.log10(float(settings.f_max))],
        tickvals=[50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000],
        ticktext=["50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"],
        gridcolor="rgba(255,255,255,0.16)",
        zeroline=False,
        color="#d8dde2",
        title_font={"size": 15},
        tickfont={"size": 13},
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True, "responsive": True})


def _wavelet_peak_trace(wavelet_map: WaveletMap, mode: str) -> tuple[np.ndarray, np.ndarray]:
    peak_time = getattr(wavelet_map, "peak_time_ms", None)
    if peak_time is not None and mode != "Source - Target":
        peak = np.asarray(peak_time, dtype=float)
        freq = np.asarray(wavelet_map.frequency_hz, dtype=float)
        confidence = getattr(wavelet_map, "confidence", None)
        if confidence is None:
            valid = np.isfinite(peak) & np.isfinite(freq)
        else:
            conf = np.asarray(confidence, dtype=float)
            valid = np.isfinite(peak) & np.isfinite(freq) & np.isfinite(conf) & (conf > 0.02)
        if peak.size == freq.size and np.count_nonzero(valid) >= 2:
            return peak[valid], freq[valid]

    time = np.asarray(wavelet_map.time_ms, dtype=float)
    freq = np.asarray(wavelet_map.frequency_hz, dtype=float)
    level = np.asarray(wavelet_map.level_db, dtype=float)
    if time.size == 0 or freq.size == 0 or level.ndim != 2 or level.shape != (freq.size, time.size):
        return np.array([], dtype=float), np.array([], dtype=float)

    peak_times: list[float] = []
    peak_freqs: list[float] = []
    signed_mode = mode == "Source - Target"
    for row, frequency in zip(level, freq):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        row_values = np.abs(row) if signed_mode else row
        row_values = np.where(finite, row_values, -np.inf)
        peak_index = int(np.argmax(row_values))
        if signed_mode and not row_values[peak_index] > 1e-9:
            continue
        peak_times.append(float(time[peak_index]))
        peak_freqs.append(float(frequency))
    return np.asarray(peak_times, dtype=float), np.asarray(peak_freqs, dtype=float)


def _wavelet_centroid_trace(wavelet_map: WaveletMap) -> tuple[np.ndarray, np.ndarray]:
    centroid = getattr(wavelet_map, "centroid_time_ms", None)
    if centroid is None:
        return np.array([], dtype=float), np.array([], dtype=float)
    centroid_time = np.asarray(centroid, dtype=float)
    freq = np.asarray(wavelet_map.frequency_hz, dtype=float)
    confidence = getattr(wavelet_map, "confidence", None)
    if confidence is None:
        valid = np.isfinite(centroid_time) & np.isfinite(freq)
    else:
        conf = np.asarray(confidence, dtype=float)
        valid = np.isfinite(centroid_time) & np.isfinite(freq) & np.isfinite(conf) & (conf > 0.08)
    if centroid_time.size != freq.size or np.count_nonzero(valid) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)
    return centroid_time[valid], freq[valid]


def _wavelet_window_preview_traces(
    wavelet_map: WaveletMap,
    *,
    spread_multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroid = getattr(wavelet_map, "centroid_time_ms", None)
    spread = getattr(wavelet_map, "spread_ms", None)
    if centroid is None or spread is None:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    center = np.asarray(centroid, dtype=float)
    width = np.asarray(spread, dtype=float) * max(float(spread_multiplier), 1.0)
    freq = np.asarray(wavelet_map.frequency_hz, dtype=float)
    confidence = getattr(wavelet_map, "confidence", None)
    if confidence is None:
        conf = np.ones_like(freq, dtype=float)
    else:
        conf = np.asarray(confidence, dtype=float)
    if not (center.size == width.size == freq.size == conf.size):
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    valid_mask = (
        np.isfinite(center)
        & np.isfinite(width)
        & np.isfinite(freq)
        & np.isfinite(conf)
        & (width > 0.0)
        & (conf > 0.02)
    )
    if np.count_nonzero(valid_mask) < 2:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    start = center[valid_mask] - width[valid_mask]
    end = center[valid_mask] + width[valid_mask]
    return start, end, freq[valid_mask], np.clip(conf[valid_mask], 0.0, 1.0)


def _plot_window_preview_matplotlib(
    ax,
    time_ms: np.ndarray,
    frequency_hz: np.ndarray,
    confidence: np.ndarray,
    *,
    label: str,
) -> None:
    high = confidence >= 0.35
    low = ~high
    if np.count_nonzero(low) >= 2:
        ax.plot(
            time_ms[low],
            frequency_hz[low],
            color=WAVELET_WINDOW_TRACE_COLOR,
            label=None,
            linewidth=0.7,
            alpha=0.22,
            linestyle="--",
        )
    if np.count_nonzero(high) >= 2:
        ax.plot(
            time_ms[high],
            frequency_hz[high],
            color=WAVELET_WINDOW_TRACE_COLOR,
            label=label,
            linewidth=0.85,
            alpha=0.7,
            linestyle="--",
        )


def _add_window_preview_plotly(
    fig: go.Figure,
    time_ms: np.ndarray,
    frequency_hz: np.ndarray,
    confidence: np.ndarray,
    name: str,
) -> None:
    high = confidence >= 0.35
    low = ~high
    if np.count_nonzero(low) >= 2:
        fig.add_trace(
            go.Scatter(
                x=time_ms[low],
                y=frequency_hz[low],
                mode="lines",
                name=f"{name} low confidence",
                showlegend=False,
                line={"color": WAVELET_WINDOW_TRACE_COLOR, "width": 1.0, "dash": "dash"},
                opacity=0.22,
                hovertemplate=f"{name}: %{{x:.2f}} ms<br>Frequency: %{{y:.0f}} Hz<extra></extra>",
            )
        )
    if np.count_nonzero(high) >= 2:
        fig.add_trace(
            go.Scatter(
                x=time_ms[high],
                y=frequency_hz[high],
                mode="lines",
                name=name,
                showlegend=True,
                line={"color": WAVELET_WINDOW_TRACE_COLOR, "width": 1.5, "dash": "dash"},
                opacity=0.7,
                hovertemplate=f"{name}: %{{x:.2f}} ms<br>Frequency: %{{y:.0f}} Hz<extra></extra>",
            )
        )


def _plotly_absolute_colorscale() -> list[list[float | str]]:
    return plotly_absolute_colorscale()


def _matplotlib_absolute_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("responsefir_wavelet_dark_turbo", WAVELET_ABSOLUTE_COLORS)


def _wavelet_bins_for_graph_points(graph_max_points: int) -> tuple[int, int]:
    selected = int(graph_max_points)
    options = sorted(WAVELET_GRAPH_POINTS_FREQUENCY_BINS)
    nearest = min(options, key=lambda value: abs(value - selected))
    return WAVELET_GRAPH_POINTS_FREQUENCY_BINS[nearest], WAVELET_GRAPH_POINTS_TIME_BINS[nearest]


def _render_wavelet_details(
    result: WaveletRenderResult,
    settings: WaveletSettings,
    auto_align: bool,
    *,
    status_note: str,
    resolution_note: str,
    bundle_note: str = "",
) -> None:
    if not _wavelet_card_open("wavelet_settings_status", "Wavelet settings / status", default_open=False):
        return
    with st.container(border=True):
        st.caption(status_note)
        st.caption(resolution_note)
        if bundle_note:
            st.caption(bundle_note)
        cols = st.columns(5)
        cols[0].metric(ui_message('ui.16b6668d9831a1'), f"{settings.f_min:.0f}-{settings.f_max:.0f} Hz")
        cols[1].metric(ui_message('ui.33b93476cf597a'), f"{settings.t_min_ms:.0f} to {settings.t_max_ms:.0f} ms")
        cols[2].metric(ui_message('ui.5de74a81b19230'), f"{settings.dynamic_range_db:.0f} dB")
        cols[3].metric(ui_message('ui.eedf3216f00aef'), f"{settings.frequency_bins} x {settings.time_bins}")
        cols[4].metric(ui_message('ui.b1b1b4e5e8d796'), "ON" if auto_align else "OFF")
        metric_summary = _wavelet_metric_summary(result.map)
        metric_cols = st.columns(6)
        metric_cols[0].metric(ui_message('ui.355aca53f2aa06'), metric_summary["valid_bins"])
        metric_cols[1].metric(ui_message('ui.8a95c00f40bc3e'), metric_summary["peak_median_ms"])
        metric_cols[2].metric(ui_message('ui.b11c3e758b0c45'), metric_summary["centroid_median_ms"])
        metric_cols[3].metric(ui_message('ui.e31af36a9fe1fc'), metric_summary["spread_median_ms"])
        metric_cols[4].metric(ui_message('ui.7538c2135b030e'), metric_summary["reflection_median"])
        metric_cols[5].metric(ui_message('ui.6422e3e78f5e5f'), metric_summary["confidence_median"])
        alignment_confidence = _wavelet_alignment_confidence(result)
        guidance = _wavelet_guidance(metric_summary, alignment_confidence=alignment_confidence)
        quality_cols = st.columns([0.18, 0.18, 0.64])
        quality_cols[0].metric(ui_message('ui.1b2c08a8733d7f'), f"{guidance['score']:.0f}/100")
        quality_cols[1].metric(ui_message('ui.9f29530464f730'), guidance["overall"])
        quality_cols[2].caption(
            ui_message('ui.23be8a4405c759', p0=f"{guidance['reflection_rating']}", p1=f"{guidance['alignment_label']}")
        )
        st.caption(ui_message('ui.ad6d6a5edee399', p0=f"{guidance['summary']}"))
        st.caption(guidance["reading"])
        alignment_notes = []
        if result.source_alignment is not None:
            alignment_notes.append(_wavelet_alignment_summary("Source", result.source_alignment))
        if result.system_alignment is not None:
            alignment_notes.append(_wavelet_alignment_summary("System", result.system_alignment))
        if alignment_notes:
            st.caption(ui_message('ui.d8d67ccefc8505') + " / ".join(alignment_notes))


def _wavelet_metric_summary(wavelet_map: WaveletMap) -> dict[str, str]:
    peak = np.asarray(
        wavelet_map.peak_time_ms
        if wavelet_map.peak_time_ms is not None
        else np.asarray([], dtype=float),
        dtype=float,
    )
    centroid = np.asarray(
        wavelet_map.centroid_time_ms
        if wavelet_map.centroid_time_ms is not None
        else np.asarray([], dtype=float),
        dtype=float,
    )
    spread = np.asarray(
        wavelet_map.spread_ms
        if wavelet_map.spread_ms is not None
        else np.asarray([], dtype=float),
        dtype=float,
    )
    confidence = np.asarray(
        wavelet_map.confidence
        if wavelet_map.confidence is not None
        else np.asarray([], dtype=float),
        dtype=float,
    )
    reflection = np.asarray(
        wavelet_map.reflection_index
        if getattr(wavelet_map, "reflection_index", None) is not None
        else np.asarray([], dtype=float),
        dtype=float,
    )
    size = max(peak.size, centroid.size, spread.size, confidence.size, reflection.size)
    valid = np.ones(size, dtype=bool)
    for values in (peak, centroid, spread, confidence):
        if values.size == size:
            valid &= np.isfinite(values)
    if confidence.size == size:
        valid &= confidence > 0.0

    def median_label(values: np.ndarray, suffix: str, precision: int = 2) -> str:
        if values.size != size:
            return "-"
        local_valid = valid & np.isfinite(values)
        if not np.any(local_valid):
            return "-"
        return f"{float(np.nanmedian(values[local_valid])):.{precision}f} {suffix}".strip()

    def median_value(values: np.ndarray) -> float:
        if values.size != size:
            return float("nan")
        local_valid = valid & np.isfinite(values)
        if not np.any(local_valid):
            return float("nan")
        return float(np.nanmedian(values[local_valid]))

    return {
        "valid_bins": f"{int(np.count_nonzero(valid))}/{size}" if size else "0/0",
        "peak_median_ms": median_label(peak, "ms"),
        "centroid_median_ms": median_label(centroid, "ms"),
        "spread_median_ms": median_label(spread, "ms"),
        "reflection_median": median_label(reflection, "", precision=2),
        "confidence_median": median_label(confidence, "", precision=2),
        "peak_median_value": median_value(peak),
        "centroid_median_value": median_value(centroid),
        "spread_median_value": median_value(spread),
        "reflection_median_value": median_value(reflection),
        "confidence_median_value": median_value(confidence),
    }


def _wavelet_guidance(
    summary: dict[str, str | float],
    *,
    alignment_confidence: float | None = None,
) -> dict[str, str | float]:
    peak = _summary_float(summary, "peak_median_value")
    centroid = _summary_float(summary, "centroid_median_value")
    spread = _summary_float(summary, "spread_median_value")
    reflection = _summary_float(summary, "reflection_median_value")
    confidence = _summary_float(summary, "confidence_median_value")
    alignment = float(alignment_confidence) if alignment_confidence is not None else float("nan")
    reflection_rating = _reflection_rating(reflection)
    score = _wavelet_quality_score(
        reflection=reflection,
        confidence=confidence,
        spread=spread,
        alignment=alignment,
    )

    issues: list[str] = []
    strengths: list[str] = []
    if np.isfinite(confidence):
        if confidence >= 0.75:
            strengths.append("confidence is high")
        elif confidence < 0.35:
            issues.append("confidence is low")
    if np.isfinite(spread):
        if spread <= 3.0:
            strengths.append("spread is compact")
        elif spread > 8.0:
            issues.append("spread is wide")
    if np.isfinite(reflection):
        if reflection >= 0.45:
            issues.append("reflection is high")
        elif reflection <= 0.20:
            strengths.append("reflection is low")
    if np.isfinite(peak) and abs(peak) > 2.0:
        issues.append("peak is away from 0 ms")
    if np.isfinite(centroid) and abs(centroid) > 3.0:
        issues.append("centroid is delayed")

    if np.isfinite(score):
        if score >= 90.0:
            overall = "Excellent"
        elif score >= 75.0:
            overall = "Good"
        elif score >= 60.0:
            overall = "Watch"
        else:
            overall = "Check"
    elif len(issues) >= 2 or (np.isfinite(confidence) and confidence < 0.25):
        overall = "Check"
    elif issues:
        overall = "Watch"
    else:
        overall = "Good"
    if issues:
        summary_text = ", ".join(issues[:3])
    else:
        score_text = f"quality score {score:.0f}" if np.isfinite(score) else "quality score unavailable"
        summary_text = ", ".join(strengths[:3]) if strengths else "metrics are within a normal range"
        summary_text = f"{summary_text}; {score_text}"

    reading = (
        "Guide: Peak/centroid near 0 ms means timing is centered; "
        "smaller spread means a tighter main response; lower reflection means fewer late components; "
        "higher confidence and alignment confidence mean the timing estimate is more reliable."
    )
    return {
        "overall": overall,
        "summary": summary_text,
        "reading": reading,
        "score": float(score) if np.isfinite(score) else 0.0,
        "reflection_rating": reflection_rating,
        "alignment_label": _alignment_rating(alignment),
    }


def _wavelet_quality_score(
    *,
    reflection: float,
    confidence: float,
    spread: float,
    alignment: float,
) -> float:
    parts: list[tuple[float, float]] = []
    if np.isfinite(reflection):
        parts.append((0.40, _reflection_score(reflection)))
    if np.isfinite(confidence):
        parts.append((0.30, np.clip(confidence, 0.0, 1.0) * 100.0))
    if np.isfinite(spread):
        parts.append((0.20, _spread_score(spread)))
    if np.isfinite(alignment):
        parts.append((0.10, np.clip(alignment, 0.0, 1.0) * 100.0))
    if not parts:
        return float("nan")
    weight_sum = float(sum(weight for weight, _ in parts))
    return float(sum(weight * score for weight, score in parts) / max(weight_sum, 1e-9))


def _reflection_score(reflection: float) -> float:
    if reflection <= 0.15:
        return 100.0
    if reflection <= 0.30:
        return 85.0 - (reflection - 0.15) / 0.15 * 15.0
    if reflection <= 0.50:
        return 70.0 - (reflection - 0.30) / 0.20 * 25.0
    return max(0.0, 45.0 - min(reflection - 0.50, 1.0) * 45.0)


def _spread_score(spread_ms: float) -> float:
    if spread_ms <= 1.5:
        return 100.0
    if spread_ms <= 3.0:
        return 92.0 - (spread_ms - 1.5) / 1.5 * 12.0
    if spread_ms <= 8.0:
        return 80.0 - (spread_ms - 3.0) / 5.0 * 35.0
    return max(0.0, 45.0 - min(spread_ms - 8.0, 12.0) / 12.0 * 45.0)


def _reflection_rating(reflection: float) -> str:
    if not np.isfinite(reflection):
        return "Unknown"
    if reflection < 0.15:
        return "Excellent"
    if reflection < 0.30:
        return "Good"
    if reflection < 0.50:
        return "Moderate"
    return "High Reflection"


def _alignment_rating(alignment: float) -> str:
    if not np.isfinite(alignment):
        return "Unknown"
    if alignment >= 0.85:
        return "High"
    if alignment >= 0.60:
        return "Good"
    if alignment >= 0.35:
        return "Watch"
    return "Low"


def _wavelet_alignment_confidence(result: WaveletRenderResult) -> float:
    values = []
    for alignment in (result.source_alignment, result.system_alignment):
        if alignment is not None and np.isfinite(float(alignment.confidence)):
            values.append(float(alignment.confidence))
    if not values:
        return float("nan")
    return float(np.nanmedian(values))


def _summary_float(summary: dict[str, str | float], key: str) -> float:
    try:
        return float(summary.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _wavelet_alignment_summary(label: str, alignment) -> str:
    fallback = ", fallback" if bool(getattr(alignment, "fallback_used", False)) else ""
    return (
        f"{label} {alignment.method} "
        f"{float(alignment.delay_ms):+.3f} ms "
        f"(conf {float(alignment.confidence):.2f}{fallback})"
    )


def _wavelet_card_open(card_id: str, title: str, *, default_open: bool = False) -> bool:
    state_key = f"{CARD_STATE_PREFIX}{card_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = bool(default_open)
    opened = bool(st.session_state[state_key])
    cols = st.columns([0.09, 0.91], gap="small", vertical_alignment="center")
    with cols[0]:
        if localized_button(st, "▼" if opened else "▶", key=f"{CARD_TOGGLE_PREFIX}{card_id}", help=ui_message('ui.358f60870d7f41')):
            st.session_state[state_key] = not opened
            st.rerun()
    with cols[1]:
        st.markdown(ui_message('ui.061bd94945ee5b', p0=f'{title}'))
    return opened


def _require_ir(values: np.ndarray | None, label: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{label} IR is not available.")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{label} IR is empty.")
    return array


def _settings_fingerprint(mode: Any, settings: WaveletSettings, auto_align: bool) -> str:
    payload = (
        WAVELET_RENDER_ALGORITHM_VERSION,
        str(mode),
        float(settings.f_min),
        float(settings.f_max),
        float(settings.t_min_ms),
        float(settings.t_max_ms),
        float(settings.bandwidth_oct),
        int(settings.frequency_bins),
        int(settings.time_bins),
        bool(auto_align),
    )
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def _format_frequency_tick(value: float) -> str:
    return f"{value / 1000:.0f}k" if value >= 1000 else f"{value:.0f}"
