from __future__ import annotations

from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.list_menu_ui import selectbox as shared_selectbox, radio as shared_radio, segmented_control as shared_segmented_control

from utils.wavelet_plot_style import (
    WAVELET_BORDER_COLOR,
    WAVELET_CENTROID_TRACE_COLOR,
    WAVELET_FOREGROUND,
    WAVELET_PAPER_BACKGROUND,
    WAVELET_PEAK_TRACE_COLOR,
    WAVELET_PLOT_BACKGROUND,
    WAVELET_TITLE_COLOR,
    WAVELET_WINDOW_TRACE_COLOR,
    plotly_absolute_colorscale,
    wavelet_color_domain,
)

from wavelet_analysis import (
    WAVELET_BANDWIDTH_OPTIONS,
    WAVELET_PRESETS,
    WaveletMap,
    bandwidth_label_to_oct,
    complex_morlet_scalogram,
    settings_from_preset,
    wavelet_difference_map,
)


_BINS = {1024: (512, 1024), 2048: (768, 1448), 4096: (1024, 2048)}


@st.cache_data(max_entries=12, show_spinner=False)
def _map_cached(
    impulse: np.ndarray,
    sample_rate_hz: int,
    settings: WaveletSettings,
    label: str,
) -> WaveletMap:
    return complex_morlet_scalogram(impulse, sample_rate_hz, settings, label=label)


def render_composite_wavelet(
    source_ir: np.ndarray,
    target_ir: np.ndarray | None,
    sample_rate_hz: int,
    *,
    key_prefix: str,
) -> None:
    st.session_state.setdefault(f"{key_prefix}_mode", "Source")
    st.session_state.setdefault(f"{key_prefix}_preset", "Overview")
    st.session_state.setdefault(f"{key_prefix}_bandwidth", "1/3 oct")
    st.session_state.setdefault(f"{key_prefix}_range", 40)
    st.session_state.setdefault(f"{key_prefix}_points", 1024)
    st.session_state.setdefault(f"{key_prefix}_window", False)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        mode = shared_segmented_control(
            "Wavelet Mode", ["Source", "Target", "Source - Target"],
            key=f"{key_prefix}_mode", width="stretch",
        ) or "Source"
        preset = shared_selectbox("Preset", list(WAVELET_PRESETS), key=f"{key_prefix}_preset")
        bandwidth = shared_selectbox(
            "Bandwidth", WAVELET_BANDWIDTH_OPTIONS, key=f"{key_prefix}_bandwidth",
        )
        dynamic_range = shared_selectbox("Range", [20, 40, 60], key=f"{key_prefix}_range")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        points = shared_selectbox("Graph points", [1024, 2048, 4096], key=f"{key_prefix}_points")
        window_preview = st.checkbox("Window Preview", key=f"{key_prefix}_window")
    if mode != "Source" and target_ir is None:
        st.info("Target IRがないためSourceを表示します。")
        mode = "Source"
    base = settings_from_preset(str(preset))
    frequency_bins, time_bins = _BINS[int(points)]
    settings = replace(
        base,
        bandwidth_oct=bandwidth_label_to_oct(str(bandwidth)),
        dynamic_range_db=float(dynamic_range),
        frequency_bins=frequency_bins,
        time_bins=time_bins,
    )
    source_map = _map_cached(np.asarray(source_ir, dtype=float), int(sample_rate_hz), settings, "Composite")
    if mode == "Source":
        selected = source_map
    else:
        target_map = _map_cached(np.asarray(target_ir, dtype=float), int(sample_rate_hz), settings, "Group Target")
        selected = target_map if mode == "Target" else wavelet_difference_map(source_map, target_map)
    color_min, color_max = wavelet_color_domain(float(dynamic_range), mode)
    figure = go.Figure(go.Heatmap(
        x=selected.time_ms,
        y=selected.frequency_hz,
        z=selected.level_db,
        colorscale=plotly_absolute_colorscale(),
        zmin=color_min,
        zmax=color_max,
        colorbar={
            "title": {"text": "dB", "font": {"color": WAVELET_FOREGROUND}},
            "tickfont": {"color": WAVELET_FOREGROUND, "size": 13},
        },
        hovertemplate="Time: %{x:.2f} ms<br>Frequency: %{y:.0f} Hz<br>Level: %{z:.2f} dB<extra></extra>",
    ))
    if selected.peak_time_ms is not None:
        figure.add_trace(go.Scatter(
            x=selected.peak_time_ms, y=selected.frequency_hz,
            mode="lines", name="Peak time", line={"color": WAVELET_PEAK_TRACE_COLOR, "width": 2.0},
        ))
    if selected.centroid_time_ms is not None:
        figure.add_trace(go.Scatter(
            x=selected.centroid_time_ms, y=selected.frequency_hz,
            mode="lines", name="Centroid time", line={"color": WAVELET_CENTROID_TRACE_COLOR, "width": 1.6},
        ))
        if window_preview and selected.spread_ms is not None:
            for sign, name in ((-1.0, "Window start"), (1.0, "Window end")):
                figure.add_trace(go.Scatter(
                    x=selected.centroid_time_ms + sign * selected.spread_ms,
                    y=selected.frequency_hz, mode="lines", name=name,
                    line={"color": WAVELET_WINDOW_TRACE_COLOR, "width": 1.5, "dash": "dash"},
                ))
    figure.update_layout(
        height=720, xaxis_title="Time [ms]", yaxis_title="Frequency [Hz]",
        yaxis_type="log", margin={"l": 60, "r": 20, "t": 20, "b": 50},
        paper_bgcolor=WAVELET_PAPER_BACKGROUND,
        plot_bgcolor=WAVELET_PLOT_BACKGROUND,
        font={"color": WAVELET_FOREGROUND, "size": 14},
        hovermode="closest",
        dragmode="zoom",
        legend={
            "bgcolor": "rgba(16,22,26,0.82)",
            "bordercolor": WAVELET_BORDER_COLOR,
            "borderwidth": 1,
            "font": {"color": WAVELET_TITLE_COLOR, "size": 12},
        },
    )
    figure.update_xaxes(
        gridcolor="rgba(255,255,255,0.16)", zeroline=False,
        color=WAVELET_FOREGROUND,
    )
    figure.update_yaxes(
        gridcolor="rgba(255,255,255,0.16)", zeroline=False,
        color=WAVELET_FOREGROUND,
        tickvals=[50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000],
        ticktext=["50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"],
    )
    st.plotly_chart(
        figure, width="stretch", key=f"{key_prefix}_chart",
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    valid = np.asarray(selected.confidence, dtype=float) if selected.confidence is not None else np.array([])
    reflection = np.asarray(selected.reflection_index, dtype=float) if selected.reflection_index is not None else np.array([])
    spread = np.asarray(selected.spread_ms, dtype=float) if selected.spread_ms is not None else np.array([])
    with st.container(horizontal=True):
        st.metric("Confidence", f"{float(np.nanmedian(valid)):.2f}" if valid.size else "—", border=True)
        st.metric("Reflection", f"{float(np.nanmedian(reflection)):.2f}" if reflection.size else "—", border=True)
        st.metric("Spread", f"{float(np.nanmedian(spread)):.2f} ms" if spread.size else "—", border=True)
    st.caption("Alignment: Composite Engine tap center固定（Peak／Cross Correlation／Auto Delayなし）")
