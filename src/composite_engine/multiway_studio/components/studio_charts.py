"""Renderer-independent Studio chart projection and Streamlit renderers.

The projector owns every display-only numeric decision (cropping, masking and
sampling).  Renderers are deliberately small consumers of the immutable
bundle: they may translate labels and style tokens, but never recalculate DSP
or infer semantics from another renderer's objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from response_display import phase_segment_ids, sample_series
from utils.ui_localization import display_text, ui_message
from utils.ui_semantic_colors import semantic_chart_colors


PROJECTION_REVISION = 1
CHART_IDS = (
    "band_impulse",
    "band_step",
    "relative_group_delay",
    "system_impulse",
    "system_gain_phase",
    "crossover_design",
)


@dataclass(frozen=True)
class AxisSpec:
    axis_id: str
    scale: str
    domain: tuple[float, float]
    label_key: str
    side: str


@dataclass(frozen=True)
class ChartSeries:
    series_id: str
    label_key_or_text: str
    role: str
    axis_id: str
    x: np.ndarray
    y: np.ndarray
    style_token: str
    legend_visible: bool = True
    dash: str = "solid"
    width: float = 1.5
    opacity: float = 1.0
    segment_id: np.ndarray | None = None


@dataclass(frozen=True)
class ChartMarker:
    marker_id: str
    axis: str
    value: float
    style_token: str
    legend_visible: bool = False


@dataclass(frozen=True)
class StudioChart:
    chart_id: str
    title_key: str
    x_axis: AxisSpec
    y_axes: tuple[AxisSpec, ...]
    series: tuple[ChartSeries, ...]
    markers: tuple[ChartMarker, ...]
    interaction: str


@dataclass(frozen=True)
class StudioChartBundle:
    source_revision: str
    projection_revision: int
    charts: tuple[StudioChart, ...]

    def chart(self, chart_id: str) -> StudioChart:
        return self.charts[CHART_IDS.index(chart_id)]


def _readonly(values, *, dtype=float) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True).reshape(-1)
    result.setflags(write=False)
    return result


def _segments(values) -> np.ndarray:
    segment_ids, _ = phase_segment_ids(np.asarray(values, dtype=float))
    return _readonly(segment_ids, dtype=np.int64)


def _series(
    series_id, label, role, axis_id, x, y, style_token, *,
    legend_visible=True, dash="solid", width=1.5, opacity=1.0,
    segmented=False,
):
    x_values = _readonly(x)
    y_values = _readonly(y)
    return ChartSeries(
        str(series_id), str(label), str(role), str(axis_id), x_values, y_values,
        str(style_token), bool(legend_visible), str(dash), float(width),
        float(opacity), _segments(y_values) if segmented else None,
    )


def _axis(axis_id, scale, domain, label_key, side):
    return AxisSpec(
        str(axis_id), str(scale), (float(domain[0]), float(domain[1])),
        str(label_key), str(side),
    )


def _center_impulses(impulses: Mapping[str, np.ndarray]):
    if not impulses:
        return {}, 1
    length = max(len(np.asarray(value).reshape(-1)) for value in impulses.values())
    centered = {}
    for name, values in impulses.items():
        source = np.asarray(values, dtype=float).reshape(-1)
        total = length - source.size
        centered[name] = np.pad(source, (total // 2, (total + 1) // 2))
    return centered, length


def _time_values(impulses, sample_rate_hz):
    centered, length = _center_impulses(impulses)
    center = (length - 1) // 2
    x = (np.arange(length) - center) / float(sample_rate_hz) * 1000.0
    visible = (x >= -2.0) & (x <= 10.0)
    return centered, x, visible


def _sample_frequency(x, y, max_points, *, phase=False):
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.size == 0:
        return x_values.copy(), y_values.copy()
    return sample_series(x_values, y_values, max_points, wrapped_phase=phase)


def build_studio_chart_bundle(
    impulses: Mapping[str, np.ndarray],
    frequency_hz: np.ndarray,
    responses: Mapping[str, np.ndarray],
    *,
    sample_rate_hz: int,
    sum_groups: Mapping[str, Sequence[str]],
    way_sums: Mapping[str, np.ndarray],
    db_min: float,
    db_max: float,
    phase_gain_mask_db: float,
    phaseeq_responses: Mapping[str, np.ndarray] | None = None,
    speaker_responses: Mapping[str, np.ndarray] | None = None,
    baffle_response: np.ndarray | None = None,
    baffle_label: str | None = None,
    max_points: int = 1024,
    source_revision: str = "",
) -> StudioChartBundle:
    """Project the six Studio graphs without importing a plotting library."""
    frequency = np.asarray(frequency_hz, dtype=float).reshape(-1)
    fs = float(sample_rate_hz)
    limit = min(4096, max(2, int(max_points)))
    centered, time_ms, time_visible = _time_values(impulses, fs)
    band_names = tuple(centered)
    groups = dict(sum_groups or {"Sum (All Bands)": tuple(band_names)})

    def grouped_impulse(keys):
        selected = [centered[key] for key in keys if key in centered]
        return np.sum(selected, axis=0) if selected else None

    impulse_series = []
    step_series = []
    for index, name in enumerate(band_names):
        impulse_series.append(_series(
            f"way:{name}:ir", name, "WAY", "amplitude",
            time_ms[time_visible], centered[name][time_visible], f"band:{index}",
        ))
        step_series.append(_series(
            f"way:{name}:step", name, "WAY", "integral",
            time_ms[time_visible], np.cumsum(centered[name])[time_visible],
            f"band:{index}",
        ))
    system_impulse_series = []
    for index, (label, keys) in enumerate(groups.items()):
        combined = grouped_impulse(keys)
        if combined is None:
            continue
        token = "sum" if index == 0 else "band:3"
        impulse_series.append(_series(
            f"sum:{label}:ir", label, "SYSTEM_SUM", "amplitude",
            time_ms[time_visible], combined[time_visible], token,
            width=2.0, opacity=0.9,
        ))
        step_series.append(_series(
            f"sum:{label}:step", label, "SYSTEM_SUM", "integral",
            time_ms[time_visible], np.cumsum(combined)[time_visible], token,
            width=2.0, opacity=0.9,
        ))
        system_impulse_series.append(_series(
            f"sum:{label}:system-ir", label, "SYSTEM_SUM", "amplitude",
            time_ms[time_visible], combined[time_visible], token, width=2.0,
        ))

    time_x = _axis("time", "linear", (-2.0, 10.0), "ui.78f2c5fb6fe469", "bottom")
    marker = ChartMarker("time-zero", "x", 0.0, "center")

    gd_series = []
    if frequency.size >= 2:
        w_rad = 2.0 * np.pi * frequency / fs
        for index, (name, response) in enumerate(responses.items()):
            values = np.asarray(response, dtype=np.complex128).reshape(-1)
            if values.size != frequency.size:
                continue
            delay = -np.diff(np.unwrap(np.angle(values))) / np.diff(w_rad) / fs * 1000.0
            delay[np.abs(delay) < 1e-9] = 0.0
            magnitude = 20.0 * np.log10(np.abs(values[1:]) + 1e-12)
            valid = (
                np.isfinite(delay) & (magnitude > float(phase_gain_mask_db))
                & (frequency[1:] >= 10.0) & (frequency[1:] <= fs / 2.0)
            )
            x, y = _sample_frequency(frequency[1:][valid], delay[valid], limit)
            gd_series.append(_series(
                f"way:{name}:gd", name, "WAY", "delay", x, y, f"band:{index}",
            ))
        for index, (label, keys) in enumerate(groups.items()):
            selected = [np.asarray(responses[key], dtype=np.complex128) for key in keys if key in responses]
            if not selected:
                continue
            values = np.sum(selected, axis=0)
            delay = -np.diff(np.unwrap(np.angle(values))) / np.diff(w_rad) / fs * 1000.0
            magnitude = 20.0 * np.log10(np.abs(values[1:]) + 1e-12)
            valid = (
                np.isfinite(delay) & (magnitude > float(phase_gain_mask_db))
                & (frequency[1:] >= 10.0) & (frequency[1:] <= fs / 2.0)
            )
            x, y = _sample_frequency(frequency[1:][valid], delay[valid], limit)
            gd_series.append(_series(
                f"sum:{label}:gd", label, "SYSTEM_SUM", "delay", x, y,
                "sum" if index == 0 else "band:3", width=2.0,
            ))

    frequency_domain = (10.0, fs / 2.0)
    gain_phase_series = []
    frequency_mask = (frequency >= 10.0) & (frequency <= fs / 2.0)
    for index, (label, keys) in enumerate(groups.items()):
        selected = [np.asarray(responses[key], dtype=np.complex128) for key in keys if key in responses]
        if not selected:
            continue
        values = np.sum(selected, axis=0)
        magnitude = 20.0 * np.log10(np.maximum(np.abs(values), 1e-12))
        phase = np.rad2deg(np.angle(values))
        phase = np.where(frequency_mask & (magnitude >= float(phase_gain_mask_db)), phase, np.nan)
        token = "sum" if index == 0 else "band:3"
        x_gain, y_gain = _sample_frequency(frequency[frequency_mask], magnitude[frequency_mask], limit)
        x_phase, y_phase = _sample_frequency(
            frequency[frequency_mask], phase[frequency_mask], limit, phase=True,
        )
        gain_phase_series.extend((
            _series(f"sum:{label}:gain", label, "SYSTEM_SUM", "gain", x_gain, y_gain, token, width=2.0),
            _series(f"sum:{label}:phase", label, "PHASE", "phase", x_phase, y_phase, token,
                    legend_visible=False, dash="dot", width=2.0, opacity=0.7, segmented=True),
        ))

    design_series = []
    positive = frequency > 0.0
    design_domain = (10.0, float(frequency[-1]) if frequency.size else fs / 2.0)

    def add_response_pair(series_id, label, role, values, token, *,
                          phase_token=None,
                          gain_legend=True, phase_legend=False,
                          gain_dash="solid", phase_dash="dot",
                          gain_width=1.5, phase_width=1.5,
                          gain_opacity=1.0, phase_opacity=0.7):
        array = np.asarray(values, dtype=np.complex128).reshape(-1)
        if array.size != frequency.size:
            return
        magnitude = 20.0 * np.log10(np.maximum(np.abs(array), 1e-12))
        phase = np.where(
            positive & (magnitude >= float(phase_gain_mask_db)),
            np.rad2deg(np.angle(array)), np.nan,
        )
        x_gain, y_gain = _sample_frequency(frequency[positive], magnitude[positive], limit)
        x_phase, y_phase = _sample_frequency(frequency[positive], phase[positive], limit, phase=True)
        design_series.extend((
            _series(f"{series_id}:gain", label, role, "design_gain", x_gain, y_gain, token,
                    legend_visible=gain_legend, dash=gain_dash, width=gain_width, opacity=gain_opacity),
            _series(f"{series_id}:phase", label, "PHASE", "design_phase", x_phase, y_phase, phase_token or token,
                    legend_visible=phase_legend, dash=phase_dash, width=phase_width,
                    opacity=phase_opacity, segmented=True),
        ))

    for index, (name, values) in enumerate(responses.items()):
        token = f"band:{index}"
        add_response_pair(f"way:{name}", name, "WAY", values, token)
        phaseeq = (phaseeq_responses or {}).get(name)
        if phaseeq is not None:
            add_response_pair(
                f"phaseeq:{name}", f"{name} · PhaseEQ EQ", "PHASEEQ", phaseeq,
                "input", phase_token="correction",
                gain_dash="dash", gain_width=1.8, gain_opacity=0.95,
                phase_dash="dashdot", phase_width=1.8, phase_opacity=0.95,
                phase_legend=True,
            )
        speaker = (speaker_responses or {}).get(name)
        if speaker is not None:
            array = np.asarray(speaker, dtype=np.complex128).reshape(-1)
            if array.size == frequency.size:
                speaker_db = 20.0 * np.log10(np.maximum(np.abs(array), 1e-12))
                x, y = _sample_frequency(frequency[positive], speaker_db[positive], limit)
                design_series.append(_series(
                    f"speaker:{name}:gain", f"{name} · Speaker", "SPEAKER",
                    "design_gain", x, y, token, dash="dot", width=1.5, opacity=0.9,
                ))
    if baffle_response is not None:
        values = np.asarray(baffle_response, dtype=np.complex128).reshape(-1)
        if values.size == frequency.size:
            add_response_pair(
                "baffle", baffle_label or "Baffle compensation", "BAFFLE",
                values, "baffle", gain_dash="dashdot", gain_width=1.7,
                phase_dash="dot", phase_opacity=0.75,
            )
    sums = dict(way_sums or {})
    for index, (label, values) in enumerate(sums.items()):
        add_response_pair(
            f"design-sum:{label}", label, "SYSTEM_SUM", values,
            "sum" if index == 0 else "band:3", gain_width=2.0, phase_width=2.0,
        )

    charts = (
        StudioChart("band_impulse", "ui.f165a57d1c1366", time_x,
                    (_axis("amplitude", "linear", _data_domain(impulse_series), "ui.f18f28ad51c0cd", "left"),),
                    tuple(impulse_series), (marker,), "x"),
        StudioChart("band_step", "ui.36e8e9a9bd5e3f", time_x,
                    (_axis("integral", "linear", _data_domain(step_series), "ui.e732bb5b771580", "left"),),
                    tuple(step_series), (marker,), "x"),
        StudioChart("relative_group_delay", "ui.909cfbf5960044",
                    _axis("frequency", "log", frequency_domain, "ui.f11c5c8bac5aeb", "bottom"),
                    (_axis("delay", "linear", (-2.0, 10.0), "ui.1bb9aa84fccb4c", "left"),),
                    tuple(gd_series), (), "both"),
        StudioChart("system_impulse", "ui.ab0248fb56262b", time_x,
                    (_axis("amplitude", "linear", _data_domain(system_impulse_series), "ui.f18f28ad51c0cd", "left"),),
                    tuple(system_impulse_series), (), "x"),
        StudioChart("system_gain_phase", "ui.9fb0cd78c608c2",
                    _axis("frequency", "log", frequency_domain, "ui.f11c5c8bac5aeb", "bottom"),
                    (_axis("gain", "linear", (db_min, db_max), "ui.390fb3129a3e48", "left"),
                     _axis("phase", "linear", (-180.0, 180.0), "ui.7fd2691857ae18", "right")),
                    tuple(gain_phase_series), (), "both"),
        StudioChart("crossover_design", "ui.476dbeb78d9836",
                    _axis("frequency", "log", design_domain, "ui.f11c5c8bac5aeb", "bottom"),
                    (_axis("design_gain", "linear", (db_min, db_max), "ui.08624e6af38efb", "left"),
                     _axis("design_phase", "linear", (-180.0, 180.0), "ui.70b2ffec8b1a1c", "right")),
                    tuple(design_series), (), "both"),
    )
    bundle = StudioChartBundle(str(source_revision), PROJECTION_REVISION, charts)
    validate_studio_chart_bundle(bundle)
    return bundle


def _data_domain(series):
    finite = [item.y[np.isfinite(item.y)] for item in series if np.any(np.isfinite(item.y))]
    if not finite:
        return (-1.0, 1.0)
    values = np.concatenate(finite)
    low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        margin = max(abs(low) * 0.05, 1.0)
        return (low - margin, high + margin)
    margin = (high - low) * 0.05
    return (low - margin, high + margin)


def validate_studio_chart_bundle(bundle: StudioChartBundle) -> None:
    if tuple(chart.chart_id for chart in bundle.charts) != CHART_IDS:
        raise ValueError("Studio chart bundle must contain the six charts in fixed order")
    for chart in bundle.charts:
        axis_ids = {axis.axis_id for axis in chart.y_axes}
        if chart.x_axis.domain[0] >= chart.x_axis.domain[1]:
            raise ValueError(f"{chart.chart_id}: invalid x-axis domain")
        for item in chart.series:
            if item.axis_id not in axis_ids:
                raise ValueError(f"{chart.chart_id}/{item.series_id}: unknown y-axis")
            if item.x.ndim != 1 or item.y.ndim != 1 or item.x.size != item.y.size:
                raise ValueError(f"{chart.chart_id}/{item.series_id}: mismatched x/y")
            if not np.all(np.isfinite(item.x)):
                raise ValueError(f"{chart.chart_id}/{item.series_id}: non-finite x")
            if item.segment_id is not None and item.segment_id.size != item.x.size:
                raise ValueError(f"{chart.chart_id}/{item.series_id}: mismatched segments")
            if item.x.flags.writeable or item.y.flags.writeable:
                raise ValueError(f"{chart.chart_id}/{item.series_id}: mutable values")


def _label(value: str) -> str:
    return ui_message(value) if value.startswith("ui.") else display_text(value)


def _color(style_token: str, colors) -> str:
    if style_token.startswith("band:"):
        return colors["bands"][int(style_token.split(":", 1)[1]) % len(colors["bands"])]
    return str(colors.get(style_token, colors["text"]))


def _plotly_transfer_values(values) -> np.ndarray:
    """Build a renderer-local float32 array without changing the bundle."""
    return np.array(values, dtype=np.float32, order="C", copy=True)


def _altair_frame_values(item: ChartSeries) -> dict[str, np.ndarray]:
    """Keep discontinuity markers in one renderer-local line dataset."""
    return {"x": item.x, "y": item.y}


def plotly_studio_spec(chart: StudioChart, *, theme="light") -> dict:
    colors = semantic_chart_colors(theme)
    traces = []
    for item in chart.series:
        trace = {
            "type": "scatter", "mode": "lines",
            "x": _plotly_transfer_values(item.x),
            "y": _plotly_transfer_values(item.y),
            "name": _label(item.label_key_or_text),
            "showlegend": item.legend_visible,
            "line": {"color": _color(item.style_token, colors), "width": item.width, "dash": item.dash},
            "opacity": item.opacity,
            "connectgaps": False,
        }
        if item.axis_id == chart.y_axes[-1].axis_id and len(chart.y_axes) > 1:
            trace["yaxis"] = "y2"
        traces.append(trace)
    for marker in chart.markers:
        if marker.axis == "x":
            traces.append({
                "type": "scatter", "mode": "lines", "x": [marker.value, marker.value],
                "y": list(chart.y_axes[0].domain), "showlegend": False,
                "hoverinfo": "skip", "line": {"color": _color(marker.style_token, colors), "width": 0.8, "dash": "dash"},
            })
    x_axis = {
        "title": {
            "text": _label(chart.x_axis.label_key),
            "font": {"color": colors["text"]},
        },
        "type": chart.x_axis.scale,
        "range": ([float(np.log10(chart.x_axis.domain[0])), float(np.log10(chart.x_axis.domain[1]))]
                  if chart.x_axis.scale == "log" else list(chart.x_axis.domain)),
        "gridcolor": colors["grid"], "linecolor": colors["spine"],
        "tickfont": {"color": colors["text"]},
    }
    layout = {
        "xaxis": x_axis, "dragmode": "zoom", "hovermode": "x unified",
        # Match the Light renderer so changing renderer never changes the
        # document height or pushes later graphs vertically.
        "height": 430 if len(chart.y_axes) > 1 else 280,
        "margin": {"l": 55, "r": 45, "t": 55, "b": 50},
        "title": {"text": _label(chart.title_key), "x": 0.5, "font": {"color": colors["text"]}},
        "paper_bgcolor": colors["figure"], "plot_bgcolor": colors["axes"],
        "font": {"color": colors["text"]},
        "legend": {"font": {"color": colors["text"]}},
    }
    for index, axis in enumerate(chart.y_axes):
        spec = {
            "title": {
                "text": _label(axis.label_key),
                "font": {"color": colors["text"]},
            },
            "range": list(axis.domain),
            "fixedrange": chart.interaction == "x", "gridcolor": colors["grid"],
            "linecolor": colors["spine"], "tickfont": {"color": colors["text"]},
        }
        if index:
            spec.update({"anchor": "x", "overlaying": "y", "side": "right", "showgrid": False})
        layout["yaxis2" if index else "yaxis"] = spec
    return {"data": traces, "layout": layout}


def altair_studio_chart(chart: StudioChart, *, theme="light"):
    import altair as alt
    import pandas as pd

    colors = semantic_chart_colors(theme)
    x_encoding = alt.X(
        "x:Q", title=_label(chart.x_axis.label_key),
        scale=alt.Scale(type=chart.x_axis.scale, domain=list(chart.x_axis.domain)),
    )
    axis_layers = []
    for axis_index, axis in enumerate(chart.y_axes):
        layers = []
        axis_series = [item for item in chart.series if item.axis_id == axis.axis_id]
        visible_items = [item for item in axis_series if item.legend_visible]
        legend_domain = [_label(item.label_key_or_text) for item in visible_items]
        legend_range = [_color(item.style_token, colors) for item in visible_items]
        if visible_items:
            # Keep legend labels in a tiny dataset.  Repeating a long label for
            # every sample inflates Arrow payload without changing the graph.
            legend_frame = pd.DataFrame({"series": legend_domain})
            layers.append(
                alt.Chart(legend_frame)
                .transform_filter("false")
                .mark_line()
                .encode(color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=legend_domain, range=legend_range),
                    legend=alt.Legend(title=None, orient="top"),
                ))
            )
        for item in axis_series:
            segments = item.segment_id
            frame = pd.DataFrame(_altair_frame_values(item), copy=False)
            dash = {
                "solid": [1, 0], "dash": [8, 5],
                "dot": [2, 4], "dashdot": [8, 4, 2, 4],
            }.get(item.dash, [1, 0])
            mark_options = {
                "clip": True,
                "color": _color(item.style_token, colors),
                "strokeDash": dash,
                "strokeWidth": item.width,
                "opacity": item.opacity,
            }
            if segments is not None:
                # Vega-Lite keeps each series as one scenegraph item and uses
                # the retained NaNs as path breaks. Encoding every phase
                # segment as a detail group creates thousands of empty SVG
                # groups and clip paths for high-delay filters.
                mark_options["invalid"] = "break-paths-show-domains"
            layer = alt.Chart(frame).mark_line(**mark_options).encode(
                x=x_encoding,
                y=alt.Y(
                    "y:Q", title=_label(axis.label_key),
                    scale=alt.Scale(domain=list(axis.domain)),
                    axis=alt.Axis(orient=axis.side),
                ),
            )
            layers.append(layer)
        if axis_index == 0:
            for marker in chart.markers:
                if marker.axis == "x":
                    layers.append(
                        alt.Chart(alt.Data(values=[{"x": marker.value}])).mark_rule(
                            color=_color(marker.style_token, colors),
                            strokeDash=[8, 5], strokeWidth=0.8,
                        ).encode(x=x_encoding)
                    )
        if not layers:
            layers.append(
                alt.Chart(alt.Data(values=[])).mark_line().encode(x=x_encoding)
            )
        # Series using the same semantic axis share one scale.  Only the
        # outer gain/phase axis groups are independent.
        axis_layers.append(alt.layer(*layers))
    result = alt.layer(*axis_layers)
    if len(axis_layers) > 1:
        result = result.resolve_scale(y="independent")
    result = result.properties(
        title=_label(chart.title_key),
        height=430 if len(chart.y_axes) > 1 else 280,
        background=colors["figure"],
    )
    return result.configure_axis(
        labelColor=colors["text"], titleColor=colors["text"],
        gridColor=colors["grid"], domainColor=colors["spine"],
        tickColor=colors["spine"],
    ).configure_title(color=colors["text"], anchor="middle").configure_legend(
        labelColor=colors["text"], titleColor=colors["text"],
    ).configure_view(fill=colors["axes"], stroke=colors["spine"])


def render_studio_chart(chart: StudioChart, *, mode="Light", theme="light") -> None:
    """Render one chart from the bundle; no renderer artifact is retained."""
    import streamlit as st

    if str(mode) == "Interactive":
        st.plotly_chart(
            plotly_studio_spec(chart, theme=theme),
            config={"displaylogo": False, "scrollZoom": True},
            width="stretch", key=f"studio_plotly_{chart.chart_id}",
        )
        return
    st.altair_chart(
        altair_studio_chart(chart, theme=theme),
        width="stretch", key=f"studio_altair_{chart.chart_id}",
    )
