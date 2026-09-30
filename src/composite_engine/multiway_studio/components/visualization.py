from utils.ui_localization import ui_message, display_text
from utils.ui_language import current_language
import numpy as np
import matplotlib

# Streamlitへ画像として渡すため、OSのGUIバックエンドに依存させない。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import freqz
import os
import warnings
from contextlib import contextmanager
from utils.plot_fonts import get_japanese_font
from utils.ui_semantic_colors import semantic_chart_colors

GROUP_DELAY_DISPLAY_REVISION = 2
TIME_RESPONSE_DISPLAY_MIN_MS = -2.0
TIME_RESPONSE_DISPLAY_MAX_MS = 10.0
GROUP_DELAY_DISPLAY_MIN_MS = -2.0
GROUP_DELAY_DISPLAY_MAX_MS = 10.0
GAIN_Y_MIN_OPTIONS = (-144, -40, -30, -20, -10)
GAIN_Y_MAX_OPTIONS = (10, 20, 30, 40)
DEFAULT_GAIN_Y_MIN_DB = -20
DEFAULT_GAIN_Y_MAX_DB = 20
PHASE_GAIN_MASK_OPTIONS_DB = (-60, -80)
DEFAULT_PHASE_GAIN_MASK_DB = -60
GRAPH_MODE_OPTIONS = ("Light", "Interactive")
DEFAULT_GRAPH_MODE = "Light"
GRAPH_MAX_POINTS_OPTIONS = (1024, 2048, 4096)
DEFAULT_GRAPH_MAX_POINTS = 1024


def studio_figure_revision(inputs):
    """Key a managed, completed figure from its source inputs, never its callbacks."""
    import hashlib
    import pickle
    try:
        return hashlib.sha256(pickle.dumps(
            ('studio-render-v1', matplotlib.__version__, inputs), protocol=5,
        )).hexdigest()
    except (pickle.PickleError, TypeError, AttributeError, ImportError):
        return None


def _source_keyed_figure(function):
    from functools import wraps
    @wraps(function)
    def wrapped(*args, **kwargs):
        figure = function(*args, **kwargs)
        figure._studio_source_revision = studio_figure_revision((function.__name__, args, kwargs))
        return figure
    return wrapped


# Compatibility re-export: the implementation is shared with PhaseEQ.
from response_display import log_spaced_sample_indices, sample_series


@contextmanager
def _limited_figure_lines(figure, max_points, *, preserve_visible_x_samples=False):
    """Temporarily thin rendered series without changing cached/numeric results."""
    saved = []
    limit = max(2, int(max_points))
    for axis in figure.axes:
        x_min, x_max = axis.get_xlim()
        for line in axis.lines:
            x_values = np.asarray(line.get_xdata())
            y_values = np.asarray(line.get_ydata())
            if x_values.size != y_values.size:
                continue
            if axis.get_xscale() == "log":
                visible = np.isfinite(x_values) & (x_values >= max(float(x_min), np.finfo(float).tiny)) & (x_values <= float(x_max))
                if not np.any(visible):
                    continue
                label = axis.get_ylabel().lower()
                phase = "位相" in label or "phase" in label
                sampled_x, sampled_y = sample_series(x_values[visible], y_values[visible], limit, wrapped_phase=phase)
                saved.append((line, x_values, y_values))
                line.set_data(sampled_x, sampled_y)
                continue
            if preserve_visible_x_samples:
                indices = np.flatnonzero((x_values >= x_min) & (x_values <= x_max))
                if indices.size == 0 or indices.size == x_values.size:
                    continue
            elif x_values.size > limit:
                indices = np.linspace(0, x_values.size - 1, limit, dtype=int)
            else:
                continue
            saved.append((line, x_values, y_values))
            line.set_data(x_values[indices], y_values[indices])
    try:
        yield
    finally:
        for line, x_values, y_values in saved:
            line.set_data(x_values, y_values)


@contextmanager
def _localized_figure_text(figure):
    """Translate presentation temporarily; cached source figures stay unchanged."""
    from matplotlib.text import Text
    original = [(artist, artist.get_text()) for artist in figure.findobj(Text)]
    try:
        for artist, text in original:
            artist.set_text(display_text(text))
        yield
    finally:
        for artist, text in original:
            artist.set_text(text)


def render_studio_figure(
    figure, *, mode="Light", interaction="both", clear_figure=False,
    max_points=DEFAULT_GRAPH_MAX_POINTS,
    cache_key=None,
):
    """Render one cached Matplotlib figure in the selected PhaseEQ-style mode."""
    import streamlit as st

    selected_mode = str(mode)
    # The browser can report its color scheme after a cached figure was built.
    # Re-apply the current canvas colors at render time so the lightweight PNG
    # never keeps a stale white background after switching the UI to dark mode.
    runtime_theme = _streamlit_runtime_theme(st)
    if runtime_theme in _PLOT_THEMES:
        _apply_plot_theme(figure, list(figure.axes), theme=runtime_theme)
    display_limit = min(4096, int(max_points))
    with _localized_figure_text(figure), _limited_figure_lines(
        figure,
        display_limit,
        preserve_visible_x_samples=(interaction == "x"),
    ):
        if selected_mode != "Interactive":
            if cache_key is not None:
                import io
                from utils.ui_work_cache import prepared_value
                def png():
                    buffer = io.BytesIO()
                    figure.savefig(buffer, dpi=200, bbox_inches="tight", format="png")
                    return buffer.getvalue()
                image = prepared_value(
                    st.session_state, '_studio_managed_png_v1',
                    (cache_key, current_language(), runtime_theme, display_limit, interaction,
                     dict(matplotlib.rcParams), _PLOT_THEMES),
                    png, max_entries=16, max_bytes=32*1024*1024,
                )
                st.image(image, width="stretch")
                if clear_figure:
                    figure.clear()
                return
            st.pyplot(figure, clear_figure=clear_figure, width="stretch")
            return
        interactive = interactive_studio_figure(figure, interaction=interaction)
        st.plotly_chart(
            interactive,
            config={"displaylogo": False, "scrollZoom": True},
            width="stretch",
        )


def _streamlit_runtime_theme(streamlit_module):
    """Read the browser theme without coupling plot tests to a Streamlit run."""
    try:
        return str(streamlit_module.context.theme.type or "").lower()
    except (AttributeError, TypeError):
        return ""


def interactive_studio_figure(figure, *, interaction="both"):
    """Convert a Studio figure and apply PhaseEQ-compatible zoom constraints."""
    import matplotlib.colors as mpl_colors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    axes = list(figure.axes)
    has_secondary_axis = len(axes) > 1
    interactive = make_subplots(specs=[[{"secondary_y": has_secondary_axis}]])
    dash_styles = {
        "-": "solid", "solid": "solid",
        "--": "dash", "dashed": "dash",
        ":": "dot", "dotted": "dot",
        "-.": "dashdot", "dashdot": "dashdot",
    }
    for axis_index, axis in enumerate(axes[:2]):
        for line in axis.lines:
            x_values = np.asarray(line.get_xdata())
            y_values = np.asarray(line.get_ydata())
            if x_values.size == 0 or x_values.size != y_values.size:
                continue
            color = mpl_colors.to_hex(line.get_color(), keep_alpha=False)
            label = str(line.get_label())
            interactive.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=label if label and not label.startswith("_") else None,
                    showlegend=bool(label and not label.startswith("_")),
                    line={
                        "color": color,
                        "width": float(line.get_linewidth()),
                        "dash": dash_styles.get(str(line.get_linestyle()), "solid"),
                    },
                    opacity=float(line.get_alpha()) if line.get_alpha() is not None else 1.0,
                ),
                secondary_y=bool(axis_index),
            )
    primary_axis = axes[0]
    x_min, x_max = primary_axis.get_xlim()
    x_range = (
        [float(np.log10(x_min)), float(np.log10(x_max))]
        if primary_axis.get_xscale() == "log" and x_min > 0
        else [float(x_min), float(x_max)]
    )
    interactive.update_xaxes(
        title_text=primary_axis.get_xlabel(),
        type="log" if primary_axis.get_xscale() == "log" else "linear",
        range=x_range,
        showgrid=True,
    )
    for axis_index, axis in enumerate(axes[:2]):
        y_min, y_max = axis.get_ylim()
        interactive.update_yaxes(
            title_text=axis.get_ylabel(),
            range=[float(y_min), float(y_max)],
            showgrid=True,
            secondary_y=bool(axis_index),
        )
    interactive.update_layout(
        dragmode="zoom",
        hovermode="x unified",
        margin=dict(l=55, r=35, t=55, b=50),
        title={"text": primary_axis.get_title(), "x": 0.5},
    )
    if interaction == "x":
        interactive.update_xaxes(fixedrange=False)
        interactive.update_yaxes(fixedrange=True)
    return interactive


def set_time_response_xlim(axis):
    """Apply the shared Multiway impulse/step display window in milliseconds."""
    axis.set_xlim(TIME_RESPONSE_DISPLAY_MIN_MS, TIME_RESPONSE_DISPLAY_MAX_MS)


def _visible_time_response_samples(x_values, y_values):
    """Crop a time response to the shared window without dropping any taps."""
    x_array = np.asarray(x_values)
    y_array = np.asarray(y_values)
    visible = (
        (x_array >= TIME_RESPONSE_DISPLAY_MIN_MS)
        & (x_array <= TIME_RESPONSE_DISPLAY_MAX_MS)
    )
    return x_array[visible], y_array[visible]


def set_group_delay_ylim(axis):
    """Apply the shared Multiway group-delay display window in milliseconds."""
    axis.set_ylim(GROUP_DELAY_DISPLAY_MIN_MS, GROUP_DELAY_DISPLAY_MAX_MS)

_PLOT_THEMES = {
    mode: semantic_chart_colors(mode) for mode in ("light", "dark")
}


def _plot_theme(theme):
    return _PLOT_THEMES["dark" if theme == "dark" else "light"]


def _apply_plot_theme(fig, axes, theme="light", font_scale=1.0):
    colors = _plot_theme(theme)
    fig.patch.set_facecolor(colors["figure"])
    for ax in axes:
        ax.set_facecolor(colors["axes"])
        ax.title.set_color(colors["text"])
        ax.title.set_fontsize(13 * font_scale)
        ax.xaxis.label.set_color(colors["text"])
        ax.yaxis.label.set_color(colors["text"])
        ax.xaxis.label.set_fontsize(11.5 * font_scale)
        ax.yaxis.label.set_fontsize(11.5 * font_scale)
        ax.tick_params(
            axis="both",
            colors=colors["text"],
            labelsize=10.5 * font_scale,
        )
        for spine in ax.spines.values():
            spine.set_color(colors["spine"])
        ax.grid(True, which="both", color=colors["grid"], alpha=0.35)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(colors["axes"])
            legend.get_frame().set_edgecolor(colors["spine"])
            for text in legend.get_texts():
                text.set_color(colors["text"])
                text.set_fontsize(10.5 * font_scale)
    return colors

# --- 日本語フォント自動検出 ---
jp_font = get_japanese_font()
if jp_font is not None:
    plt.rcParams["font.family"] = jp_font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
else:
    warnings.warn(
        "日本語対応フォントが見つかりません。グラフ内の日本語が文字化けする場合は、"
        "Noto Sans CJK、IPAexゴシック、メイリオ等をインストールしてください。",
        RuntimeWarning,
        stacklevel=2,
    )

def set_jp_font(ax):
    if jp_font is not None:
        try:
            def apply_font(text):
                size = text.get_fontsize()
                text.set_fontproperties(jp_font)
                text.set_fontsize(size)

            apply_font(ax.title)
            apply_font(ax.xaxis.label)
            apply_font(ax.yaxis.label)
            if ax.get_legend() is not None:
                for t in ax.get_legend().get_texts():
                    apply_font(t)
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                apply_font(t)
        except Exception:
            pass

# --- フィルタ長・中心合わせ（必ず奇数、pad_left/pad_right同数） ---
def center_fir_lengths_old(fir_list):
    lengths = [len(h) for h in fir_list]
    max_len = max(lengths)
    if max_len % 2 == 0:
        max_len += 1  # 必ず奇数長に
    centered_list = []
    for h in fir_list:
        pad_total = max_len - len(h)
        pad_left = pad_total // 2
        pad_right = pad_total // 2
        h_centered = np.pad(h, (pad_left, pad_right), mode='constant')
        centered_list.append(h_centered)
    return centered_list

def center_fir_lengths(fir_list):
    """
    FIRフィルタ群を最大長に中心揃えパディングする
    Args:
        fir_list: list of 1D np.ndarray (各帯域FIR)
    Returns:
        list of 1D np.ndarray（全てmax長で中央揃え）
    """
    lengths = [len(h) for h in fir_list]
    max_len = max(lengths)
    return [
        np.pad(h, ((max_len - len(h)) // 2, (max_len - len(h) + 1) // 2))
        for h in fir_list
    ]

def match_fir_length(fir, target_length):
    """
    FIRの長さをtarget_lengthに揃える（pad or crop, 両端均等, 奇数対応）
    Args:
        fir: 1D np.ndarray
        target_length: int
    Returns:
        1D np.ndarray
    """
    current = len(fir)
    if current < target_length:
        pad = (target_length - current) // 2
        return np.pad(fir, (pad, target_length - current - pad))
    elif current > target_length:
        start = (current - target_length) // 2
        return fir[start:start + target_length]
    return fir


@_source_keyed_figure
def plot_impulse_response(
    fir_dict, fs=48000, orig_lens=None, overlay_sum=False, theme="light",
    sum_groups=None,
):
    """
    帯域ごとのインパルス応答（中央揃え）を表示。
    overlay_sum=True のとき、全帯域の合成波も太線で重ね描き。
    x軸は中心=0。
    """
    fir_list = list(fir_dict.values())
    band_list = list(fir_dict.keys())

    max_len = max((len(fir) for fir in fir_list), default=1)
    center = (max_len - 1) // 2
    x = np.arange(-center, max_len - center) / float(fs) * 1000.0

    fig, ax = plt.subplots(figsize=(8, 3))
    for band, fir in zip(band_list, fir_list):
        label = f"{band} ({orig_lens[band]} taps)" if orig_lens else band
        color = _plot_theme(theme)["bands"][len(ax.lines) % 5]
        visible_x, visible_fir = _visible_time_response_samples(x, fir)
        ax.plot(visible_x, visible_fir, label=label, color=color)

    grouped_sums = sum_groups or (
        {"Sum (All Bands)": tuple(fir_dict)} if overlay_sum and len(fir_list) >= 2 else {}
    )
    for sum_label, keys in grouped_sums.items():
        selected = [fir_dict[key] for key in keys if key in fir_dict]
        if not selected:
            continue
        combined = np.sum(selected, axis=0)
        visible_x, visible_combined = _visible_time_response_samples(x, combined)
        ax.plot(
            visible_x,
            visible_combined,
            label=sum_label,
            color=_plot_theme(theme)["sum"],
            linewidth=2,
            alpha=0.9,
        )

    ax.set_title(ui_message('ui.f165a57d1c1366'))
    ax.set_xlabel(ui_message('ui.78f2c5fb6fe469'))
    ax.set_ylabel(ui_message('ui.f18f28ad51c0cd'))
    set_time_response_xlim(ax)
    ax.axvline(
        0,
        color=_plot_theme(theme)["center"],
        linestyle='--',
        linewidth=0.8,
    )
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    _apply_plot_theme(fig, [ax], theme)
    set_jp_font(ax)
    fig.tight_layout()
    fig.canvas.draw()
    return fig


@_source_keyed_figure
def plot_step_response_centered(
    fir_dict, fs=48000, orig_lens=None, overlay_sum=False, theme="light",
    sum_groups=None,
):
    """
    帯域ごとのステップ応答（中央揃え）を表示。
    overlay_sum=True のとき、合成波のステップ応答も太線で重ね描き。
    x軸は中心=0。
    """
    fir_list = list(fir_dict.values())
    band_list = list(fir_dict.keys())

    max_len = max((len(fir) for fir in fir_list), default=1)
    center = (max_len - 1) // 2
    x = np.arange(-center, max_len - center) / float(fs) * 1000.0

    fig, ax = plt.subplots(figsize=(8, 3))
    for band, fir in zip(band_list, fir_list):
        step = np.cumsum(fir)
        label = f"{band} ({orig_lens[band]} taps)" if orig_lens else band
        color = _plot_theme(theme)["bands"][len(ax.lines) % 5]
        visible_x, visible_step = _visible_time_response_samples(x, step)
        ax.plot(visible_x, visible_step, label=label, color=color)

    grouped_sums = sum_groups or (
        {"Sum (All Bands)": tuple(fir_dict)} if overlay_sum and len(fir_list) >= 2 else {}
    )
    for sum_label, keys in grouped_sums.items():
        selected = [fir_dict[key] for key in keys if key in fir_dict]
        if not selected:
            continue
        combined = np.sum(selected, axis=0)
        combined_step = np.cumsum(combined)
        visible_x, visible_step = _visible_time_response_samples(x, combined_step)
        ax.plot(
            visible_x,
            visible_step,
            label=sum_label,
            color=_plot_theme(theme)["sum"],
            linewidth=2,
            alpha=0.9,
        )

    ax.set_title(ui_message('ui.36e8e9a9bd5e3f'))
    ax.set_xlabel(ui_message('ui.78f2c5fb6fe469'))
    ax.set_ylabel(ui_message('ui.e732bb5b771580'))
    set_time_response_xlim(ax)
    ax.axvline(
        0,
        color=_plot_theme(theme)["center"],
        linestyle='--',
        linewidth=0.8,
    )
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    _apply_plot_theme(fig, [ax], theme)
    set_jp_font(ax)
    fig.tight_layout()
    return fig


def plot_group_delay_centered(
    fir_dict, fs=48000, orig_lens=None, theme="light"
):
    fir_list = list(fir_dict.values())
    band_list = list(fir_dict.keys())
    fir_centered = center_fir_lengths(fir_list)
    fig, ax = plt.subplots(figsize=(8, 3))
    w_rad = np.linspace(0, np.pi, 4096)
    w_hz = w_rad * fs / (2 * np.pi)
    max_len = len(fir_centered[0])
    center_delay = (max_len - 1) / 2
    for index, (band, fir) in enumerate(zip(band_list, fir_centered)):
        H = freqz(fir, worN=w_rad)[1]
        mag = 20 * np.log10(np.abs(H) + 1e-12)
        group_delay_ms = (
            -np.diff(np.unwrap(np.angle(H))) / np.diff(w_rad)
            - center_delay
        ) / float(fs) * 1000.0
        w_c = w_hz[1:]
        mag_c = mag[1:]
        valid = (
            (mag_c > -60.0)
            & (w_c >= 10)
            & (w_c <= fs / 2)
        )
        label = f"{band} ({orig_lens[band]} taps)" if orig_lens else band
        color = _plot_theme(theme)["bands"][index % 5]
        ax.plot(w_c[valid], group_delay_ms[valid], label=label, color=color)
    ax.set_xscale('log')
    ax.set_xlim(10, fs/2)
    ax.set_title(ui_message('ui.909cfbf5960044'))
    ax.set_xlabel(ui_message('ui.f11c5c8bac5aeb'))
    ax.set_ylabel(ui_message('ui.1bb9aa84fccb4c'))
    set_group_delay_ylim(ax)
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    _apply_plot_theme(fig, [ax], theme)
    set_jp_font(ax)
    fig.tight_layout()
    return fig


@_source_keyed_figure
def plot_group_delay_from_responses(
    frequency_hz, responses, fs=48000, theme="light", gain_mask_db=-60.0,
    sum_groups=None,
):
    """Plot relative delay directly from center-referenced realized responses."""
    frequency = np.asarray(frequency_hz, dtype=float)
    w_rad = 2.0 * np.pi * frequency / float(fs)
    fig, ax = plt.subplots(figsize=(8, 3))
    for index, (band, response) in enumerate(responses.items()):
        values = np.asarray(response, dtype=np.complex128)
        if values.size != frequency.size or values.size < 2:
            continue
        group_delay_ms = (
            -np.diff(np.unwrap(np.angle(values))) / np.diff(w_rad)
        ) / float(fs) * 1000.0
        group_delay_ms[np.abs(group_delay_ms) < 1e-9] = 0.0
        frequency_center = frequency[1:]
        magnitude_db = 20.0 * np.log10(np.abs(values[1:]) + 1e-12)
        valid = (
            np.isfinite(group_delay_ms)
            & (magnitude_db > float(gain_mask_db))
            & (frequency_center >= 10.0)
            & (frequency_center <= fs / 2.0)
        )
        color = _plot_theme(theme)["bands"][index % 5]
        ax.plot(
            frequency_center[valid], group_delay_ms[valid],
            label=band, color=color,
        )
    for sum_label, keys in (sum_groups or {}).items():
        selected = [np.asarray(responses[key], dtype=np.complex128) for key in keys if key in responses]
        if not selected:
            continue
        values = np.sum(selected, axis=0)
        group_delay_ms = (-np.diff(np.unwrap(np.angle(values))) / np.diff(w_rad)) / float(fs) * 1000.0
        magnitude_db = 20.0 * np.log10(np.abs(values[1:]) + 1e-12)
        valid = (
            np.isfinite(group_delay_ms)
            & (magnitude_db > float(gain_mask_db))
            & (frequency[1:] >= 10.0)
            & (frequency[1:] <= fs / 2.0)
        )
        ax.plot(
            frequency[1:][valid], group_delay_ms[valid], label=sum_label,
            color=_plot_theme(theme)["sum"], linewidth=2.0,
        )
    ax.set_xscale("log")
    ax.set_xlim(10, fs / 2)
    ax.set_title(ui_message('ui.909cfbf5960044'))
    ax.set_xlabel(ui_message('ui.f11c5c8bac5aeb'))
    ax.set_ylabel(ui_message('ui.1bb9aa84fccb4c'))
    set_group_delay_ylim(ax)
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    _apply_plot_theme(fig, [ax], theme)
    set_jp_font(ax)
    fig.tight_layout()
    return fig


@_source_keyed_figure
def plot_sum_freq_from_responses(
    frequency_hz, responses, *, fs=48000, db_min=-30, db_max=20,
    phase_gain_mask_db=-60, theme="light", sum_groups=None,
):
    """Plot the realized complex sum without an even-length IR round trip."""
    frequency = np.asarray(frequency_hz, dtype=float)
    frequency_mask = (frequency >= 10.0) & (frequency <= float(fs) / 2.0)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    grouped_sums = sum_groups or {"Sum (All Bands)": tuple(responses)}
    for index, (sum_label, keys) in enumerate(grouped_sums.items()):
        realized = [np.asarray(responses[key], dtype=np.complex128) for key in keys if key in responses]
        if not realized:
            continue
        combined = np.sum(tuple(realized), axis=0)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(combined), 1e-12))
        phase_deg = np.rad2deg(np.angle(combined))
        phase_mask = frequency_mask & (magnitude_db >= float(phase_gain_mask_db))
        color = _plot_theme(theme)["sum"] if index == 0 else _plot_theme(theme)["bands"][3]
        ax1.plot(
            frequency[frequency_mask], magnitude_db[frequency_mask],
            label=sum_label, color=color, linewidth=2.0,
        )
        ax2.plot(
            frequency[frequency_mask], np.where(phase_mask, phase_deg, np.nan)[frequency_mask],
            linestyle="dotted", color=color, alpha=0.7,
        )
    ax1.set_xscale("log")
    ax1.set_xlim(10, float(fs) / 2.0)
    ax1.set_ylim(db_min, db_max)
    ax1.set_xlabel(ui_message('ui.f11c5c8bac5aeb'))
    ax1.set_ylabel(ui_message('ui.390fb3129a3e48'))
    ax2.set_ylabel(ui_message('ui.7fd2691857ae18'))
    ax2.set_ylim(-180, 180)
    if ax1.get_legend_handles_labels()[1]:
        ax1.legend(loc="upper right")
    ax1.set_title(ui_message('ui.9fb0cd78c608c2'))
    _apply_plot_theme(fig, [ax1, ax2], theme, font_scale=1.2)
    ax2.grid(False)
    set_jp_font(ax1)
    set_jp_font(ax2)
    fig.tight_layout()
    return fig


@_source_keyed_figure
def plot_sum_impulse_centered(fir_dict, fs=48000, theme="light", sum_groups=None):
    fir_list = list(fir_dict.values())
    fir_centered = center_fir_lengths(fir_list) if fir_list else []
    centered_by_name = dict(zip(fir_dict, fir_centered))
    fig, ax = plt.subplots(figsize=(8, 3))
    N = len(fir_centered[0]) if fir_centered else 1
    # Match individual IR/step axes; (-N)//2 shifts odd lengths one sample early.
    center = (N - 1) // 2
    x = (np.arange(N) - center) / float(fs) * 1000.0
    grouped_sums = sum_groups or {"Sum (All Bands)": tuple(fir_dict)}
    for index, (sum_label, keys) in enumerate(grouped_sums.items()):
        selected = [centered_by_name[key] for key in keys if key in centered_by_name]
        if not selected:
            continue
        combined = np.sum(selected, axis=0)
        color = _plot_theme(theme)["sum"] if index == 0 else _plot_theme(theme)["bands"][3]
        ax.plot(x, combined, label=sum_label, color=color, linewidth=2.0)
    ax.set_title(ui_message('ui.ab0248fb56262b'))
    ax.set_xlabel(ui_message('ui.78f2c5fb6fe469'))
    ax.set_ylabel(ui_message('ui.f18f28ad51c0cd'))
    set_time_response_xlim(ax)
    if ax.get_legend_handles_labels()[1]:
        ax.legend()
    _apply_plot_theme(fig, [ax], theme)
    set_jp_font(ax)
    fig.tight_layout()
    return fig

def plot_sum_freq_centered(
    fir_dict, fs=48000, orig_lens=None, db_min=-30, db_max=20,
    phase_gain_mask_db=-60, theme="light",
):
    fir_list = list(fir_dict.values())
    band_list = list(fir_dict.keys())
    fir_centered = center_fir_lengths(fir_list)
    combined = np.sum(fir_centered, axis=0)
    w_rad = np.linspace(0, np.pi, 4096)
    w_hz = w_rad * fs / (2 * np.pi)
    mask = w_hz <= fs / 2
    max_len = len(combined)
    delay = (max_len - 1) / 2
    H_sum = freqz(combined, worN=w_rad)[1]
    mag_sum = 20 * np.log10(np.abs(H_sum) + 1e-12)
    phase_sum = np.unwrap(np.angle(H_sum))
    phase_sum_deg = np.rad2deg(phase_sum + w_rad * delay)
    phase_sum_deg = (phase_sum_deg + 180) % 360 - 180
    valid = (mag_sum >= float(phase_gain_mask_db)) & mask
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    sum_color = _plot_theme(theme)["sum"]
    ax1.plot(w_hz[mask], mag_sum[mask], label="Sum (All Bands)", color=sum_color)
    ax2.plot(
        w_hz[mask],
        np.where(valid, phase_sum_deg, np.nan)[mask],
        linestyle='dotted',
        color=sum_color,
        alpha=0.7,
    )
    ax1.set_xscale('log')
    ax1.set_xlim(10, fs / 2)
    ax1.set_ylim(db_min, db_max)
    ax1.set_xlabel(ui_message('ui.f11c5c8bac5aeb'))
    ax1.set_ylabel(ui_message('ui.390fb3129a3e48'))
    ax2.set_ylabel(ui_message('ui.8abcf431a76fac'))
    ax2.set_ylim(-180, 180)
    ax1.legend(loc="upper right")
    ax1.set_title(ui_message('ui.178068dc43f1e3'))
    _apply_plot_theme(fig, [ax1, ax2], theme, font_scale=1.2)
    ax2.grid(False)
    set_jp_font(ax1)
    set_jp_font(ax2)
    plt.tight_layout()
    return fig

def plot_frequency_response_with_sum_centered(
    fir_dict,
    fs=48000,
    orig_lens=None,
    show_sum=True,
    db_min=-30,
    db_max=20,
    phase_gain_mask_db=-60,
    theme="light",
):
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    color_map = _plot_theme(theme)["bands"]
    w_rad = np.linspace(0, np.pi, 4096)
    w_hz = w_rad * fs / (2 * np.pi)
    mask = w_hz <= fs / 2

    band_list = list(fir_dict.keys())
    fir_list_raw = list(fir_dict.values())
    fir_list = center_fir_lengths(fir_list_raw)
    max_len = len(fir_list[0])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    delay = (max_len - 1) / 2  # 全帯域で共通

    # 各帯域 FIR
    for i, (band, fir, fir_raw) in enumerate(zip(band_list, fir_list, fir_list_raw)):
        orig_len = orig_lens[band] if orig_lens else len(fir_raw)
        H = freqz(fir, worN=w_rad)[1]
        mag = 20 * np.log10(np.abs(H) + 1e-12)
        phase = np.unwrap(np.angle(H))
        phase_deg = np.rad2deg(phase + w_rad * delay)
        phase_deg = (phase_deg + 180) % 360 - 180
        valid = (mag >= float(phase_gain_mask_db)) & mask
        label = f"{band} ({orig_len} taps)"
        color = color_map[i % len(color_map)]
        ax1.plot(w_hz[mask], mag[mask], label=label, color=color)
        ax2.plot(w_hz[mask], np.where(valid, phase_deg, np.nan)[mask], linestyle='dotted', color=color, alpha=0.7)

    # 合成波
    if show_sum and len(fir_list) >= 2:
        combined = np.sum(fir_list, axis=0)
        H_sum = freqz(combined, worN=w_rad)[1]
        mag_sum = 20 * np.log10(np.abs(H_sum) + 1e-12)
        phase_sum = np.unwrap(np.angle(H_sum))
        phase_sum_deg = np.rad2deg(phase_sum + w_rad * delay)
        phase_sum_deg = (phase_sum_deg + 180) % 360 - 180
        valid = (mag_sum >= float(phase_gain_mask_db)) & mask
        sum_color = _plot_theme(theme)["sum"]
        ax1.plot(
            w_hz[mask],
            mag_sum[mask],
            label="Sum (All Bands)",
            color=sum_color,
            linewidth=2,
        )
        ax2.plot(
            w_hz[mask],
            np.where(valid, phase_sum_deg, np.nan)[mask],
            linestyle='dotted',
            color=sum_color,
            linewidth=2,
            alpha=0.7,
        )

    ax1.set_xscale('log')
    ax1.set_xlim(10, fs / 2)
    ax1.set_ylim(db_min, db_max)
    ax1.set_xlabel(ui_message('ui.f11c5c8bac5aeb'))
    ax1.set_ylabel(ui_message('ui.390fb3129a3e48'))
    ax2.set_ylabel(ui_message('ui.8abcf431a76fac'))
    ax2.set_ylim(-180, 180)
    ax1.legend(loc="upper right")
    ax1.set_title(ui_message('ui.b7487e9da3073c'))
    _apply_plot_theme(fig, [ax1, ax2], theme, font_scale=1.2)
    ax2.grid(False)
    set_jp_font(ax1)
    set_jp_font(ax2)
    plt.tight_layout()
    return fig


@_source_keyed_figure
def plot_crossover_design_response(
    frequency_hz,
    responses,
    way_sum,
    *,
    db_min=-30,
    db_max=20,
    phase_gain_mask_db=-60,
    theme="light",
    phaseeq_responses=None,
    speaker_responses=None,
    way_sums=None,
    baffle_response=None,
    baffle_label=None,
):
    """Plot realized Studio Way responses, including assigned response stages."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    mask = frequency_hz > 0.0
    color_map = _plot_theme(theme)["bands"]
    for index, (way, response) in enumerate(responses.items()):
        response = np.asarray(response, dtype=np.complex128)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
        phase_deg = np.rad2deg(np.angle(response))
        visible_phase = mask & (magnitude_db >= float(phase_gain_mask_db))
        color = color_map[index % len(color_map)]
        ax1.plot(frequency_hz[mask], magnitude_db[mask], label=way, color=color)
        ax2.plot(
            frequency_hz[mask], np.where(visible_phase, phase_deg, np.nan)[mask],
            linestyle="dotted", color=color, alpha=0.7,
        )
        phaseeq_response = (phaseeq_responses or {}).get(way)
        if phaseeq_response is not None:
            phaseeq_values = np.asarray(phaseeq_response, dtype=np.complex128)
            phaseeq_db = 20.0 * np.log10(
                np.maximum(np.abs(phaseeq_values), 1e-12)
            )
            phaseeq_phase_deg = np.rad2deg(np.angle(phaseeq_values))
            phaseeq_phase_mask = mask & (
                phaseeq_db >= float(phase_gain_mask_db)
            )
            ax1.plot(
                frequency_hz[mask], phaseeq_db[mask],
                label=f"{way} · PhaseEQ EQ",
                color=_plot_theme(theme)["input"],
                linestyle="--", linewidth=1.8, alpha=0.95,
            )
            # A phase-only FIR is intentionally almost flat in Magnitude.  Its
            # effect must still be inspectable instead of looking disconnected.
            ax2.plot(
                frequency_hz[mask],
                np.where(phaseeq_phase_mask, phaseeq_phase_deg, np.nan)[mask],
                label=f"{way} · PhaseEQ EQ · Phase",
                color=_plot_theme(theme)["correction"],
                linestyle="dashdot", linewidth=1.8, alpha=0.95,
            )
        speaker_response = (speaker_responses or {}).get(way)
        if speaker_response is not None:
            speaker_db = 20.0 * np.log10(
                np.maximum(np.abs(np.asarray(speaker_response)), 1e-12)
            )
            ax1.plot(
                frequency_hz[mask], speaker_db[mask],
                label=f"{way} · Speaker",
                color=color, linestyle=":", linewidth=1.5, alpha=0.9,
            )
    if baffle_response is not None:
        baffle_values = np.asarray(baffle_response, dtype=np.complex128)
        if baffle_values.shape == frequency_hz.shape:
            baffle_db = 20.0 * np.log10(np.maximum(np.abs(baffle_values), 1e-12))
            baffle_phase = np.rad2deg(np.angle(baffle_values))
            baffle_phase_mask = mask & (baffle_db >= float(phase_gain_mask_db))
            target_color = _plot_theme(theme)["baffle"]
            ax1.plot(
                frequency_hz[mask], baffle_db[mask],
                label=baffle_label or "Baffle compensation",
                color=target_color, linestyle="-.", linewidth=1.7,
            )
            ax2.plot(
                frequency_hz[mask], np.where(baffle_phase_mask, baffle_phase, np.nan)[mask],
                color=target_color, linestyle=":", alpha=0.75,
            )
    sums = way_sums or {"Design Way Sum": np.asarray(way_sum, dtype=np.complex128)}
    for index, (sum_label, sum_values) in enumerate(sums.items()):
        sum_response = np.asarray(sum_values, dtype=np.complex128)
        sum_db = 20.0 * np.log10(np.maximum(np.abs(sum_response), 1e-12))
        sum_phase_deg = np.rad2deg(np.angle(sum_response))
        sum_color = _plot_theme(theme)["sum"] if index == 0 else _plot_theme(theme)["bands"][3]
        ax1.plot(
            frequency_hz[mask], sum_db[mask], label=sum_label,
            color=sum_color, linewidth=2,
        )
        sum_phase_mask = mask & (sum_db >= float(phase_gain_mask_db))
        ax2.plot(
            frequency_hz[mask], np.where(sum_phase_mask, sum_phase_deg, np.nan)[mask],
            linestyle="dotted", color=sum_color, linewidth=2, alpha=0.7,
        )
    ax1.set_xscale("log")
    ax1.set_xlim(10, float(frequency_hz[-1]))
    ax1.set_ylim(db_min, db_max)
    ax1.set_xlabel(ui_message('ui.f11c5c8bac5aeb'))
    ax1.set_ylabel(ui_message('ui.08624e6af38efb'))
    ax2.set_ylabel(ui_message('ui.70b2ffec8b1a1c'))
    ax2.set_ylim(-180, 180)
    gain_handles, gain_labels = ax1.get_legend_handles_labels()
    phase_handles, phase_labels = ax2.get_legend_handles_labels()
    ax1.legend(
        [*gain_handles, *phase_handles], [*gain_labels, *phase_labels],
        loc="upper right",
    )
    ax1.set_title(ui_message('ui.476dbeb78d9836'))
    _apply_plot_theme(fig, [ax1, ax2], theme, font_scale=1.2)
    ax2.grid(False)
    set_jp_font(ax1)
    set_jp_font(ax2)
    fig.tight_layout()
    return fig
