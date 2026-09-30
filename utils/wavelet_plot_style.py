from __future__ import annotations


WAVELET_ABSOLUTE_COLORS = (
    "#00020a",
    "#020b2f",
    "#053a83",
    "#00a6d6",
    "#f2e85e",
    "#f47b20",
    "#b40000",
)
WAVELET_PAPER_BACKGROUND = "#10161a"
WAVELET_PLOT_BACKGROUND = "#071019"
WAVELET_FOREGROUND = "#d8dde2"
WAVELET_TITLE_COLOR = "#f0f3f5"
WAVELET_BORDER_COLOR = "#69727a"
WAVELET_PEAK_TRACE_COLOR = "#fff3b0"
WAVELET_CENTROID_TRACE_COLOR = "#9de8ff"
WAVELET_WINDOW_TRACE_COLOR = "#b5f27a"


def plotly_absolute_colorscale() -> list[list[float | str]]:
    last = len(WAVELET_ABSOLUTE_COLORS) - 1
    return [[index / last, color] for index, color in enumerate(WAVELET_ABSOLUTE_COLORS)]


def wavelet_color_domain(dynamic_range_db: float, mode: str) -> tuple[float, float]:
    if str(mode) == "Source - Target":
        maximum = max(8.0, abs(float(dynamic_range_db)) / 2.0)
        return -maximum, maximum
    return -abs(float(dynamic_range_db)), 0.0
