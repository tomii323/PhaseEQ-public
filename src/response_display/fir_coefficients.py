"""Coefficient inspection, independent of frequency-response analysis."""
import numpy as np
import plotly.graph_objects as go
from fir_output_window import output_window


def coefficient_figure(fir, *, window_enabled=False, colors=None, height=360):
    values = np.asarray(fir, dtype=float)
    window = output_window(values.size, window_enabled)
    palette = {"text": "#454545", "chart_grid": "#dddddd", "border": "#999999",
               "series_delta": "#d58b36", "series_result": "#438b70",
               "series_gain_error": "#688fbc"} | (colors or {})
    x = np.arange(values.size)
    fig = go.Figure()
    for name, y, color, axis in (
        ("linear value", values, palette["series_result"], "y"),
        ("dB re 1.0", 20*np.log10(np.maximum(np.abs(values), 1e-15)), palette["series_gain_error"], "y2"),
        ("window function", window, palette["series_delta"], "y"),
    ):
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", yaxis=axis,
                                line=dict(color=color, width=2)))
    axis = dict(showline=True, linecolor=palette["border"], gridcolor=palette["chart_grid"],
                zeroline=False, fixedrange=False)
    fig.update_layout(
        title="FIR Filter cofs", template="none", height=height+110,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"], size=16), margin=dict(l=65,r=80,t=55,b=75),
        xaxis=axis | dict(title="Tap (samples)", range=[0, max(values.size-1, 1)]),
        yaxis=axis | dict(title="linear value / window function"),
        yaxis2=axis | dict(title="dB re 1.0", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=-.22), dragmode="zoom",
        uirevision=f"fir-cofs-{values.size}",
    )
    return fig
