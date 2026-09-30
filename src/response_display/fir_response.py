"""Display the realized final FIR, never redesign or apply an output window here."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from composite_engine.fft import fir_complex_response
from response_display import sample_series
from response_display.cache import content_cached


@content_cached(revision="final-fir-response-v1", max_entries=32, max_bytes=32*1024*1024)
def final_fir_response(fir, sample_rate):
    values = np.asarray(fir, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("FIR coefficients must be finite and nonempty")
    if int(sample_rate) < 1:
        raise ValueError("sample_rate must be positive")
    # About 1 Hz; odd length and no truncation of long exported coefficients.
    n = max(int(sample_rate), values.size) | 1
    frequency, response = fir_complex_response(values, int(sample_rate), n,
                                               center_position=(values.size-1)/2)
    return frequency, 20*np.log10(np.maximum(np.abs(response), 1e-15)), np.angle(response, deg=True)


def final_fir_response_figure(fir, sample_rate, *, colors, max_points=4096,
                              phase_mask_db=-80, gain_range=(-144, 10)):
    frequency, gain, phase = final_fir_response(fir, sample_rate)
    visible = frequency >= 2
    frequency, gain, phase = frequency[visible], gain[visible], phase[visible].copy()
    phase[gain < phase_mask_db] = np.nan
    xf, yf = sample_series(frequency, gain, max_points)
    xp, yp = sample_series(frequency, phase, max_points, wrapped_phase=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.10)
    for row, label, x, y, color in (
        (1, "Gain", xf, yf, colors["result"]),
        (2, "Phase", xp, yp, colors["phase_error"]),
    ):
        fig.add_trace(go.Scatter(x=x, y=y, name=label, mode="lines", connectgaps=False,
                                 line=dict(color=color, width=2)), row=row, col=1)
    fig.update_layout(template="none", height=560, showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=colors["text"], size=16), dragmode="zoom",
                      margin=dict(l=70,r=25,t=20,b=55),
                      uirevision=f"final-fir-{len(fir)}-{sample_rate}-{gain_range}-{phase_mask_db}")
    fig.update_xaxes(type="log", range=[np.log10(2), np.log10(sample_rate/2)],
                     gridcolor=colors["grid"], fixedrange=False)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_yaxes(gridcolor=colors["grid"], fixedrange=False, zeroline=False)
    fig.update_yaxes(title_text="Gain (dB)", range=list(gain_range), row=1, col=1)
    fig.update_yaxes(title_text="Phase (deg)", range=[-180, 180], row=2, col=1)
    return fig
