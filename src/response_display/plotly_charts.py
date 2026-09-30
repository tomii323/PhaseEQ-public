"""Interactive rendering of prepared PhaseEQ display data; no DSP or resampling."""
import math
import json

import plotly.graph_objects as go


PLOTLY_CONFIG = dict(scrollZoom=True, displaylogo=False, responsive=True,
                     doubleClick="reset", modeBarButtonsToRemove=["select2d", "lasso2d"])


def _rgba(color, alpha):
    value = color.lstrip("#")
    return f"rgba({int(value[:2], 16)},{int(value[2:4], 16)},{int(value[4:6], 16)},{alpha})"


def series_figure(frame, *, x_name, x_domain, y_domain, colors, series_colors,
                  styles, height, highlight_series=None, x_zoom_only=False,
                  markers=None, bands=None, secondary_domain=None):
    """Keep segment boundaries and all supplied samples, including fill gaps."""
    fig = go.Figure()
    names = list(dict.fromkeys(frame["series"].tolist()))
    # Separate fill polygons prevent bridging phase wraps or masked intervals.
    if highlight_series in names:
        for _, part in frame[frame.series == highlight_series].groupby("segment", sort=False):
            fig.add_trace(go.Scatter(
                x=part[x_name].tolist(), y=part.value.tolist(), mode="lines",
                line=dict(width=0, color=colors["series_delta"]), fill="tozeroy",
                fillcolor=_rgba(colors["series_delta"], .18), hoverinfo="skip",
                name=highlight_series, legendgroup=highlight_series, showlegend=False,
                connectgaps=False))
    for name in names:
        xs, ys = [], []
        for _, part in frame[frame.series == name].groupby("segment", sort=False):
            if xs:
                xs.append(None)
                ys.append(None)
            xs.extend(part[x_name].tolist())
            ys.extend(part.value.tolist())
        dash, width = styles.get(name, ([], 2.0))
        highlighted = name == highlight_series
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=name, legendgroup=name, mode="lines", connectgaps=False,
            showlegend=name != "Current page delta",
            line=dict(color=colors["series_delta"] if highlighted else series_colors[name],
                      width=3.0 if highlighted else width,
                      dash="solid" if highlighted or not dash else ",".join(f"{n}px" for n in dash)),
            opacity=.96 if highlighted else 1.0,
            yaxis="y2" if secondary_domain is not None and name == "Phase Error" else "y",
            hovertemplate="%{x:.0f}<br>%{y:.3f}<extra>%{fullData.name}</extra>"))
    axis = dict(showline=True, mirror=True, linecolor=colors["border"],
                gridcolor=colors["chart_grid"], tickcolor=colors["border"],
                tickfont=dict(size=16, color=colors["text"]), zeroline=False,
                title=dict(font=dict(size=17, color=colors["muted"])), automargin=True)
    frequency = x_name == "frequency"
    xr = [math.log10(v) for v in x_domain] if frequency else list(x_domain)
    fig.update_layout(
        template="none", height=height + 110, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="sans-serif", size=16, color=colors["text"]),
        margin=dict(l=65, r=30, t=45, b=65), dragmode="zoom", hovermode="closest",
        legend=dict(x=1.02, y=1, font=dict(size=16, color=colors["text"])),
        xaxis=axis | dict(type="log" if frequency else "linear", range=xr,
                         tickformat=".0f", fixedrange=False),
        yaxis=axis | dict(range=y_domain, tickformat=".2f", fixedrange=x_zoom_only),
        uirevision=json.dumps([x_name, x_domain, y_domain, secondary_domain, names]),
    )
    fig.update_xaxes(title_text="Frequency (Hz)" if frequency else x_name)
    if frequency:
        fig.update_xaxes(minor=dict(showgrid=True, gridcolor=colors["chart_grid"]))
    if secondary_domain is not None:
        fig.update_layout(yaxis2=axis | dict(overlaying="y", side="right", mirror=False,
                                           showgrid=False, range=secondary_domain, tickformat=".2f"))
        fig.update_yaxes(title_text="Gain Error (dB)")
        fig.layout.yaxis2.title.text = "Phase Error (deg)"
        fig.layout.legend.x = 1.15
    if frequency:
        for band in bands or []:
            color = "#6299ad" if band["label"] == "P0補完" else "#a891b7"
            fig.add_vrect(x0=band["start"], x1=band["end"], fillcolor=color,
                          opacity=.12, line_width=0, layer="below")
            fig.add_vline(x=band["start"], line_color=color, line_dash="3px,3px", opacity=.5)
        for marker in markers or []:
            x = float(marker.get("frequency", float("nan")))
            if not math.isfinite(x) or x <= 0 or not marker.get("label"):
                continue
            fig.add_vline(x=x, line_color=colors["muted"], line_dash="5px,4px",
                          line_width=1.5, opacity=.82)
            fig.add_annotation(x=math.log10(x), y=1, xref="x", yref="paper",
                               text=marker["label"], showarrow=False, xanchor="left",
                               yanchor="top", xshift=5, font=dict(size=12, color=colors["muted"]))
    return fig
