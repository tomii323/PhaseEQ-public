"""Compact preview of the exact Group Target sent to PhaseEQ."""
from utils.ui_localization import ui_message, display_text, localized_formatter
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from target_engine.realization import realize_target_definition


def render_target_preview(definition, sample_rate, *, key):
    if definition is None:
        st.info(ui_message('ui.6b8e4a8d8a7b37'))
        return
    try:
        response = realize_target_definition(definition, int(sample_rate))
    except (ValueError, OSError) as exc:
        st.error(ui_message('ui.a4edd118be7a19', p0=f'{exc}'))
        return
    with st.expander(ui_message('ui.2f004c1df92314', p0=f"{definition.get('name', 'Target')}"), expanded=True):
        view = st.segmented_control(ui_message('ui.d939e652e6812e'), ["振幅", "位相"], default="振幅",
                                    key=f"{key}_view", format_func=localized_formatter(str)) or "振幅"
        frequency = np.asarray(response.frequency)
        values = np.asarray(response.gain_db if view == "振幅" else response.phase_deg)
        valid = (frequency > 0) & np.isfinite(values)
        frame = pd.DataFrame({"frequency_hz": frequency[valid], "value": values[valid]})
        if len(frame) > 2048:
            frame = frame.iloc[np.unique(np.linspace(0, len(frame)-1, 2048).astype(int))]
        unit = "Gain [dB]" if view == "振幅" else "Phase [°]"
        chart = alt.Chart(frame).mark_line().encode(
            x=alt.X("frequency_hz:Q", title=ui_message('ui.f11c5c8bac5aeb'), scale=alt.Scale(type="log")),
            y=alt.Y("value:Q", title=unit, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("frequency_hz:Q", title=ui_message('ui.f11c5c8bac5aeb'), format=".1f"), alt.Tooltip("value:Q", title=unit, format=".2f")],
        ).properties(height=210)
        st.altair_chart(chart, width="stretch")
        st.caption(ui_message('ui.f9ec9b586682c8'))
