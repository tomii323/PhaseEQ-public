"""Per-channel FIR capability control; DSP and persistence live outside the UI."""
import streamlit as st
from utils.ui_localization import ui_message


def render_channel_dc_gain(prefix, *, fir_enabled):
    key = f'{prefix}_dc_gain_normalize'
    st.session_state.setdefault(key, False)
    return st.checkbox(
        ui_message('ui.e734cb6705fef7'), key=key, disabled=not fir_enabled,
        help=ui_message('ui.c34df0cad4fef6'),
    )
