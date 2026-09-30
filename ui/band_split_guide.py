"""Design guidance for PhaseEQ's output boundaries, separate from DSP application."""
from utils.ui_localization import ui_message, display_text
import streamlit as st

from composite_engine.kaiser_help import KAISER_SLOPE_GUIDE
from composite_engine.multiway_studio.utils.kaiser import (
    kaiser_attenuation_usage_hint, kaiser_beta_to_attenuation_db,
)
from crossover_engine.lr2_taps import automatic_lr2_taps
from phase_fir_designer.config import LINEAR_FIR_CYCLES_MIN, LINEAR_FIR_CYCLES_MAX
from phase_fir_designer.linear_fir import multiway_kaiser_taps


def render_band_split_guide(*, method, sample_rate, crossover_hz, cycles, beta,
                            overlap_oct=0.0, current_taps=None, current_auto_taps=False):
    if method in {"kaiser", "Kaiser FIR"}:
        attenuation = kaiser_beta_to_attenuation_db(beta)
        taps = multiway_kaiser_taps(sample_rate, crossover_hz, cycles)
        st.caption(
            ui_message('ui.7e7b80474e9282', p0=f'{attenuation:.0f}', p1=display_text(kaiser_attenuation_usage_hint(attenuation)), p2=f'{taps:,}')
            + (ui_message('ui.5fa0cea795b7ea', p0=f'{current_taps:,}') if current_taps is not None else ""),
            help=ui_message('ui.b2120eb112db24'),
        )
        with st.popover(ui_message('ui.cecdffbccd548d')):
            # Studio's input range is wider; keep the shared table without
            # promising that every reference value can be entered here.
            lines = [line for line in display_text(KAISER_SLOPE_GUIDE).splitlines()
                     if not line.startswith(("周期数は2.7", "Cycles accept direct entry"))]
            st.markdown("\n".join(lines))
            st.caption(ui_message('ui.d4c03ac7b71f1d', p0=f'{LINEAR_FIR_CYCLES_MIN:g}', p1=f'{LINEAR_FIR_CYCLES_MAX:g}'))
    elif method in {"linear_phase_lr2", "Linear-phase LR2 FIR", "linear_phase_lr4", "Linear-phase LR4 FIR"}:
        from crossover_engine.lr4_taps import automatic_lr4_taps
        is_lr4 = method in {"linear_phase_lr4", "Linear-phase LR4 FIR"}
        name = "LR4" if is_lr4 else "LR2"
        floor = "100" if is_lr4 else "80"
        planner = automatic_lr4_taps if is_lr4 else automatic_lr2_taps
        try:
            plan = planner(int(sample_rate), float(crossover_hz), float(overlap_oct))
        except ValueError as exc:
            st.warning(ui_message('ui.5da8392fac8dd7', p0=f'{name}', p1=f'{exc}'))
            return
        if current_auto_taps:
            current_taps = plan.taps
        st.caption(
            ui_message('ui.36dd2b07d81493', p0=f'{name}', p1=f'{plan.taps:,}', p2=f'{plan.max_error_db:.4f}')
            + (ui_message('ui.5fa0cea795b7ea', p0=f'{current_taps:,}') if current_taps is not None else ""),
            help=ui_message('ui.4a54d8e4b20cab', p0=f'{floor}'),
        )

        if is_lr4:
            st.caption(ui_message('ui.c25455c3ab80ae'))
