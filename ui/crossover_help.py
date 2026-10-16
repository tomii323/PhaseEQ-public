"""Render shared crossover help without mutating design or widget state."""
import streamlit as st

from composite_engine.crossover_help import (
    acoustic_flow_text, filter_comparison_text, kaiser_adjustment_text,
    method_usage_text, setup_guide_text,
)
from composite_engine.kaiser_help import KAISER_SLOPE_GUIDE
from composite_engine.phase_alignment_help import structural_phase_guide_text, structural_phase_labels
from utils.ui_language import current_language
from utils.ui_localization import display_text, ui_message


def render_structural_phase_guide() -> None:
    language = current_language()
    with st.expander(structural_phase_labels(language)['title'], expanded=False):
        st.markdown(structural_phase_guide_text(language))


def render_filter_comparison(*, expandable: bool = False) -> None:
    content = setup_guide_text(current_language()) + "\n" + filter_comparison_text(current_language())
    if expandable:
        label = "Choosing a crossover filter" if current_language() == "en" else "帯域分割フィルターの使い分け"
        with st.expander(label, expanded=False):
            st.markdown(content)
    else:
        st.markdown(content)


def render_kaiser_usage_guide(*, phaseeq: bool = False) -> None:
    content = display_text(KAISER_SLOPE_GUIDE)
    if phaseeq:
        # PhaseEQ renders its own entry-range note below the shared content.
        content = "\n".join(line for line in content.splitlines()
                            if not line.startswith(("周期数は2.7", "Cycles are displayed")))
    with st.expander(ui_message('ui.f753e49eab7a4f'), expanded=False):
        st.markdown(kaiser_adjustment_text(current_language()))
        st.markdown(content)
        if phaseeq:
            from phase_fir_designer.config import LINEAR_FIR_CYCLES_MIN, LINEAR_FIR_CYCLES_MAX
            st.caption(ui_message('ui.02000caed3f3ee', p0=f'{LINEAR_FIR_CYCLES_MIN:.2f}', p1=f'{LINEAR_FIR_CYCLES_MAX:.2f}'))


def render_acoustic_target_guide(label: str, *, studio: bool, method: str | None = None) -> None:
    with st.expander(ui_message('ui.2bbd78a88b6990', p0=label), expanded=False):
        if method is not None:
            st.markdown(method_usage_text(method, current_language()))
        st.markdown(acoustic_flow_text(studio=studio, language=current_language()))
        if studio:
            st.markdown(ui_message('ui.f316df0f81421e'))
