"""Read-only PhaseEQ view of the Studio-owned alignment recipe."""
import streamlit as st

from crossover_engine.recipe import iir_config
from utils.ui_localization import ui_message, display_text


def render_output_iir_summary(recipe):
    """Show the direct DSP chain independently of FIR generation and target settings."""
    from crossover_engine.alignment import output_sos, output_polarity, target_crossover_polarity, separate_polarity
    import pandas as pd
    sos = output_sos(recipe)
    st.markdown(display_text("**出力帯域分割 IIR（Biquad）**"))
    st.caption(display_text("LR2の極性補償は帯域分割、All Passは位相・時間整合のターゲット設定に従います。手動Polarityは独立した直接設定です。"))
    if separate_polarity(recipe):
        st.caption(ui_message("ui.10d28c53aebd4e", p0=f"{output_polarity(recipe):+d}",
                              p1=f"{target_crossover_polarity(recipe):+d}",
                              p2=f"{recipe.get('phase_alignment', {}).get('polarity', 1):+d}"))
    if not sos.size:
        st.caption(display_text("直接適用するIIRはありません。"))
        return
    st.caption(ui_message("ui.911be1385ddefc", p0=len(sos)))
    st.dataframe(pd.DataFrame(sos, columns=["b0", "b1", "b2", "a0", "a1", "a2"]),
                 hide_index=True, width="stretch")
    st.caption(display_text("書き出しの「出力帯域分割」または「IIR EQ＋出力帯域分割」から取得できます。極性は係数に含まれるため、DSP側で重ねて反転しないでください。"))


def render_alignment_definition(recipe):
    from crossover_engine.alignment import separate_polarity
    value = recipe.get("phase_alignment")
    if value is None:
        return
    st.markdown(ui_message("ui.4207770a291167"))
    st.checkbox(ui_message("ui.f0531283f6c414"), value=value["acoustic_target"],
                disabled=True, key=f"recipe_alignment_target_{value['acoustic_target']}",
                help=ui_message("ui.051c94b46a54d8" if separate_polarity(recipe) else "ui.0742ddceef0d48"))
    if not separate_polarity(recipe):
        st.caption(ui_message("ui.e3450a1ee53f1e", p0=f"{value['polarity']:+d}",
                              p1=f"{iir_config(recipe).lr2_polarity:+d}"))
        st.caption(display_text("旧Recipeの極性適用ルールを保持しています。新しいルールへ移行する場合はマルチウェイから再送信してください。"))
    for item in value["allpass"]:
        if separate_polarity(recipe):
            st.caption(ui_message("ui.f45bcebea7a888", p0=item["order"],
                                  p1=f"{item['frequency_hz']:g}", p2=f"{item['q']:g}"))
        else:
            st.caption(ui_message("ui.5fdd4a866748ce", p0=item["order"],
                                  p1=f"{item['frequency_hz']:g}", p2=f"{item['q']:g}",
                                  p3=f"{item['polarity']:+d}"))
    if not value["allpass"]:
        st.caption(ui_message("ui.c402e5379394e8"))
