from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ui_localization import display_text, ui_message
from ..processing.output_summary import output_timing_rows


_TAP_COUNT_COLUMNS = (
    "最終FIRタップ数", "補正用割り当てタップ数",
    "帯域分割FIRタップ数", "分割＋追加EQタップ数",
    "Adaptive削減 [taps]",
)


def output_timing_frame(rows):
    """Keep numerical results intact; normalize mixed tap/placeholder display columns."""
    frame = pd.DataFrame(rows)
    for column in _TAP_COUNT_COLUMNS:
        if column in frame:
            frame[column] = frame[column].map(str)
    return frame.rename(columns=display_text)


def render_band_tap_counts(package, hosts):
    counts = {
        channel.way: len(channel.final_fir) if channel.final_fir is not None else None
        for channel in package.channels
    }
    for band, host in hosts.items():
        if band not in counts:
            value = display_text("出力チャンネルなし")
        elif counts[band] is None:
            value = "FIR OFF"
        else:
            value = f"{counts[band]:,} taps"
        host.caption(display_text("現在の最終FIRタップ数：{taps}").format(taps=value))


def render_output_timing(package, *, assigned_taps, split_lengths, intermediate_lengths):
    summary, details = output_timing_rows(
        package, assigned_taps=assigned_taps, split_lengths=split_lengths,
        intermediate_lengths=intermediate_lengths,
    )
    st.subheader(ui_message('ui.d52c03834c9fd6'))
    st.caption(display_text(
        "最終FIRタップ数は保存・出力するFIRと同じです。DSP追加Delayは手動・位相整合を含む合計値です。msかsamplesのどちらかで入力し、FIR自体の遅延を重ねて加えないでください。"
    ))
    st.dataframe(output_timing_frame(summary), hide_index=True, width="stretch")
    with st.expander(display_text("タップ数・Delayの明細"), expanded=False):
        st.caption(display_text(
            "補正用割り当てはPhaseEQで補正を設計する長さ、帯域分割FIRは分割フィルター単体の長さです。分割＋追加EQはPhaseEQ返却前の中間結果です。グラフ用の表示IR長はDSPへ入力しません。"
        ))
        st.dataframe(output_timing_frame(details), hide_index=True, width="stretch")
