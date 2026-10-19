from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ui_localization import display_text, ui_message
from ..processing.output_summary import output_timing_rows, delay_breakdown_rows


_TAP_COUNT_COLUMNS = (
    "最終FIRタップ数", "補正用割り当てタップ数",
    "帯域分割FIRタップ数", "分割＋追加EQタップ数",
    "コサインテーパー保護枠 [taps]",
    "整列前FIRタップ数", "整列追加zero [taps]", "Adaptive削減 [taps]",
    "固定枠追加zero [taps]", "固定長切り出し [taps]",
)

WAITING_MESSAGES = {
    "recalculate": "現在の最終FIRタップ数：再計算待ち。「再計算」を押してください。",
    "restore": "現在の最終FIRタップ数：復元した補正の返却待ち。PhaseEQで再生成して返却してください。",
    "invalid_settings": "現在の最終FIRタップ数：設定の確認が必要です。画面のエラーを修正し、再計算してください。",
    "generation_error": "現在の最終FIRタップ数：生成できません。画面のエラーを確認し、原因を解消して再計算してください。",
}


def render_waiting_band_tap_counts(hosts, reason):
    for host in (hosts or {}).values():
        host.caption(display_text(WAITING_MESSAGES[reason]))


def render_pending_corrections(rows):
    """A missing correction is advisory; existing processing remains usable."""
    from ..processing.alignment_target import pending_reason
    grouped = {"return_missing": [], "settings_changed": []}
    for row in rows:
        if not isinstance(row, dict) or row.get("source") != "generated_band" or not row.get("enabled", True):
            continue
        reason = pending_reason(row)
        if reason:
            grouped[reason].append(str(row.get("name") or row.get("way", "")))
    messages = {
        "return_missing": "補正結果待ち：{channels}。FIRの理論遅延差を補償して計算・出力します。位相目標への補正には、PhaseEQで測定を適用し、補正を設定・更新して返却してください。",
        "settings_changed": "補正更新待ち：{channels}。FIRの理論遅延差を補償して計算・出力します。新しい条件への補正には、PhaseEQへ再送信し、補正を更新して返却してください。",
    }
    for reason, names in grouped.items():
        if names:
            st.warning(display_text(messages[reason]).format(channels="、".join(names)))


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


def render_output_timing(
    package, *, assigned_taps, split_lengths, intermediate_lengths,
    align_output_taps, taper_protection_taps=None,
):
    summary, details = output_timing_rows(
        package, assigned_taps=assigned_taps, split_lengths=split_lengths,
        intermediate_lengths=intermediate_lengths,
        align_output_taps=align_output_taps,
        taper_protection_taps=taper_protection_taps,
    )
    st.subheader(ui_message('ui.d52c03834c9fd6'))
    st.caption(display_text(
        "最終FIRタップ数は保存・出力するFIRと同じです。FIR使用時のDSP追加Delayは、積算した追加Delayの最小値を全出力から差し引いた相対値です。msかsamplesのどちらかで入力し、FIR自体の遅延を重ねて加えないでください。"
    ))
    st.caption(display_text("共通基準差引はFIR自体の遅延を含みません。FIRの時間差補償を維持し、最小のDSP追加Delayを0にします。最小値が既に0の場合は減りません。機種別のDelay上限は出力前に確認してください。"))
    st.dataframe(output_timing_frame(summary), hide_index=True, width="stretch")
    breakdown = delay_breakdown_rows(package)
    if breakdown:
        st.caption(display_text("Delayの内訳：FIR起点補償＋FIR理論差補償＋測定補正＋Way相対Delay＋整合Delay残り＋共通オフセット−共通基準差引。測定到達差は補正前の相対値です。"))
        st.dataframe(output_timing_frame(breakdown), hide_index=True, width="stretch")
        terms = [(channel.adaptive_crop_metadata or {}).get("delay_breakdown", {}) for channel in package.channels]
        if any(item.get("legacy_unclassified") for item in terms):
            st.warning(display_text("旧設定の整合Delayに測定補正が含まれるか確認が必要です。値を維持しています。「位相・時間整合」を再適用して内訳を確定してください。"))
        if any(item.get("measurement_source") == "retained" for item in terms):
            st.caption(display_text("共通基準の測定到達差が揃わないため、保存済みの測定補正を維持しています。測定を更新した場合は、全チャンネルの測定基準を確認して再計算してください。"))
    with st.expander(display_text("タップ数・Delayの明細"), expanded=False):
        st.caption(ui_message('ui.6c1c481f75aba4'))
        st.dataframe(output_timing_frame(details), hide_index=True, width="stretch")
