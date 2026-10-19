from __future__ import annotations

from collections.abc import Mapping

from composite_engine.dsp_export.models import CanonicalDSPPackage


def output_timing_rows(
    package: CanonicalDSPPackage, *, assigned_taps: Mapping[str, int | None],
    split_lengths: Mapping[str, int], intermediate_lengths: Mapping[str, int],
    align_output_taps: bool,
    taper_protection_taps: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Project the actual export artifact and delays into summary/detail rows."""
    summary, details = [], []
    for channel in package.channels:
        taps = len(channel.final_fir) if channel.final_fir is not None else None
        pre_alignment_taps = (
            channel.pre_alignment_tap_count
            if channel.pre_alignment_tap_count is not None else taps
        )
        alignment_padding = (
            taps - pre_alignment_taps
            if taps is not None and pre_alignment_taps is not None else None
        )
        taper_protection = (taper_protection_taps or {}).get(channel.way)
        identity = {"出力CH": channel.output_index, "グループ": channel.group, "チャンネル": channel.name}
        terms = (channel.adaptive_crop_metadata or {}).get("delay_breakdown", {})
        metadata = channel.adaptive_crop_metadata or {}
        fixed_frame = metadata.get("fixed_frame", {})
        summary.append({
            **identity,
            "最終FIRタップ数": taps if taps is not None else "—",
            "DSP追加Delay [ms]": round(channel.total_delay_samples / channel.sample_rate_hz * 1000, 6),
            "DSP追加Delay [samples]": round(channel.total_delay_samples, 6),
        })
        details.append({
            **identity,
            "補正用割り当てタップ数": assigned_taps.get(channel.way) or "—",
            "帯域分割FIRタップ数": split_lengths.get(channel.way, "—") if taps is not None else "—",
            "分割＋追加EQタップ数": intermediate_lengths.get(channel.way, "—") if taps is not None else "—",
            "コサインテーパー保護枠 [taps]": (
                taper_protection if taper_protection is not None and taps is not None else "—"
            ),
            "最長タップ整列": "ON" if align_output_taps else "OFF",
            "整列前FIRタップ数": pre_alignment_taps if pre_alignment_taps is not None else "—",
            "整列追加zero [taps]": alignment_padding if alignment_padding is not None else "—",
            "最終FIRタップ数": taps if taps is not None else "—",
            "固定枠追加zero [taps]": (
                int(fixed_frame["padding_left"] + fixed_frame["padding_right"])
                if fixed_frame else "—"
            ),
            "固定長切り出し [taps]": (
                int(fixed_frame["input_taps"] - (fixed_frame["source_stop"] - fixed_frame["source_start"]))
                if fixed_frame else "—"
            ),
            "FIR長差Delay [samples]": round(float(terms.get("fir_base", channel.fir_alignment_delay_samples)), 6),
            "FIR理論差補償 [samples]": round(float(
                (channel.adaptive_crop_metadata or {}).get("theoretical_alignment_delay_samples", 0.0)
            ), 6),
            "FIR起点補償 [samples]": round(float(terms.get("fir_origin", 0.0)), 6),
            "測定到達差 [samples]": terms.get("speaker_arrival", "—") if terms.get("speaker_arrival") is not None else "—",
            "測定補正 [samples]": round(float(terms.get("measurement", 0.0)), 6),
            "共通オフセット [samples]": round(float(terms.get("common", 0.0)), 6),
            "共通基準差引 [samples]": round(channel.output_reference_delay_samples, 6),
            "DSP追加Delay合計 [samples]": round(channel.total_delay_samples, 6),
            "手動Delay [samples]": round(channel.manual_delay_samples, 6),
            "整合Delay [samples]": round(channel.phase_alignment_delay_samples, 6),
            "Adaptive削減 [taps]": (
                int(
                    channel.adaptive_crop_metadata.get("original_taps", taps)
                    - channel.adaptive_crop_metadata.get("final_taps", taps)
                )
                if metadata.get("mode") == "adaptive" and taps is not None else "—"
            ),
            "Adaptive時間補償": (
                channel.adaptive_crop_metadata.get("timing_mode", "—")
                if metadata.get("mode") == "adaptive" else "—"
            ),
        })
    return summary, details


def delay_breakdown_rows(package):
    """Render correction terms from the committed artifact, never recalculate them."""
    labels = (("fir_origin", "FIR起点補償"), ("fir_difference", "FIR理論差補償"),
              ("measurement", "測定補正"), ("way", "Way相対Delay"),
              ("alignment", "整合Delay残り"), ("common", "共通オフセット"))
    rows = []
    for channel in package.channels:
        terms = (channel.adaptive_crop_metadata or {}).get("delay_breakdown", {})
        if not terms:
            continue
        scale = 1000.0 / channel.sample_rate_hz
        row = {"出力CH": channel.output_index, "チャンネル": channel.name}
        row.update({label + " [ms]": round(float(terms[key]) * scale, 6) for key, label in labels})
        row["共通基準差引 [ms]"] = round(channel.output_reference_delay_samples * scale, 6)
        row["DSP追加Delay合計 [ms]"] = round(channel.total_delay_samples * scale, 6)
        row["測定到達差 [ms]"] = (f"{terms['speaker_arrival'] * scale:.6f}"
                                  if terms.get("speaker_arrival") is not None else "—")
        rows.append(row)
    return rows
