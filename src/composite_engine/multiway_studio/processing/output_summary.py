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
            "FIR長差Delay [samples]": round(channel.fir_alignment_delay_samples, 6),
            "手動Delay [samples]": round(channel.manual_delay_samples, 6),
            "整合Delay [samples]": round(channel.phase_alignment_delay_samples, 6),
            "Adaptive削減 [taps]": (
                int(
                    channel.adaptive_crop_metadata.get("original_taps", taps)
                    - channel.adaptive_crop_metadata.get("final_taps", taps)
                )
                if isinstance(channel.adaptive_crop_metadata, dict) and taps is not None else "—"
            ),
            "Adaptive時間補償": (
                channel.adaptive_crop_metadata.get("timing_mode", "—")
                if isinstance(channel.adaptive_crop_metadata, dict) else "—"
            ),
        })
    return summary, details
