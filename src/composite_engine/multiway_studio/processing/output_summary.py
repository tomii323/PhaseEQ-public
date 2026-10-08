from __future__ import annotations

from collections.abc import Mapping

from composite_engine.dsp_export.models import CanonicalDSPPackage


def output_timing_rows(
    package: CanonicalDSPPackage, *, assigned_taps: Mapping[str, int | None],
    split_lengths: Mapping[str, int], intermediate_lengths: Mapping[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Project the actual export artifact and delays into summary/detail rows."""
    summary, details = [], []
    for channel in package.channels:
        taps = len(channel.final_fir) if channel.final_fir is not None else None
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
