from __future__ import annotations

import csv
import io
import re

import numpy as np

from composite_engine.iir_crossover import iir_crossover_sos, sos_is_stable
from composite_engine.phase_alignment import allpass_sos

from .models import CanonicalDSPChannel, CanonicalDSPPackage, CompatibilityIssue, CompatibilityReport, DSPProfile


def safe_stem(value: str, fallback: str = "item") -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    return stem or fallback


def channel_root(channel: CanonicalDSPChannel) -> str:
    return f"ch{channel.output_index:02d}_{safe_stem(channel.group, 'Group')}_{safe_stem(channel.way, 'Way')}"


def normalized_sos(rows: object) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.size == 0:
        return np.empty((0, 6), dtype=float)
    values = values.reshape(-1, 6).copy()
    values /= values[:, 3:4]
    return values


def crossover_sos(channel: CanonicalDSPChannel) -> np.ndarray:
    return normalized_sos(iir_crossover_sos(channel.crossover, channel.sample_rate_hz))


def allpass_sos_rows(channel: CanonicalDSPChannel) -> np.ndarray:
    rows = [allpass_sos(section, channel.sample_rate_hz) for section in channel.crossover_allpass]
    return normalized_sos(np.vstack(rows)) if rows else np.empty((0, 6), dtype=float)


def csv_bytes(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def validate_package(package: CanonicalDSPPackage, profile: DSPProfile) -> CompatibilityReport:
    issues: list[CompatibilityIssue] = []
    rows: list[dict[str, object]] = []
    try:
        sample_rate = package.sample_rate_hz
    except ValueError as exc:
        return CompatibilityReport(profile.adapter_id, profile.profile_id, (
            CompatibilityIssue("error", "sample_rate_mismatch", str(exc)),
        ), ())
    if profile.sample_rates_hz and sample_rate not in profile.sample_rates_hz:
        issues.append(CompatibilityIssue(
            "error", "unsupported_sample_rate",
            f"{sample_rate:,} Hz is not supported by {profile.display_name}",
        ))
    seen_indices: set[int] = set()
    for channel in package.channels:
        if channel.output_index in seen_indices:
            issues.append(CompatibilityIssue("error", "duplicate_output_index", "duplicate output index", channel.channel_id))
        seen_indices.add(channel.output_index)
        fir_taps = int(channel.final_fir.size) if channel.final_fir is not None else 0
        phaseeq = normalized_sos(channel.phaseeq_iir_sos)
        baffle = normalized_sos(channel.baffle_iir_sos)
        crossover = crossover_sos(channel)
        allpass = allpass_sos_rows(channel)
        section_count = len(phaseeq) + len(baffle) + len(crossover) + len(allpass)
        if fir_taps and not profile.supports_fir:
            issues.append(CompatibilityIssue("error", "fir_unsupported", "Final FIR is not supported", channel.channel_id, "final_fir"))
        if profile.max_fir_taps and fir_taps > profile.max_fir_taps:
            issues.append(CompatibilityIssue("error", "fir_taps_exceeded", f"FIR taps {fir_taps} exceed {profile.max_fir_taps}", channel.channel_id, "final_fir"))
        if profile.max_iir_sections and section_count > profile.max_iir_sections:
            issues.append(CompatibilityIssue("error", "iir_sections_exceeded", f"IIR sections {section_count} exceed {profile.max_iir_sections}", channel.channel_id, "iir"))
        if len(allpass) and not profile.supports_allpass:
            issues.append(CompatibilityIssue("error", "allpass_unsupported", "Crossover compensation All-pass is not supported", channel.channel_id, "crossover_allpass"))
        if channel.total_delay_samples < 0:
            issues.append(CompatibilityIssue("error", "negative_delay", "Total delay must not be negative", channel.channel_id, "delay"))
        if not profile.supports_fractional_delay and not np.isclose(channel.total_delay_samples, round(channel.total_delay_samples)):
            issues.append(CompatibilityIssue("warning", "fractional_delay", "Fractional delay requires manual realization", channel.channel_id, "delay"))
        for stage, sos in (("phaseeq_iir", phaseeq), ("baffle_iir", baffle), ("crossover_iir", crossover), ("crossover_allpass", allpass)):
            if sos.size and not sos_is_stable(sos):
                issues.append(CompatibilityIssue("error", "unstable_iir", f"Unstable {stage}", channel.channel_id, stage))
        rows.append({
            "Output": channel.output_index, "Channel": channel.name, "Group": channel.group,
            "Way": channel.way, "FIR taps": fir_taps,
            "PhaseEQ IIR": len(phaseeq), "Baffle IIR": len(baffle),
            "Crossover IIR": len(crossover),
            "All-pass": len(allpass), "IIR total": section_count,
            "Gain [dB]": channel.gain_db, "Polarity": channel.polarity,
            "Delay [samples]": channel.total_delay_samples,
        })
    return CompatibilityReport(profile.adapter_id, profile.profile_id, tuple(issues), tuple(rows))
