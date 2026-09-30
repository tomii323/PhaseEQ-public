"""Deterministic Markdown and JSON documentation for saved Multiway Systems."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from composite_engine.dsp_export.models import CanonicalDSPPackage, CompatibilityReport


SYSTEM_SPECIFICATION_FORMAT = "phaseeq-multiway-system-specification"
SYSTEM_SPECIFICATION_FORMAT_VERSION = 1


def build_system_specification(
    package: CanonicalDSPPackage,
    *,
    adapter_id: str,
    profile_id: str,
    report: CompatibilityReport,
) -> dict[str, Any]:
    workspace = dict(package.workspace)
    target = workspace.get("target") if isinstance(workspace.get("target"), dict) else {}
    assignments: list[dict[str, Any]] = []
    workspace_channels = workspace.get("channels", [])
    if isinstance(workspace_channels, list):
        for item in workspace_channels:
            if not isinstance(item, dict):
                continue
            assignment_id = str(
                item.get("phaseeq_assignment_id", item.get("latest_assignment_id", ""))
            ).strip()
            if assignment_id:
                assignments.append({
                    "assignment_id": assignment_id,
                    "channel_id": str(item.get("channel_id", "")),
                    "state": str(item.get("phaseeq_status_state", "")),
                })
    channels = []
    for channel in package.channels:
        fir_taps = int(channel.final_fir.size) if channel.final_fir is not None else 0
        channels.append({
            "output_index": channel.output_index,
            "channel_id": channel.channel_id,
            "name": channel.name,
            "group": channel.group,
            "way": channel.way,
            "sample_rate_hz": channel.sample_rate_hz,
            "fir_tap_count": fir_taps,
            "fir_intrinsic_center_delay_samples": (
                (fir_taps - 1) / 2.0 if fir_taps else 0.0
            ),
            "phaseeq_iir_sections": len(channel.phaseeq_iir_sos),
            "baffle_iir": {
                "enabled": bool(channel.baffle_iir_sos),
                "parameters": channel.baffle_iir_parameters,
                "sections": len(channel.baffle_iir_sos),
            },
            "crossover": channel.crossover.to_dict(),
            "crossover_allpass": [item.to_dict() for item in channel.crossover_allpass],
            "gain_db": channel.gain_db,
            "polarity": channel.polarity,
            "delay": {
                "fir_alignment_samples": channel.fir_alignment_delay_samples,
                "manual_samples": channel.manual_delay_samples,
                "phase_alignment_samples": channel.phase_alignment_delay_samples,
                "dsp_additional_total_samples": channel.total_delay_samples,
                "dsp_additional_total_ms": (
                    channel.total_delay_samples / channel.sample_rate_hz * 1000.0
                ),
            },
        })
    issues = [asdict(issue) for issue in report.issues]
    return {
        "format": SYSTEM_SPECIFICATION_FORMAT,
        "format_version": SYSTEM_SPECIFICATION_FORMAT_VERSION,
        "system": {
            "system_id": package.system_id,
            "management_no": package.management_no,
            "name": package.system_name,
            "revision_number": package.system_revision_number,
            "content_hash": package.system_content_hash,
            "mode": package.mode,
            "sample_rate_hz": package.sample_rate_hz,
        },
        "target": target,
        "groups": sorted({channel.group for channel in package.channels}),
        "channels": channels,
        "assignments": assignments,
        "dsp_export": {
            "adapter_id": adapter_id,
            "profile_id": profile_id,
            "source_signature": package.source_signature,
        },
        "validation": {
            "status": "pass" if report.compatible else "error",
            "issues": issues,
        },
        "generated_at": package.generated_at,
    }


def render_system_specification_json(model: dict[str, Any]) -> bytes:
    return (json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def render_system_summary_markdown(model: dict[str, Any]) -> str:
    system = model["system"]
    validation = model["validation"]
    channels = model["channels"]
    target = model.get("target", {})
    target_name = ""
    if isinstance(target, dict):
        definition = target.get("preset_definition")
        if isinstance(definition, dict):
            target_name = str(definition.get("name", ""))
    lines = [
        "# Multiway System summary",
        "",
        f"- Management No: {system.get('management_no') or 'Unassigned'}",
        f"- System: {system.get('name', '')}",
        f"- Revision: {system.get('revision_number') or 'Draft / standalone'}",
        f"- Mode: {system.get('mode', '')}",
        f"- Sample rate: {system.get('sample_rate_hz', 0):,} Hz",
        f"- Channels: {len(channels)}",
        f"- Target: {target_name or 'None / file snapshot'}",
        f"- Validation: {validation.get('status', 'unknown')} "
        f"({len(validation.get('issues', []))} issues)",
    ]
    return "\n".join(lines) + "\n"


def render_system_specification_markdown(model: dict[str, Any]) -> str:
    system = model["system"]
    lines = [
        "# Multiway System specification", "",
        "## Identity", "",
        f"- System ID: `{system.get('system_id', '')}`",
        f"- Management No: `{system.get('management_no') or 'Unassigned'}`",
        f"- Name: {system.get('name', '')}",
        f"- Revision: {system.get('revision_number') or 'Draft / standalone'}",
        f"- Content hash: `{system.get('content_hash', '')}`",
        f"- Mode: {system.get('mode', '')}",
        f"- Sample rate: {system.get('sample_rate_hz', 0):,} Hz", "",
        "## Channels", "",
        "| Output | Channel | Group | Way | FIR taps | Baffle | FIR center | DSP delay | Gain | Polarity |",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for channel in model["channels"]:
        lines.append(
            f"| {channel['output_index']} | {channel['name']} | {channel['group']} | "
            f"{channel['way']} | {channel['fir_tap_count']} | "
            f"{'IIR Shelf' if channel['baffle_iir']['enabled'] else 'FIR/OFF'} | "
            f"{channel['fir_intrinsic_center_delay_samples']:.3f} samples | "
            f"{channel['delay']['dsp_additional_total_samples']:.3f} samples | "
            f"{channel['gain_db']:+.3f} dB | {channel['polarity']:+d} |"
        )
    lines.extend(["", "## DSP export", ""])
    dsp_export = model["dsp_export"]
    lines.extend([
        f"- Adapter: {dsp_export['adapter_id']}",
        f"- Profile: {dsp_export['profile_id']}",
        f"- Source signature: `{dsp_export['source_signature']}`", "",
        "## Validation", "",
        f"- Status: {model['validation']['status']}",
    ])
    for issue in model["validation"]["issues"]:
        lines.append(
            f"- {str(issue.get('severity', 'info')).upper()}: "
            f"{issue.get('code', '')} — {issue.get('message', '')}"
        )
    return "\n".join(lines) + "\n"
