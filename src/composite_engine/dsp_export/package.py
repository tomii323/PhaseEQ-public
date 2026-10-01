from __future__ import annotations

from dataclasses import asdict
import datetime
import hashlib
import io
import json
import zipfile

import numpy as np

from .adapters.base import DSPExportAdapter
from .common import channel_root, csv_bytes, safe_stem
from .models import CanonicalDSPPackage, DSPProfile
from composite_engine.system_specification import (
    build_system_specification,
    render_system_specification_json,
    render_system_specification_markdown,
    render_system_summary_markdown,
)


DSP_EXPORT_FORMAT = "phaseeq-multiway-dsp-export"
DSP_EXPORT_FORMAT_VERSION = 3


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"omitted_binary_bytes": len(value)}
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return str(value)


def package_filename(package: CanonicalDSPPackage, adapter: DSPExportAdapter) -> str:
    timestamp = datetime.datetime.fromisoformat(package.generated_at).strftime("%Y%m%d_%H%M%S")
    fs_label = f"{round(package.sample_rate_hz / 1000):g}k"
    return (
        f"PhaseEQ_MultiwayDSP_{safe_stem(package.system_name, 'System')}_"
        f"{safe_stem(package.mode)}_{adapter.adapter_id}_{fs_label}_{timestamp}.zip"
    )


def build_dsp_export_zip(
    package: CanonicalDSPPackage,
    adapter: DSPExportAdapter,
    profile: DSPProfile,
) -> tuple[bytes, str]:
    exported = adapter.export(package, profile)
    files = dict(exported.files)
    for channel in package.channels:
        root = f"configuration/channels/{channel_root(channel)}"
        files[f"{root}/phaseeq_iir.json"] = json.dumps({
            "stage": "phaseeq_iir",
            "parameters": channel.phaseeq_iir_parameters,
            "sos": channel.phaseeq_iir_sos,
        }, ensure_ascii=False, indent=2, default=_json_default).encode()
        files[f"{root}/baffle_iir.json"] = json.dumps({
            "stage": "baffle_iir",
            "enabled": bool(channel.baffle_iir_sos),
            "parameters": channel.baffle_iir_parameters,
            "sos": channel.baffle_iir_sos,
        }, ensure_ascii=False, indent=2, default=_json_default).encode()
        files[f"{root}/crossover_iir.json"] = json.dumps({
            "processing_order": ["iir_crossover", "crossover_phase_compensation_allpass"],
            "iir_crossover": channel.crossover.to_dict(),
            "crossover_phase_compensation_allpass": [
                section.to_dict() for section in channel.crossover_allpass
            ],
        }, ensure_ascii=False, indent=2).encode()
        files[f"{root}/channel_config.json"] = json.dumps({
            "output_index": channel.output_index, "channel_id": channel.channel_id,
            "name": channel.name, "group": channel.group, "way": channel.way,
            "sample_rate_hz": channel.sample_rate_hz,
            "processing_order": ["final_fir", "phaseeq_iir", "baffle_iir", "iir_crossover", "crossover_allpass", "gain", "polarity", "delay"],
            "gain_db": channel.gain_db, "polarity": channel.polarity,
            "delay": {
                "fir_alignment_samples": channel.fir_alignment_delay_samples,
                "manual_samples": channel.manual_delay_samples,
                "phase_alignment_samples": channel.phase_alignment_delay_samples,
                "total_samples": channel.total_delay_samples,
                "total_ms": channel.total_delay_samples / channel.sample_rate_hz * 1000.0,
            },
            "timing_provenance": channel.timing_provenance,
        }, ensure_ascii=False, indent=2).encode()
    files["documentation/README_FIRST.txt"] = (
        f"System: {package.system_name}\nTarget DSP: {adapter.display_name}\n"
        f"Profile: {profile.display_name}\nSample rate: {package.sample_rate_hz} Hz\n"
        f"Channels: {len(package.channels)}\nGenerated: {package.generated_at}\n\n"
        "Processing order: Final FIR -> PhaseEQ IIR -> Baffle IIR -> IIR crossover -> "
        "crossover compensation All-pass -> Gain -> Polarity -> Delay.\n"
    ).encode("utf-8-sig")
    files["documentation/channel_settings.csv"] = csv_bytes(
        tuple(exported.report.channel_rows[0].keys()) if exported.report.channel_rows else ("Channel",),
        [tuple(row.values()) for row in exported.report.channel_rows],
    )
    files["documentation/verification_report.json"] = json.dumps({
        "compatible": exported.report.compatible,
        "issues": [asdict(issue) for issue in exported.report.issues],
        "channels": list(exported.report.channel_rows),
    }, ensure_ascii=False, indent=2).encode()
    system_specification = build_system_specification(
        package, adapter_id=adapter.adapter_id,
        profile_id=profile.profile_id, report=exported.report,
    )
    files["documentation/system_summary.md"] = render_system_summary_markdown(
        system_specification
    ).encode("utf-8")
    files["documentation/system_specification.md"] = (
        render_system_specification_markdown(system_specification).encode("utf-8")
    )
    files["documentation/system_specification.json"] = (
        render_system_specification_json(system_specification)
    )
    files["configuration/workspace.json"] = json.dumps(
        package.workspace, ensure_ascii=False, indent=2, default=_json_default,
    ).encode()
    from phase_fir_designer.output_terms import OUTPUT_TERMS_FILENAME, OUTPUT_TERMS_BYTES
    files[OUTPUT_TERMS_FILENAME] = OUTPUT_TERMS_BYTES
    file_index = [
        {"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in sorted(files.items())
    ]
    manifest = {
        "format": DSP_EXPORT_FORMAT, "format_version": DSP_EXPORT_FORMAT_VERSION,
        "package_role": "dsp_delivery", "generated_at": package.generated_at,
        "system_name": package.system_name, "mode": package.mode,
        "system_id": package.system_id,
        "management_no": package.management_no,
        "system_revision_number": package.system_revision_number,
        "system_content_hash": package.system_content_hash,
        "sample_rate_hz": package.sample_rate_hz, "source_signature": package.source_signature,
        "adapter": adapter.adapter_id, "profile": asdict(profile),
        "compatible": exported.report.compatible,
        "issues": [asdict(issue) for issue in exported.report.issues],
        "channels": list(exported.report.channel_rows), "files": file_index,
    }
    files["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    files["checksums.sha256"] = ("\n".join(
        f"{hashlib.sha256(data).hexdigest()}  {path}" for path, data in sorted(files.items())
    ) + "\n").encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path, data in sorted(files.items()):
            output.writestr(path, data)
    return archive.getvalue(), package_filename(package, adapter)
