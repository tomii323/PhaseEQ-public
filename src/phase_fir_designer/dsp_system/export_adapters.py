from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import io
import json
import re
import zipfile

import numpy as np

from ..export import export_bin, export_wav
from ..iir import biquad_rows, export_biquad_csv, export_biquad_json, export_minidsp_biquads, export_sigmastudio_biquads
from .output_pipeline import DSPSystemOutput, DSPWayOutput
from .signal_path import build_final_signal_path_traces


@dataclass(frozen=True)
class DSPExportPackage:
    files: dict[str, bytes]
    manifest: dict[str, object]

    def zip_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, data in sorted(self.files.items()):
                archive.writestr(path, data)
        return buffer.getvalue()


def _slug(value: str, fallback: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip()).strip("._-")
    return clean or fallback


def _wav_bytes(coefficients: np.ndarray, sample_rate: int) -> bytes:
    output = io.BytesIO()
    export_wav(output, np.asarray(coefficients, dtype=float), int(sample_rate))
    return output.getvalue()


def _bin_bytes(coefficients: np.ndarray) -> bytes:
    output = io.BytesIO()
    export_bin(output, np.asarray(coefficients, dtype=float))
    return output.getvalue()


def _coefficient_text(coefficients: np.ndarray) -> bytes:
    return ("\n".join(f"{float(value):.16g}" for value in coefficients) + "\n").encode()


def _rate_label(sample_rate: int) -> str:
    rate = int(sample_rate)
    return f"{rate // 1000}k" if rate % 1000 == 0 else f"{rate}Hz"


def _stamp(timestamp: datetime) -> str:
    return timestamp.strftime("%y%m%d-%H%M") + f"-{timestamp.microsecond % 4096:03X}"


def _way_stem(way: DSPWayOutput) -> str:
    return "_".join(
        [
            _slug(way.device_name, "DSP"),
            f"CH{way.output_channel + 1:02d}",
            _slug(way.way_name, "Way"),
        ]
    )


def _camilladsp_yaml(output: DSPSystemOutput, ways: list[DSPWayOutput], dsp_id: str, stamp: str) -> str:
    sample_rate = int(ways[0].sample_rate)
    relevant_routes = [route for route in output.routes if any(way.way_id == route.way_id for way in ways)]
    inputs = {item.id: item for item in output.inputs}
    active_routes = [
        route for route in relevant_routes
        if route.enabled and not route.muted
        and route.input_id in inputs and not inputs[route.input_id].muted
    ]
    bindings = {(item.input_id, item.dsp_id): item.input_channel for item in output.input_bindings}
    input_channels = [bindings[(route.input_id, dsp_id)] for route in active_routes if (route.input_id, dsp_id) in bindings]
    mixer_name = f"phaseeq_input_matrix_{_slug(dsp_id, 'dsp')}"
    lines = ["# PhaseEQ CamillaDSP filter/pipeline snippet"]
    if relevant_routes:
        lines.extend([
            "mixers:", f"  {mixer_name}:", "    channels:",
            f"      in: {max(input_channels, default=0) + 1}",
            f"      out: {max(way.output_channel for way in ways) + 1}",
            "    mapping:",
        ])
        for way in ways:
            sources = [route for route in active_routes if route.way_id == way.way_id]
            lines.extend([f"      - dest: {way.output_channel}", "        sources:"])
            for route in sources:
                channel = bindings.get((route.input_id, dsp_id))
                if channel is None:
                    raise ValueError(f"CamillaDSP Routing export is missing an input binding for {route.input_id}")
                lines.extend([
                    f"          - channel: {channel}",
                    f"            gain: {float(route.gain_db) + float(inputs.get(route.input_id).gain_db if inputs.get(route.input_id) else 0.0):.16g}",
                    f"            inverted: {'true' if bool(inputs.get(route.input_id).polarity_invert if inputs.get(route.input_id) else False) else 'false'}",
                ])
    lines.append("filters:")
    input_chains: list[tuple[int, list[str]]] = []
    for input_item in output.inputs:
        channel = bindings.get((input_item.id, dsp_id))
        filters = tuple(item for item in input_item.iir_filters if item.enabled)
        if channel is None or not filters or input_item.muted:
            continue
        input_chain: list[str] = []
        input_name = f"phaseeq_input_{_slug(input_item.id, 'input')}"
        for row in biquad_rows(filters, sample_rate):
            iir_name = f"{input_name}_iir_{row['section']}"
            input_chain.append(iir_name)
            lines.extend([
                f"  {iir_name}:", "    type: DiffEq", "    parameters:",
                f"      a: [1.0, {float(row['a1']):.16g}, {float(row['a2']):.16g}]",
                f"      b: [{float(row['b0']):.16g}, {float(row['b1']):.16g}, {float(row['b2']):.16g}]",
            ])
        input_chains.append((channel, input_chain))
    names: list[tuple[DSPWayOutput, list[str]]] = []
    for way in ways:
        name = f"phaseeq_{_slug(way.way_id, 'way')}"
        chain: list[str] = []
        for row in biquad_rows(way.iir_filters, way.sample_rate):
            iir_name = f"{name}_iir_{row['section']}"
            chain.append(iir_name)
            lines.extend([
                f"  {iir_name}:", "    type: DiffEq", "    parameters:",
                f"      a: [1.0, {float(row['a1']):.16g}, {float(row['a2']):.16g}]",
                f"      b: [{float(row['b0']):.16g}, {float(row['b1']):.16g}, {float(row['b2']):.16g}]",
            ])
        if way.fir_enabled:
            chain.append(name)
            lines.extend([
                f"  {name}:", "    type: Conv", "    parameters:", "      type: Wav",
                f"      filename: fir/{_way_stem(way)}_FIR_{_rate_label(way.sample_rate)}_{stamp}.wav",
                "      channel: 0",
            ])
        output_gain_db = float(way.way_gain_db) + float(output.common_auto_gain_db)
        if abs(output_gain_db) > 1e-12 or way.polarity_invert:
            gain_name = f"{name}_output_gain"
            chain.append(gain_name)
            lines.extend([
                f"  {gain_name}:", "    type: Gain", "    parameters:",
                f"      gain: {output_gain_db:.16g}",
                f"      inverted: {'true' if way.polarity_invert else 'false'}",
                "      mute: false", "      scale: dB",
            ])
        if float(way.delay_ms) > 1e-12:
            delay_name = f"{name}_output_delay"
            chain.append(delay_name)
            lines.extend([
                f"  {delay_name}:", "    type: Delay", "    parameters:",
                f"      delay: {float(way.delay_ms):.16g}",
                "      unit: ms", "      subsample: false",
            ])
        if chain:
            names.append((way, chain))
    lines.append("pipeline:")
    for channel, chain in input_chains:
        lines.extend([
            "  - type: Filter",
            f"    channels: [{channel}]",
            f"    names: [{', '.join(chain)}]",
        ])
    if relevant_routes:
        lines.extend(["  - type: Mixer", f"    name: {mixer_name}"])
    for way, chain in names:
        lines.extend(["  - type: Filter", f"    channels: [{way.output_channel}]", f"    names: [{', '.join(chain)}]"])
    return "\n".join(lines) + "\n"


def _routing_text(output: DSPSystemOutput, dsp_id: str) -> str:
    ways = {way.way_id: way for way in output.ways if way.dsp_id == dsp_id}
    inputs = {item.id: item for item in output.inputs}
    bindings = {(item.input_id, item.dsp_id): item.input_channel for item in output.input_bindings}
    lines = [
        "# PhaseEQ final signal path",
        "# Input -> Input IIR -> Route -> Way IIR -> FIR -> Way output -> Common output",
    ]
    for route in output.routes:
        way = ways.get(route.way_id)
        if way is None:
            continue
        source = inputs.get(route.input_id)
        channel = bindings.get((route.input_id, dsp_id))
        lines.append(
            f"{source.name if source else route.input_id} (Input {channel + 1 if channel is not None else '?'})"
            f" -> {way.way_name} (Output {way.output_channel + 1})"
            f" / Input {(source.gain_db if source else 0.0):+.2f} dB"
            f" / Route {route.gain_db:+.2f} dB"
            f" / Way {way.way_gain_db:+.2f} dB"
            f" / Common {output.common_auto_gain_db:+.2f} dB"
            f" / Total {((source.gain_db if source else 0.0) + route.gain_db + way.way_gain_db + output.common_auto_gain_db):+.2f} dB"
            f" / {'Invert' if (bool(source.polarity_invert if source else False) ^ way.polarity_invert) else 'Normal'}"
            f" / Way delay {way.delay_ms:+.3f} ms"
            f" / {'Mute' if route.muted or not route.enabled or bool(source.muted if source else False) else 'On'}"
        )
    return "\n".join(lines) + "\n"


def build_dsp_export_package(
    output: DSPSystemOutput,
    *,
    generated_at: datetime | None = None,
    expected_design_signature: str | None = None,
    iir_safety: dict[str, object] | None = None,
) -> DSPExportPackage:
    if output.design_signature and expected_design_signature is None:
        raise ValueError(
            "expected Design Signature is required for DSP System export"
        )
    if expected_design_signature is not None:
        expected = str(expected_design_signature).strip()
        generated = str(output.design_signature).strip()
        if not expected or not generated or generated != expected:
            raise ValueError(
                "Design Signature mismatch: regenerate the DSP System before export"
            )
    generated_at = generated_at or datetime.now()
    stamp = _stamp(generated_at)
    files: dict[str, bytes] = {}
    devices: dict[str, list[DSPWayOutput]] = {}
    for way in output.ways:
        devices.setdefault(way.dsp_id, []).append(way)
    device_manifest: list[dict[str, object]] = []
    composite_channels: list[dict[str, object]] = []
    for dsp_id, ways in sorted(devices.items()):
        ways.sort(key=lambda item: item.output_channel)
        first = ways[0]
        if any(float(way.delay_ms) < -1e-12 for way in ways):
            raise ValueError("DSP output Delay cannot be negative; apply a common offset before export")
        root = f"devices/{_slug(dsp_id, 'dsp')}_{_slug(first.device_name, 'DSP')}"
        target = first.target.strip().lower()
        if "camilla" in target:
            adapter = "camilladsp"
            adapter_root = f"{root}/{adapter}"
            files[f"{adapter_root}/CamillaDSP_Filters_{stamp}.yml"] = _camilladsp_yaml(output, ways, dsp_id, stamp).encode()
            files[f"{adapter_root}/CamillaDSP_Routing_{stamp}.txt"] = _routing_text(output, dsp_id).encode()
            for way in ways:
                stem = _way_stem(way)
                if way.fir_enabled:
                    files[f"{adapter_root}/fir/{stem}_FIR_{_rate_label(way.sample_rate)}_{stamp}.wav"] = _wav_bytes(way.coefficients, way.sample_rate)
                if way.iir_filters:
                    files[f"{adapter_root}/iir/{stem}_IIR_CamillaDSP_{_rate_label(way.sample_rate)}_{stamp}.json"] = export_biquad_json(way.iir_filters, way.sample_rate).encode()
        elif "sigma" in target:
            adapter = "sigmadsp"
            adapter_root = f"{root}/{adapter}"
            files[f"{adapter_root}/SigmaStudio_Routing_{stamp}.txt"] = _routing_text(output, dsp_id).encode()
            for way in ways:
                stem = _way_stem(way)
                if way.fir_enabled:
                    files[f"{adapter_root}/fir/{stem}_FIR_SigmaStudio_{_rate_label(way.sample_rate)}_{stamp}.txt"] = _coefficient_text(way.coefficients)
                    files[f"{adapter_root}/fir/{stem}_FIR_MemRev_{_rate_label(way.sample_rate)}_{stamp}.txt"] = _coefficient_text(way.coefficients[::-1])
                if way.iir_filters:
                    files[f"{adapter_root}/iir/{stem}_IIR_SigmaStudio_{_rate_label(way.sample_rate)}_{stamp}.txt"] = export_sigmastudio_biquads(way.iir_filters, way.sample_rate).encode()
        else:
            adapter = "minidsp"
            adapter_root = f"{root}/{adapter}"
            files[f"{adapter_root}/miniDSP_Routing_{stamp}.txt"] = _routing_text(output, dsp_id).encode()
            for way in ways:
                stem = _way_stem(way)
                if way.fir_enabled:
                    files[f"{adapter_root}/fir/{stem}_FIR_F32LE_{_rate_label(way.sample_rate)}_{stamp}.bin"] = _bin_bytes(way.coefficients)
                    files[f"{adapter_root}/fir/{stem}_FIR_Paste_{_rate_label(way.sample_rate)}_{stamp}.txt"] = _coefficient_text(way.coefficients)
                if way.iir_filters:
                    files[f"{adapter_root}/iir/{stem}_IIR_miniDSP_{_rate_label(way.sample_rate)}_{stamp}.txt"] = export_minidsp_biquads(way.iir_filters, way.sample_rate).encode()
        for way in ways:
            composite_path = f"composite/{_slug(way.composite_group, 'Main')}/{_way_stem(way)}_{_rate_label(way.sample_rate)}.wav"
            if way.fir_enabled:
                files[composite_path] = _wav_bytes(way.coefficients, way.sample_rate)
            composite_channels.append({
                "name": way.composite_channel_name or way.way_id,
                "way": way.way_name, "group": way.composite_group,
                "gain": float(way.way_gain_db) + float(output.common_auto_gain_db),
                "polarity": -1 if way.polarity_invert else 1,
                "delay": float(way.delay_ms) * float(way.sample_rate) / 1000.0,
                "wav": composite_path if way.fir_enabled else None, "frd": None,
                "tap_count": int(len(way.coefficients)),
                "center_position": (int(len(way.coefficients)) - 1) / 2.0,
                "time_reference": "tap_center",
                "delay_application": "after_center_alignment",
                "sample_rate_hz": int(way.sample_rate),
            })
            if way.iir_filters:
                stem = _way_stem(way)
                files[f"{root}/generic_iir/{stem}_IIR_Biquad_{_rate_label(way.sample_rate)}_{stamp}.csv"] = export_biquad_csv(way.iir_filters, way.sample_rate).encode()
                files[f"{root}/generic_iir/{stem}_IIR_Biquad_{_rate_label(way.sample_rate)}_{stamp}.json"] = export_biquad_json(way.iir_filters, way.sample_rate).encode()
        bound_inputs = [
            item for item in output.inputs
            if any(binding.input_id == item.id and binding.dsp_id == dsp_id for binding in output.input_bindings)
        ]
        for input_item in bound_inputs:
            if not input_item.iir_filters:
                continue
            stem = _slug(input_item.name, "Input")
            filters = tuple(item for item in input_item.iir_filters if item.enabled)
            files[f"{root}/input_iir/{stem}_Input_IIR_{_rate_label(first.sample_rate)}_{stamp}.json"] = export_biquad_json(filters, first.sample_rate).encode()
            files[f"{root}/input_iir/{stem}_Input_IIR_{_rate_label(first.sample_rate)}_{stamp}.csv"] = export_biquad_csv(filters, first.sample_rate).encode()
            if adapter == "camilladsp":
                files[f"{root}/input_iir/{stem}_Input_IIR_CamillaDSP_{_rate_label(first.sample_rate)}_{stamp}.json"] = export_biquad_json(filters, first.sample_rate).encode()
            elif adapter == "sigmadsp":
                files[f"{root}/input_iir/{stem}_Input_IIR_SigmaStudio_{_rate_label(first.sample_rate)}_{stamp}.txt"] = export_sigmastudio_biquads(filters, first.sample_rate).encode()
            else:
                files[f"{root}/input_iir/{stem}_Input_IIR_miniDSP_{_rate_label(first.sample_rate)}_{stamp}.txt"] = export_minidsp_biquads(filters, first.sample_rate).encode()
        device_manifest.append({
            "id": dsp_id,
            "name": first.device_name,
            "target": first.target,
            "adapter": adapter,
            "adapter_execution": "pipeline" if adapter == "camilladsp" else "manual_import",
            "sample_rate": first.sample_rate,
            "max_input_channels": first.max_input_channels,
            "max_output_channels": first.max_output_channels,
            "channels": [way.output_channel for way in ways],
            "way_fir": {
                str(way.output_channel): bool(way.fir_enabled)
                for way in ways
            },
            "way_design_signatures": {
                str(way.output_channel): way.design_signature
                for way in ways
            },
            "way_controls": {
                str(way.output_channel): {
                    "gain_db": way.way_gain_db,
                    "polarity_invert": way.polarity_invert,
                    "delay_ms": way.delay_ms,
                    "common_output_gain_db": output.common_auto_gain_db,
                }
                for way in ways
            },
        })
    manifest: dict[str, object] = {
        "schema_version": 8,
        "system_revision": output.revision,
        "output_revision": output.revision,
        "design_signature": output.design_signature,
        "signatures": {
            "fir": dict(output.fir_signatures or {}),
            "alignment": output.alignment_signature,
            "export": output.design_signature,
        },
        "common_auto_gain_db": output.common_auto_gain_db,
        "channel_numbering": {"manifest": "zero-based", "filenames": "one-based"},
        "devices": device_manifest,
        "routing": {
            "inputs": [asdict(item) for item in output.inputs],
            "bindings": [item.__dict__ for item in output.input_bindings],
            "routes": [item.__dict__ for item in output.routes],
        },
        "signal_paths": [
            asdict(item) for item in build_final_signal_path_traces(output)
        ],
        "iir_export_safety": dict(iir_safety or {}),
        "composite_engine": {
            "format_version": 2,
            "engine": "Multiway FIR Composite Engine",
            "normalization": "none",
            "composite_groups": list(dict.fromkeys(str(item["group"]) for item in composite_channels)),
            "channels": composite_channels,
        },
    }
    manifest["generated_at"] = generated_at.isoformat(timespec="milliseconds")
    composite_rates = sorted({int(item["sample_rate_hz"]) for item in composite_channels})
    for rate in composite_rates:
        composite_manifest = {
            "format_version": 2, "sample_rate_hz": rate,
            "composite_groups": list(dict.fromkeys(
                str(item["group"])
                for item in composite_channels
                if int(item["sample_rate_hz"]) == rate and item["wav"] is not None
            )),
            "channels": [
                {key: value for key, value in item.items() if key != "sample_rate_hz"}
                for item in composite_channels
                if int(item["sample_rate_hz"]) == rate and item["wav"] is not None
            ],
        }
        path = "manifest.json" if len(composite_rates) == 1 else f"composite/manifest_{_rate_label(rate)}.json"
        files[path] = json.dumps(composite_manifest, ensure_ascii=False, indent=2, sort_keys=True).encode()
    files[f"DSP-System_Manifest_{stamp}.json"] = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode()
    return DSPExportPackage(files=files, manifest=manifest)
