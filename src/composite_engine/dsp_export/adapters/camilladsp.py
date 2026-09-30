from __future__ import annotations

import io
import json

import numpy as np
import yaml
from scipy.io import wavfile

from ..common import allpass_sos_rows, channel_root, crossover_sos, normalized_sos
from ..models import CanonicalDSPPackage, DSPProfile, ExportedDSPTarget
from .base import DSPExportAdapter


def _free_biquad(row: np.ndarray, description: str) -> dict[str, object]:
    b0, b1, b2, _a0, a1, a2 = row
    return {"type": "Biquad", "description": description, "parameters": {
        "type": "Free", "b0": float(b0), "b1": float(b1), "b2": float(b2),
        "a1": float(a1), "a2": float(a2),
    }}


class CamillaDSPAdapter(DSPExportAdapter):
    adapter_id = "camilladsp"
    display_name = "CamillaDSP"

    def profiles(self) -> tuple[DSPProfile, ...]:
        return (DSPProfile(
            "camilladsp_standard", self.adapter_id, "Standard",
            supports_fractional_delay=True,
            options={"fir_format": "wav_float32", "config": "yaml"},
        ),)

    def export(self, package: CanonicalDSPPackage, profile: DSPProfile) -> ExportedDSPTarget:
        report = self.validate(package, profile)
        if not report.compatible:
            raise ValueError("CamillaDSP compatibility validation failed")
        files: dict[str, bytes] = {}
        filters: dict[str, object] = {}
        phaseeq_filters: dict[str, object] = {}
        crossover_filters: dict[str, object] = {}
        pipeline: list[dict[str, object]] = []
        channel_count = len(package.channels)
        groups = list(dict.fromkeys(channel.group for channel in package.channels))
        for channel in package.channels:
            stem = channel_root(channel)
            channel_number = channel.output_index - 1
            names: list[str] = []
            if channel.final_fir is not None:
                fir_path = f"coefficients/{stem}_final_fir.wav"
                buffer = io.BytesIO()
                wavfile.write(buffer, channel.sample_rate_hz, np.asarray(channel.final_fir, dtype=np.float32))
                files[fir_path] = buffer.getvalue()
                name = f"{stem}_final_fir"
                filters[name] = {"type": "Conv", "parameters": {"type": "Wav", "filename": fir_path, "channel": 0}}
                names.append(name)
            stages = (
                ("phaseeq_iir", normalized_sos(channel.phaseeq_iir_sos), phaseeq_filters),
                ("baffle_iir", normalized_sos(channel.baffle_iir_sos), phaseeq_filters),
                ("crossover_iir", crossover_sos(channel), crossover_filters),
                ("crossover_allpass", allpass_sos_rows(channel), crossover_filters),
            )
            for stage, sos, partial in stages:
                for index, row in enumerate(sos, start=1):
                    name = f"{stem}_{stage}_{index:02d}"
                    definition = _free_biquad(row, stage)
                    filters[name] = definition
                    partial[name] = definition
                    names.append(name)
            gain_name = f"{stem}_gain_polarity"
            filters[gain_name] = {"type": "Gain", "parameters": {
                "gain": float(channel.gain_db), "inverted": channel.polarity < 0,
                "mute": False, "scale": "dB",
            }}
            names.append(gain_name)
            if channel.total_delay_samples > 0:
                delay_name = f"{stem}_delay"
                filters[delay_name] = {"type": "Delay", "parameters": {
                    "delay": float(channel.total_delay_samples), "unit": "samples",
                    "subsample": not np.isclose(channel.total_delay_samples, round(channel.total_delay_samples)),
                }}
                names.append(delay_name)
            pipeline.append({"type": "Filter", "channels": [channel_number], "names": names})
        mixer = {
            "channels": {"in": len(groups), "out": channel_count},
            "mapping": [
                {"dest": channel.output_index - 1, "sources": [{
                    "channel": groups.index(channel.group), "gain": 0.0, "inverted": False,
                }]}
                for channel in package.channels
            ],
        }
        pipeline.insert(0, {"type": "Mixer", "name": "input_to_multiway_outputs"})
        config = {
            "title": f"PhaseEQ {package.system_name}",
            "description": f"Generated {package.generated_at}",
            "devices": {"samplerate": package.sample_rate_hz, "chunksize": 2048,
                        "capture": {"type": "Stdin", "channels": len(groups), "format": "FLOAT32LE"},
                        "playback": {"type": "Stdout", "channels": channel_count, "format": "FLOAT32LE"}},
            "mixers": {"input_to_multiway_outputs": mixer},
            "filters": filters, "pipeline": pipeline,
        }
        files["camilladsp.yml"] = yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode()
        files["filters_phaseeq_iir.yml"] = yaml.safe_dump({"filters": phaseeq_filters}, sort_keys=False).encode()
        files["filters_crossover_iir.yml"] = yaml.safe_dump({"filters": crossover_filters}, sort_keys=False).encode()
        files["validation.json"] = json.dumps({"yaml_valid": True, "camilladsp_check": "not_run"}, indent=2).encode()
        files["setup_guide.txt"] = (
            "Keep camilladsp.yml and coefficients/ together. Review device names, then run camilladsp --check camilladsp.yml.\n"
        ).encode()
        return ExportedDSPTarget(self.adapter_id, profile.profile_id, files, report)
