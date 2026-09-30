from __future__ import annotations

import json

import numpy as np

from ..common import allpass_sos_rows, channel_root, crossover_sos, csv_bytes, normalized_sos
from ..models import CanonicalDSPPackage, DSPProfile, ExportedDSPTarget
from .base import DSPExportAdapter


def _sigma_coefficients(sos: np.ndarray) -> bytes:
    # General 2nd Order load order is B0, B1, B2, A1, A2. SigmaStudio stores
    # feedback additions, therefore scipy denominator signs are inverted.
    values = [value for row in sos for value in (row[0], row[1], row[2], -row[4], -row[5])]
    return (("\n".join(f"{float(value):.15g}" for value in values) + "\n") if values else "").encode()


def _fixed_5_23(values: bytes) -> bytes:
    numbers = [float(line) for line in values.decode().splitlines() if line.strip()]
    rows = []
    for value in numbers:
        clipped = min(max(value, -16.0), 15.99999988079071)
        raw = int(round(clipped * (1 << 23))) & 0x0FFFFFFF
        rows.append(f"{raw:08X}")
    return (("\n".join(rows) + "\n") if rows else "").encode()


class SigmaStudioAdapter(DSPExportAdapter):
    adapter_id = "sigmastudio"
    display_name = "SigmaStudio"

    def profiles(self) -> tuple[DSPProfile, ...]:
        return (DSPProfile(
            "sigmastudio_5_23", self.adapter_id, "SigmaDSP 5.23",
            max_iir_sections=20,
            supports_fractional_delay=False,
            options={"numeric_format": "5.23", "coefficient_order": ["B0", "B1", "B2", "A1", "A2"]},
        ),)

    def export(self, package: CanonicalDSPPackage, profile: DSPProfile) -> ExportedDSPTarget:
        report = self.validate(package, profile)
        if not report.compatible:
            raise ValueError("SigmaStudio compatibility validation failed")
        files: dict[str, bytes] = {"device_profile.json": json.dumps(profile.__dict__, ensure_ascii=False, indent=2).encode()}
        block_rows: list[tuple[object, ...]] = []
        for channel in package.channels:
            root = f"channels/{channel_root(channel)}"
            fir = np.asarray(channel.final_fir if channel.final_fir is not None else [], dtype=float)
            fir_text = (("\n".join(f"{value:.17g}" for value in fir) + "\n") if len(fir) else "").encode()
            if channel.final_fir is not None:
                files[f"{root}/final_fir_float.txt"] = fir_text
                files[f"{root}/final_fir_5_23.hex"] = _fixed_5_23(fir_text)
            phaseeq = normalized_sos(channel.phaseeq_iir_sos)
            baffle = normalized_sos(channel.baffle_iir_sos)
            crossover = crossover_sos(channel)
            allpass = allpass_sos_rows(channel)
            phaseeq_text = _sigma_coefficients(phaseeq)
            baffle_text = _sigma_coefficients(baffle)
            crossover_text = _sigma_coefficients(
                np.vstack([crossover, allpass]) if len(crossover) or len(allpass) else np.empty((0, 6))
            )
            files[f"{root}/phaseeq_iir_coefficients.txt"] = phaseeq_text
            files[f"{root}/baffle_iir_coefficients.txt"] = baffle_text
            files[f"{root}/crossover_iir_coefficients.txt"] = crossover_text
            files[f"{root}/phaseeq_iir_5_23.hex"] = _fixed_5_23(phaseeq_text)
            files[f"{root}/baffle_iir_5_23.hex"] = _fixed_5_23(baffle_text)
            files[f"{root}/crossover_iir_5_23.hex"] = _fixed_5_23(crossover_text)
            section = 0
            for stage, values in (("phaseeq_iir", phaseeq), ("baffle_iir", baffle), ("crossover_iir", crossover), ("crossover_allpass", allpass)):
                for _row in values:
                    section += 1
                    block_rows.append((channel.output_index, channel.name, section, stage, f"{channel_root(channel)}_{stage}_{section:02d}"))
            files[f"{root}/channel_settings.csv"] = csv_bytes(
                ("Channel", "Gain dB", "Polarity", "Delay samples"),
                [(channel.name, channel.gain_db, channel.polarity, channel.total_delay_samples)],
            )
        files["block_map.csv"] = csv_bytes(("Output", "Channel", "Section", "Stage", "Suggested block"), block_rows)
        files["setup_guide.txt"] = (
            "Load FIR and General 2nd Order coefficients into the matching SigmaStudio blocks.\n"
            "Coefficient order: B0, B1, B2, A1, A2. Recompile and Export System Files in SigmaStudio.\n"
        ).encode()
        return ExportedDSPTarget(self.adapter_id, profile.profile_id, files, report)
