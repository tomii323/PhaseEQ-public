from __future__ import annotations

import json

from ..common import allpass_sos_rows, channel_root, crossover_sos, csv_bytes, normalized_sos
from ..models import CanonicalDSPPackage, DSPProfile, ExportedDSPTarget
from .base import DSPExportAdapter


class GenericSOSAdapter(DSPExportAdapter):
    adapter_id = "generic_sos"
    display_name = "Generic SOS"

    def profiles(self) -> tuple[DSPProfile, ...]:
        return (DSPProfile("generic_sos", self.adapter_id, "Standard SOS"),)

    def export(self, package: CanonicalDSPPackage, profile: DSPProfile) -> ExportedDSPTarget:
        report = self.validate(package, profile)
        files: dict[str, bytes] = {}
        for channel in package.channels:
            root = f"channels/{channel_root(channel)}"
            for filename, sos in (
                ("phaseeq_iir_sos.csv", normalized_sos(channel.phaseeq_iir_sos)),
                ("baffle_iir_sos.csv", normalized_sos(channel.baffle_iir_sos)),
                ("crossover_iir_sos.csv", crossover_sos(channel)),
                ("crossover_allpass_sos.csv", allpass_sos_rows(channel)),
            ):
                files[f"{root}/{filename}"] = csv_bytes(
                    ("section", "b0", "b1", "b2", "a0", "a1", "a2"),
                    [(index, *row) for index, row in enumerate(sos, start=1)],
                )
        return ExportedDSPTarget(self.adapter_id, profile.profile_id, files, report)


class GenericParametersAdapter(DSPExportAdapter):
    adapter_id = "generic_parameters"
    display_name = "Generic Parameters"

    def profiles(self) -> tuple[DSPProfile, ...]:
        return (DSPProfile(
            "generic_parameters_iir_only", self.adapter_id, "IIR only · Hz / dB / Q",
            supports_fir=False, supports_allpass=True, supports_fractional_delay=False,
        ),)

    def export(self, package: CanonicalDSPPackage, profile: DSPProfile) -> ExportedDSPTarget:
        report = self.validate(package, profile)
        if not report.compatible:
            raise ValueError("Generic Parameters compatibility validation failed")
        files: dict[str, bytes] = {}
        for channel in package.channels:
            root = f"channels/{channel_root(channel)}"
            parameter_rows = []
            for index, item in enumerate(channel.phaseeq_iir_parameters, start=1):
                parameter_rows.append((index, item.get("type", ""), item.get("frequency_hz", item.get("freq", "")), item.get("gain_db", item.get("gain", "")), item.get("q", "")))
            files[f"{root}/phaseeq_iir_parameters.csv"] = csv_bytes(
                ("section", "type", "frequency_hz", "gain_db", "q"), parameter_rows,
            )
            baffle = channel.baffle_iir_parameters or {}
            files[f"{root}/baffle_iir_parameters.csv"] = csv_bytes(
                ("section", "type", "frequency_hz", "gain_db", "q"),
                ([
                    (1, baffle.get("type", "high_shelf"),
                     baffle.get("frequency_hz", ""), baffle.get("gain_db", ""),
                     baffle.get("q", ""))
                ] if baffle else []),
            )
            files[f"{root}/crossover_parameters.json"] = json.dumps({
                "iir_crossover": channel.crossover.to_dict(),
                "crossover_allpass": [section.to_dict() for section in channel.crossover_allpass],
            }, ensure_ascii=False, indent=2).encode()
        return ExportedDSPTarget(self.adapter_id, profile.profile_id, files, report)
