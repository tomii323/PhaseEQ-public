from __future__ import annotations

import json

import numpy as np

from ..common import allpass_sos_rows, channel_root, crossover_sos, normalized_sos
from ..models import CanonicalDSPPackage, DSPProfile, ExportedDSPTarget
from .base import DSPExportAdapter


def _minidsp_biquads(sos: np.ndarray) -> bytes:
    blocks: list[str] = []
    for index, row in enumerate(sos, start=1):
        b0, b1, b2, _a0, a1, a2 = row
        # miniDSP Advanced Biquad uses feedback signs opposite scipy SOS.
        blocks.append(
            f"biquad{index},\n"
            f"b0={b0:.15g},\nb1={b1:.15g},\nb2={b2:.15g},\n"
            f"a1={-a1:.15g},\na2={-a2:.15g},\n"
        )
    return ("\n".join(blocks) + ("\n" if blocks else "")).encode()


class MiniDSPAdapter(DSPExportAdapter):
    adapter_id = "minidsp"
    display_name = "miniDSP"

    def profiles(self) -> tuple[DSPProfile, ...]:
        return (DSPProfile(
            "minidsp_user_defined", self.adapter_id, "User-defined",
            sample_rates_hz=(48_000, 96_000),
            max_iir_sections=16, supports_allpass=True,
            supports_fractional_delay=True,
            options={"fir_format": "float32_raw_le", "iir_format": "advanced_biquad"},
        ),)

    def export(self, package: CanonicalDSPPackage, profile: DSPProfile) -> ExportedDSPTarget:
        report = self.validate(package, profile)
        if not report.compatible:
            raise ValueError("miniDSP compatibility validation failed")
        files: dict[str, bytes] = {}
        for channel in package.channels:
            root = f"channels/{channel_root(channel)}"
            fir = np.asarray(channel.final_fir if channel.final_fir is not None else [], dtype="<f4")
            if channel.final_fir is not None:
                files[f"{root}/final_fir.bin"] = fir.tobytes()
            phaseeq = normalized_sos(channel.phaseeq_iir_sos)
            baffle = normalized_sos(channel.baffle_iir_sos)
            xover = crossover_sos(channel)
            allpass = allpass_sos_rows(channel)
            files[f"{root}/phaseeq_iir_biquad.txt"] = _minidsp_biquads(phaseeq)
            files[f"{root}/baffle_iir_biquad.txt"] = _minidsp_biquads(baffle)
            files[f"{root}/crossover_iir_biquad.txt"] = _minidsp_biquads(
                np.vstack([xover, allpass]) if len(xover) or len(allpass) else np.empty((0, 6))
            )
            files[f"{root}/channel_settings.json"] = json.dumps({
                "output_index": channel.output_index, "channel": channel.name,
                "sample_rate_hz": channel.sample_rate_hz,
                "fir_enabled": channel.final_fir is not None,
                "fir_taps": len(fir) if channel.final_fir is not None else None,
                "phaseeq_iir_sections": len(phaseeq), "baffle_iir_sections": len(baffle),
                "crossover_iir_sections": len(xover),
                "crossover_allpass_sections": len(allpass), "gain_db": channel.gain_db,
                "polarity": channel.polarity, "delay_samples": channel.total_delay_samples,
                "delay_ms": channel.total_delay_samples / channel.sample_rate_hz * 1000.0,
            }, ensure_ascii=False, indent=2).encode()
        files["device_profile.json"] = json.dumps(profile.__dict__, ensure_ascii=False, indent=2).encode()
        files["setup_guide.txt"] = (
            "For channels that contain final_fir.bin, load it into the FIR block.\n"
            "Load phaseeq_iir_biquad.txt, baffle_iir_biquad.txt and crossover_iir_biquad.txt into separate Advanced Biquad blocks.\n"
            "Then apply Gain, Polarity and Delay from channel_settings.json.\n"
        ).encode()
        return ExportedDSPTarget(self.adapter_id, profile.profile_id, files, report)
