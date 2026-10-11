"""Display exported FIR and IIR separately and compose their complex responses."""
from dataclasses import dataclass, replace

import numpy as np

from phase_fir_designer.iir import cascade_iir_sos
from phase_fir_designer.linear_fir import band_split_output_sos


@dataclass(frozen=True)
class FIRPreviewPolicy:
    """Display decisions derived from effective FIR state, never saved settings."""

    stage: str
    fir_enabled: bool

    @property
    def iir_only_output(self):
        return not self.fir_enabled and self.stage not in {
            "Speaker/Input", "Target Response", "IIR EQ",
        }

    @property
    def output_series_required(self):
        return self.stage == "Export FIR" or self.iir_only_output

    @property
    def chart_uses_input_before_iir(self):
        return self.stage == "Export FIR" or not self.fir_enabled

    @property
    def analysis_uses_input_before_iir(self):
        return self.stage in {"IIR EQ", "Export FIR"} or not self.fir_enabled

    def detail_available(self, detail):
        if self.fir_enabled:
            return True
        return detail != "FIR Filter cofs" and not (
            self.iir_only_output and detail in {"Error", "Response Advisor"}
        )


def export_iir_sos(config, legacy_crossover=None):
    """Use the same EQ, crossover, alignment and polarity as Biquad export."""
    sections = [cascade_iir_sos(config.iir_filters, config.sample_rate),
                band_split_output_sos(config)]
    # A shared Recipe owns its boundaries; old assignment metadata must not
    # apply the same crossover twice.
    if config.band_split_recipe is None and isinstance(legacy_crossover, dict) and legacy_crossover.get("enabled"):
        if int(legacy_crossover.get("coefficient_sample_rate_hz", 0)) != config.sample_rate:
            raise ValueError("Composite IIR SOSのサンプルレートが一致しません。")
        sos = np.asarray(legacy_crossover.get("sos", []), float).reshape(-1, 6).copy()
        if sos.size:
            sos[0, :3] *= int(legacy_crossover.get("lr2_polarity", 1))
            sections.append(sos)
    return np.concatenate(sections)


def add_export_output_series(chart, result, iir_response, *, fir_enabled=True):
    """The chart input is the speaker before IIR EQ, so every stage occurs once."""
    gain, phase = dict(chart.gain_series), dict(chart.phase_series)
    fir_gain = np.asarray(result.realized_gain_db) if fir_enabled else np.zeros_like(chart.frequency)
    fir_phase = np.asarray(result.realized_phase_unwrapped_deg) if fir_enabled else np.zeros_like(chart.frequency)
    iir_gain = 20 * np.log10(np.maximum(abs(iir_response), 1e-12))
    iir_phase = np.rad2deg(np.unwrap(np.angle(iir_response)))
    for values, fir, iir in ((gain, fir_gain, iir_gain), (phase, fir_phase, iir_phase)):
        values.pop("Correction Filter Realized", None)
        if fir_enabled:
            values["FIR Output"] = fir
            values["FIR + IIR Output"] = fir + iir
        else:
            # Internal design results may still contain FIR corrections. None
            # of those curves describe the active output when FIR is bypassed.
            for name in ("FIR Output", "FIR + IIR Output", "Correction Filter",
                         "Correction Filter Realized + IIR Crossover",
                         "Current page delta", "Auto EQ Calculation Input"):
                values.pop(name, None)
        values["IIR Output"] = iir
        if "Speaker/Input" in values:
            values["Speaker + Realized"] = values["Speaker/Input"] + fir + iir
    return replace(chart, gain_series=gain, phase_series=phase)
