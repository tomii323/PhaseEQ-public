"""Display exported FIR and IIR separately and compose their complex responses."""
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import freqz, sosfreqz
import response_analysis as analysis

from phase_fir_designer.iir import cascade_iir_sos
from phase_fir_designer.linear_fir import band_split_output_sos, linear_fir_overlay
from phase_fir_designer.phase_curves import phase_error_deg


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
    def output_comparison_required(self):
        return self.stage in {"Linear FIR", "Export FIR"}

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
            self.iir_only_output and detail == "Response Advisor"
        )


def output_band_split_sos(config, legacy_crossover=None):
    """Resolve output crossover, alignment and polarity using export ownership."""
    sections = [band_split_output_sos(config)]
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


def export_iir_sos(config, legacy_crossover=None):
    """Use the same EQ, crossover, alignment and polarity as Biquad export."""
    return np.concatenate([
        cascade_iir_sos(config.iir_filters, config.sample_rate),
        output_band_split_sos(config, legacy_crossover),
    ])


def add_iir_band_split_series(chart, config, legacy_crossover=None):
    """Evaluate the output-only IIR on the final chart grid without changing DSP."""
    gain, phase = dict(chart.gain_series), dict(chart.phase_series)
    label = "IIR Band Split"
    # Removal and generation share the current coefficient source, including
    # target-only and empty configurations after a context change.
    gain.pop(label, None)
    phase.pop(label, None)
    sos = output_band_split_sos(config, legacy_crossover)
    if sos.size:
        response = sosfreqz(sos, worN=chart.frequency, fs=config.sample_rate)[1]
        gain[label] = 20 * np.log10(np.maximum(abs(response), 1e-12))
        phase[label] = np.rad2deg(np.unwrap(np.angle(response)))
    return replace(chart, gain_series=gain, phase_series=phase)


OUTPUT_TARGET_LABEL = "Target + Band Split"


def output_band_split_response(config, frequency, legacy_crossover=None, *, fir_enabled, cache=None):
    """Use direct output stages only; acoustic-target stages are already in Target."""
    cache = cache if cache is not None else analysis.AnalysisCache()
    frequency = np.asarray(frequency, float)
    sos = output_band_split_sos(config, legacy_crossover)
    key = analysis.digest("output_band_split", config.sample_rate,
                          config.linear_fir_filters, config.band_split_recipe,
                          frequency, sos, fir_enabled)

    def evaluate():
        response = (sosfreqz(sos, worN=frequency, fs=config.sample_rate)[1] if sos.size
                    else np.ones_like(frequency, complex))
        # Bypassed FIR must not generate or read a hidden FIR design.
        if fir_enabled:
            overlay = linear_fir_overlay(config)
            if overlay is not None:
                n = 2 * frequency.size - 1
                if len(overlay) <= n and np.allclose(
                        frequency, np.fft.rfftfreq(n, 1 / config.sample_rate)):
                    fir = analysis.fir_response(cache, overlay, config.sample_rate, n).response
                else:
                    fir = freqz(overlay, worN=frequency, fs=config.sample_rate)[1]
                    fir *= np.exp(2j * np.pi * frequency * (len(overlay)-1) / 2 / config.sample_rate)
                response = response * fir
        return analysis.frozen(response)

    return cache.get(key, evaluate)


def add_output_target_series(chart, config, legacy_crossover=None, *, fir_enabled, cache=None):
    """Compose the visible acoustic target with the effective output band split."""
    gain, phase = dict(chart.gain_series), dict(chart.phase_series)
    gain.pop(OUTPUT_TARGET_LABEL, None)
    phase.pop(OUTPUT_TARGET_LABEL, None)
    if "Target Response" in gain:
        response = output_band_split_response(
            config, chart.frequency, legacy_crossover, fir_enabled=fir_enabled, cache=cache)
        gain[OUTPUT_TARGET_LABEL] = gain["Target Response"] + 20*np.log10(np.maximum(abs(response), 1e-12))
        if "Target Response" in phase:
            phase[OUTPUT_TARGET_LABEL] = phase["Target Response"] + np.rad2deg(np.unwrap(np.angle(response)))
    return replace(chart, gain_series=gain, phase_series=phase)


def output_error_metrics(chart):
    """Compare Speaker + Realized against the same target shown in the chart."""
    missing = np.full_like(chart.frequency, np.nan, dtype=float)
    actual_gain = np.asarray(chart.gain_series.get("Speaker + Realized", missing))
    target_gain = np.asarray(chart.gain_series.get(OUTPUT_TARGET_LABEL, missing))
    actual_phase = np.asarray(chart.phase_series.get("Speaker + Realized", missing))
    target_phase = np.asarray(chart.phase_series.get(OUTPUT_TARGET_LABEL, missing))
    return actual_gain - target_gain, phase_error_deg(target_phase, actual_phase), np.minimum(actual_gain, target_gain)


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
