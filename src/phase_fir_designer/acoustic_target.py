"""Derive an acoustic goal without changing the user's base target."""
from dataclasses import replace

import numpy as np

from crossover_engine.acoustic_target import boundary_response, enabled as recipe_enabled, response
from .config import AutoEQConfig, SpeakerResponse


def enabled(config):
    if config.band_split_recipe is not None:
        return recipe_enabled(config.band_split_recipe)
    return any(item.enabled and item.acoustic_target for item in config.linear_fir_filters)


def crossover_response(config, frequency):
    if config.band_split_recipe is not None:
        return response(config.band_split_recipe, frequency)
    h = np.ones(np.asarray(frequency).shape, complex)
    for item in config.linear_fir_filters:
        if item.enabled and item.acoustic_target:
            h *= boundary_response(config.sample_rate, frequency, item.fc,
                                   "highpass" if item.mode == "hp" else "lowpass",
                                   4 if "4" in item.response else 2,
                                   item.response.startswith("linear_phase"))
    return h


def effective_target(config):
    """Used by both target plots and correction evaluation; raw target survives OFF."""
    if not enabled(config):
        return config.target_response
    base = config.target_response
    frequency = np.unique(np.concatenate((np.geomspace(1, config.sample_rate/2, 8193),
                                         np.asarray(base.frequency) if base else [])))
    frequency = frequency[(frequency > 0) & (frequency <= config.sample_rate/2)]
    h = crossover_response(config, frequency)
    gain = 20*np.log10(np.maximum(np.abs(h), 1e-12))
    phase = np.rad2deg(np.unwrap(np.angle(h)))
    if base is not None:
        # DC is valid in FFT-derived targets, but has no logarithmic position.
        # Hold the first positive-frequency value below the measured band.
        base_frequency = np.asarray(base.frequency, dtype=float)
        positive = base_frequency > 0
        if not np.any(positive):
            raise ValueError("Targetには0 Hzより高い周波数のデータが必要です。")
        from response_completion import complete_response
        base_gain, base_phase, _ = complete_response(
            base.frequency, base.gain_db, base.phase_deg, frequency, gain_axis="log")
        gain += base_gain
        if base_phase is not None:
            phase += base_phase
    return SpeakerResponse(frequency=frequency.tolist(), gain_db=gain.tolist(), phase_deg=phase.tolist())


def automatic_correction_active(config):
    """A target definition alone does not enable automatic FIR matching."""
    return bool(config.acoustic_correction_enabled and enabled(config)
                and config.speaker_response is not None
                and config.speaker_response.phase_deg is not None)


def correction_config(config):
    """Explicit acoustic matching uses measured range and both response components."""
    if not automatic_correction_active(config):
        return config
    speaker = config.speaker_response
    lo = max(5.0, min(speaker.frequency))
    hi = min(config.sample_rate/2, max(speaker.frequency))
    return replace(config, auto_eq=AutoEQConfig(
        gain_enabled=True, phase_enabled=True, f_min=lo, f_max=hi,
        smoothing_fraction=0.0, edge_smooth_percent=0.0,
        gain_edge_smooth_oct=0.0, phase_edge_smooth_oct=0.0,
        match_target_level=True, max_boost_db=6.0, max_cut_db=120.0,
        limit_mode="full"))


def continue_boundary_gain_correction(config, frequency, correction, manual_gain=None):
    """Hold edge attenuation through DC/Nyquist without extrapolating measurements."""
    speaker = config.speaker_response
    if (not config.acoustic_boundary_continuation_enabled
        or not automatic_correction_active(config)):
        return correction
    f = np.asarray(frequency, dtype=float)
    valid = np.flatnonzero((f >= max(5.0, min(speaker.frequency)))
                          & (f <= min(config.sample_rate/2, max(speaker.frequency))))
    if not valid.size:
        return correction
    manual = np.zeros_like(f) if manual_gain is None else np.asarray(manual_gain)
    result = np.asarray(correction).copy()
    for index, outside in (
        (valid[0], (f >= 0) & (f < f[valid[0]])),
        (valid[-1], (f > f[valid[-1]]) & (f <= config.sample_rate/2)),
    ):
        edge = min(0.0, float(correction[index] + manual[index]))
        # Preserve stronger manual cuts and never introduce out-of-range gain.
        # Unity at either endpoint leaks into a short FIR's attenuation band.
        result[outside] = np.minimum(edge - manual[outside], 0.0)
    return result
