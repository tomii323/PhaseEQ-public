"""Acoustic LR goals: IIR magnitude, optionally with phase set to zero."""
import numpy as np
from scipy.signal import butter, sosfreqz


def enabled(recipe):
    if not isinstance(recipe, dict):
        return False
    from .alignment import target_enabled, target_crossover_polarity
    if target_enabled(recipe) or target_crossover_polarity(recipe) == -1:
        return True
    ways = recipe.get("ordered_ways", [])
    if recipe.get("way") not in ways:
        return False
    index = ways.index(recipe["way"])
    boundaries = recipe.get("boundaries", [])
    return any(boundaries[j].get("acoustic_target", False)
               for j in (index-1, index) if 0 <= j < len(boundaries))


def boundary_response(sample_rate, frequency, cutoff, side, order, linear_phase):
    sos = butter(order//2, cutoff, btype=side, fs=sample_rate, output="sos")
    h = sosfreqz(np.concatenate((sos, sos)), worN=np.asarray(frequency, float), fs=sample_rate)[1]
    return np.abs(h).astype(complex) if linear_phase else h


def response(recipe, frequency):
    from .recipe import validate
    from .filters import kaiser_overlap_edges_hz
    validate(recipe)
    frequency = np.asarray(frequency, float)
    result = np.ones(frequency.shape, complex)
    index = recipe["ordered_ways"].index(recipe["way"])
    for side, j in (("highpass", index-1), ("lowpass", index)):
        if not 0 <= j < len(recipe["boundaries"]):
            continue
        b = recipe["boundaries"][j]
        if not b.get("acoustic_target", False):
            continue
        method = b["method"]
        if method == "Through":
            continue
        if method == "Kaiser FIR":
            raise ValueError("音響ターゲットにはLR2／LR4を選択してください。")
        order = 4 if "4" in method else 2
        lp, hp = kaiser_overlap_edges_hz(b["base_crossover_hz"], b["overlap_oct"])
        # User-requested construction: take the *digital IIR* magnitude;
        # zero phase is explicit, never a minimum-phase reconstruction.
        result *= boundary_response(recipe["sample_rate_hz"], frequency,
                                    hp if side == "highpass" else lp, side, order,
                                    method.startswith("Linear-phase"))
    from .alignment import alignment_sos, target_enabled, separate_polarity, target_crossover_polarity
    result *= target_crossover_polarity(recipe)
    if target_enabled(recipe):
        from .recipe import iir_config
        if not separate_polarity(recipe):
            result *= iir_config(recipe).lr2_polarity
        sos = alignment_sos(recipe, target=True)
        if sos.size:
            result *= sosfreqz(sos, worN=frequency, fs=recipe["sample_rate_hz"])[1]
    return result
