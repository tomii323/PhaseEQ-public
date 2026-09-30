"""Versioned Studio bank dispatch; preserved from PhaseEQ v1.19.9."""
import numpy as np
from .filters import (
    generate_2way_filters, generate_3way_filters, generate_4way_filters,
    generate_exclusive_kaiser_firs, generate_residual_multiband_filters,
)
DEFAULT_FIR_CYCLES = 3.5
DEFAULT_FIR_BETA = 12.0
MODE_BANDS = {
    "Fullrange": ["Fullrange"],
    "Fullrange+SUB": ["SUB", "Fullrange"],
    "2Way": ["Low", "High"],
    "2Way+SUB": ["SUB", "Low", "High"],
    "3Way": ["Low", "Mid", "High"],
    "3Way+SUB": ["SUB", "Low", "Mid", "High"],
    "4Way": ["Low", "Low-Mid", "High-Mid", "High"],
    "4Way+SUB": ["SUB", "Low", "Low-Mid", "High-Mid", "High"],
}

def _mode_boundary_values(mode_key, conf):
    """Return ordered cycles, beta and overlap values for every boundary."""
    count = len(MODE_BANDS[mode_key]) - 1
    cycles = list(conf.get("boundary_cycles", ()))
    betas = list(conf.get("boundary_betas", ()))
    overlaps = list(conf.get("boundary_overlaps_oct", ()))
    fallback_cycles = float(conf.get("cycles", DEFAULT_FIR_CYCLES))
    fallback_beta = float(conf.get("beta", DEFAULT_FIR_BETA))
    cycles = (cycles + [fallback_cycles] * count)[:count]
    betas = (betas + [fallback_beta] * count)[:count]
    overlaps = (overlaps + [0.0] * count)[:count]
    return tuple(map(float, cycles)), tuple(map(float, betas)), tuple(map(float, overlaps))

def generate_split_filter_bank(mode_key, fs, conf, crossover_methods):
    """Generate the exact split FIR bank shared by display, Assignment and export."""
    from functools import partial
    from .filters import generate_exclusive_kaiser_firs as exclusive_bank
    generate_exclusive_kaiser_firs = partial(exclusive_bank, lr2_auto_taps=conf.get("lr2_auto_taps", True))
    bands = MODE_BANDS[mode_key]
    cycles = float(conf.get("cycles", DEFAULT_FIR_CYCLES))
    beta = float(conf.get("beta", DEFAULT_FIR_BETA))
    targets = conf.get("boundary_acoustic_targets", [])
    methods = tuple("Through" if i < len(targets) and targets[i] else str(method)
                    for i, method in enumerate(crossover_methods))
    all_kaiser = all(method == "Kaiser FIR" for method in methods)
    if mode_key == "Fullrange":
        unity = {"Fullrange": np.asarray([1.0], dtype=float)}
        return unity, {key: value.copy() for key, value in unity.items()}
    if mode_key == "Fullrange+SUB":
        boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(mode_key, conf)
        before = generate_exclusive_kaiser_firs(
            fs, ("SUB", "Fullrange"), tuple(conf["cross_freqs"]), methods,
            (0.0,), boundary_cycles, boundary_betas,
        )
        realized = generate_exclusive_kaiser_firs(
            fs, ("SUB", "Fullrange"), tuple(conf["cross_freqs"]), methods,
            boundary_overlaps, boundary_cycles, boundary_betas,
        )
        return before, realized
    if mode_key == "2Way":
        fc = conf["cross_freqs"][0]
        if all_kaiser:
            return (
                generate_2way_filters(fs, fc, cycles, beta, overlap_hz=0),
                generate_2way_filters(
                    fs, fc, cycles, beta,
                    overlap_oct=conf.get("kaiser_overlap_oct", 0.0),
                ),
            )
        return (
            generate_exclusive_kaiser_firs(
                fs, ("Low", "High"), (fc,), methods, (0.0,), (cycles,), (beta,),
            ),
            generate_exclusive_kaiser_firs(
                fs, ("Low", "High"), (fc,), methods,
                (conf.get("kaiser_overlap_oct", 0.0),), (cycles,), (beta,),
            ),
        )
    if mode_key == "3Way":
        fc1, fc2 = conf["cross_freqs"]
        low_cycles = float(conf.get("cycles_low", cycles))
        high_cycles = float(conf.get("cycles_high", cycles))
        low_beta = float(conf.get("beta_low", beta))
        high_beta = float(conf.get("beta_high", beta))
        if all_kaiser:
            common = {
                "cycles_low": low_cycles, "beta_low": low_beta,
                "cycles_high": high_cycles, "beta_high": high_beta,
            }
            return (
                generate_3way_filters(
                    fs, fc1, fc2, cycles, beta,
                    overlap1_hz=0, overlap2_hz=0, **common,
                ),
                generate_3way_filters(
                    fs, fc1, fc2, cycles, beta,
                    overlap1_oct=conf.get("kaiser_overlap_low_mid_oct", 0.0),
                    overlap2_oct=conf.get("kaiser_overlap_mid_high_oct", 0.0),
                    **common,
                ),
            )
        cycles_by_boundary = (low_cycles, high_cycles)
        betas_by_boundary = (low_beta, high_beta)
        return (
            generate_exclusive_kaiser_firs(
                fs, ("Low", "Mid", "High"), (fc1, fc2), methods,
                (0.0, 0.0), cycles_by_boundary, betas_by_boundary,
            ),
            generate_exclusive_kaiser_firs(
                fs, ("Low", "Mid", "High"), (fc1, fc2), methods,
                (conf.get("kaiser_overlap_low_mid_oct", 0.0),
                 conf.get("kaiser_overlap_mid_high_oct", 0.0)),
                cycles_by_boundary, betas_by_boundary,
            ),
        )
    if mode_key == "3Way+SUB":
        fc1, fc2, fc3 = conf["cross_freqs"]
        common = {
            "cycles_sub_low": conf.get("cycles_sub_low", cycles),
            "beta_sub_low": conf.get("beta_sub_low", beta),
            "cycles_low_mid": conf.get("cycles_low_mid", cycles),
            "beta_low_mid": conf.get("beta_low_mid", beta),
            "cycles_mid_high": conf.get("cycles_mid_high", cycles),
            "beta_mid_high": conf.get("beta_mid_high", beta),
        }
        if all_kaiser:
            return (
                generate_4way_filters(
                    fs, fc1, fc2, fc3,
                    overlap1_hz=0, overlap2_hz=0, overlap3_hz=0, **common,
                ),
                generate_4way_filters(
                    fs, fc1, fc2, fc3,
                    overlap1_oct=conf.get("kaiser_overlap_sub_low_oct", 0.0),
                    overlap2_oct=conf.get("kaiser_overlap_low_mid_oct", 0.0),
                    overlap3_oct=conf.get("kaiser_overlap_mid_high_oct", 0.0),
                    **common,
                ),
            )
        cycles_by_boundary = (
            common["cycles_sub_low"], common["cycles_low_mid"], common["cycles_mid_high"],
        )
        betas_by_boundary = (
            common["beta_sub_low"], common["beta_low_mid"], common["beta_mid_high"],
        )
        return (
            generate_exclusive_kaiser_firs(
                fs, ("SUB", "Low", "Mid", "High"), (fc1, fc2, fc3), methods,
                (0.0, 0.0, 0.0), cycles_by_boundary, betas_by_boundary,
            ),
            generate_exclusive_kaiser_firs(
                fs, ("SUB", "Low", "Mid", "High"), (fc1, fc2, fc3), methods,
                (conf.get("kaiser_overlap_sub_low_oct", 0.0),
                 conf.get("kaiser_overlap_low_mid_oct", 0.0),
                 conf.get("kaiser_overlap_mid_high_oct", 0.0)),
                cycles_by_boundary, betas_by_boundary,
            ),
        )
    boundary_cycles, boundary_betas, boundary_overlaps = _mode_boundary_values(mode_key, conf)
    if all_kaiser:
        return (
            generate_residual_multiband_filters(
                fs, tuple(bands), tuple(conf["cross_freqs"]),
                boundary_cycles, boundary_betas, tuple(0.0 for _ in boundary_overlaps),
            ),
            generate_residual_multiband_filters(
                fs, tuple(bands), tuple(conf["cross_freqs"]),
                boundary_cycles, boundary_betas, boundary_overlaps,
            ),
        )
    return (
        generate_exclusive_kaiser_firs(
            fs, tuple(bands), tuple(conf["cross_freqs"]), methods,
            tuple(0.0 for _ in boundary_overlaps), boundary_cycles, boundary_betas,
        ),
        generate_exclusive_kaiser_firs(
            fs, tuple(bands), tuple(conf["cross_freqs"]), methods,
            boundary_overlaps, boundary_cycles, boundary_betas,
        ),
    )
