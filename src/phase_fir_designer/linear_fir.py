from __future__ import annotations

import numpy as np
from scipy import signal

from .config import DesignConfig, IIRFilter, LinearFIRFilter


FIR_BAND_SPLIT_METHODS = frozenset({"kaiser", "linear_phase_lr2", "linear_phase_lr4"})
IIR_BAND_SPLIT_METHODS = frozenset({"iir_lr2", "iir_lr4"})


def multiway_kaiser_taps(sample_rate: float, fc: float, cycles: float) -> int:
    from crossover_engine.filters import odd_number
    return max(3, odd_number(float(sample_rate) / max(float(fc), 1.0) * float(cycles)))


def multiway_clip_cutoff_hz(fc: float, sample_rate: float) -> float:
    nyquist = float(sample_rate) / 2.0
    return float(min(max(float(fc), 1.0), nyquist - 1.0))


def apply_linear_fir_filters(config: DesignConfig, fir: np.ndarray) -> np.ndarray:
    """Convolve the unmasked crossover after EQ-only masking and FIR design."""
    output = finite_fir(fir)
    overlay = linear_fir_overlay(config)
    if overlay is None:
        return output
    overlay = finite_fir(overlay)
    # Truncating the convolution to the correction length destroys the
    # crossover's stopband attenuation. Keep the complete output boundary.
    return signal.fftconvolve(output, overlay)


def linear_fir_overlay(config: DesignConfig) -> np.ndarray | None:
    if config.band_split_recipe is not None:
        from crossover_engine.recipe import generate
        recipe = config.band_split_recipe
        if not recipe.get("fir_enabled", True):
            return None
        return generate(recipe)[recipe["way"]]
    linear_filters = [
        item
        for item in config.linear_fir_filters
        if item.enabled and not item.acoustic_target and item.response in FIR_BAND_SPLIT_METHODS
    ]
    overlay: np.ndarray | None = None
    for item in linear_filters:
        item_overlay = design_linear_fir(config, item)
        if overlay is None:
            overlay = item_overlay
        else:
            overlay = signal.fftconvolve(finite_fir(overlay), finite_fir(item_overlay))
    return overlay


def add_linear_fir_target_response(
    config: DesignConfig,
    target_gain_db: np.ndarray,
    target_phase_unwrapped_deg: np.ndarray,
    fft_size: int,
    realized_response_func,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = linear_fir_overlay(config)
    if overlay is None:
        return target_gain_db, target_phase_unwrapped_deg
    overlay_response = realized_response_func(overlay, config.sample_rate, fft_size)
    overlay_gain_db = 20.0 * np.log10(np.maximum(np.abs(overlay_response), 1e-12))
    overlay_phase_unwrapped_deg = np.rad2deg(np.unwrap(np.angle(overlay_response)))
    return (
        np.asarray(target_gain_db, dtype=float) + overlay_gain_db,
        np.asarray(target_phase_unwrapped_deg, dtype=float) + overlay_phase_unwrapped_deg,
    )


def design_kaiser_linear_fir(config: DesignConfig, mode: str, fc: float, cycles: float, beta: float) -> np.ndarray:
    taps = multiway_kaiser_taps(config.sample_rate, fc, cycles)
    cutoff = multiway_clip_cutoff_hz(fc, config.sample_rate)
    return signal.firwin(
        taps,
        cutoff,
        window=("kaiser", float(beta)),
        pass_zero=(mode == "lp"),
        scale=True,
        fs=float(config.sample_rate),
    ).astype(np.float64, copy=False)


def design_linear_fir(
    config: DesignConfig,
    item: LinearFIRFilter,
    *,
    mode: str | None = None,
) -> np.ndarray:
    """Generate the coefficient response named by ``LinearFIRFilter.response``."""
    output_mode = item.mode if mode is None else mode
    if config.band_split_recipe is not None:
        from crossover_engine.recipe import boundary_details
        from crossover_engine.filters import _linear_phase_lr2_fir, _linear_phase_lr4_fir, firwin, _clip_cutoff
        recipe = config.band_split_recipe
        index = recipe["ordered_ways"].index(recipe["way"])
        boundary_index = index - 1 if item.mode == "hp" else index
        detail = boundary_details(recipe)[boundary_index]
        cutoff = detail["highpass_cutoff_hz" if output_mode == "hp" else "lowpass_cutoff_hz"]
        taps = detail["natural_taps"]
        if taps is None:
            raise ValueError("IIR boundary cannot generate a FIR section")
        if detail["method"] == "Linear-phase LR4 FIR":
            return _linear_phase_lr4_fir(config.sample_rate, output_mode, cutoff, taps)
        if detail["method"] == "Linear-phase LR2 FIR":
            return _linear_phase_lr2_fir(config.sample_rate, output_mode, cutoff, taps)
        return firwin(taps, _clip_cutoff(cutoff, config.sample_rate)/(config.sample_rate/2),
                      window=("kaiser", detail["beta"]), pass_zero=output_mode == "lp")
    if item.response == "linear_phase_lr4":
        from crossover_engine.lr4_taps import automatic_lr4_taps
        from crossover_engine.filters import _linear_phase_lr4_fir
        plan = automatic_lr4_taps(int(config.sample_rate),
                                  float(item.base_crossover_hz or item.fc), item.overlap_oct)
        return _linear_phase_lr4_fir(config.sample_rate, output_mode, item.fc, plan.taps)
    if item.response == "linear_phase_lr2":
        if item.lr2_auto_taps:
            from crossover_engine.lr2_taps import automatic_lr2_taps
            from crossover_engine.filters import _linear_phase_lr2_fir
            plan = automatic_lr2_taps(int(config.sample_rate),
                                      float(item.base_crossover_hz or item.fc), item.overlap_oct)
            return _linear_phase_lr2_fir(config.sample_rate, output_mode, item.fc, plan.taps)
        if item.base_crossover_hz is not None:
            from crossover_engine.filters import _linear_phase_lr2_fir
            return _linear_phase_lr2_fir(config.sample_rate, output_mode, item.fc,
                multiway_kaiser_taps(config.sample_rate, item.base_crossover_hz, item.cycles))
        return design_linear_phase_lr2_fir(
            config.sample_rate,
            output_mode,
            item.fc,
            taps=config.taps,
        )
    if item.response == "kaiser":
        if item.base_crossover_hz is not None:
            from crossover_engine.filters import firwin, _clip_cutoff
            taps = multiway_kaiser_taps(config.sample_rate, item.base_crossover_hz, item.cycles)
            return firwin(taps, _clip_cutoff(item.fc, config.sample_rate)/(config.sample_rate/2),
                          window=("kaiser", item.beta), pass_zero=output_mode == "lp")
        return design_kaiser_linear_fir(
            config,
            output_mode,
            item.fc,
            item.cycles,
            item.beta,
        )
    raise ValueError(f"unsupported Linear FIR response: {item.response}")


def band_split_iir_filters(config: DesignConfig) -> tuple[IIRFilter, ...]:
    """Return only the output-band-split Biquads, in HP then LP order."""
    if config.band_split_recipe is not None:
        from crossover_engine.recipe import iir_config
        realized = iir_config(config.band_split_recipe)
        return tuple(IIRFilter(kind=kind, fc=fc, family="linkwitz_riley", order=order)
                     for kind, fc, order in (
                         ("high_pass", realized.highpass_hz, realized.highpass_order),
                         ("low_pass", realized.lowpass_hz, realized.lowpass_order)) if order)
    filters: list[IIRFilter] = []
    for item in sorted(
        (item for item in config.linear_fir_filters if item.enabled),
        key=lambda value: 0 if value.mode == "hp" else 1,
    ):
        if item.acoustic_target or item.response not in IIR_BAND_SPLIT_METHODS:
            continue
        filters.append(
            IIRFilter(
                kind="high_pass" if item.mode == "hp" else "low_pass",
                fc=float(item.fc),
                family="linkwitz_riley",
                order=2 if item.response == "iir_lr2" else 4,
                enabled=True,
                origin="manual",
            )
        )
    return tuple(filters)


def design_linear_phase_lr2_fir(
    sample_rate: int,
    mode: str,
    frequency_hz: float,
    *,
    taps: int,
) -> np.ndarray:
    """Generate a complementary FIR from the exact Linear Phase LR2 target."""
    rate = int(sample_rate)
    frequency = float(frequency_hz)
    output_taps = int(taps)
    if rate <= 0:
        raise ValueError("sample rate must be positive")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("LR2 crossover frequency must be finite and positive")
    if frequency >= rate / 2.0:
        raise ValueError("Linear Phase LR2 frequency must be below Nyquist")
    if mode not in {"lp", "hp"}:
        raise ValueError("Linear Phase LR2 mode must be lp or hp")
    if output_taps < 3 or output_taps % 2 == 0:
        raise ValueError("Linear Phase LR2 taps must be an odd integer of at least 3")

    # The Target editor defines LR2 directly as 1 / (1 + (f / fc)^2).
    # Sample that same magnitude on a dense grid and turn it into a Type-I
    # linear-phase FIR.  No Kaiser window or Kaiser cycles/beta are involved.
    grid_intervals = 1
    while grid_intervals < max(output_taps, 16_384):
        grid_intervals *= 2
    grid_size = grid_intervals + 1
    grid_frequency = np.linspace(0.0, rate / 2.0, grid_size)
    target_low_pass = 1.0 / (1.0 + (grid_frequency / frequency) ** 2)
    low_pass = signal.firwin2(
        output_taps,
        grid_frequency,
        target_low_pass,
        nfreqs=grid_size,
        window=None,
        fs=float(rate),
    ).astype(np.float64, copy=False)
    low_pass = 0.5 * (low_pass + low_pass[::-1])
    if mode == "lp":
        return low_pass
    delta = np.zeros(output_taps, dtype=np.float64)
    delta[output_taps // 2] = 1.0
    return delta - low_pass


def design_mid_complement_linear_fir(
    config: DesignConfig,
    hp_filter: LinearFIRFilter,
    lp_filter: LinearFIRFilter,
) -> np.ndarray:
    # Mid is a complement band: low-side LP and high-side HP are subtracted
    # from a centered delta so Low + Mid + High stays structurally flat.
    low_side, high_side = sorted((hp_filter, lp_filter), key=lambda item: float(item.fc))
    low = design_linear_fir(config, low_side, mode="lp")
    high = design_linear_fir(config, high_side, mode="hp")
    target_size = max(low.size, high.size)
    if target_size % 2 == 0:
        target_size += 1
    low = center_or_crop_fir(low, target_size)
    high = center_or_crop_fir(high, target_size)
    delta = np.zeros(target_size, dtype=np.float64)
    delta[target_size // 2] = 1.0
    return delta - low - high


def center_or_crop_fir(fir: np.ndarray, target_size: int) -> np.ndarray:
    source = np.asarray(fir, dtype=np.float64)
    target_size = int(target_size)
    if source.size == target_size:
        return source.copy()
    if source.size < target_size:
        pad_total = target_size - source.size
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(source, (pad_left, pad_right))
    start = (source.size - target_size) // 2
    return source[start : start + target_size].copy()


def finite_fir(fir: np.ndarray) -> np.ndarray:
    source = np.asarray(fir, dtype=np.float64)
    if source.size == 0 or np.all(np.isfinite(source)):
        return source.copy()
    return np.where(np.isfinite(source), source, 0.0).astype(np.float64, copy=False)


def band_split_alignment_sos(config: DesignConfig) -> np.ndarray:
    from crossover_engine.alignment import alignment_sos
    return alignment_sos(config.band_split_recipe)


def band_split_output_auxiliary_sos(config: DesignConfig) -> np.ndarray:
    from crossover_engine.alignment import output_auxiliary_sos
    return output_auxiliary_sos(config.band_split_recipe)


def band_split_output_sos(config: DesignConfig) -> np.ndarray:
    """Complete output Biquads, valid with FIR output disabled."""
    from .iir import cascade_iir_sos
    return np.concatenate((cascade_iir_sos(band_split_iir_filters(config), config.sample_rate),
                           band_split_output_auxiliary_sos(config)))
