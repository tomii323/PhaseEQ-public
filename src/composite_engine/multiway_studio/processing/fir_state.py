from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


FIR_CROSSOVER_METHODS = frozenset({"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"})
DEFAULT_MANUAL_FIR_TAPS = 1023


@dataclass(frozen=True)
class BandFIRState:
    enabled: bool
    tap_source: Literal["none", "auto_split", "manual"]
    tap_count: int | None
    split_fir_required: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "tap_source": self.tap_source,
            "tap_count": self.tap_count,
            "split_fir_required": self.split_fir_required,
        }


def bands_requiring_split_fir(
    ordered_bands: Sequence[str], crossover_methods: Sequence[str],
) -> frozenset[str]:
    bands = tuple(str(value) for value in ordered_bands)
    methods = tuple(str(value) for value in crossover_methods)
    if len(methods) != max(0, len(bands) - 1):
        raise ValueError("crossover method count does not match bands")
    required: set[str] = set()
    for index, method in enumerate(methods):
        if method in FIR_CROSSOVER_METHODS:
            required.update((bands[index], bands[index + 1]))
    return frozenset(required)


def resolve_band_fir_states(
    ordered_bands: Sequence[str],
    crossover_methods: Sequence[str],
    configured_taps: Mapping[str, int],
    *,
    fir_output_enabled: bool,
    split_firs: Mapping[str, np.ndarray] | None = None,
    auto_crop: bool = False,
) -> dict[str, BandFIRState]:
    bands = tuple(str(value) for value in ordered_bands)
    required = bands_requiring_split_fir(bands, crossover_methods)
    states: dict[str, BandFIRState] = {}
    for band in bands:
        split_required = band in required
        if not fir_output_enabled:
            states[band] = BandFIRState(False, "none", None, split_required)
            continue
        manual_taps = int(configured_taps.get(band, 0) or 0)
        if manual_taps > 0:
            states[band] = BandFIRState(True, "manual", manual_taps, split_required)
            continue
        if not split_required:
            states[band] = BandFIRState(False, "none", None, False)
            continue
        if split_firs is None or band not in split_firs:
            states[band] = BandFIRState(True, "auto_split", None, True)
            continue
        coefficients = np.asarray(split_firs[band], dtype=float)
        if coefficients.ndim != 1 or coefficients.size < 1 or not np.isfinite(coefficients).all():
            raise ValueError(f"{band}: generated split FIR is invalid")
        # Reserve correction headroom regardless of automatic cropping. The split
        # coefficients and the crop search's minimum length stay unchanged.
        taps = int(coefficients.size) * 2 + 1
        states[band] = BandFIRState(True, "auto_split", taps, True)
    return states


def inferred_fir_output_enabled(
    ordered_bands: Sequence[str],
    crossover_methods: Sequence[str],
    configured_taps: Mapping[str, int],
    *,
    baffle_enabled: bool = False,
    eq_files: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    if bands_requiring_split_fir(ordered_bands, crossover_methods):
        return True
    if any(int(configured_taps.get(str(band), 0) or 0) > 0 for band in ordered_bands):
        return True
    return any(
        any(str(filename).strip() for filename in (eq_files or {}).get(str(band), ()))
        for band in ordered_bands
    )


def initialize_manual_taps_for_enable(
    ordered_bands: Sequence[str],
    crossover_methods: Sequence[str],
    configured_taps: Mapping[str, int],
    *,
    default_taps: int = DEFAULT_MANUAL_FIR_TAPS,
) -> dict[str, int]:
    required = bands_requiring_split_fir(ordered_bands, crossover_methods)
    result = {str(band): int(configured_taps.get(str(band), 0) or 0) for band in ordered_bands}
    for band in result:
        if band not in required and result[band] < 1:
            result[band] = int(default_taps)
    return result
