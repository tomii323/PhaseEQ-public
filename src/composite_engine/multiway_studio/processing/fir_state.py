from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from fir_design_common.output_tap_policy import FixedWindowMode
from fir_design_common.fir_state import (
    DEFAULT_MANUAL_FIR_TAPS,
    FIRFrameState,
    boundary_requires_direct_fir,
    direct_fir_split_family,
    initialize_taps_for_fir_enable,
    plan_acoustic_target_taps,
    resolve_fir_frame_state,
)


FIR_CROSSOVER_METHODS = frozenset({"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR"})
KAISER_CROSSOVER_METHOD = "Kaiser FIR"
class BandFIRState(FIRFrameState):
    def to_dict(self) -> dict[str, object]:
        return dict(
            enabled=self.enabled,
            tap_source=self.tap_source,
            tap_count=self.tap_count,
            split_fir_required=self.split_fir_required,
        )


def bands_requiring_split_fir(
    ordered_bands: Sequence[str], crossover_methods: Sequence[str],
) -> frozenset[str]:
    bands = tuple(str(value) for value in ordered_bands)
    methods = tuple(str(value) for value in crossover_methods)
    if len(methods) != max(0, len(bands) - 1):
        raise ValueError("crossover method count does not match bands")
    required: set[str] = set()
    for index, method in enumerate(methods):
        if boundary_requires_direct_fir(method):
            required.update((bands[index], bands[index + 1]))
    return frozenset(required)


def bands_using_kaiser_split(
    ordered_bands: Sequence[str], crossover_methods: Sequence[str],
) -> frozenset[str]:
    """Return bands touching at least one Kaiser FIR boundary."""
    bands = tuple(str(value) for value in ordered_bands)
    methods = tuple(str(value) for value in crossover_methods)
    if len(methods) != max(0, len(bands) - 1):
        raise ValueError("crossover method count does not match bands")
    required: set[str] = set()
    for index, method in enumerate(methods):
        if direct_fir_split_family(method) == "kaiser":
            required.update((bands[index], bands[index + 1]))
    return frozenset(required)


def resolve_fixed_window_mode(tap_source: str, *, auto_crop: bool) -> FixedWindowMode:
    """Keep zero-tap allocation independent from manual final windowing."""
    return "direct" if tap_source == "manual" and not auto_crop else "protected"


def resolve_band_fir_states(
    ordered_bands: Sequence[str],
    crossover_methods: Sequence[str],
    configured_taps: Mapping[str, int],
    *,
    fir_output_enabled: bool,
    split_firs: Mapping[str, np.ndarray] | None = None,
    automatic_taps: Mapping[str, int] | None = None,
    auto_crop: bool = False,
) -> dict[str, BandFIRState]:
    bands = tuple(str(value) for value in ordered_bands)
    required = bands_requiring_split_fir(bands, crossover_methods)
    states: dict[str, BandFIRState] = {}
    for band in bands:
        split_required = band in required
        try:
            resolved = resolve_fir_frame_state(
                int(configured_taps.get(band, 0) or 0),
                fir_output_enabled=fir_output_enabled,
                split_fir=(split_firs or {}).get(band),
                split_fir_required=split_required,
                automatic_taps=int((automatic_taps or {}).get(band, 0) or 0),
            )
        except ValueError as exc:
            raise ValueError(f"{band}: {exc}") from exc
        states[band] = BandFIRState(
            resolved.enabled,
            resolved.tap_source,
            resolved.tap_count,
            resolved.split_fir_required,
        )
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
        result[band] = initialize_taps_for_fir_enable(
            result[band],
            split_fir_required=band in required,
            default_taps=default_taps,
        )
    return result
