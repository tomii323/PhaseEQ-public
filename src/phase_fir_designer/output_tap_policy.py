"""Standalone final-output tap policy.

The policy is deliberately independent from Streamlit and export formats.  It
receives the fully composed FIR, decides its real support and final taper; scalar gain remains explicit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fir_design_common.fir_state import (
    DEFAULT_MANUAL_FIR_TAPS,
    FIRFrameState,
    acoustic_target_frame_taps,
    boundary_requires_direct_fir,
    resolve_fir_frame_state,
)
from fir_design_common.output_tap_policy import (
    FinalOutputTapResult,
    apply_final_output_tap_policy,
)

from .linear_fir import linear_fir_overlay


StandaloneOutputTapResult = FinalOutputTapResult


def standalone_design_input_taps(
    configured_taps: int,
    *,
    recommended_taps: int = DEFAULT_MANUAL_FIR_TAPS,
) -> int:
    """Return a positive correction-design seed before split-FIR generation.

    Zero is resolved after the split FIR is generated.  A 1023-tap seed keeps
    that generation independent from UI state and matches Multiway's first-ON
    value for outputs without a FIR crossover.
    """
    configured = int(configured_taps)
    if configured < 0:
        raise ValueError("configured taps must be >= 0")
    if configured > 0:
        return configured
    recommended = int(recommended_taps)
    if recommended < 1:
        raise ValueError("recommended taps must be >= 1")
    return recommended


def standalone_split_fir(config) -> np.ndarray | None:
    """Return the generated output-band FIR used to resolve a zero setting."""
    return linear_fir_overlay(config)


def _filter_value(item: Any, key: str, default: Any = None) -> Any:
    """Read one normalized filter field from a record or config object."""
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def standalone_split_fir_required_for(
    linear_fir_filters: Sequence[Any],
    *,
    band_split_recipe: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether the current output-band split contributes FIR taps.

    This is the shared capability decision used by both the Streamlit controls
    and final FIR-state resolution.  Keeping it free of session state prevents
    the UI and export pipeline from disagreeing after a page transition.
    """
    if band_split_recipe is not None:
        return bool(band_split_recipe.get("fir_enabled", True))
    return any(
        bool(_filter_value(item, "enabled", True))
        and boundary_requires_direct_fir(
            _filter_value(item, "response", ""),
            acoustic_target=bool(_filter_value(item, "acoustic_target", False)),
        )
        for item in linear_fir_filters
    )


def standalone_acoustic_target_required_for(
    linear_fir_filters: Sequence[Any],
) -> bool:
    """Return whether an LR2/LR4 acoustic target can supply an auto frame."""
    return any(
        bool(_filter_value(item, "enabled", True))
        and bool(_filter_value(item, "acoustic_target", False))
        and str(_filter_value(item, "response", ""))
        in {"linear_phase_lr2", "linear_phase_lr4"}
        for item in linear_fir_filters
    )


def standalone_automatic_frame_available_for(
    linear_fir_filters: Sequence[Any],
    *,
    band_split_recipe: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a zero tap setting can resolve to a positive FIR frame."""
    return standalone_split_fir_required_for(
        linear_fir_filters,
        band_split_recipe=band_split_recipe,
    ) or standalone_acoustic_target_required_for(linear_fir_filters)


def standalone_split_fir_required(config) -> bool:
    return standalone_split_fir_required_for(
        config.linear_fir_filters,
        band_split_recipe=config.band_split_recipe,
    )


def standalone_acoustic_target_taps(config) -> int:
    """Resolve Standalone target headroom with the Multiway calculation."""
    target_taps = [
        acoustic_target_frame_taps(
            config.sample_rate,
            item.base_crossover_hz or item.fc,
            item.response,
            overlap_oct=item.overlap_oct,
        )
        for item in config.linear_fir_filters
        if item.enabled
        and item.acoustic_target
        and item.response in {"linear_phase_lr2", "linear_phase_lr4"}
    ]
    return max(target_taps, default=0)


def resolve_standalone_fir_state(
    config,
    configured_taps: int,
    *,
    fir_output_enabled: bool,
) -> FIRFrameState:
    """Resolve Standalone through the same zero-tap policy as Multiway."""
    split_required = standalone_split_fir_required(config)
    split_fir = standalone_split_fir(config) if split_required else None
    return resolve_fir_frame_state(
        configured_taps,
        fir_output_enabled=fir_output_enabled,
        split_fir=split_fir,
        split_fir_required=split_required,
        automatic_taps=standalone_acoustic_target_taps(config),
    )


def apply_standalone_output_tap_policy(
    coefficients: np.ndarray,
    output_taps: int,
    *,
    auto_crop: bool = False,
    minimum_taps: object = 1,
    output_window_enabled: bool = False,
    kaiser_band_split_active: bool = False,
) -> StandaloneOutputTapResult:
    """Finalize real support and taper using the common coefficient policy."""
    return apply_final_output_tap_policy(
        coefficients,
        output_taps,
        adaptive=auto_crop,
        minimum_taps=minimum_taps,
        output_window_enabled=output_window_enabled,
        kaiser_band_split_active=kaiser_band_split_active,
    )
