"""Application-independent FIR output enable and provisional-frame policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping, Sequence

import numpy as np


DEFAULT_MANUAL_FIR_TAPS = 1023

_DIRECT_FIR_SPLIT_METHODS = {
    "kaiser": "kaiser",
    "Kaiser FIR": "kaiser",
    "linear_phase_lr2": "lr2",
    "Linear-phase LR2 FIR": "lr2",
    "linear_phase_lr4": "lr4",
    "Linear-phase LR4 FIR": "lr4",
}


def direct_fir_split_family(method: object) -> Literal["kaiser", "lr2", "lr4"] | None:
    """Normalize Standalone and Multiway names for one direct FIR boundary."""
    return _DIRECT_FIR_SPLIT_METHODS.get(str(method))


def boundary_requires_direct_fir(
    method: object,
    *,
    acoustic_target: bool = False,
) -> bool:
    """Return whether a boundary contributes coefficients to the FIR output.

    An acoustic-target boundary contributes to the target rather than the
    direct split stage. Standalone and Multiway use different persisted method
    names, so both are normalized at this shared policy boundary.
    """
    return not bool(acoustic_target) and direct_fir_split_family(method) is not None


def automatic_lr_natural_taps(
    sample_rate_hz: int,
    crossover_hz: float,
    method: str,
    *,
    overlap_oct: float = 0.0,
) -> int:
    """Return the one shared LR2/LR4 natural split-filter length."""
    method_name = str(method).lower()
    if "lr4" in method_name or "phase_lr4" in method_name:
        from crossover_engine.lr4_taps import automatic_lr4_taps

        plan = automatic_lr4_taps(
            int(sample_rate_hz), float(crossover_hz), float(overlap_oct)
        )
    elif "lr2" in method_name or "phase_lr2" in method_name:
        from crossover_engine.lr2_taps import automatic_lr2_taps

        plan = automatic_lr2_taps(
            int(sample_rate_hz), float(crossover_hz), float(overlap_oct)
        )
    else:
        raise ValueError("automatic LR method must be LR2 or LR4")
    return int(plan.taps)


def acoustic_target_frame_taps(
    sample_rate_hz: int,
    crossover_hz: float,
    method: str,
    *,
    overlap_oct: float = 0.0,
) -> int:
    """Return the shared acoustic-target work frame."""
    return 2 * automatic_lr_natural_taps(
        sample_rate_hz,
        crossover_hz,
        # This is a provisional correction frame, not an NTM4 FIR prototype.
        # Reuse the LR4 frame allocation for the fourth-order IIR target.
        "LR4" if str(method).lower() in {"ntm4", "iir_ntm4"} else method,
        overlap_oct=overlap_oct,
    ) + 1


def plan_acoustic_target_taps(
    ordered_bands: Sequence[str],
    crossover_methods: Sequence[str],
    acoustic_targets: Sequence[bool],
    crossover_frequencies_hz: Sequence[float],
    configured_taps: Mapping[str, int],
    *,
    sample_rate_hz: int,
    crossover_overlaps_oct: Sequence[float] | None = None,
) -> dict[str, int]:
    """Plan transient per-band target headroom without changing saved taps."""
    bands = tuple(str(value) for value in ordered_bands)
    methods = tuple(str(value) for value in crossover_methods)
    targets = tuple(bool(value) for value in acoustic_targets)
    frequencies = tuple(float(value) for value in crossover_frequencies_hz)
    boundary_count = max(0, len(bands) - 1)
    overlap_values = () if crossover_overlaps_oct is None else crossover_overlaps_oct
    overlaps = tuple(float(value) for value in overlap_values)
    if not overlaps:
        overlaps = (0.0,) * boundary_count
    if not (
        len(methods)
        == len(targets)
        == len(frequencies)
        == len(overlaps)
        == boundary_count
    ):
        raise ValueError("acoustic target boundary count does not match bands")

    configured = {band: int(configured_taps.get(band, 0) or 0) for band in bands}
    result = dict.fromkeys(bands, 0)
    for index, enabled in enumerate(targets):
        if not enabled:
            continue
        target_taps = acoustic_target_frame_taps(
            sample_rate_hz,
            frequencies[index],
            methods[index],
            overlap_oct=overlaps[index],
        )
        for band in bands[index:index + 2]:
            if configured[band] == 0:
                result[band] = max(result[band], target_taps)
    return result


@dataclass(frozen=True)
class FIRFrameState:
    """Resolved FIR state before final composition and optional cropping."""

    enabled: bool
    tap_source: Literal["none", "auto_split", "auto_target", "manual"]
    tap_count: int | None
    split_fir_required: bool


@dataclass(frozen=True)
class FIROutputStatus:
    """One inspectable enablement decision, separate from stored preferences."""

    owner: Literal["standalone", "assignment"]
    requested_enabled: bool
    configured_taps: int
    automatic_frame_available: bool
    available: bool
    effective_enabled: bool
    reason: Literal["enabled", "requested_off", "no_tap_frame", "assignment_off"]

    def snapshot(self, frame: FIRFrameState | None = None) -> dict[str, object]:
        result = asdict(self)
        if frame is not None:
            result["frame"] = asdict(frame)
            result["consistent"] = self.effective_enabled == frame.enabled
        return result


def resolve_fir_output_status(
    configured_taps: int,
    *,
    requested_enabled: bool,
    automatic_frame_available: bool = False,
    assignment_taps: int | None = None,
) -> FIROutputStatus:
    """Resolve UI/output capability using the same inputs as frame allocation."""
    inherited = assignment_taps is not None
    taps = int(assignment_taps if inherited else configured_taps)
    if taps < 0:
        raise ValueError("configured taps must be >= 0")
    automatic = bool(automatic_frame_available) and not inherited
    available = taps > 0 or automatic
    requested = available if inherited else bool(requested_enabled)
    effective = requested and available
    reason = ("enabled" if effective else "assignment_off" if inherited
              else "requested_off" if not requested else "no_tap_frame")
    return FIROutputStatus(
        "assignment" if inherited else "standalone", requested, taps,
        automatic, available, effective, reason,
    )


def resolve_fir_frame_state(
    configured_taps: int,
    *,
    fir_output_enabled: bool,
    split_fir: np.ndarray | None = None,
    split_fir_required: bool = False,
    automatic_taps: int = 0,
) -> FIRFrameState:
    """Resolve one output using the policy shared by Standalone and Multiway.

    A positive value is an explicit frame.  Zero is not an Adaptive-crop
    switch: for a generated FIR crossover it reserves twice the split length
    plus one; without a FIR crossover (or another automatic target) it means
    FIR OFF.  Adaptive cropping is an independent finalization option.
    """
    configured = int(configured_taps or 0)
    if configured < 0:
        raise ValueError("configured taps must be >= 0")
    required = bool(split_fir_required)
    automatic = int(automatic_taps or 0)
    status = resolve_fir_output_status(
        configured, requested_enabled=fir_output_enabled,
        automatic_frame_available=required or automatic > 0,
    )
    if not status.effective_enabled:
        return FIRFrameState(False, "none", None, required)
    if configured > 0:
        return FIRFrameState(True, "manual", configured, required)
    if automatic > 0:
        return FIRFrameState(True, "auto_target", automatic, required)
    if not required:
        return FIRFrameState(False, "none", None, False)
    if split_fir is None:
        return FIRFrameState(True, "auto_split", None, True)
    coefficients = np.asarray(split_fir, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 1 or not np.isfinite(coefficients).all():
        raise ValueError("generated split FIR is invalid")
    return FIRFrameState(True, "auto_split", int(coefficients.size) * 2 + 1, True)


def initialize_taps_for_fir_enable(
    configured_taps: int,
    *,
    split_fir_required: bool,
    automatic_frame_available: bool = False,
    default_taps: int = DEFAULT_MANUAL_FIR_TAPS,
) -> int:
    """Return the first-ON setting without replacing an automatic zero frame.

    ``split_fir_required`` covers a directly emitted crossover FIR, while
    ``automatic_frame_available`` covers another runtime allocator such as an
    LR2/LR4 acoustic target.  Both keep the persisted zero setting unchanged.
    """
    configured = int(configured_taps or 0)
    if configured < 0:
        raise ValueError("configured taps must be >= 0")
    if configured > 0 or split_fir_required or automatic_frame_available:
        return configured
    default = int(default_taps)
    if default < 1:
        raise ValueError("default taps must be positive")
    return default


def fir_output_available_for_tap_setting(
    configured_taps: int,
    *,
    automatic_frame_available: bool,
) -> bool:
    """Return whether a tap setting can produce an FIR output frame.

    Availability is independent from the user's persisted FIR ON/OFF choice.
    A caller must not overwrite that choice merely because a zero setting has
    temporarily lost its automatic frame source.
    """
    return resolve_fir_output_status(
        int(configured_taps or 0), requested_enabled=True,
        automatic_frame_available=automatic_frame_available,
    ).available
