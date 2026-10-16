"""Pure policies for Multiway regeneration and recalculation notices."""

from collections.abc import Mapping, Sequence

from fir_design_common.fir_state import boundary_requires_direct_fir, direct_fir_split_family


def should_warn_about_kaiser_recalculation(
    *, fir_output_enabled: bool, auto_update: bool,
    sample_rate_hz: int, band_filters: Mapping[str, Sequence[Mapping[str, object]]],
) -> bool:
    """Use the active Kaiser work-length heuristic, never saved inactive cycles.

    This is a recalculation hint, not a final output length or DSP capacity check.
    LR FIR uses a different length planner and is outside this cycle-based hint.
    """
    if not fir_output_enabled or not auto_update:
        return False
    for filters in band_filters.values():
        for item in filters:
            method = item.get("response", "kaiser")
            if (not item.get("enabled", True)
                    or direct_fir_split_family(method) != "kaiser"
                    or not boundary_requires_direct_fir(
                        method, acoustic_target=bool(item.get("acoustic_target", False)))):
                continue
            estimated_taps = int(sample_rate_hz / max(float(item["fc"]), 1.) * float(item["cycles"]))
            if estimated_taps > 1800:
                return True
    return False


def should_generate_result(
    *,
    configuration_valid: bool,
    has_cached_result: bool,
    manual_refresh: bool,
    auto_update: bool,
    signature_changed: bool,
) -> bool:
    """Return whether the current configuration requires a fresh result.

    Rendering duration is intentionally excluded.  A previous slow render may
    justify a performance notice, but must not leave controls and results out of
    sync while automatic updating is enabled.
    """
    if not configuration_valid:
        return False
    if not has_cached_result or manual_refresh:
        return True
    return bool(auto_update and signature_changed)
