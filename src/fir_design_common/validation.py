"""Coefficient and response checks, independent of design, UI and export."""
from dataclasses import dataclass

import numpy as np

from .adaptive_crop import ACTIVE_FLOOR_DB, _metrics, _passes


class FIRValidationError(ValueError):
    """A candidate must not become the authoritative output."""


@dataclass(frozen=True)
class FIRResponseValidation:
    taps: int
    peak_magnitude: float
    max_abs_phase_deg: float
    min_group_delay_samples: float
    max_group_delay_samples: float
    max_group_delay_error_samples: float = 0.0


def _response(values, size):
    response = np.fft.rfft(values, size)
    peak = max(float(np.max(np.abs(response))), 1e-15)
    mask = np.abs(response) >= peak * 10 ** (ACTIVE_FLOOR_DB / 20)
    weighted = np.fft.rfft(np.arange(values.size) * values, size)
    delay = np.zeros(response.size)
    delay[mask] = np.real(weighted[mask] / response[mask])
    return response, mask, delay


def validate_basic_fir(coefficients):
    """Inspect taps, amplitude, phase and group delay without modifying FIR."""
    if np.iscomplexobj(coefficients):
        raise FIRValidationError("FIR coefficients must be real")
    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise FIRValidationError("FIR coefficients must be a nonempty finite vector")
    size = 1 << max(4, (values.size * 8 - 1).bit_length())
    response, mask, delay = _response(values, size)
    phase = np.unwrap(np.angle(response))
    if not np.isfinite(response).all() or not np.isfinite(delay[mask]).all():
        raise FIRValidationError("FIR response contains invalid amplitude or group delay")
    active_delay = delay[mask] if np.any(mask) else np.array([0.0])
    return FIRResponseValidation(values.size, float(np.max(np.abs(response))),
                                 float(np.max(np.abs(np.rad2deg(phase)))),
                                 float(np.min(active_delay)), float(np.max(active_delay)))


def validate_final_fir(source, plan, start, stop, output, *, window,
                       enforce_precision):
    """Validate protected-core finalization, including Adaptive precision."""
    return _validate_final_fir(source, plan, start, stop, output, window=window,
                               enforce_precision=enforce_precision, preserve_core=True)


def validate_fixed_fir(source, plan, start, stop, output, *, window):
    """Validate direct windowing of the exact fixed range without a core guard."""
    if (start, stop) != (plan.start, plan.stop) or plan.adopted:
        raise FIRValidationError("fixed FIR must retain its planned range")
    return _validate_final_fir(source, plan, start, stop, output, window=window,
                               enforce_precision=False, preserve_core=False)


def validate_fixed_frame(source, plan, output, *, window):
    """Verify the planned crop/zero frame, window and original time reference."""
    from .fixed_frame import plan_fixed_fir_frame
    if plan != plan_fixed_fir_frame(len(source), plan.output_taps):
        raise FIRValidationError("fixed FIR frame differs from its centered plan")
    if len(output) != plan.output_taps or len(window) != plan.output_taps:
        raise FIRValidationError("fixed FIR must have the configured length")
    expected = np.zeros(plan.output_taps, dtype=float)
    expected[plan.padding_left:plan.output_taps-plan.padding_right] = source[
        plan.source_start:plan.source_stop]
    if not np.array_equal(output, expected * window):
        raise FIRValidationError("fixed FIR differs from the one frame/window operation")
    report = validate_basic_fir(output)
    size = 1 << max(4, (max(len(source), len(output)) * 8 - 1).bit_length())
    _, ref_mask, ref_delay = _response(source, size)
    _, got_mask, got_delay = _response(output, size)
    active = ref_mask & got_mask
    error = (float(np.max(np.abs(got_delay[active] + plan.origin_samples - ref_delay[active])))
             if np.any(active) else 0.0)
    from dataclasses import replace
    return replace(report, max_group_delay_error_samples=error)


def _validate_final_fir(source, plan, start, stop, output, *, window,
                        enforce_precision, preserve_core):
    """Verify the actual DSP coefficients and their compensated time origin.

    Existing amplitude/phase/complex/energy limits remain authoritative.
    Group delay is measured on active reference bins, without inventing a new
    numerical acceptance threshold or changing the optimization target.
    """
    report = validate_basic_fir(output)
    if output.size != stop - start or not 0 <= start <= plan.start < plan.stop <= stop <= source.size:
        raise FIRValidationError("final FIR does not match its crop range")
    expected = source[start:stop] * window
    if not np.array_equal(output, expected):
        raise FIRValidationError("final FIR differs from the one crop/window operation")
    first, last = plan.start - start, plan.stop - start
    if preserve_core and not np.array_equal(window[first:last], np.ones(plan.taps)):
        raise FIRValidationError("Adaptive core window must be exactly 1.0")
    if preserve_core and not np.array_equal(output[first:last], source[plan.start:plan.stop]):
        raise FIRValidationError("Adaptive core coefficients have changed")
    metrics = _metrics(source, output, centered_shift_samples=(source.size-stop-start)/2)
    removed = float(np.dot(source[:start], source[:start]) + np.dot(source[stop:], source[stop:]))
    from dataclasses import replace
    metrics = replace(metrics, removed_energy_ratio=removed / max(float(np.dot(source, source)), 1e-300))
    if enforce_precision and not _passes(metrics):
        raise FIRValidationError(
            f"final FIR validation failed: gain={metrics.gain_error_db:.6g} dB, "
            f"phase={metrics.phase_error_deg:.6g} deg, complex={metrics.complex_error:.6g}; "
            "no FIR output was adopted")
    size = 1 << max(4, (source.size * 8 - 1).bit_length())
    ref, mask, ref_delay = _response(source, size)
    got, got_mask, got_delay = _response(output, size)
    active = mask & got_mask
    error = float(np.max(np.abs(got_delay[active] + start - ref_delay[active]))) if np.any(active) else 0.0
    return replace(report, max_group_delay_error_samples=error)
