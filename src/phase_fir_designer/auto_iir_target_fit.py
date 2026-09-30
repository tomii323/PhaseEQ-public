"""Conventional magnitude-target PEQ fitting, independent of perceptual modes."""
from dataclasses import dataclass, asdict, field
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from .auto_iir_input_shaping import (
    InputShapingSettings, SHAPING_VERSION, input_signature,
)

from .config import IIRFilter, SpeakerResponse, IIR_PEQ_Q_MAX, quantize_iir_q
from .iir import (
    AutoIIRSettings, _auto_reliable_range, _response_gain_on_log_axis,
    _enforce_auto_iir_filter_headroom, iir_frequency_response,
)


@dataclass(frozen=True)
class TargetFitSettings:
    f_min: float = 20.0
    f_max: float = 20000.0
    max_filters: int = 6
    max_boost_db: float = 6.0
    max_cut_db: float = 12.0
    max_q: float = 5.0
    headroom_db: float = 12.0
    reliable_f_min_hz: float | None = None
    reliable_f_max_hz: float | None = None
    shaping: InputShapingSettings = field(default_factory=InputShapingSettings)


@dataclass(frozen=True)
class TargetFitResult:
    filters: tuple[IIRFilter, ...]
    diagnostics: dict


def _fit_target_peq_existing(speaker: SpeakerResponse, target: SpeakerResponse, sample_rate: int,
                   *, existing_filters=(), settings=TargetFitSettings()) -> TargetFitResult:
    """Minimize equal-log-frequency RMS error; do not normalize either input level.

    Residual extrema seed successive PEQs. Joint bounded least squares refines
    Fc, gain and Q, then the exported Q precision and cascade headroom are
    evaluated before accepting each stage. Phase is not an optimization target.
    """
    s = settings
    values = [sample_rate, s.f_min, s.f_max, s.max_boost_db, s.max_cut_db, s.max_q, s.headroom_db]
    if (not np.all(np.isfinite(values)) or sample_rate <= 0 or s.f_min <= 0
            or s.f_max <= s.f_min or not 0 <= s.max_filters <= 16
            or int(s.max_filters) != s.max_filters or s.max_boost_db < 0 or s.max_cut_db <= 0
            or not 0.3 <= s.max_q <= IIR_PEQ_Q_MAX or s.headroom_db < 0):
        raise ValueError("Invalid target-following PEQ settings")
    common = AutoIIRSettings(reliable_f_min_hz=s.reliable_f_min_hz,
        reliable_f_max_hz=s.reliable_f_max_hz, filter_headroom_limit_db=s.headroom_db)
    lo, hi = _auto_reliable_range(speaker, target, sample_rate, common)
    lo, hi = max(lo, s.f_min), min(hi, s.f_max)
    if hi <= lo:
        raise ValueError("No common Input/Target data in the requested correction band")
    frequency = np.geomspace(lo, hi, max(128, min(768, int(np.log2(hi / lo) * 48) + 1)))
    manual = tuple(existing_filters)

    def gain(filters, axis=frequency):
        return 20.0 * np.log10(np.maximum(np.abs(iir_frequency_response(filters, sample_rate, axis)), 1e-15))

    before = _response_gain_on_log_axis(speaker, frequency) + gain(manual)
    target_gain = _response_gain_on_log_axis(target, frequency)
    wanted = target_gain - before
    accepted = []
    best_rms = float(np.sqrt(np.mean(wanted ** 2)))
    initial_rms = best_rms
    # Check the entire digital band, including filter centers, after fitting.
    guard_axis = np.unique(np.r_[0.0, np.geomspace(0.1, sample_rate / 2, 4096), sample_rate / 2])
    lower_gain, upper_gain = -s.max_cut_db, max(s.max_boost_db, 1e-12)

    def unpack(vector, quantized=False):
        return [IIRFilter(kind="peq", fc=float(np.exp(fc)),
            q=min(np.floor(s.max_q * 10 + 1e-9) / 10, quantize_iir_q("peq", float(np.exp(q)))) if quantized else float(np.exp(q)),
            gain_db=float(g), origin="auto") for fc, g, q in np.asarray(vector).reshape(-1, 3)]

    for _ in range(int(s.max_filters)):
        residual = wanted - gain(accepted)
        eligible = (residual < -0.25) | ((residual > 0.25) & (s.max_boost_db > 0))
        if not np.any(eligible):
            break
        peak = int(np.argmax(np.where(eligible, np.abs(residual), -1.0)))
        previous = [[np.log(f.fc), f.gain_db, np.log(f.q)] for f in accepted]
        count = len(accepted) + 1
        bounds = (np.tile([np.log(lo), lower_gain, np.log(0.3)], count),
                  np.tile([np.log(hi), upper_gain, np.log(s.max_q)], count))
        # A nonzero Q interval is needed by least_squares at the lowest UI limit.
        bounds[1][2::3] = np.maximum(bounds[1][2::3], bounds[0][2::3] + 1e-10)
        winner, winner_rms = None, best_rms
        for seed_q in sorted(set(min(q, s.max_q) for q in (0.7, 2.0, 5.0))):
            seed = np.asarray(previous + [[np.log(frequency[peak]),
                np.clip(residual[peak], lower_gain, upper_gain), np.log(seed_q)]]).ravel()
            fit = least_squares(lambda x: gain(unpack(x)) - wanted,
                np.clip(seed, *bounds), bounds=bounds, max_nfev=100,
                ftol=1e-6, xtol=1e-6, gtol=1e-6)
            trial = unpack(fit.x, quantized=True)
            check_axis = np.unique(np.r_[guard_axis, [f.fc for f in trial], [f.fc for f in manual]])
            trial = _enforce_auto_iir_filter_headroom(trial, manual, sample_rate, check_axis, common)
            trial = [f for f in trial if abs(f.gain_db) >= 0.01]
            rms = float(np.sqrt(np.mean((gain(trial) - wanted) ** 2)))
            if rms < winner_rms - 0.005:
                winner, winner_rms = trial, rms
        if winner is None:
            break
        accepted, best_rms = winner, winner_rms
    after = before + gain(accepted)
    headroom = max(0.0, float(np.max(gain([*manual, *accepted], guard_axis))))
    return TargetFitResult(tuple(accepted), {
        "algorithm": "target_fit", "peq_count": len(accepted), "shelf_count": 0,
        "before_weighted_rms_db": initial_rms, "after_weighted_rms_db": best_rms,
        "correction_f_min_hz": lo, "correction_f_max_hz": hi,
        "filter_headroom_db": headroom, "filter_headroom_limit_db": s.headroom_db,
        "auto_iir_curve_frequency_hz": frequency.tolist(),
        "auto_iir_preview_gain_db": after.tolist(),
        "max_boost_db": s.max_boost_db, "max_cut_db": s.max_cut_db, "max_q": s.max_q,
    })


def prepare_eq_speaker_input(speaker, target, sample_rate, settings):
    from .auto_iir_input_shaping import prepare_speaker_input
    return prepare_speaker_input(speaker, target, sample_rate, settings.shaping,
        f_min=settings.f_min, f_max=settings.f_max,
        reliable_f_min_hz=settings.reliable_f_min_hz,
        reliable_f_max_hz=settings.reliable_f_max_hz)


def fit_target_peq(speaker: SpeakerResponse, target: SpeakerResponse, sample_rate: int,
                   *, existing_filters=(), settings=TargetFitSettings()) -> TargetFitResult:
    """Prepare the EQ-only input, then run the existing PEQ fitter unchanged."""
    if not settings.shaping.enabled:
        return _fit_target_peq_existing(speaker, target, sample_rate,
                                       existing_filters=existing_filters, settings=settings)
    started = perf_counter()
    prepared, dips = prepare_eq_speaker_input(speaker, target, sample_rate, settings)
    preparation_seconds = perf_counter()-started
    manual = tuple(existing_filters)
    result = _fit_target_peq_existing(prepared, target, sample_rate,
                                     existing_filters=manual, settings=settings)
    diagnostics = dict(result.diagnostics)
    frequency = np.asarray(diagnostics['auto_iir_curve_frequency_hz'])
    source = _response_gain_on_log_axis(speaker, frequency)
    target_gain = _response_gain_on_log_axis(target, frequency)
    def gain(filters):
        return 20*np.log10(np.maximum(abs(iir_frequency_response(filters, sample_rate, frequency)), 1e-15))
    before = source+gain(manual)
    after = before+gain(result.filters)
    diagnostics.update({
        'input_shaping_version': SHAPING_VERSION,
        'input_shaping_signature': input_signature(speaker, target, sample_rate, manual),
        'input_shaping_fit_settings': asdict(settings),
        'input_shaping_settings': asdict(settings.shaping),
        'input_shaping_dips': dips,
        'input_shaping_seconds': preparation_seconds,
        'auto_iir_calculation_input_gain_db': _response_gain_on_log_axis(prepared, frequency).tolist(),
        'auto_iir_preview_gain_db': after.tolist(),
        'original_before_rms_db': float(np.sqrt(np.mean((before-target_gain)**2))),
        'original_after_rms_db': float(np.sqrt(np.mean((after-target_gain)**2))),
    })
    return TargetFitResult(result.filters, diagnostics)
