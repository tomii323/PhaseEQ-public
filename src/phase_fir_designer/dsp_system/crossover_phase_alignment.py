from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize, signal

from ..config import IIRFilter
from ..iir import iir_frequency_response
from .latency import effective_fir_support_taps


AlignmentMethod = Literal["delay", "iir_allpass", "fir_phase", "none"]


@dataclass(frozen=True)
class PhaseAlignmentMetrics:
    phase_rms_deg: float
    phase_p95_deg: float
    sum_ripple_db: float
    worst_sum_db: float
    points: int


@dataclass(frozen=True)
class PhaseAlignmentRecommendation:
    method: AlignmentMethod
    compensate: Literal["lower", "upper", "none"]
    delay_ms: float
    polarity_invert: bool
    allpass_filters: tuple[IIRFilter, ...]
    fir_correction: tuple[float, ...]
    fir_fits_tap_budget: bool
    baseline: PhaseAlignmentMetrics
    realized: PhaseAlignmentMetrics
    active_frequency_hz: tuple[float, float] | None
    warnings: tuple[str, ...]


def crossover_active_mask(
    lower_response: np.ndarray,
    upper_response: np.ndarray,
    *,
    floor_db: float = -40.0,
) -> np.ndarray:
    lower = np.abs(np.asarray(lower_response, dtype=complex))
    upper = np.abs(np.asarray(upper_response, dtype=complex))
    if lower.shape != upper.shape:
        raise ValueError("crossover responses must have the same shape")
    lower_peak = max(float(np.nanmax(lower)), 1e-15)
    upper_peak = max(float(np.nanmax(upper)), 1e-15)
    threshold = 10.0 ** (float(floor_db) / 20.0)
    return (
        np.isfinite(lower)
        & np.isfinite(upper)
        & (lower >= lower_peak * threshold)
        & (upper >= upper_peak * threshold)
    )


def _metrics(lower: np.ndarray, upper: np.ndarray, mask: np.ndarray) -> PhaseAlignmentMetrics:
    difference = np.rad2deg(np.angle(lower[mask] * np.conj(upper[mask])))
    absolute = np.abs(difference)
    combined = np.abs(lower[mask] + upper[mask])
    individual_reference = np.maximum(np.abs(lower[mask]) + np.abs(upper[mask]), 1e-15)
    relative_sum_db = 20.0 * np.log10(np.maximum(combined / individual_reference, 1e-15))
    return PhaseAlignmentMetrics(
        phase_rms_deg=float(np.sqrt(np.mean(np.square(difference)))),
        phase_p95_deg=float(np.percentile(absolute, 95.0)),
        sum_ripple_db=float(np.percentile(relative_sum_db, 95.0) - np.percentile(relative_sum_db, 5.0)),
        worst_sum_db=float(np.min(relative_sum_db)),
        points=int(np.count_nonzero(mask)),
    )


def _score(metrics: PhaseAlignmentMetrics) -> float:
    cancellation = max(0.0, -metrics.worst_sum_db - 6.0)
    # Phase-curve agreement is the primary objective. Sum uniformity is a
    # guard/secondary term so a deep cancellation cannot be "fixed" merely by
    # rotating both curves farther apart.
    return metrics.phase_rms_deg + 0.25 * metrics.phase_p95_deg + 0.5 * metrics.sum_ripple_db + 0.5 * cancellation


def _phase_improves(candidate: PhaseAlignmentMetrics, reference: PhaseAlignmentMetrics) -> bool:
    return (
        candidate.phase_rms_deg < reference.phase_rms_deg * 0.995
        and candidate.phase_p95_deg <= reference.phase_p95_deg + 0.25
        and candidate.worst_sum_db >= reference.worst_sum_db - 0.25
    )


def _apply_increment(
    lower: np.ndarray,
    upper: np.ndarray,
    frequency: np.ndarray,
    *,
    compensate: Literal["lower", "upper"],
    delay_ms: float,
    polarity_invert: bool,
    extra_response: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    phase_delay = np.exp(-1j * 2.0 * np.pi * frequency * float(delay_ms) / 1000.0)
    increment = phase_delay * (-1.0 if polarity_invert else 1.0)
    if extra_response is not None:
        increment = increment * extra_response
    if compensate == "lower":
        return lower * increment, upper
    return lower, upper * increment


def _best_delay(
    lower: np.ndarray,
    upper: np.ndarray,
    frequency: np.ndarray,
    mask: np.ndarray,
    *,
    max_delay_ms: float,
) -> tuple[Literal["lower", "upper"], float, bool, PhaseAlignmentMetrics]:
    candidates = [
        _best_delay_for_compensation(
            lower,
            upper,
            frequency,
            mask,
            compensate=compensate,
            max_delay_ms=max_delay_ms,
        )
        for compensate in ("lower", "upper")
    ]
    return min(candidates, key=lambda item: _score(item[3]))


def _best_delay_for_compensation(
    lower: np.ndarray,
    upper: np.ndarray,
    frequency: np.ndarray,
    mask: np.ndarray,
    *,
    compensate: Literal["lower", "upper"],
    max_delay_ms: float,
) -> tuple[Literal["lower", "upper"], float, bool, PhaseAlignmentMetrics]:
    best: tuple[Literal["lower", "upper"], float, bool, PhaseAlignmentMetrics] | None = None
    for polarity in (False, True):
        def objective(delay: float) -> float:
            candidate = _apply_increment(
                lower, upper, frequency,
                compensate=compensate,
                delay_ms=float(delay),
                polarity_invert=polarity,
            )
            return _score(_metrics(*candidate, mask))

        optimized = optimize.minimize_scalar(
            objective,
            bounds=(0.0, max(float(max_delay_ms), 0.0)),
            method="bounded",
            options={"xatol": 1e-4},
        )
        delay_candidates = (0.0, max(0.0, float(optimized.x)), max(float(max_delay_ms), 0.0))
        for delay in delay_candidates:
            candidate = _apply_increment(
                lower, upper, frequency,
                compensate=compensate,
                delay_ms=delay,
                polarity_invert=polarity,
            )
            metrics = _metrics(*candidate, mask)
            if best is None or _score(metrics) < _score(best[3]):
                best = (compensate, delay, polarity, metrics)
    assert best is not None
    return best


def _greedy_allpass(
    lower: np.ndarray,
    upper: np.ndarray,
    frequency: np.ndarray,
    mask: np.ndarray,
    sample_rate: int,
    *,
    compensate: Literal["lower", "upper"],
    delay_ms: float,
    polarity_invert: bool,
    max_sections: int,
) -> tuple[tuple[IIRFilter, ...], PhaseAlignmentMetrics]:
    base_lower, base_upper = _apply_increment(
        lower, upper, frequency,
        compensate=compensate,
        delay_ms=delay_ms,
        polarity_invert=polarity_invert,
    )
    selected: list[IIRFilter] = []
    best_metrics = _metrics(base_lower, base_upper, mask)
    active_frequency = frequency[mask]
    lo = max(float(active_frequency[0]), 1.0)
    hi = min(float(active_frequency[-1]), sample_rate / 2.0 * 0.98)
    if hi <= lo:
        return (), best_metrics
    fc_grid = np.geomspace(lo, hi, min(32, max(8, int(np.sqrt(active_frequency.size) * 2))))
    q_grid = (0.35, 0.5, 0.7071, 1.0, 1.4, 2.0, 3.0, 5.0)
    for _ in range(max(0, int(max_sections))):
        current_score = _score(best_metrics)
        candidate_best: tuple[IIRFilter, PhaseAlignmentMetrics] | None = None
        for fc in fc_grid:
            for q in q_grid:
                item = IIRFilter(kind="allpass", fc=float(fc), q=float(q), origin="auto")
                filters = tuple(selected + [item])
                extra = iir_frequency_response(filters, sample_rate, frequency)
                candidate = _apply_increment(
                    lower, upper, frequency,
                    compensate=compensate,
                    delay_ms=delay_ms,
                    polarity_invert=polarity_invert,
                    extra_response=extra,
                )
                metrics = _metrics(*candidate, mask)
                if candidate_best is None or _score(metrics) < _score(candidate_best[1]):
                    candidate_best = (item, metrics)
        if candidate_best is None or _score(candidate_best[1]) >= current_score * 0.98:
            break
        selected.append(candidate_best[0])
        best_metrics = candidate_best[1]
    return tuple(selected), best_metrics


def design_fir_phase_correction(
    frequency_hz: np.ndarray,
    desired_response: np.ndarray,
    sample_rate: int,
    taps: int,
    *,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Design a real, causal, phase-focused FIR with a unity-magnitude target."""

    frequency = np.asarray(frequency_hz, dtype=float)
    desired = np.asarray(desired_response, dtype=complex)
    mask = np.asarray(active_mask, dtype=bool)
    length = max(3, int(taps) | 1)
    n_fft = max(1024, int(2 ** np.ceil(np.log2(max(length * 8, frequency.size * 2)))))
    fft_frequency = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    phase = np.zeros_like(fft_frequency)
    source_phase = np.unwrap(np.angle(desired[mask]))
    phase_active = np.interp(fft_frequency, frequency[mask], source_phase, left=source_phase[0], right=source_phase[-1])
    lo = float(frequency[mask][0])
    hi = float(frequency[mask][-1])
    inside = (fft_frequency >= lo) & (fft_frequency <= hi)
    phase[inside] = phase_active[inside]
    target = np.exp(1j * phase)
    impulse = np.fft.fftshift(np.fft.irfft(target, n=n_fft))
    center = n_fft // 2
    start = center - length // 2
    correction = impulse[start:start + length] * signal.windows.tukey(length, alpha=0.15)
    reference = float(np.sum(correction))
    if abs(reference) > 1e-9:
        correction = correction / reference
    return np.asarray(correction, dtype=float)


def recommend_crossover_phase_alignment(
    frequency_hz: np.ndarray,
    lower_response: np.ndarray,
    upper_response: np.ndarray,
    sample_rate: int,
    *,
    lower_fir: np.ndarray | None = None,
    upper_fir: np.ndarray | None = None,
    tap_budget: int | None = None,
    floor_db: float = -40.0,
    max_delay_ms: float = 20.0,
    max_allpass_sections: int = 2,
    fir_correction_taps: int = 129,
) -> PhaseAlignmentRecommendation:
    frequency = np.asarray(frequency_hz, dtype=float)
    lower = np.asarray(lower_response, dtype=complex)
    upper = np.asarray(upper_response, dtype=complex)
    if frequency.ndim != 1 or frequency.size < 8 or lower.shape != frequency.shape or upper.shape != frequency.shape:
        raise ValueError("phase alignment requires matching one-dimensional responses")
    mask = crossover_active_mask(lower, upper, floor_db=floor_db)
    warnings: list[str] = []
    if np.count_nonzero(mask) < 8:
        empty = PhaseAlignmentMetrics(180.0, 180.0, 0.0, -300.0, int(np.count_nonzero(mask)))
        return PhaseAlignmentRecommendation("none", "none", 0.0, False, (), (), False, empty, empty, None, ("−40 dB以上の共通帯域が不足しています。",))
    baseline = _metrics(lower, upper, mask)
    compensate, delay_ms, polarity, delay_metrics = _best_delay(
        lower, upper, frequency, mask, max_delay_ms=max_delay_ms,
    )
    allpass_candidates = []
    for allpass_compensate in ("lower", "upper"):
        candidate_delay = _best_delay_for_compensation(
            lower,
            upper,
            frequency,
            mask,
            compensate=allpass_compensate,
            max_delay_ms=max_delay_ms,
        )
        seeds = {
            (candidate_delay[1], candidate_delay[2]),
            (0.0, False),
            (0.0, True),
        }
        for seed_delay, seed_polarity in seeds:
            candidate_filters, candidate_metrics = _greedy_allpass(
                lower, upper, frequency, mask, int(sample_rate),
                compensate=allpass_compensate,
                delay_ms=seed_delay,
                polarity_invert=seed_polarity,
                max_sections=max_allpass_sections,
            )
            allpass_candidates.append((
                allpass_compensate,
                seed_delay,
                seed_polarity,
                candidate_filters,
                candidate_metrics,
            ))
    best_allpass = min(allpass_candidates, key=lambda item: _score(item[4]))
    if best_allpass[3] and _score(best_allpass[4]) < _score(delay_metrics):
        compensate, delay_ms, polarity = best_allpass[:3]
        allpass, allpass_metrics = best_allpass[3], best_allpass[4]
    else:
        allpass, allpass_metrics = (), delay_metrics
    method: AlignmentMethod = "iir_allpass" if allpass and _score(allpass_metrics) < _score(delay_metrics) else "delay"
    realized = allpass_metrics if method == "iir_allpass" else delay_metrics
    fir_values: tuple[float, ...] = ()
    fir_fits = False
    base_lower, base_upper = _apply_increment(
        lower, upper, frequency,
        compensate=compensate,
        delay_ms=delay_ms,
        polarity_invert=polarity,
    )
    current = base_lower if compensate == "lower" else base_upper
    target = base_upper if compensate == "lower" else base_lower
    desired = np.exp(1j * np.angle(target / np.where(np.abs(current) > 1e-15, current, 1.0)))
    correction = design_fir_phase_correction(
        frequency,
        desired,
        int(sample_rate),
        int(fir_correction_taps),
        active_mask=mask,
    )
    existing = lower_fir if compensate == "lower" else upper_fir
    if tap_budget is not None and existing is not None:
        required = effective_fir_support_taps(existing) + effective_fir_support_taps(correction) - 1
        fir_fits = required <= int(tap_budget)
        if not fir_fits:
            warnings.append(f"FIR位相補正には実効{required} tapsが必要なため使用しません。")
    if fir_fits:
        _w, fir_response = signal.freqz(correction, worN=np.clip(frequency, 0.0, sample_rate / 2.0), fs=sample_rate)
        fir_candidate = _apply_increment(
            lower, upper, frequency,
            compensate=compensate,
            delay_ms=delay_ms,
            polarity_invert=polarity,
            extra_response=fir_response,
        )
        fir_metrics = _metrics(*fir_candidate, mask)
        if _phase_improves(fir_metrics, realized) and _score(fir_metrics) < _score(realized) * 0.98:
            method = "fir_phase"
            realized = fir_metrics
            fir_values = tuple(float(value) for value in correction)
            allpass = ()
    if not _phase_improves(realized, baseline) or _score(realized) >= _score(baseline) * 0.995:
        method = "none"
        compensate = "none"
        delay_ms = 0.0
        polarity = False
        allpass = ()
        fir_values = ()
        realized = baseline
        warnings.append("Delay／All Pass／FIRで有効な改善が得られませんでした。")
    active_frequency = frequency[mask]
    return PhaseAlignmentRecommendation(
        method=method,
        compensate=compensate,
        delay_ms=float(delay_ms),
        polarity_invert=bool(polarity),
        allpass_filters=tuple(allpass),
        fir_correction=fir_values,
        fir_fits_tap_budget=bool(fir_fits),
        baseline=baseline,
        realized=realized,
        active_frequency_hz=(float(active_frequency[0]), float(active_frequency[-1])),
        warnings=tuple(warnings),
    )
