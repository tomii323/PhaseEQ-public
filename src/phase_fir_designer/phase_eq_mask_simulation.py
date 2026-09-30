from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from octave_boundary_smoothing import OneSidedBoundarySpec, apply_one_sided_boundary

from .phase_eq_mask_adaptive import (
    PhaseMaskBParameters,
    apply_phase_eq_mask_b_low,
    closest_zero_equivalent_phase,
)


@dataclass(frozen=True)
class PhaseMaskSimulationCase:
    name: str
    category: str
    boundary_hz: float
    smoothing_oct: float
    phase_deg: np.ndarray


@dataclass(frozen=True)
class PhaseMaskMetrics:
    effective_taps_99_9: int
    tail_energy_1025: float
    group_delay_peak_ms: float
    group_delay_roughness_ms: float
    protected_phase_error_deg: float


@dataclass(frozen=True)
class PhaseMaskSimulationRecord:
    case_name: str
    category: str
    parameters: PhaseMaskBParameters
    method: str
    width_scale: float
    confidence: float
    neutral_phase_deg: float
    residual_phase_deg: float
    fade_slope_deg_per_oct: float
    runtime_ms: float
    current: PhaseMaskMetrics
    candidate: PhaseMaskMetrics

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "case": self.case_name,
            "category": self.category,
            "parameters": self.parameters.key,
            **asdict(self.parameters),
            "method": self.method,
            "width_scale": self.width_scale,
            "confidence": self.confidence,
            "neutral_phase_deg": self.neutral_phase_deg,
            "residual_phase_deg": self.residual_phase_deg,
            "fade_slope_deg_per_oct": self.fade_slope_deg_per_oct,
            "runtime_ms": self.runtime_ms,
        }
        row.update({f"current_{key}": value for key, value in asdict(self.current).items()})
        row.update({f"candidate_{key}": value for key, value in asdict(self.candidate).items()})
        return row


@dataclass(frozen=True)
class PhaseMaskParameterScore:
    parameters: PhaseMaskBParameters
    tap_score: float
    smoothness_score: float
    load_score: float
    da_score: float
    worst_case_score: float
    median_effective_tap_ratio: float
    median_tail_energy_ratio: float
    median_runtime_ms: float
    fallback_rate: float

    def as_row(self) -> dict[str, object]:
        return {
            "parameters": self.parameters.key,
            **asdict(self.parameters),
            "tap_score": self.tap_score,
            "smoothness_score": self.smoothness_score,
            "load_score": self.load_score,
            "da_score": self.da_score,
            "worst_case_score": self.worst_case_score,
            "median_effective_tap_ratio": self.median_effective_tap_ratio,
            "median_tail_energy_ratio": self.median_tail_energy_ratio,
            "median_runtime_ms": self.median_runtime_ms,
            "fallback_rate": self.fallback_rate,
        }


@dataclass(frozen=True)
class PhaseMaskSimulationResult:
    records: tuple[PhaseMaskSimulationRecord, ...]
    scores: tuple[PhaseMaskParameterScore, ...]

    @property
    def best(self) -> PhaseMaskParameterScore:
        if not self.scores:
            raise ValueError("simulation has no parameter scores")
        return self.scores[0]


def simulation_parameter_grid(*, quick: bool = False) -> tuple[PhaseMaskBParameters, ...]:
    if quick:
        return (
            PhaseMaskBParameters(15.0, 60.0, 10.0, 1.0 / 24.0),
            PhaseMaskBParameters(15.0, 90.0, 10.0, 1.0 / 12.0),
            PhaseMaskBParameters(20.0, 60.0, 20.0, 1.0 / 24.0),
        )
    values: list[PhaseMaskBParameters] = []
    for base_slope in (10.0, 15.0, 20.0):
        for max_slope in (15.0, 30.0, 45.0, 60.0, 90.0):
            if max_slope < base_slope:
                continue
            for landing_hz in (5.0, 10.0, 20.0):
                for reference_oct in (1.0 / 48.0, 1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0):
                    values.append(
                        PhaseMaskBParameters(base_slope, max_slope, landing_hz, reference_oct)
                    )
    return tuple(values)


def build_adversarial_phase_cases(
    frequency: np.ndarray,
    *,
    seed: int = 20260714,
    quick: bool = False,
) -> tuple[PhaseMaskSimulationCase, ...]:
    freq = np.asarray(frequency, dtype=float)
    rng = np.random.default_rng(seed)
    boundaries = (500.0, 2000.0) if quick else (100.0, 500.0, 2000.0)
    cases: list[PhaseMaskSimulationCase] = []
    for boundary in boundaries:
        x = np.log2(np.maximum(freq, 1.0) / boundary)
        linear_x = (freq - boundary) / max(boundary, 1.0)
        width = 1.0 / 3.0 if boundary <= 500.0 else 0.5
        cases.extend(
            [
                PhaseMaskSimulationCase(
                    f"constant_60_{boundary:g}", "normal", boundary, width, np.full_like(freq, 60.0)
                ),
                PhaseMaskSimulationCase(
                    f"multiturn_400_{boundary:g}", "multi-turn", boundary, width, 400.0 + 12.0 * x
                ),
                PhaseMaskSimulationCase(
                    f"multiturn_minus520_{boundary:g}", "multi-turn", boundary, width, -520.0 - 18.0 * x
                ),
                PhaseMaskSimulationCase(
                    f"coherent_steep_{boundary:g}",
                    "steep",
                    boundary,
                    width,
                    140.0 + 150.0 * np.tanh(5.0 * linear_x),
                ),
                PhaseMaskSimulationCase(
                    f"high_q_turn_{boundary:g}",
                    "steep",
                    boundary,
                    width,
                    220.0 + 330.0 * np.arctan(35.0 * linear_x) / np.pi,
                ),
                PhaseMaskSimulationCase(
                    f"oscillatory_{boundary:g}",
                    "adversarial",
                    boundary,
                    width,
                    380.0 + 55.0 * np.sin(18.0 * x) * np.exp(-0.35 * np.maximum(x, 0.0)),
                ),
            ]
        )

        noise = rng.normal(0.0, 8.0, size=freq.size)
        correlated_noise = np.convolve(noise, np.ones(5) / 5.0, mode="same")
        noisy = 400.0 + 35.0 * x + correlated_noise
        cases.append(
            PhaseMaskSimulationCase(f"noisy_{boundary:g}", "adversarial", boundary, width, noisy)
        )

        spikes = 360.0 + 25.0 * x
        local_indices = np.flatnonzero((freq >= boundary) & (freq <= boundary * (2.0**0.5)))
        if local_indices.size:
            spike_count = max(1, min(8, local_indices.size // 8))
            chosen = rng.choice(local_indices, size=spike_count, replace=False)
            spikes = spikes.copy()
            spikes[chosen] += rng.choice((-240.0, 240.0), size=spike_count)
        cases.append(
            PhaseMaskSimulationCase(f"sparse_spikes_{boundary:g}", "adversarial", boundary, width, spikes)
        )

        alternating = 360.0 + 20.0 * x
        alternating = alternating.copy()
        alternating[local_indices[::2]] += 75.0
        alternating[local_indices[1::2]] -= 75.0
        cases.append(
            PhaseMaskSimulationCase(f"alternating_bins_{boundary:g}", "adversarial", boundary, width, alternating)
        )

        wrapped_source = 350.0 + 90.0 * x + 25.0 * np.sin(5.0 * x)
        wrapped_source = (wrapped_source + 180.0) % 360.0 - 180.0
        cases.append(
            PhaseMaskSimulationCase(f"wrapped_branch_{boundary:g}", "branch", boundary, width, wrapped_source)
        )
    if quick:
        keep = ("normal", "multi-turn", "adversarial", "branch")
        selected: list[PhaseMaskSimulationCase] = []
        for category in keep:
            selected.append(next(case for case in cases if case.category == category))
        return tuple(selected)
    return tuple(cases)


def run_phase_eq_mask_b_simulation(
    *,
    sample_rate: int = 48_000,
    fft_size: int = 4096,
    seed: int = 20260714,
    seeds: Iterable[int] | None = None,
    quick: bool = False,
) -> PhaseMaskSimulationResult:
    frequency = np.fft.rfftfreq(int(fft_size), d=1.0 / int(sample_rate))
    seed_values = tuple(int(value) for value in seeds) if seeds is not None else (int(seed),)
    if not seed_values:
        raise ValueError("seeds must contain at least one value")
    parameters = simulation_parameter_grid(quick=quick)
    records: list[PhaseMaskSimulationRecord] = []
    for seed_value in seed_values:
        cases = build_adversarial_phase_cases(frequency, seed=seed_value, quick=quick)
        for case in cases:
            case_name = (
                case.name if len(seed_values) == 1 else f"seed_{seed_value}:{case.name}"
            )
            current_phase = _legacy_phase_low_mask_15(
                frequency,
                case.phase_deg,
                boundary_hz=case.boundary_hz,
                smoothing_oct=case.smoothing_oct,
            )
            current_metrics = phase_mask_metrics(
                frequency,
                current_phase,
                original_phase_deg=case.phase_deg,
                boundary_hz=case.boundary_hz,
                smoothing_oct=case.smoothing_oct,
            )
            for parameter in parameters:
                candidate = apply_phase_eq_mask_b_low(
                    frequency,
                    case.phase_deg,
                    boundary_hz=case.boundary_hz,
                    smoothing_oct=case.smoothing_oct,
                    parameters=parameter,
                )
                candidate_metrics = phase_mask_metrics(
                    frequency,
                    candidate.phase_deg,
                    original_phase_deg=case.phase_deg,
                    boundary_hz=case.boundary_hz,
                    smoothing_oct=case.smoothing_oct,
                )
                records.append(
                    PhaseMaskSimulationRecord(
                        case_name,
                        case.category,
                        parameter,
                        candidate.method,
                        candidate.width_scale,
                        candidate.confidence,
                        candidate.neutral_phase_deg,
                        candidate.residual_phase_deg,
                        candidate.fade_slope_deg_per_oct,
                        candidate.runtime_ms,
                        current_metrics,
                        candidate_metrics,
                    )
                )
    scores = _score_parameters(records)
    return PhaseMaskSimulationResult(tuple(records), tuple(scores))


def phase_mask_metrics(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    *,
    original_phase_deg: np.ndarray,
    boundary_hz: float,
    smoothing_oct: float,
) -> PhaseMaskMetrics:
    freq = np.asarray(frequency, dtype=float)
    phase = _unwrap_phase(phase_deg)
    original = _unwrap_phase(original_phase_deg)
    local = (
        (freq > 0.0)
        & (freq >= float(boundary_hz) / 2.0)
        & (freq <= float(boundary_hz) * (2.0 ** max(2.0 * float(smoothing_oct), 0.5)))
    )
    local_freq = freq[local]
    local_phase = phase[local]
    if local_freq.size >= 4:
        group_delay_ms = -np.gradient(local_phase, local_freq, edge_order=1) / 360.0 * 1000.0
        group_delay_peak = float(np.percentile(np.abs(group_delay_ms), 99.0))
        group_delay_roughness = float(np.percentile(np.abs(np.diff(group_delay_ms)), 95.0))
    else:
        group_delay_peak = float("inf")
        group_delay_roughness = float("inf")
    protected_start = float(boundary_hz) * (2.0 ** max(2.0 * float(smoothing_oct), 0.5))
    protected = freq >= protected_start
    protected_error = (
        float(np.max(np.abs(phase[protected] - original[protected]))) if np.any(protected) else 0.0
    )
    impulse = _phase_only_impulse(phase)
    return PhaseMaskMetrics(
        effective_taps_99_9=_minimum_circular_energy_span(impulse, 0.999),
        tail_energy_1025=_minimum_circular_tail_energy(impulse, min(1025, impulse.size)),
        group_delay_peak_ms=group_delay_peak,
        group_delay_roughness_ms=group_delay_roughness,
        protected_phase_error_deg=protected_error,
    )


def _phase_only_impulse(phase_deg: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_deg, dtype=float)
    response = np.exp(1j * np.deg2rad(phase))
    response = response.copy()
    response[0] = 1.0 + 0.0j
    if response.size > 1:
        response[-1] = (1.0 if np.cos(np.deg2rad(phase[-1])) >= 0.0 else -1.0) + 0.0j
    return np.fft.irfft(response)


def _minimum_circular_energy_span(impulse: np.ndarray, fraction: float) -> int:
    energy = np.square(np.asarray(impulse, dtype=float))
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0
    target = min(max(float(fraction), 0.0), 1.0) * total
    doubled = np.concatenate([energy, energy])
    end = 0
    accumulated = 0.0
    best = energy.size
    for start in range(energy.size):
        while end < start + energy.size and accumulated < target:
            accumulated += float(doubled[end])
            end += 1
        if accumulated >= target:
            best = min(best, end - start)
        accumulated -= float(doubled[start])
    return int(best)


def _minimum_circular_tail_energy(impulse: np.ndarray, taps: int) -> float:
    energy = np.square(np.asarray(impulse, dtype=float))
    total = float(np.sum(energy))
    if total <= 0.0 or taps >= energy.size:
        return 0.0
    doubled = np.concatenate([energy, energy])
    window = float(np.sum(doubled[:taps]))
    best = window
    for start in range(1, energy.size):
        window += float(doubled[start + taps - 1]) - float(doubled[start - 1])
        best = max(best, window)
    return max(0.0, 1.0 - best / total)


def _score_parameters(records: list[PhaseMaskSimulationRecord]) -> list[PhaseMaskParameterScore]:
    if not records:
        return []
    grouped: dict[str, list[PhaseMaskSimulationRecord]] = {}
    parameters_by_key: dict[str, PhaseMaskBParameters] = {}
    for record in records:
        grouped.setdefault(record.parameters.key, []).append(record)
        parameters_by_key[record.parameters.key] = record.parameters

    per_case: dict[str, list[PhaseMaskSimulationRecord]] = {}
    for record in records:
        per_case.setdefault(record.case_name, []).append(record)
    tap_scores: dict[tuple[str, str], float] = {}
    smooth_scores: dict[tuple[str, str], float] = {}
    case_scores: dict[tuple[str, str], float] = {}
    for case_name, case_records in per_case.items():
        tap_cost = np.asarray(
            [
                record.candidate.effective_taps_99_9
                + 2048.0 * record.candidate.tail_energy_1025
                for record in case_records
            ],
            dtype=float,
        )
        smooth_cost = np.asarray(
            [
                record.candidate.group_delay_peak_ms
                + 2.0 * record.candidate.group_delay_roughness_ms
                + 1000.0 * record.candidate.protected_phase_error_deg
                for record in case_records
            ],
            dtype=float,
        )
        tap_quality = _rank_quality(tap_cost)
        smooth_quality = _rank_quality(smooth_cost)
        for index, record in enumerate(case_records):
            key = (record.parameters.key, case_name)
            tap_scores[key] = float(tap_quality[index])
            smooth_scores[key] = float(smooth_quality[index])
            case_scores[key] = 0.5 * float(tap_quality[index]) + 0.5 * float(smooth_quality[index])

    runtimes = np.asarray(
        [np.median([record.runtime_ms for record in grouped[key]]) for key in grouped], dtype=float
    )
    # Microbenchmark ordering below 1 ms is dominated by scheduler and cache noise.
    # Treat all candidates inside the interactive budget as equivalent, then
    # degrade linearly until 5 ms.
    load_quality = np.clip((5.0 - runtimes) / 4.0, 0.0, 1.0)
    load_quality[runtimes <= 1.0] = 1.0
    load_by_key = {key: float(load_quality[index]) for index, key in enumerate(grouped)}

    scores: list[PhaseMaskParameterScore] = []
    for key, parameter_records in grouped.items():
        case_keys = [(key, record.case_name) for record in parameter_records]
        tap_score = float(np.mean([tap_scores[item] for item in case_keys]))
        smooth_score = float(np.mean([smooth_scores[item] for item in case_keys]))
        load_score = load_by_key[key]
        da_score = 0.4 * tap_score + 0.4 * smooth_score + 0.2 * load_score
        worst_case = float(np.percentile([case_scores[item] for item in case_keys], 10.0))
        tap_ratios = [
            record.candidate.effective_taps_99_9 / max(record.current.effective_taps_99_9, 1)
            for record in parameter_records
        ]
        tail_ratios = [
            (record.candidate.tail_energy_1025 + 1.0e-12)
            / (record.current.tail_energy_1025 + 1.0e-12)
            for record in parameter_records
        ]
        fallback_rate = float(
            np.mean(["fallback" in record.method or not record.method for record in parameter_records])
        )
        scores.append(
            PhaseMaskParameterScore(
                parameters_by_key[key],
                tap_score,
                smooth_score,
                load_score,
                da_score,
                worst_case,
                float(np.median(tap_ratios)),
                float(np.median(tail_ratios)),
                float(np.median([record.runtime_ms for record in parameter_records])),
                fallback_rate,
            )
        )
    scores.sort(
        key=lambda score: (
            -score.da_score,
            -score.worst_case_score,
            score.parameters.max_slope_deg_per_oct,
            score.parameters.base_slope_deg_per_oct,
            score.median_runtime_ms,
        )
    )
    return scores


def _rank_quality(cost: np.ndarray) -> np.ndarray:
    values = np.asarray(cost, dtype=float)
    if values.size <= 1 or np.allclose(values, values[0], equal_nan=True):
        return np.ones_like(values)
    finite = np.isfinite(values)
    fallback = float(np.max(values[finite])) * 2.0 if np.any(finite) else 1.0
    safe = np.where(finite, values, fallback)
    order = np.argsort(safe, kind="stable")
    sorted_values = safe[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * float(start + stop - 1)
        start = stop
    return 1.0 - ranks / max(values.size - 1, 1)


def _unwrap_phase(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    output = source.copy()
    finite = np.isfinite(output)
    if np.count_nonzero(finite) >= 2:
        output[finite] = np.rad2deg(np.unwrap(np.deg2rad(output[finite])))
    return output


def _legacy_phase_low_mask_15(
    frequency: np.ndarray,
    phase_deg: np.ndarray,
    *,
    boundary_hz: float,
    smoothing_oct: float,
) -> np.ndarray:
    """Reproduce the pre-adaptive 15 deg/oct baseline for stable comparisons."""

    freq = np.asarray(frequency, dtype=float)
    source = _unwrap_phase(phase_deg)
    boundary = float(boundary_hz)
    hold = float(np.interp(boundary, freq, source))
    output = source.copy()
    outside = freq < boundary
    if np.any(outside):
        distance_oct = np.maximum(
            np.log2(boundary / np.maximum(freq[outside], 1.0e-9)),
            0.0,
        )
        magnitude = np.maximum(abs(hold) - 15.0 * distance_oct, 0.0)
        total_oct = abs(hold) / 15.0
        edge_oct = min(1.0 / 6.0, total_oct * 0.5)
        edge_start = max(total_oct - edge_oct, 0.0)
        edge = (distance_oct > edge_start) & (distance_oct < total_oct)
        if np.any(edge):
            t = (distance_oct[edge] - edge_start) / edge_oct
            h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
            h10 = t**3 - 2.0 * t**2 + t
            magnitude[edge] = h00 * 15.0 * edge_oct + h10 * edge_oct * -15.0
        output[outside] = np.copysign(magnitude, hold)
    if smoothing_oct <= 0.0:
        return output
    outer_slope = (
        0.0
        if hold == 0.0
        else np.copysign(15.0, hold) / max(boundary * np.log(2.0), 1.0e-12)
    )
    connection = apply_one_sided_boundary(
        freq,
        source,
        OneSidedBoundarySpec(
            fb_hz=boundary,
            delta_oct=float(smoothing_oct),
            side="lower",
            axis="linear",
            outer_value=hold,
            outer_slope=outer_slope,
            reference_oct=1.0 / 24.0,
            name="Legacy Phase EQ mask 15 deg/oct Lo",
        ),
    )
    item = connection.item
    if item.applied and item.f_start_hz is not None and item.f_end_hz is not None:
        transition = (freq >= float(item.f_start_hz)) & (freq <= float(item.f_end_hz))
        output[transition] = connection.values[transition]
    return output


def records_as_rows(records: Iterable[PhaseMaskSimulationRecord]) -> list[dict[str, object]]:
    return [record.as_row() for record in records]


def scores_as_rows(scores: Iterable[PhaseMaskParameterScore]) -> list[dict[str, object]]:
    return [score.as_row() for score in scores]
