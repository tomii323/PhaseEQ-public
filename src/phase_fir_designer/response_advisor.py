from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SEVERITY_RANK = {
    "danger": 0,
    "warning": 1,
    "info": 2,
    "good": 3,
}


@dataclass(frozen=True)
class DiagnosisCard:
    id: str
    severity: str
    category: str
    f_start_hz: float
    f_end_hz: float
    f_center_hz: float
    title: str
    summary: str
    recommendation: str
    confidence: float
    action: str
    metrics: dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.0


@dataclass(frozen=True)
class AdvisorSettings:
    peak_threshold_db: float = 2.0
    dip_threshold_db: float = 3.0
    confidence_threshold: float = 0.7
    reflection_threshold: float = 0.6
    phase_threshold_deg: float = 60.0
    group_delay_threshold_ms: float = 2.0
    max_cards: int = 12
    show_good_cards: bool = True
    nyquist_edge_exclusion_ratio: float = 0.995


def generate_response_advisor_cards(
    frequency_hz: np.ndarray,
    gain_error_db: np.ndarray,
    phase_error_deg: np.ndarray | None = None,
    *,
    group_delay_ms: np.ndarray | None = None,
    wavelet_frequency_hz: np.ndarray | None = None,
    wavelet_confidence: np.ndarray | None = None,
    wavelet_reflection_index: np.ndarray | None = None,
    wavelet_spread_ms: np.ndarray | None = None,
    wavelet_integrated_energy_db: np.ndarray | None = None,
    wavelet_peak_time_ms: np.ndarray | None = None,
    wavelet_centroid_time_ms: np.ndarray | None = None,
    taps: int | None = None,
    sample_rate: int | None = None,
    settings: AdvisorSettings | None = None,
) -> list[DiagnosisCard]:
    settings = settings or AdvisorSettings()
    freq = np.asarray(frequency_hz, dtype=float)
    gain_error = np.asarray(gain_error_db, dtype=float)
    phase_error = _same_shape_or_default(phase_error_deg, freq, 0.0)
    group_delay = _same_shape_or_default(group_delay_ms, freq, np.nan)
    if freq.shape != gain_error.shape:
        raise ValueError("frequency_hz and gain_error_db must have the same shape")

    valid = np.isfinite(freq) & np.isfinite(gain_error) & (freq > 0.0)
    if np.count_nonzero(valid) < 2:
        return []
    confidence = _metric_on_axis(freq, wavelet_frequency_hz, wavelet_confidence, 0.65)
    reflection = _metric_on_axis(freq, wavelet_frequency_hz, wavelet_reflection_index, 0.25)
    spread = _metric_on_axis(freq, wavelet_frequency_hz, wavelet_spread_ms, 0.0)
    integrated_energy = _metric_on_axis(freq, wavelet_frequency_hz, wavelet_integrated_energy_db, np.nan)
    peak_time = _metric_on_axis(freq, wavelet_frequency_hz, wavelet_peak_time_ms, np.nan)
    centroid_time = _metric_on_axis(freq, wavelet_frequency_hz, wavelet_centroid_time_ms, np.nan)
    in_band = np.ones_like(freq, dtype=bool)
    stable_time_band = in_band & _below_nyquist_edge_mask(freq, sample_rate, settings.nyquist_edge_exclusion_ratio)

    cards: list[DiagnosisCard] = []
    cards.extend(_direct_peak_cards(freq, gain_error, confidence, reflection, in_band, settings))
    cards.extend(_reflection_dip_cards(freq, gain_error, confidence, reflection, in_band, settings))
    cards.extend(_broad_tonal_dip_cards(freq, gain_error, confidence, reflection, in_band, settings))
    cards.extend(_narrow_ringing_cards(freq, gain_error, confidence, reflection, spread, in_band, settings))
    cards.extend(
        _phase_instability_cards(
            freq,
            phase_error,
            group_delay,
            confidence,
            reflection,
            spread,
            peak_time,
            centroid_time,
            stable_time_band,
            settings,
        )
    )
    cards.extend(_excess_time_energy_cards(freq, gain_error, integrated_energy, confidence, reflection, spread, stable_time_band))
    cards.extend(_delayed_energy_cards(freq, gain_error, confidence, reflection, spread, peak_time, centroid_time, stable_time_band))
    if settings.show_good_cards:
        cards.extend(_good_region_cards(freq, gain_error, confidence, reflection, in_band))
    return _sort_and_limit_cards(cards, settings.max_cards)


def _same_shape_or_default(values: np.ndarray | None, reference: np.ndarray, default: float) -> np.ndarray:
    if values is None:
        return np.full_like(reference, float(default), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape != reference.shape:
        return np.full_like(reference, float(default), dtype=float)
    return array


def _metric_on_axis(
    frequency: np.ndarray,
    metric_frequency: np.ndarray | None,
    metric_value: np.ndarray | None,
    default: float,
) -> np.ndarray:
    if metric_frequency is None or metric_value is None:
        return np.full_like(frequency, float(default), dtype=float)
    source_freq = np.asarray(metric_frequency, dtype=float)
    source_value = np.asarray(metric_value, dtype=float)
    valid = np.isfinite(source_freq) & np.isfinite(source_value) & (source_freq > 0.0)
    if np.count_nonzero(valid) < 2:
        return np.full_like(frequency, float(default), dtype=float)
    order = np.argsort(source_freq[valid])
    source_freq = source_freq[valid][order]
    source_value = source_value[valid][order]
    return np.interp(
        np.asarray(frequency, dtype=float),
        source_freq,
        source_value,
        left=float(source_value[0]),
        right=float(source_value[-1]),
    )


def _below_nyquist_edge_mask(frequency: np.ndarray, sample_rate: int | None, ratio: float) -> np.ndarray:
    if sample_rate is None or int(sample_rate) <= 0:
        return np.ones_like(frequency, dtype=bool)
    nyquist = float(sample_rate) / 2.0
    edge_ratio = float(np.clip(ratio, 0.0, 1.0))
    return np.asarray(frequency, dtype=float) < nyquist * edge_ratio


def _regions(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    return [part for part in np.split(indices, split) if part.size > 0]


def _region_bandwidth_oct(frequency: np.ndarray, indices: np.ndarray) -> float:
    if indices.size <= 1:
        return 0.0
    f_start = max(float(frequency[indices[0]]), 1e-9)
    f_end = max(float(frequency[indices[-1]]), f_start)
    return float(np.log2(f_end / f_start)) if f_end > f_start else 0.0


def _region_center(frequency: np.ndarray, values: np.ndarray, indices: np.ndarray) -> float:
    local = np.asarray(values[indices], dtype=float)
    if local.size == 0 or not np.any(np.isfinite(local)):
        return float(frequency[indices[indices.size // 2]])
    peak = int(np.nanargmax(np.abs(local)))
    return float(frequency[indices[peak]])


def _card(
    *,
    category: str,
    severity: str,
    frequency: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    title: str,
    summary: str,
    recommendation: str,
    confidence: float,
    action: str,
    metrics: dict[str, Any],
    impact_score: float,
) -> DiagnosisCard:
    center = _region_center(frequency, values, indices)
    return DiagnosisCard(
        id=f"{category}:{center:.1f}",
        severity=severity,
        category=category,
        f_start_hz=float(frequency[indices[0]]),
        f_end_hz=float(frequency[indices[-1]]),
        f_center_hz=center,
        title=title,
        summary=summary,
        recommendation=recommendation,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        action=action,
        metrics=metrics,
        impact_score=float(max(impact_score, 0.0)),
    )


def _direct_peak_cards(
    freq: np.ndarray,
    gain_error: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    in_band: np.ndarray,
    settings: AdvisorSettings,
) -> list[DiagnosisCard]:
    mask = (
        (gain_error > settings.peak_threshold_db)
        & (confidence >= settings.confidence_threshold)
        & (reflection < 0.4)
        & in_band
    )
    cards = []
    for region in _regions(mask):
        peak = float(np.nanmax(gain_error[region]))
        conf = float(np.nanmedian(confidence[region]))
        refl = float(np.nanmedian(reflection[region]))
        cards.append(
            _card(
                category="direct_peak",
                severity="warning",
                frequency=freq,
                values=gain_error,
                indices=region,
                title="Direct peak",
                summary="Targetより大きいピークで、直接音主体として扱いやすい領域です。",
                recommendation="自然さを保つため、まず軽いCut候補として検討してください。",
                confidence=conf,
                action="cut",
                metrics={"peak_error_db": round(peak, 2), "confidence": round(conf, 2), "reflection": round(refl, 2)},
                impact_score=abs(peak) * conf * (1.0 - refl),
            )
        )
    return cards


def _reflection_dip_cards(
    freq: np.ndarray,
    gain_error: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    in_band: np.ndarray,
    settings: AdvisorSettings,
) -> list[DiagnosisCard]:
    mask = (
        (gain_error < -abs(settings.dip_threshold_db))
        & ((reflection >= settings.reflection_threshold) | (confidence < 0.5))
        & in_band
    )
    cards = []
    for region in _regions(mask):
        dip = float(np.nanmin(gain_error[region]))
        conf = float(np.nanmedian(confidence[region]))
        refl = float(np.nanmedian(reflection[region]))
        cards.append(
            _card(
                category="reflection_dip",
                severity="danger",
                frequency=freq,
                values=gain_error,
                indices=region,
                title="Reflection dip",
                summary="反射やキャンセル由来の可能性があるディップです。",
                recommendation="Boostは避け、Target調整や測定条件の確認を優先してください。",
                confidence=max(refl, 1.0 - conf),
                action="avoid_boost",
                metrics={"dip_error_db": round(dip, 2), "confidence": round(conf, 2), "reflection": round(refl, 2)},
                impact_score=abs(dip) * max(refl, 1.0 - conf),
            )
        )
    return cards


def _broad_tonal_dip_cards(
    freq: np.ndarray,
    gain_error: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    in_band: np.ndarray,
    settings: AdvisorSettings,
) -> list[DiagnosisCard]:
    mask = (gain_error < -2.0) & (confidence > 0.75) & (reflection < 0.4) & in_band
    cards = []
    for region in _regions(mask):
        if _region_bandwidth_oct(freq, region) < 0.5:
            continue
        dip = float(np.nanmin(gain_error[region]))
        conf = float(np.nanmedian(confidence[region]))
        cards.append(
            _card(
                category="broad_tonal_dip",
                severity="info",
                frequency=freq,
                values=gain_error,
                indices=region,
                title="Broad tonal dip",
                summary="広めの落ち込みで、音色バランスとして聞こえやすい可能性があります。",
                recommendation="強いBoostではなく、Targetを少し緩めるか軽い補正から試してください。",
                confidence=conf,
                action="adjust_target",
                metrics={"dip_error_db": round(dip, 2), "bandwidth_oct": round(_region_bandwidth_oct(freq, region), 2)},
                impact_score=abs(dip) * conf,
            )
        )
    return cards


def _narrow_ringing_cards(
    freq: np.ndarray,
    gain_error: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    spread: np.ndarray,
    in_band: np.ndarray,
    settings: AdvisorSettings,
) -> list[DiagnosisCard]:
    mask = (gain_error > 1.5) & (spread >= 8.0) & (confidence > 0.55) & (reflection < 0.7) & in_band
    cards = []
    for region in _regions(mask):
        peak = float(np.nanmax(gain_error[region]))
        spread_value = float(np.nanmedian(spread[region]))
        cards.append(
            _card(
                category="narrow_ringing",
                severity="warning",
                frequency=freq,
                values=gain_error,
                indices=region,
                title="Narrow ringing",
                summary="狭めのピークにWavelet spreadの長さが重なり、共振や鳴きとして耳につく可能性があります。",
                recommendation="メリハリを保つため、強いBoostは避け、軽いCutや制振、測定確認を検討してください。",
                confidence=float(np.nanmedian(confidence[region])),
                action="cut",
                metrics={"peak_error_db": round(peak, 2), "spread_ms": round(spread_value, 2), "reflection": round(float(np.nanmedian(reflection[region])), 2)},
                impact_score=abs(peak) * min(spread_value / 8.0, 2.0),
            )
        )
    return cards


def _phase_instability_cards(
    freq: np.ndarray,
    phase_error: np.ndarray,
    group_delay: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    spread: np.ndarray,
    peak_time: np.ndarray,
    centroid_time: np.ndarray,
    in_band: np.ndarray,
    settings: AdvisorSettings,
) -> list[DiagnosisCard]:
    gd_deviation = _local_abs_deviation(group_delay)
    centroid_deviation = _local_abs_deviation(centroid_time)
    peak_deviation = _local_abs_deviation(peak_time)
    peak_to_centroid_gap = centroid_time - peak_time
    time_axis_unstable = (
        (centroid_deviation > 2.0)
        | (peak_deviation > 1.5)
        | (peak_to_centroid_gap > 4.0)
    )
    mask = (
        (
            (np.abs(phase_error) > settings.phase_threshold_deg)
            | (gd_deviation > settings.group_delay_threshold_ms)
            | time_axis_unstable
        )
        & ((gd_deviation > settings.group_delay_threshold_ms) | (confidence < 0.55) | (spread > 10.0) | time_axis_unstable)
        & in_band
    )
    cards = []
    for region in _regions(mask):
        phase_value = float(np.nanmax(np.abs(phase_error[region])))
        gd_value = _nanmedian_or_default(gd_deviation[region], 0.0)
        centroid_dev_value = _nanmedian_or_default(centroid_deviation[region], 0.0)
        time_gap_value = _nanmedian_or_default(peak_to_centroid_gap[region], 0.0)
        instability_confidence = max(
            phase_value / 180.0,
            gd_value / max(settings.group_delay_threshold_ms * 3.0, 1e-9),
            centroid_dev_value / 6.0,
            max(time_gap_value, 0.0) / 8.0,
            1.0 - _nanmedian_or_default(confidence[region], 0.5),
        )
        cards.append(
            _card(
                category="phase_instability",
                severity="warning",
                frequency=freq,
                values=phase_error,
                indices=region,
                title="Phase instability",
                summary="Phase Error、Group Delay、Wavelet時間軸のいずれかが不安定で、位相補正が自然さを損なう可能性があります。",
                recommendation="Phase補正は控えめにし、まず直接音/反射/測定窓の状態を確認してください。",
                confidence=float(np.clip(instability_confidence, 0.0, 1.0)),
                action="reduce_phase_correction",
                metrics={
                    "phase_error_deg": round(phase_value, 1),
                    "group_delay_dev_ms": round(gd_value, 2),
                    "centroid_dev_ms": round(centroid_dev_value, 2),
                    "centroid_peak_gap_ms": round(time_gap_value, 2),
                    "reflection": round(float(np.nanmedian(reflection[region])), 2),
                },
                impact_score=phase_value / 30.0 + gd_value + max(centroid_dev_value, 0.0),
            )
        )
    return cards


def _excess_time_energy_cards(
    freq: np.ndarray,
    gain_error: np.ndarray,
    integrated_energy: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    spread: np.ndarray,
    in_band: np.ndarray,
) -> list[DiagnosisCard]:
    if not np.any(np.isfinite(integrated_energy)):
        return []
    excess = integrated_energy - _local_median(integrated_energy)
    mask = (excess > 4.0) & (spread > 6.0) & (confidence > 0.45) & in_band
    cards = []
    for region in _regions(mask):
        excess_value = float(np.nanmax(excess[region]))
        cards.append(
            _card(
                category="excess_time_energy",
                severity="info",
                frequency=freq,
                values=excess,
                indices=region,
                title="Excess time energy",
                summary="周波数応答の見た目以上に、時間方向へ残るエネルギーが多い領域です。",
                recommendation="音量感や残響感が強い場合は、軽いCutや測定/設置の確認を検討してください。",
                confidence=float(np.nanmedian(confidence[region])),
                action="check_measurement",
                metrics={"excess_energy_db": round(excess_value, 2), "spread_ms": round(float(np.nanmedian(spread[region])), 2), "reflection": round(float(np.nanmedian(reflection[region])), 2)},
                impact_score=excess_value * float(np.nanmedian(confidence[region])),
            )
        )
    return cards


def _delayed_energy_cards(
    freq: np.ndarray,
    gain_error: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    spread: np.ndarray,
    peak_time: np.ndarray,
    centroid_time: np.ndarray,
    in_band: np.ndarray,
) -> list[DiagnosisCard]:
    if not (np.any(np.isfinite(peak_time)) or np.any(np.isfinite(centroid_time))):
        return []
    delay_score = np.maximum(
        np.nan_to_num(peak_time, nan=-np.inf),
        np.nan_to_num(centroid_time, nan=-np.inf),
    )
    delay_score[~np.isfinite(delay_score)] = np.nan
    centroid_gap = centroid_time - peak_time
    mask = (
        ((delay_score > 4.0) | (centroid_gap > 3.0))
        & (confidence > 0.4)
        & ((reflection > 0.45) | (spread > 6.0) | (centroid_gap > 4.0))
        & in_band
    )
    cards = []
    for region in _regions(mask):
        delay_value = _nanmedian_or_default(delay_score[region], 0.0)
        gap_value = _nanmedian_or_default(centroid_gap[region], 0.0)
        refl = _nanmedian_or_default(reflection[region], 0.0)
        severity = "warning" if refl > 0.55 or delay_value > 6.0 else "info"
        cards.append(
            _card(
                category="delayed_energy",
                severity=severity,
                frequency=freq,
                values=np.nan_to_num(delay_score, nan=0.0),
                indices=region,
                title="Delayed energy",
                summary="直接音より遅れたエネルギーが支配的で、FR上の凹凸以上に聞こえ方へ影響する可能性があります。",
                recommendation="Boostや強いPhase補正は慎重にし、反射、設置、測定窓、Target側の調整を優先してください。",
                confidence=_nanmedian_or_default(confidence[region], 0.0),
                action="avoid_boost",
                metrics={
                    "delay_ms": round(delay_value, 2),
                    "centroid_peak_gap_ms": round(gap_value, 2),
                    "spread_ms": round(_nanmedian_or_default(spread[region], 0.0), 2),
                    "reflection": round(refl, 2),
                },
                impact_score=delay_value * max(refl, 0.4) + max(gap_value, 0.0),
            )
        )
    return cards


def _good_region_cards(
    freq: np.ndarray,
    gain_error: np.ndarray,
    confidence: np.ndarray,
    reflection: np.ndarray,
    in_band: np.ndarray,
) -> list[DiagnosisCard]:
    mask = (np.abs(gain_error) < 1.5) & (confidence > 0.75) & (reflection < 0.35) & in_band
    regions = [region for region in _regions(mask) if _region_bandwidth_oct(freq, region) >= 1.0]
    if not regions:
        return []
    region = max(regions, key=lambda item: _region_bandwidth_oct(freq, item))
    return [
        _card(
            category="stable_region",
            severity="good",
            frequency=freq,
            values=gain_error,
            indices=region,
            title="Stable region",
            summary="Targetとの差が小さく、直接音主体で安定している帯域です。",
            recommendation="自然さを保つため、この帯域は大きく触らず基準として使えます。",
            confidence=float(np.nanmedian(confidence[region])),
            action="ignore",
            metrics={"bandwidth_oct": round(_region_bandwidth_oct(freq, region), 2), "median_error_db": round(float(np.nanmedian(gain_error[region])), 2)},
            impact_score=0.1,
        )
    ]


def _local_abs_deviation(values: np.ndarray, window: int = 9) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return np.full_like(values, np.nan)
    return np.abs(values - _local_median(values, window=window))


def _local_median(values: np.ndarray, window: int = 9) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan, dtype=float)
    if values.size == 0:
        return output
    half = max(int(window) // 2, 1)
    for idx in range(values.size):
        start = max(idx - half, 0)
        end = min(idx + half + 1, values.size)
        segment = values[start:end]
        finite = segment[np.isfinite(segment)]
        output[idx] = float(np.median(finite)) if finite.size else np.nan
    return output


def _nanmedian_or_default(values: np.ndarray, default: float) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(default)
    return float(np.median(finite))


def _sort_and_limit_cards(cards: list[DiagnosisCard], max_cards: int) -> list[DiagnosisCard]:
    unique: dict[str, DiagnosisCard] = {}
    for card in cards:
        existing = unique.get(card.id)
        if existing is None or card.impact_score > existing.impact_score:
            unique[card.id] = card
    sorted_cards = sorted(
        unique.values(),
        key=lambda card: (SEVERITY_RANK.get(card.severity, 99), -card.impact_score, card.f_center_hz),
    )
    return sorted_cards[: max(int(max_cards), 1)]
