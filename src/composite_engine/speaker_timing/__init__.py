from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from composite_engine.distance_timing import external_distance_timing
from timing_provenance import relative_timing_restore_samples, trusted_relative_arrivals, normalize_timing_provenance


@dataclass(frozen=True)
class SpeakerTimingProjection:
    """One exclusive physical-time projection for every active speaker Channel."""

    source: str
    samples_by_channel: dict[str, float]
    complete: bool
    warning: str = ""


def _channel_id(row: Mapping[str, Any]) -> str:
    channel_id = str(row.get("channel_id", "")).strip()
    if channel_id:
        return channel_id
    return "|".join((
        str(row.get("group", "")),
        str(row.get("way", row.get("band", ""))),
        str(row.get("name", "")),
    ))


def _active_generated_rows(
    channel_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        row for row in channel_rows
        if row.get("source") == "generated_band" and bool(row.get("enabled", True))
    )


def resolve_speaker_timing_projection(
    channel_rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any] | None,
    *,
    sample_rate_hz: int,
) -> SpeakerTimingProjection:
    """Resolve external-distance, PhaseEQ, or neutral timing without mixing them.

    External distance is an explicit override.  PhaseEQ Tweeter timing is used
    only when every active generated Channel belongs to one trusted reference
    set.  Partial or stale provenance therefore cannot shift only part of a
    system before phase-fit fallback runs.
    """

    rows = _active_generated_rows(channel_rows)
    if not rows:
        return SpeakerTimingProjection("none", {}, True)
    ids = tuple(_channel_id(row) for row in rows)
    if len(set(ids)) != len(ids):
        return SpeakerTimingProjection(
            "none", {}, False, "Channel IDが重複しているため時間基準を適用できません。",
        )

    config = (settings or {}).get("external_distance_timing", {})
    if not isinstance(config, Mapping):
        config = {}
    if bool(config.get("enabled", False)):
        stored_rows = config.get("channels", ())
        if not isinstance(stored_rows, (list, tuple)):
            stored_rows = ()
        by_id: dict[str, float] = {}
        by_identity: dict[tuple[str, str, str], float] = {}
        for item in stored_rows:
            if not isinstance(item, Mapping):
                continue
            try:
                distance = float(item.get("distance_m", 0.0))
            except (TypeError, ValueError, OverflowError):
                distance = 0.0
            by_id[str(item.get("channel_id", "")).strip()] = distance
            by_identity[(
                str(item.get("name", "")), str(item.get("way", "")),
                str(item.get("group", "")),
            )] = distance
        distances = {
            _channel_id(row): by_id.get(
                _channel_id(row),
                by_identity.get((
                    str(row.get("name", "")),
                    str(row.get("way", row.get("band", ""))),
                    str(row.get("group", "")),
                ), 0.0),
            )
            for row in rows
        }
        if any(value <= 0.0 for value in distances.values()):
            return SpeakerTimingProjection(
                "external_distance", {}, False,
                "外部測定距離が未入力のChannelがあるため、時間基準は未適用です。",
            )
        try:
            timing = external_distance_timing(
                distances,
                sample_rate_hz=int(sample_rate_hz),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return SpeakerTimingProjection("external_distance", {}, False, str(exc))
        earliest = min(timing.arrival_samples.values())
        origin_offsets = {}
        for row in rows:
            item = normalize_timing_provenance(row.get("timing_provenance"))
            # Distances replace physical arrival differences, not P0's origin
            # movement. Undo the latter without re-adding measured distance.
            origin_offsets[_channel_id(row)] = (
                (item.removed_bulk_delay_samples - item.target_arrival_sample)
                * int(sample_rate_hz) / item.sample_rate_hz
                if item is not None and item.trusted else 0.0
            )
        return SpeakerTimingProjection(
            "external_distance",
            {
                channel_id: float(arrival - earliest + origin_offsets[channel_id])
                for channel_id, arrival in timing.arrival_samples.items()
            },
            True,
        )

    provenance = {_channel_id(row): row.get("timing_provenance") for row in rows}
    trusted = trusted_relative_arrivals(
        ids, provenance, sample_rate_hz=int(sample_rate_hz),
    )
    if trusted is None:
        return SpeakerTimingProjection(
            "none", {}, False,
            "全Channelで共通のTweeter基準時間が揃っていないため、位相推定を使用します。",
        )
    restored: dict[str, float] = {}
    for row in rows:
        channel_id = _channel_id(row)
        samples = relative_timing_restore_samples(
            row.get("timing_provenance"), sample_rate_hz=int(sample_rate_hz),
        )
        if samples is None:
            return SpeakerTimingProjection("none", {}, False)
        restored[channel_id] = float(samples)
    return SpeakerTimingProjection("phaseeq_tweeter_reference", restored, True)


def speaker_timing_samples_for_row(
    projection: SpeakerTimingProjection,
    row: Mapping[str, Any],
) -> float:
    return float(projection.samples_by_channel.get(_channel_id(row), 0.0))


__all__ = [
    "SpeakerTimingProjection",
    "resolve_speaker_timing_projection",
    "speaker_timing_samples_for_row",
]
