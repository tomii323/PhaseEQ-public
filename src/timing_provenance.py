from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


TIMING_PROVENANCE_SCHEMA_VERSION = 2
TRUSTED_TIMING_CONFIDENCE = frozenset({"trusted", "derived"})


@dataclass(frozen=True)
class TimingProvenance:
    """Portable relative-arrival contract shared by PhaseEQ and Multiway."""

    timing_reference_kind: str = "unknown"
    timing_reference_session_id: str = ""
    reference_tweeter_channel_id: str = ""
    reference_measurement_id: str = ""
    source_measurement_id: str = ""
    sample_rate_hz: int = 0
    target_arrival_sample: float = float("nan")
    reference_arrival_sample: float = float("nan")
    relative_arrival_samples: float = float("nan")
    relative_arrival_ms: float = float("nan")
    removed_bulk_delay_samples: float = 0.0
    clock_adjustment_ppm: float = 0.0
    sequence_timing_valid: bool = False
    common_reference_timing_valid: bool = False
    arrival_time_valid: bool = False
    confidence: str = "invalid"
    source_ir_hash: str = ""
    measurement_recipe_hash: str = ""
    schema_version: int = TIMING_PROVENANCE_SCHEMA_VERSION
    phase_origin: str = "measured"
    p0_removed_delay_ms: float = 0.0
    response_hash: str = ""
    center_method: str = ""
    center_source: str = ""

    @property
    def trusted(self) -> bool:
        internally_consistent = bool(
            np.isfinite(self.target_arrival_sample)
            and np.isfinite(self.reference_arrival_sample)
            and np.isclose(
                self.relative_arrival_samples,
                self.target_arrival_sample - self.reference_arrival_sample,
                atol=1e-4,
                rtol=0.0,
            )
        )
        return bool(
            self.schema_version in {1, TIMING_PROVENANCE_SCHEMA_VERSION}
            and self.phase_origin == "measured"
            and np.isfinite(self.removed_bulk_delay_samples)
            and self.timing_reference_kind == "phaseeq_tweeter_reference"
            and self.timing_reference_session_id
            and self.reference_tweeter_channel_id
            and self.reference_measurement_id
            and self.source_measurement_id
            and self.sample_rate_hz > 0
            and self.sequence_timing_valid
            and self.common_reference_timing_valid
            and self.arrival_time_valid
            and self.confidence in TRUSTED_TIMING_CONFIDENCE
            and np.isfinite(self.relative_arrival_samples)
            and internally_consistent
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_timing_provenance(payload: Any) -> TimingProvenance | None:
    if isinstance(payload, TimingProvenance):
        return payload
    if not isinstance(payload, Mapping):
        return None
    try:
        version = int(payload.get("schema_version", TIMING_PROVENANCE_SCHEMA_VERSION))
        sample_rate = int(payload.get("sample_rate_hz", 0))
        relative_samples_raw = payload.get("relative_arrival_samples", float("nan"))
        relative_ms_raw = payload.get("relative_arrival_ms", float("nan"))
        relative_samples = (
            float(relative_samples_raw) if relative_samples_raw is not None else float("nan")
        )
        relative_ms = float(relative_ms_raw) if relative_ms_raw is not None else float("nan")
        if not np.isfinite(relative_ms) and sample_rate > 0 and np.isfinite(relative_samples):
            relative_ms = relative_samples * 1_000.0 / sample_rate
        return TimingProvenance(
            schema_version=version,
            timing_reference_kind=str(payload.get("timing_reference_kind", "unknown")),
            timing_reference_session_id=str(payload.get("timing_reference_session_id", "")),
            reference_tweeter_channel_id=str(payload.get("reference_tweeter_channel_id", "")),
            reference_measurement_id=str(payload.get("reference_measurement_id", "")),
            source_measurement_id=str(payload.get("source_measurement_id", "")),
            sample_rate_hz=sample_rate,
            target_arrival_sample=float(payload.get("target_arrival_sample", float("nan"))),
            reference_arrival_sample=float(payload.get("reference_arrival_sample", float("nan"))),
            relative_arrival_samples=relative_samples,
            relative_arrival_ms=relative_ms,
            removed_bulk_delay_samples=float(payload.get("removed_bulk_delay_samples", 0.0)),
            clock_adjustment_ppm=float(payload.get("clock_adjustment_ppm", 0.0)),
            sequence_timing_valid=bool(payload.get("sequence_timing_valid", False)),
            common_reference_timing_valid=bool(payload.get("common_reference_timing_valid", False)),
            arrival_time_valid=bool(payload.get("arrival_time_valid", False)),
            confidence=str(payload.get("confidence", "invalid")),
            source_ir_hash=str(payload.get("source_ir_hash", "")),
            measurement_recipe_hash=str(payload.get("measurement_recipe_hash", "")),
            phase_origin=str(payload.get("phase_origin", "measured")),
            p0_removed_delay_ms=float(payload.get("p0_removed_delay_ms", 0.0)),
            response_hash=str(payload.get("response_hash", "")),
            center_method=str(payload.get("center_method", "")),
            center_source=str(payload.get("center_source", "")),
        )
    except (TypeError, ValueError, OverflowError):
        return None


def trusted_relative_arrivals(
    channel_ids: Sequence[str],
    provenance_by_channel: Mapping[str, Any],
    *,
    sample_rate_hz: int,
) -> dict[str, float] | None:
    """Return one-session arrivals converted to the current sample rate."""

    normalized: dict[str, TimingProvenance] = {}
    for channel_id in channel_ids:
        item = normalize_timing_provenance(provenance_by_channel.get(channel_id))
        if item is None or not item.trusted:
            return None
        normalized[channel_id] = item
    sessions = {item.timing_reference_session_id for item in normalized.values()}
    references = {item.reference_tweeter_channel_id for item in normalized.values()}
    reference_measurements = {
        item.reference_measurement_id for item in normalized.values()
    }
    if (
        len(sessions) != 1
        or len(references) != 1
        or len(reference_measurements) != 1
    ):
        return None
    destination_rate = int(sample_rate_hz)
    if destination_rate <= 0:
        return None
    return {
        channel_id: float(item.relative_arrival_samples) * destination_rate / item.sample_rate_hz
        for channel_id, item in normalized.items()
    }


def trusted_reference_provenance(payload: Any) -> TimingProvenance | None:
    """Return a verified zero-time Reference Tweeter measurement."""

    item = normalize_timing_provenance(payload)
    if item is None or not item.trusted:
        return None
    if item.source_measurement_id != item.reference_measurement_id:
        return None
    if not np.isclose(item.relative_arrival_samples, 0.0, atol=1e-4, rtol=0.0):
        return None
    return item


def reference_arrival_samples_at_rate(
    payload: Any,
    *,
    sample_rate_hz: int,
) -> float | None:
    """Convert a trusted Reference Tweeter arrival to another sample rate."""

    item = trusted_reference_provenance(payload)
    destination_rate = int(sample_rate_hz)
    if item is None or destination_rate <= 0:
        return None
    return (
        float(item.reference_arrival_sample)
        * float(destination_rate)
        / float(item.sample_rate_hz)
    )


def timing_provenance_after_phase_centering(
    payload: Any,
    *,
    additional_removed_delay_ms: float | None,
    sample_rate_hz: int = 0,
    phase_origin: str = "measured",
    response_hash: str = "",
    center_method: str = "",
    center_source: str = "",
) -> dict[str, object] | None:
    """Record additional PhaseEQ centering without changing measured arrival."""

    item = normalize_timing_provenance(payload)
    if item is None:
        item = TimingProvenance(sample_rate_hz=int(sample_rate_hz))
    elif item.schema_version not in {1, TIMING_PROVENANCE_SCHEMA_VERSION}:
        # Never upgrade an unknown contract into a trusted reference.
        item = replace(item, confidence="invalid", common_reference_timing_valid=False)
    item = replace(item, schema_version=TIMING_PROVENANCE_SCHEMA_VERSION,
                   phase_origin=phase_origin, response_hash=response_hash,
                   center_method=center_method, center_source=center_source)
    delay_ms = float(additional_removed_delay_ms or 0.0)
    if not np.isfinite(delay_ms) or item.sample_rate_hz <= 0:
        return item.to_dict()
    return replace(
        item,
        p0_removed_delay_ms=delay_ms,
        removed_bulk_delay_samples=(
            float(item.removed_bulk_delay_samples)
            + delay_ms * float(item.sample_rate_hz) / 1_000.0
        ),
    ).to_dict()


def relative_timing_restore_samples(
    payload: Any,
    *,
    sample_rate_hz: int,
) -> float | None:
    """Return delay needed to restore a centered response to reference-relative time.

    Absolute capture latency is intentionally excluded.  Restoring only
    ``removed_bulk - reference_arrival`` keeps every Channel on the common
    Tweeter time plane while preserving the measured inter-Channel arrival.
    """

    item = normalize_timing_provenance(payload)
    destination_rate = int(sample_rate_hz)
    if item is None or not item.trusted or destination_rate <= 0:
        return None
    relative_removed = (
        float(item.removed_bulk_delay_samples)
        - float(item.reference_arrival_sample)
    )
    if not np.isfinite(relative_removed):
        return None
    return relative_removed * float(destination_rate) / float(item.sample_rate_hz)


def causal_delays_from_relative_arrivals(
    relative_arrivals_samples: Mapping[str, float],
) -> dict[str, float]:
    """Delay earlier arrivals so every Channel reaches the latest arrival."""

    if not relative_arrivals_samples:
        return {}
    values = {str(key): float(value) for key, value in relative_arrivals_samples.items()}
    if not np.isfinite(tuple(values.values())).all():
        raise ValueError("relative arrivals must be finite")
    latest = max(values.values())
    return {
        channel_id: float(np.floor((latest - arrival) + 0.5))
        for channel_id, arrival in values.items()
    }
