from __future__ import annotations

import hashlib
import json
from typing import Any

from composite_engine.multiway_studio.processing.speaker_source import (
    resolve_speaker_source,
)


def studio_dsp_input_signature(
    rows: list[dict[str, Any]],
    *,
    sample_rate_hz: int,
    crossover_frequencies_hz: tuple[float, ...] = (),
    crossover_methods: tuple[str, ...] = (),
    speaker_timing_settings: dict[str, Any] | None = None,
) -> str:
    """Hash every input that can change Studio graphs or alignment analysis."""
    payload = {
        "schema": 2,
        "sample_rate_hz": int(sample_rate_hz),
        "crossover_frequencies_hz": [float(value) for value in crossover_frequencies_hz],
        "crossover_methods": [str(value) for value in crossover_methods],
        "external_distance_timing": _active_external_distance_timing(
            speaker_timing_settings,
        ),
        "channels": [_channel_payload(row) for row in rows if isinstance(row, dict)],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _active_external_distance_timing(
    settings: dict[str, Any] | None,
) -> dict[str, Any]:
    config = (settings or {}).get("external_distance_timing", {})
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return {}
    active = dict(config)
    active.pop("temperature_c", None)
    return active


def _channel_payload(row: dict[str, Any]) -> dict[str, Any]:
    speaker = resolve_speaker_source(row)
    return {
        "channel_id": str(row.get("channel_id", "")),
        "assignment_id": str(
            row.get("phaseeq_assignment_id") or row.get("latest_assignment_id", "")
        ),
        "assignment_revision": int(row.get("phaseeq_assignment_revision", 0) or 0),
        "source": str(row.get("source", "")),
        "band": str(row.get("band", "")),
        "name": str(row.get("name", "")),
        "way": str(row.get("way", "")),
        "group": str(row.get("group", "")),
        "enabled": bool(row.get("enabled", True)),
        "gain_db": float(row.get("gain_db", 0.0)),
        "dc_gain_normalize": bool(row.get("dc_gain_normalize", False)),
        "polarity": int(row.get("polarity", 1)),
        "delay_samples": float(row.get("delay_samples", 0.0)),
        "alignment_target": bool(row.get("phase_alignment_acoustic_target", False)),
        "alignment_target_return": row.get("phaseeq_alignment_definition"),
        "alignment_target_applied": bool(row.get("phaseeq_alignment_target_applied", False)),
        "alignment_delay_samples": float(row.get("auto_alignment_delay_samples", 0.0)),
        "alignment_allpass": [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in row.get("auto_alignment_allpass", ())
            if hasattr(item, "to_dict") or isinstance(item, dict)
        ],
        "speaker": None if speaker is None else {
            "label": speaker.label,
            "revision": speaker.revision,
            "content_hash": speaker.content_hash or _hash_bytes(speaker.data),
        },
        "phaseeq_fir_hash": _mapping_hash(row.get("phaseeq_fir_response")),
        "phaseeq_target_dc_abs": (row["phaseeq_fir_response"].get("target_dc_abs")
                                  if isinstance(row.get("phaseeq_fir_response"), dict) else None),
        "phaseeq_iir_sos": [list(section) for section in row.get("phaseeq_iir_sos", ())],
        "timing_provenance": (
            dict(row["timing_provenance"])
            if isinstance(row.get("timing_provenance"), dict) else None
        ),
        "baffle_correction_mode": str(row.get("baffle_correction_mode", "OFF")),
        "baffle_iir_sos": [list(section) for section in row.get("baffle_iir_sos", ())],
        "manual_iir_hash": _upload_hash(row.get("iir_response")),
        "additional_fir_hash": _upload_hash(row.get("additional_fir")),
        "upload_hash": _upload_hash(row.get("upload")),
        "iir_crossover": (
            row["iir_crossover"].to_dict()
            if hasattr(row.get("iir_crossover"), "to_dict") else {}
        ),
    }


def _mapping_hash(value: Any) -> str:
    if not isinstance(value, dict) or not value.get("data"):
        return ""
    return str(value.get("content_hash", "")) or _hash_bytes(bytes(value["data"]))


def _upload_hash(value: Any) -> str:
    if value is None or not hasattr(value, "getvalue"):
        return ""
    return _hash_bytes(bytes(value.getvalue()))


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
