from __future__ import annotations

from typing import Any

import numpy as np

from phase_fir_designer import SpeakerResponse
from timing_provenance import normalize_timing_provenance
from utils.settings_io import speaker_response_from_payload, speaker_response_payload


SPEAKER_INPUT_BUNDLE_KIND = "speaker_input_measurement_bundle"


def speaker_calibration_state_from_measurement_payload(payload: Any) -> str:
    """Protect measured responses without interpreting arbitrary FRD as calibrated."""
    if not isinstance(payload, dict):
        return "raw"
    if payload.get("input_calibration_state") == "measurement_calibration_unknown":
        return "measurement_calibration_unknown"
    if payload.get("external_mic_calibration_applied") is True:
        return "measurement_calibrated"
    session = payload.get("measurement_session")
    if not isinstance(session, dict):
        return "raw"
    application = session.get("calibration_application")
    if isinstance(application, dict):
        if application.get("output_finalized") is True:
            return "measurement_calibrated"
        if application.get("magnitude_applied") is True:
            return "measurement_calibrated"
        if application.get("magnitude_applied") is False:
            return "raw"
    # Older Measure exports recorded the selected calibration ID, not whether
    # its curve was available. Avoid applying a second curve to those results.
    if str(session.get("mic_calibration_id") or "").strip():
        return "measurement_calibration_unknown"
    return "raw"


def speaker_allows_mic_calibration(state: str) -> bool:
    return state not in {
        "legacy_calibrated", "measurement_calibrated", "measurement_calibration_unknown",
    }


def timing_provenance_from_measurement_payload(payload: Any) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    source = payload.get("timing_provenance")
    if source is None and isinstance(payload.get("measurement_session"), dict):
        source = payload["measurement_session"].get("timing_provenance")
    normalized = normalize_timing_provenance(source)
    return normalized.to_dict() if normalized is not None else None


def speaker_input_measurement_bundle(
    *,
    speaker_input: SpeakerResponse,
    speaker_source_name: str = "",
    near_field_woofer: SpeakerResponse | None = None,
    near_field_woofer_source_name: str = "",
    near_field_port: SpeakerResponse | None = None,
    near_field_port_source_name: str = "",
    auto_iir_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": SPEAKER_INPUT_BUNDLE_KIND,
        "speaker_input": speaker_response_payload(speaker_input),
        "speaker_source_name": speaker_source_name,
    }
    metadata = normalize_auto_iir_metadata(auto_iir_metadata)
    if metadata:
        payload["auto_iir_metadata"] = metadata
    if near_field_woofer is not None:
        payload["near_field_woofer"] = speaker_response_payload(near_field_woofer)
        payload["near_field_woofer_source_name"] = near_field_woofer_source_name
    if near_field_port is not None:
        payload["near_field_port"] = speaker_response_payload(near_field_port)
        payload["near_field_port_source_name"] = near_field_port_source_name
    return payload


def is_speaker_input_measurement_bundle(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("kind") == SPEAKER_INPUT_BUNDLE_KIND


def speaker_input_from_measurement_payload(payload: Any) -> SpeakerResponse | None:
    if is_speaker_input_measurement_bundle(payload):
        return speaker_response_from_payload(payload.get("speaker_input"))
    if isinstance(payload, dict) and "impedance_ohm" in payload:
        return None
    if isinstance(payload, dict) and ("frequency" not in payload or "gain_db" not in payload):
        return None
    return speaker_response_from_payload(payload)


def near_field_woofer_from_measurement_payload(payload: Any) -> SpeakerResponse | None:
    if not is_speaker_input_measurement_bundle(payload):
        return None
    return speaker_response_from_payload(payload.get("near_field_woofer"))


def near_field_port_from_measurement_payload(payload: Any) -> SpeakerResponse | None:
    if not is_speaker_input_measurement_bundle(payload):
        return None
    return speaker_response_from_payload(payload.get("near_field_port"))


def speaker_input_source_name_from_measurement_payload(payload: Any, fallback: str = "") -> str:
    if not is_speaker_input_measurement_bundle(payload):
        return fallback
    return str(payload.get("speaker_source_name") or fallback)


def near_field_woofer_source_name_from_measurement_payload(payload: Any, fallback: str = "") -> str:
    if not is_speaker_input_measurement_bundle(payload):
        return fallback
    return str(payload.get("near_field_woofer_source_name") or fallback)


def near_field_port_source_name_from_measurement_payload(payload: Any, fallback: str = "") -> str:
    if not is_speaker_input_measurement_bundle(payload):
        return fallback
    return str(payload.get("near_field_port_source_name") or fallback)


def normalize_auto_iir_metadata(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("valid_f_min_hz", "valid_f_max_hz", "independent_resolution_hz"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number) and number > 0.0:
            result[key] = number
    source = str(payload.get("source") or "").strip()
    if source:
        result["source"] = source
    return result


def inferred_auto_iir_metadata(
    response: SpeakerResponse,
    *,
    source: str = "imported",
) -> dict[str, Any]:
    frequency = np.asarray(response.frequency, dtype=float)
    frequency = np.unique(np.sort(frequency[np.isfinite(frequency) & (frequency > 0.0)]))
    if frequency.size < 2:
        return {}
    metadata: dict[str, Any] = {
        "valid_f_min_hz": float(frequency[0]),
        "valid_f_max_hz": float(frequency[-1]),
        "source": str(source),
    }
    low = frequency[frequency <= min(float(frequency[-1]), max(200.0, float(frequency[0]) * 64.0))]
    if low.size >= 4:
        spacing = np.diff(low)
        median = float(np.median(spacing))
        if median > 0.0 and float(np.percentile(np.abs(spacing - median), 90.0)) <= median * 0.08:
            metadata["independent_resolution_hz"] = median
            if float(frequency[0]) <= max(5.0, median * 2.1):
                metadata["valid_f_min_hz"] = max(20.0, float(frequency[0]))
                metadata["source"] = f"{source}_fft_fallback"
    return metadata


def auto_iir_metadata_from_measurement_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metadata = normalize_auto_iir_metadata(payload.get("auto_iir_metadata"))
    if metadata:
        return metadata
    session = payload.get("measurement_session")
    if isinstance(session, dict):
        reference = session.get("ess_reference")
        if isinstance(reference, dict):
            derived = normalize_auto_iir_metadata(
                {
                    "valid_f_min_hz": reference.get("start_frequency_hz"),
                    "valid_f_max_hz": reference.get("end_frequency_hz"),
                    "source": "measurement_ess_band",
                }
            )
            raw_ir = session.get("combined_raw_ir")
            sample_rate = session.get("ir_sample_rate_hz")
            if isinstance(raw_ir, list) and raw_ir and sample_rate:
                try:
                    resolution = float(sample_rate) / float(len(raw_ir))
                except (TypeError, ValueError, ZeroDivisionError):
                    resolution = 0.0
                if np.isfinite(resolution) and resolution > 0.0:
                    derived["independent_resolution_hz"] = resolution
            if derived:
                return derived
    response = speaker_input_from_measurement_payload(payload)
    return inferred_auto_iir_metadata(response, source="legacy") if response is not None else {}
