from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping

from .config import DesignConfig, SpeakerResponse


ResponseAssetResolver = Callable[[Mapping[str, Any] | str], SpeakerResponse]


@dataclass(frozen=True)
class ProjectRuntimeSnapshot:
    project_id: str
    project_name: str
    revision: str
    payload: Mapping[str, Any]
    config: DesignConfig
    raw_speaker_response: SpeakerResponse | None
    raw_target_response: SpeakerResponse | None
    mic_calibration: SpeakerResponse | None
    processed_speaker_response: SpeakerResponse | None
    effective_target_response: SpeakerResponse | None
    original_speaker_impulse: tuple[float, ...] | None = None
    original_speaker_impulse_sample_rate: int | None = None


def project_payload_revision(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_project_runtime(
    payload: Mapping[str, Any],
    *,
    project_id: str = "",
    project_name: str = "",
    asset_resolver: ResponseAssetResolver | None = None,
) -> ProjectRuntimeSnapshot:
    if not isinstance(payload, Mapping):
        raise TypeError("project payload must be a mapping")
    config_payload = payload.get("config", payload)
    if not isinstance(config_payload, Mapping):
        raise ValueError("project payload does not contain a valid config")
    config = DesignConfig.from_dict(dict(config_payload)).normalized()
    io = payload.get("io", {})
    if not isinstance(io, Mapping):
        io = {}
    refs = io.get("response_refs", {})
    if not isinstance(refs, Mapping):
        refs = {}

    def response(name: str) -> SpeakerResponse | None:
        embedded = _response_from_payload(io.get(name))
        if embedded is not None:
            return embedded
        ref = refs.get(name)
        if ref is None:
            return None
        if asset_resolver is None:
            raise ValueError(f"response asset resolver is required for {name}")
        resolved = asset_resolver(ref)
        if not isinstance(resolved, SpeakerResponse):
            raise TypeError(f"asset resolver returned an invalid response for {name}")
        return resolved

    ui = payload.get("ui", {})
    stored_name = str(ui.get("project_name", "")) if isinstance(ui, Mapping) else ""
    impulse_payload = io.get("speaker_impulse_raw")
    original_impulse: tuple[float, ...] | None = None
    original_impulse_rate: int | None = None
    if isinstance(impulse_payload, Mapping):
        samples = impulse_payload.get("samples")
        sample_rate = impulse_payload.get("sample_rate")
        if isinstance(samples, (list, tuple)) and len(samples) >= 2:
            try:
                original_impulse = tuple(float(value) for value in samples)
                original_impulse_rate = int(sample_rate)
            except (TypeError, ValueError):
                original_impulse = None
                original_impulse_rate = None
            if original_impulse_rate is not None and original_impulse_rate <= 0:
                original_impulse = None
                original_impulse_rate = None
    return ProjectRuntimeSnapshot(
        project_id=str(project_id),
        project_name=str(project_name or stored_name or "Untitled Project"),
        revision=project_payload_revision(payload),
        payload=payload,
        config=config,
        raw_speaker_response=response("speaker_response_raw"),
        raw_target_response=response("target_response_raw"),
        mic_calibration=response("mic_cal_response_raw"),
        processed_speaker_response=config.speaker_response,
        effective_target_response=config.target_response,
        original_speaker_impulse=original_impulse,
        original_speaker_impulse_sample_rate=original_impulse_rate,
    )


def _response_from_payload(value: Any) -> SpeakerResponse | None:
    if isinstance(value, SpeakerResponse):
        return value
    if not isinstance(value, Mapping) or "frequency" not in value or "gain_db" not in value:
        return None
    return SpeakerResponse(
        frequency=list(value["frequency"]),
        gain_db=list(value["gain_db"]),
        phase_deg=None if value.get("phase_deg") is None else list(value["phase_deg"]),
    )
