from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from phase_fir_designer.speaker import load_speaker_response_text
from phase_fir_designer.target_extension import is_wav_target_source
from response_completion.settings import METHOD
from utils.settings_io import speaker_response_payload

from .response import normalize_target_edit_payload


def canonical_target_definition(payload: Any, *, name: str = "Global Target") -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    scopes = source.get("compatible_scopes", ("phaseeq_channel", "composite_group"))
    if not isinstance(scopes, (list, tuple)):
        scopes = ("phaseeq_channel", "composite_group")
    level_mode = str(source.get("level_mode", "手動シフト"))
    level_db = float(source.get("level_db", 0.0))
    generated_target = not isinstance(source.get("target_response_raw"), dict) and not isinstance(
        source.get("materialized_response"), dict
    )
    if level_mode.lower().startswith("auto") and generated_target and level_db <= -120.0:
        level_db = 0.0
    definition = {
        "response_extension_method": METHOD,
        "name": str(source.get("name", name)).strip() or name,
        "description": str(source.get("description", "")),
        "category": str(source.get("category", "Reference")),
        "compatible_scopes": list(scopes),
        "preset_schema_version": int(source.get("preset_schema_version", 2)),
        "source_preset_id": str(source.get("source_preset_id", source.get("id", ""))),
        "level_mode": level_mode,
        "level_db": level_db,
        "application_gain_db": float(source.get("application_gain_db", 0.0)),
        "source_content_mode": (
            "Gain only"
            if str(source.get("source_content_mode", "Gain + Phase")) == "Gain only"
            else "Gain + Phase"
        ),
        "phase_enabled": bool(source.get("phase_enabled", True)),
        "lf_extension_enabled": bool(source.get("lf_extension_enabled", False)) and not is_wav_target_source(source.get("target_source_name", "")),
        "hf_extension_enabled": bool(source.get("hf_extension_enabled", False)),
        "target_source_name": str(source.get("target_source_name", "")),
        "target_url": str(source.get("target_url", "")),
        "target_response_raw": deepcopy(source.get("target_response_raw")),
        "materialized_response": deepcopy(source.get("materialized_response")),
        "target_edit": normalize_target_edit_payload(source.get("target_edit", {})),
    }
    digest_payload = json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    definition["revision"] = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
    return definition


def target_definition_frd_bytes(
    definition: dict[str, Any], sample_rate_hz: int,
) -> bytes:
    materialized = definition.get("materialized_response")
    raw = materialized if isinstance(materialized, dict) else definition.get("target_response_raw")
    if isinstance(raw, dict):
        frequency = np.asarray(raw.get("frequency", []), dtype=float)
        gain = np.asarray(raw.get("gain_db", []), dtype=float)
        phase_payload = raw.get("phase_deg")
        phase = np.asarray(phase_payload, dtype=float) if isinstance(phase_payload, list) else None
        valid = frequency.size == gain.size and frequency.size >= 2
        if phase is not None:
            valid = valid and phase.size == frequency.size
        if not valid:
            raise ValueError("Global Target raw response is invalid")
    else:
        frequency = np.concatenate((np.array([0.0]), np.geomspace(1.0, float(sample_rate_hz) / 2.0, 4096)))
        gain = np.zeros_like(frequency)
        phase = None
    if not isinstance(materialized, dict) and not bool(definition.get("phase_enabled", True)):
        phase = None
    gain = gain + float(definition.get("application_gain_db", 0.0))
    if not isinstance(materialized, dict):
        gain = gain + float(definition.get("level_db", 0.0))
    lines = (
        (f"{f:.16g} {g:.16g} {p:.16g}" for f, g, p in zip(frequency, gain, phase, strict=True))
        if phase is not None
        else (f"{f:.16g} {g:.16g}" for f, g in zip(frequency, gain, strict=True))
    )
    return ("\n".join(lines) + "\n").encode()


def target_definition_from_asset(
    filename: str,
    data: bytes,
    *,
    sample_rate_hz: int,
) -> dict[str, Any]:
    """Import a Group Target asset into a re-editable Target definition."""
    suffix = Path(str(filename)).suffix.lower()
    if suffix == ".wav":
        rate, values = wavfile.read(io.BytesIO(data))
        samples = np.asarray(values, dtype=float)
        if samples.ndim > 1:
            samples = samples[:, 0]
        if samples.size < 2 or int(rate) <= 0:
            raise ValueError("Target WAV is empty")
        response = np.fft.rfft(samples)
        frequency = np.fft.rfftfreq(samples.size, 1.0 / float(rate))
        raw = {
            "frequency": frequency.tolist(),
            "gain_db": (20.0 * np.log10(np.maximum(np.abs(response), 1e-12))).tolist(),
            "phase_deg": np.rad2deg(np.unwrap(np.angle(response))).tolist(),
        }
    else:
        text = data.decode("utf-8-sig")
        raw = speaker_response_payload(load_speaker_response_text(io.StringIO(text)))
    return canonical_target_definition({
        "name": Path(str(filename)).stem or "Imported Group Target",
        "level_mode": "手動シフト",
        "level_db": 0.0,
        "target_response_raw": raw,
        "target_edit": {},
    })
