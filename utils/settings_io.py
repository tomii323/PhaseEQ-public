from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from phase_fir_designer import DesignConfig, SpeakerResponse


CURRENT_SETTINGS_FILE_REVISION = 2
MIN_SUPPORTED_SETTINGS_FILE_REVISION = CURRENT_SETTINGS_FILE_REVISION
CURRENT_CONFIG_SCHEMA_VERSION = 3


class UnsupportedSettingsFileRevision(ValueError):
    def __init__(self, revision: int, *, minimum: int, maximum: int) -> None:
        self.revision = int(revision)
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        if self.revision < self.minimum:
            reason = "too_old"
            message = (
                f"設定ファイルrevision {self.revision}は対応範囲外です "
                f"({self.minimum}～{self.maximum})。"
            )
        else:
            reason = "too_new"
            message = (
                f"設定ファイルrevision {self.revision}はこのアプリより新しいため読み込めません "
                f"({self.minimum}～{self.maximum})。"
            )
        self.reason = reason
        super().__init__(message)


class UnsupportedSettingsSchema(ValueError):
    def __init__(self, schema_version: int, *, required: int) -> None:
        self.schema_version = int(schema_version)
        self.required = int(required)
        self.revision = CURRENT_SETTINGS_FILE_REVISION
        self.reason = "too_old" if self.schema_version < self.required else "too_new"
        super().__init__(
            f"設定schema {self.schema_version}は対応範囲外です。"
            f"schema {self.required}が必要です。"
        )


def settings_file_revision(payload: dict[str, Any]) -> int:
    if not isinstance(payload, dict):
        raise ValueError("settings payload must be an object")
    raw = payload.get("settings_file_revision", -1)
    if isinstance(raw, bool):
        raise ValueError("settings_file_revision must be an integer")
    try:
        revision = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("settings_file_revision must be an integer") from exc
    return revision


def validate_settings_file_revision(
    payload: dict[str, Any],
    *,
    minimum: int = MIN_SUPPORTED_SETTINGS_FILE_REVISION,
    maximum: int = CURRENT_SETTINGS_FILE_REVISION,
) -> int:
    revision = settings_file_revision(payload)
    if revision < int(minimum) or revision > int(maximum):
        raise UnsupportedSettingsFileRevision(
            revision,
            minimum=int(minimum),
            maximum=int(maximum),
        )
    return revision


def quarantine_unsupported_settings_file(path: Path, *, revision: int) -> Path:
    """Remove an obsolete autosave from the active path without destroying it."""
    path = Path(path)
    quarantine = (
        path.parent
        / "legacy"
        / "decommission_pending"
        / "settings-file-revisions"
    )
    quarantine.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = quarantine / f"{path.stem}.revision-{int(revision)}.{stamp}{path.suffix}"
    sequence = 1
    while target.exists():
        target = quarantine / (
            f"{path.stem}.revision-{int(revision)}.{stamp}-{sequence}{path.suffix}"
        )
        sequence += 1
    path.replace(target)
    return target


def speaker_response_payload(response: SpeakerResponse | None) -> dict[str, Any] | None:
    return asdict(response) if response is not None else None


def speaker_response_from_payload(payload: Any) -> SpeakerResponse | None:
    if payload is None:
        return None
    if isinstance(payload, SpeakerResponse):
        return payload
    if isinstance(payload, dict):
        # Measurement payloads may carry provenance and analysis metadata next
        # to the response arrays.  Keep those fields out of the domain model.
        if "frequency" not in payload or "gain_db" not in payload:
            return None
        return SpeakerResponse(
            frequency=payload["frequency"],
            gain_db=payload["gain_db"],
            phase_deg=payload.get("phase_deg"),
        )
    return None


def build_config_payload(
    *,
    app_name: str,
    app_version: str,
    app_author: str,
    schema_version: int,
    config: DesignConfig,
    ui_payload: dict[str, Any],
    ui_profile: dict[str, Any],
    snapshot_name: str | None = None,
    raw_speaker_response: SpeakerResponse | None = None,
    raw_target_response: SpeakerResponse | None = None,
    mic_cal_response: SpeakerResponse | None = None,
    speaker_source_name: str = "",
    target_source_name: str = "",
    mic_cal_source_name: str = "",
    response_refs: dict[str, dict[str, Any] | None] | None = None,
    embed_response_data: bool = True,
    processing: dict[str, Any] | None = None,
    eq_links: dict[str, Any] | None = None,
    eq_link_channel: str | None = None,
    eq_link_channels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    serialized_config = (
        config
        if embed_response_data
        else replace(config, speaker_response=None, target_response=None)
    )
    config_payload = asdict(serialized_config)
    if raw_target_response is None:
        config_payload["target_response"] = None
    if not embed_response_data:
        # Processed responses are derived from raw assets and current UI/DSP
        # settings. Keeping them out of the frequently written settings file
        # avoids serializing megabytes on every Streamlit rerun.
        config_payload["speaker_response"] = None
        config_payload["target_response"] = None
    io_payload: dict[str, Any] = {
        "speaker_source_name": speaker_source_name,
        "target_source_name": target_source_name,
        "mic_cal_source_name": mic_cal_source_name,
    }
    if embed_response_data:
        io_payload.update(
            {
                "speaker_response_raw": speaker_response_payload(raw_speaker_response),
                "target_response_raw": speaker_response_payload(raw_target_response),
                "mic_cal_response_raw": speaker_response_payload(mic_cal_response),
            }
        )
    else:
        io_payload["response_refs"] = dict(response_refs or {})
    payload = {
        "app": app_name,
        "app_version": app_version,
        "author": app_author,
        "settings_file_revision": CURRENT_SETTINGS_FILE_REVISION,
        "schema_version": schema_version,
        "snapshot_name": snapshot_name,
        "io": io_payload,
        "ui": ui_payload,
        "ui_profile": ui_profile,
        "config": config_payload,
    }
    if eq_links is not None:
        from utils.eq_links import EQLinks
        payload["eq_links"] = EQLinks(eq_links).snapshot()
        payload["eq_link_channel"] = eq_link_channel
        if eq_link_channels is not None:
            payload["eq_link_channels"] = eq_link_channels
    payload["processing"] = processing if processing is not None else processing_manifest()
    from response_completion.settings import migrate_config_extensions
    payload = migrate_config_extensions(payload)
    payload["revision"] = payload_revision(payload)
    return payload


def payload_revision(payload: dict[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "revision"}
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def response_revision(response: SpeakerResponse | None) -> str | None:
    if response is None:
        return None
    # JSON serialization is read-only; avoid recursively copying every sample
    # with asdict while keeping the established persisted hash byte-for-byte.
    payload = {"frequency": response.frequency, "gain_db": response.gain_db,
               "phase_deg": response.phase_deg}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def processing_manifest(input_result: Any = None, target_result: Any = None) -> dict[str, Any]:
    input_stages = ("original_response", "calibrated_response", "polarity_response", "centered_response", "phase_adjusted_response", "extended_response", "smoothed_response", "processed_response")
    target_stages = ("original_response", "base_response", "shifted_response", "effective_response")
    revisions: dict[int, str | None] = {}
    def revision(response: SpeakerResponse | None) -> str | None:
        # Unchanged pipeline stages share response objects. Memoize only during
        # this call; no persistent identity cache can hide later content edits.
        key = id(response)
        if key not in revisions:
            revisions[key] = response_revision(response)
        return revisions[key]
    return {
        "schema_version": 1,
        "pipeline_revision": "input-target-v2-shared-boundary",
        "input": {name: revision(getattr(input_result, name, None)) for name in input_stages},
        "target": {name: revision(getattr(target_result, name, None)) for name in target_stages},
    }


def validate_current_config_payload(
    payload: dict[str, Any],
    *,
    expected_schema_version: int = CURRENT_CONFIG_SCHEMA_VERSION,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("settings payload must be an object")
    validate_settings_file_revision(payload)
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != int(expected_schema_version):
        raise UnsupportedSettingsSchema(
            schema_version,
            required=int(expected_schema_version),
        )
    required_objects = ("config", "io", "ui", "ui_profile", "processing")
    missing = [key for key in required_objects if not isinstance(payload.get(key), dict)]
    if missing:
        raise ValueError("Current settings payload is missing objects: " + ", ".join(missing))
    if "migration" in payload:
        raise ValueError("Migrated settings payloads are not supported")
    from response_completion.settings import migrate_config_extensions
    migrated = migrate_config_extensions(payload)
    migrated["revision"] = payload_revision(migrated)
    return migrated


def build_target_settings_payload(
    *,
    app_name: str,
    app_version: str,
    schema_version: int,
    target_payload: dict[str, Any],
) -> dict[str, Any]:
    from response_completion.settings import migrate_extension_choices
    return {
        "app": app_name,
        "app_version": app_version,
        "settings_file_revision": CURRENT_SETTINGS_FILE_REVISION,
        "schema_version": schema_version,
        "kind": "target_response_settings",
        "target": migrate_extension_choices(target_payload),
    }


def config_from_payload(
    payload: dict[str, Any], *, overrides: dict[str, Any] | None = None,
) -> DesignConfig:
    # Apply destination-owned conditions before frequency normalization.
    payload = validate_current_config_payload(payload)
    if "eq_links" in payload:
        from utils.eq_links import EQLinks
        links = EQLinks(payload["eq_links"])
        channel = payload.get("eq_link_channel")
        if channel not in links.channels:
            raise ValueError("Missing Stereo Link channel")
        payload = links.effective(channel, payload)
    data = {**payload["config"], **(overrides or {})}
    # Before FIR input shaping entered Config, its shared IIR/FIR settings
    # lived only in UI state. Restore them before Config becomes authoritative.
    # An explicit Config record (including OFF) must always win.
    if "auto_gain_input_shaping" not in data:
        from phase_fir_designer.auto_iir_input_shaping import SHAPING_UI_DEFAULTS

        profile_state = payload["ui_profile"].get("state", {})
        legacy_ui = {**(profile_state if isinstance(profile_state, dict) else {}), **payload["ui"]}
        shaping = {key.removeprefix("auto_iir_shape_"): legacy_ui[key]
                   for key in SHAPING_UI_DEFAULTS if key in legacy_ui}
        if shaping:
            data["auto_gain_input_shaping"] = shaping
    return DesignConfig.from_dict(data).normalized()
