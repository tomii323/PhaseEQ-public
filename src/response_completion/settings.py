"""Pure, idempotent migration of saved response-extension choices."""

METHOD = "shared-lo-hi-v1"
LEGACY_UI_KEYS = frozenset({
    "speaker_hf_extension_start_hz", "speaker_hf_extension_slope_db_per_oct",
    "speaker_hf_custom_slope_db_per_oct", "speaker_hf_gain_mode", "speaker_hf_phase_mode",
    "speaker_lf_extension_mode", "speaker_extension_transition_oct", "speaker_extension_strength",
    "speaker_hf_phase_fit_start_hz", "speaker_hf_min_slope_db_per_oct", "speaker_hf_max_slope_db_per_oct",
})


def migrate_extension_choices(settings):
    """Retain raw data and explicit ON/OFF; retired algorithms cannot reactivate."""
    migrated = {key: value for key, value in settings.items() if key not in LEGACY_UI_KEYS}
    migrated["response_extension_method"] = METHOD
    return migrated


def migrate_config_extensions(payload):
    migrated = dict(payload)
    migrated["ui"] = migrate_extension_choices(payload.get("ui", {}))
    profile = dict(payload.get("ui_profile", {}))
    if isinstance(profile.get("state"), dict):
        profile["state"] = migrate_extension_choices(profile["state"])
    migrated["ui_profile"] = profile
    migrated["processing"] = {**payload.get("processing", {}), "response_extension_method": METHOD}
    return migrated
