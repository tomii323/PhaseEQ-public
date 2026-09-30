"""Keep application-owned FFT settings across input/channel restoration."""
from collections.abc import Mapping


def preserve_common_analysis(payload: dict, state: Mapping) -> dict:
    """Copy only changed containers; never mutate the saved source payload."""
    fft = state.get("analysis_fft_size")
    if fft is None:
        return payload
    ui = dict(payload.get("ui", {}))
    if "analysis_fft_saved_size" in state:
        ui["analysis_fft_saved_size"] = state["analysis_fft_saved_size"]
    else:
        ui.pop("analysis_fft_saved_size", None)
    profile = dict(payload.get("ui_profile", {}))
    profile_state = dict(profile.get("state", {}))
    for key in ("analysis_fft_size", "analysis_fft_saved_size", "common_analysis_fft_size_draft"):
        profile_state.pop(key, None)
    profile["state"] = profile_state
    return {
        **payload,
        "config": {**payload.get("config", {}), "analysis_fft_size": int(fft)},
        "ui": ui,
        "ui_profile": profile,
    }
