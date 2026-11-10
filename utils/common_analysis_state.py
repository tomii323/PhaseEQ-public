"""Keep application-owned analysis/display settings across context restoration."""
from collections.abc import Mapping

GRAPH_MODE_OPTIONS = ("Light", "Interactive")
DEFAULT_GRAPH_MODE = "Light"


def current_graph_mode(state: Mapping) -> str:
    """Read committed presentation state without restoring or changing widgets."""
    mode = state.get("graph_mode")
    return mode if mode in GRAPH_MODE_OPTIONS else DEFAULT_GRAPH_MODE


def saved_graph_mode(payload: Mapping) -> str | None:
    """Read the canonical saved display mode, with legacy profile fallback."""
    mode = payload.get("ui", {}).get("graph_mode")
    if mode is None:
        mode = payload.get("ui_profile", {}).get("state", {}).get("graph_mode")
    return mode if mode in GRAPH_MODE_OPTIONS else None


def preserve_common_analysis(payload: dict, state: Mapping) -> dict:
    """Copy only changed containers; never mutate the saved source payload."""
    fft = state.get("analysis_fft_size")
    mode = state.get("graph_mode")
    has_mode = mode in GRAPH_MODE_OPTIONS
    if fft is None and not has_mode:
        return payload
    ui = dict(payload.get("ui", {}))
    keys = []
    if fft is not None:
        if "analysis_fft_saved_size" in state:
            ui["analysis_fft_saved_size"] = state["analysis_fft_saved_size"]
        else:
            ui.pop("analysis_fft_saved_size", None)
        keys.extend(("analysis_fft_size", "analysis_fft_saved_size", "common_analysis_fft_size_draft"))
    if has_mode:
        ui["graph_mode"] = mode
        keys.extend(("graph_mode", "graph_mode_widget"))
    profile = dict(payload.get("ui_profile", {}))
    profile_state = dict(profile.get("state", {}))
    for key in keys:
        profile_state.pop(key, None)
    profile["state"] = profile_state
    return {
        **payload,
        "config": ({**payload.get("config", {}), "analysis_fft_size": int(fft)}
                   if fft is not None else payload.get("config", {})),
        "ui": ui,
        "ui_profile": profile,
    }
