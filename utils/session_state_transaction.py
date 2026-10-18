"""Restore application values without replaying Streamlit input events."""
from collections.abc import Mapping, MutableMapping
from typing import Any


_TRANSIENT_VALUE_TYPES = frozenset({
    # Streamlit clears the wire type for empty widget values on rerun.
    None,
    "trigger_value", "string_trigger_value", "json_trigger_value", "chat_input_value",
    "file_uploader_state_value",
})


def transient_widget_keys(widget_metadata) -> set[str]:
    """Identify event/upload ownership even when their current value is empty."""
    from streamlit.runtime.state.common import user_key_from_element_id

    return {
        key for widget in widget_metadata
        if widget.value_type in _TRANSIENT_VALUE_TYPES
        if (key := user_key_from_element_id(widget.id)) is not None
    }


def snapshot_session_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Capture restorable values before a settings/context transaction starts.

    Streamlit owns trigger and upload values. Excluding them at capture time
    also protects nested rollback paths when a reset removes their bindings.
    Keep the existing shallow snapshot contract for application-owned objects.
    """
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.runtime.state import get_session_state

    # Isolate the Streamlit metadata dependency here. Serialized widget values
    # can omit empty uploads, so they cannot define transaction ownership.
    events = (transient_widget_keys(
        get_session_state()._state._new_widget_state.widget_metadata.values())
        if get_script_run_ctx(suppress_warning=True) is not None else set())
    return {key: state[key] for key in state if key not in events}


def rollback_session_state(state: MutableMapping[str, Any], before: Mapping[str, Any]) -> None:
    """Release new/event state, then restore changed application values."""
    for key in list(state):
        if key not in before:
            del state[key]
    for key, value in before.items():
        if key not in state or state[key] is not value:
            state[key] = value
