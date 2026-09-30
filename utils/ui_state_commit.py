"""Commit scalar UI edits before a rerun can skip the originating control."""
from typing import Any, Callable


def _commit(settings_key, widget_key, transform, callback, args, kwargs):
    import streamlit as st

    if widget_key not in st.session_state:
        return
    value = st.session_state[widget_key]
    st.session_state[settings_key] = transform(value) if transform else value
    if callback is not None:
        callback(*(args or ()), **(kwargs or {}))


def bind_setting_commit(widget_args: dict[str, Any], settings_key: str,
                        transform: Callable[[Any], Any] | None = None) -> None:
    """Chain the caller callback after domain commit, preserving its arguments."""
    callback = widget_args.pop('on_change', None)
    args = widget_args.pop('args', None)
    kwargs = widget_args.pop('kwargs', None)
    widget_args['on_change'] = _commit
    widget_args['args'] = (settings_key, widget_args['key'], transform, callback, args, kwargs)
