"""Project Channel settings to owner-bound widgets; commit explicit edits only."""
import hashlib

from utils.ui_state_commit import project_setting_widget
from utils.ui_index import register_ui_entry


def commit_channel_field(state, owner, setting_key, widget_key):
    if state.get('_multiway_channel_editor_owner') == owner and widget_key in state:
        state[setting_key] = state[widget_key]


def channel_field(renderer, label, *args, key, **kwargs):
    import streamlit as st
    owner = st.session_state['_multiway_channel_editor_owner']
    token = hashlib.sha256(repr(owner).encode()).hexdigest()[:16]
    widget_key = f'__channel_edit_{token}_{key}'
    project_setting_widget(st.session_state, key, widget_key, st.session_state[key])
    renderer(label, *args, key=widget_key,
        on_change=commit_channel_field, args=(st.session_state, owner, key, widget_key), **kwargs)
    register_ui_entry(kind='channel_field', label=label, settings_key=key,
                      widget_key=widget_key)
    # Render return values never write into the Channel model. This also keeps
    # an intentionally empty edit until validation reports the missing field.
    return st.session_state[key]
