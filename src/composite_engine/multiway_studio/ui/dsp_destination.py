"""Project saved destination settings; commit edits only through callbacks."""
import streamlit as st

from composite_engine.dsp_export.destinations import available_destinations, normalize_destination, selected_destination
from composite_engine.dsp_export.setup_policy import plan_setup, apply_setup
from utils.list_menu_ui import selectbox
from utils.ui_language import current_language
from utils.ui_localization import ui_message


DESTINATION_KEY = 'composite_studio_dsp_export_adapter'


def _commit_destination(widget_key, destination_id=None):
    settings = dict(st.session_state['settings'])
    selection = normalize_destination(settings.get('dsp_export'))
    if destination_id is None:
        selection['destination_id'] = st.session_state[widget_key]
    else:
        selection['profile_ids'][destination_id] = st.session_state[widget_key]
    settings['dsp_export'] = normalize_destination(selection)
    st.session_state['settings'] = settings


def _apply_destination_setup(profile):
    settings = st.session_state['settings']
    mode = settings.get('last_mode', '2Way')
    changes = plan_setup(settings, mode, profile)
    if changes:
        st.session_state['_pending_uploaded_settings'] = apply_setup(settings, changes)
        st.session_state['_pending_uploaded_name'] = profile.display_name
        st.session_state['_pending_uploaded_kind'] = 'dsp_setup'


def render_dsp_destination():
    english = current_language() == 'en'
    selection = normalize_destination(st.session_state['settings'].get('dsp_export'))
    destinations = available_destinations()
    labels = {item.destination_id: item.display_name for item in destinations}
    # Saved settings are the only restoration source, including when old widgets survive.
    st.session_state[DESTINATION_KEY] = selection['destination_id']
    selectbox(ui_message('ui.5928e6ccdd119b'), tuple(labels), format_func=labels.get,
              key=DESTINATION_KEY, on_change=_commit_destination, args=(DESTINATION_KEY,))
    destination, profile = selected_destination(selection)
    profile_key = f'composite_studio_dsp_export_profile_{destination.destination_id}'
    st.session_state[profile_key] = profile.profile_id
    profile_labels = {item.profile_id: item.display_name for item in destination.profiles}
    selectbox(ui_message('ui.1e80a85cbd7c37'), tuple(profile_labels), format_func=profile_labels.get,
              key=profile_key, on_change=_commit_destination, args=(profile_key, destination.destination_id))
    settings = st.session_state['settings']
    changes = plan_setup(settings, settings.get('last_mode', '2Way'), profile)
    def value_text(value):
        if isinstance(value, bool):
            return 'ON' if value else 'OFF'
        if isinstance(value, list) and all(isinstance(item, bool) for item in value):
            return ', '.join('ON' if item else 'OFF' for item in value) or '—'
        return str(value)
    if changes:
        st.caption('Changes in Multiway only:' if english else 'マルチウェイ側の変更内容：')
        st.markdown('\n'.join(f'- {change.label_en if english else change.label_ja}: '
                              f'{value_text(change.before)} → **{value_text(change.after)}**' for change in changes))
    st.button('Apply settings for this DSP' if english else 'このDSP用の設定へ変更',
              key='composite_studio_apply_dsp_setup', disabled=not changes,
              on_click=_apply_destination_setup, args=(profile,))
    st.caption('Only the listed Multiway settings are changed. Channel EQ and Working Sessions are retained.' if english else
               '表示したマルチウェイ設定だけを変更します。PhaseEQのEQ編集値・Working Sessionは保持します。')
    return destination.adapter, profile
