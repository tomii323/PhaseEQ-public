"""Apply explicit Multiway channel inputs before PhaseEQ input widgets render."""
import sqlite3
import streamlit as st
from utils.multiway_measurements import assignment_measurement_input, measurement_input_record, input_is_empty
from utils.ui_localization import display_text
from utils.session_state_transaction import snapshot_session_state, rollback_session_state


def _token(assignment, item):
    import hashlib
    import json
    generation = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
    return (assignment.assignment_id, assignment.system_id, assignment.channel_id, item['request_id'], generation)


def sync_measurement_input(root, db_path, assignment, apply_record, *, target_edit=False, clear_input=None):
    if assignment is None or target_edit:
        st.session_state.pop('_channel_measurement_input_seen', None)
        st.session_state.pop('_channel_measurement_input_request_id', None)
        return
    try:
        item = assignment_measurement_input(root, assignment)
        if item is None:
            st.session_state.pop('_channel_measurement_input_seen', None)
            st.session_state.pop('_channel_measurement_input_request_id', None)
            return
        token = _token(assignment, item)
        if st.session_state.get('_channel_measurement_input_seen') == token:
            return
        before = snapshot_session_state(st.session_state)
        try:
            if input_is_empty(item):
                if clear_input is None:
                    raise ValueError('空入力を適用する処理がありません。')
                ok, message = clear_input()
            else:
                ok, message = apply_record(measurement_input_record(item, db_path))
            if not ok:
                raise ValueError(message)
        except Exception:
            rollback_session_state(st.session_state, before)
            raise
        st.session_state['_channel_measurement_input_seen'] = token
        st.session_state['_channel_measurement_input_request_id'] = item['request_id']
        st.session_state.pop('_channel_measurement_input_failed', None)
        st.session_state['_channel_measurement_input_notice'] = item['name']
    except (OSError, ValueError, sqlite3.Error) as exc:
        if 'token' in locals():
            st.session_state['_channel_measurement_input_failed'] = token
        st.warning(display_text('チャンネルの測定データを入力へ適用できませんでした：') + str(exc))


def refresh_measurement_input_if_updated(root, assignment, *, target_edit=False):
    if assignment is None or target_edit:
        return
    try:
        item = assignment_measurement_input(root, assignment)
        if item is None:
            return
        token = _token(assignment, item)
        if token not in (st.session_state.get('_channel_measurement_input_seen'),
                         st.session_state.get('_channel_measurement_input_failed')):
            st.rerun()
    except (OSError, ValueError) as exc:
        st.warning(display_text('チャンネルの測定データを確認できませんでした：') + str(exc))
