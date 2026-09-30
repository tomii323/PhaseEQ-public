"""Refresh an open PhaseEQ editor when Multiway publishes revised conditions."""
from utils.ui_localization import ui_message
from pathlib import Path
import streamlit as st

from utils.composite_exchange import read_current_composite_assignment


def refresh_assignment_if_updated(root: Path, assignment, *, target_edit=False):
    if assignment is None or target_edit:
        return
    snapshot = st.session_state.get('_composite_context_payload')
    if snapshot is None or st.session_state.get('_pending_composite_context') is not None:
        return
    try:
        latest = read_current_composite_assignment(root, assignment.assignment_id)
        if latest is None or latest.assignment_id == assignment.assignment_id:
            return
        attempt = (assignment.assignment_id, latest.assignment_id)
        if st.session_state.get('_composite_auto_refresh_attempt') == attempt:
            return
        # Materialize the current editor before rerunning. The existing context
        # transition saves it as a Draft and overlays the destination conditions.
        payload = snapshot() if callable(snapshot) else snapshot
    except (OSError, ValueError) as exc:
        st.warning(ui_message('ui.df70c1930b2a13', p0=f'{exc}'))
        return
    st.session_state['_composite_auto_refresh_attempt'] = attempt
    st.session_state['_pending_composite_context'] = {
        'assignment_id': latest.assignment_id, 'payload': payload,
    }
    st.rerun()
