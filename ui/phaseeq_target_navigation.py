"""Open Target management without creating another tab when PhaseEQ is live."""
from utils.ui_localization import ui_message
import os
import streamlit as st
from utils.browser_presence import browser_is_open
from utils.phaseeq_target_navigation import (
    request_target_menu, target_menu_request, acknowledge_target_menu,
    target_menu_acknowledged,
)


@st.fragment(run_every=2.0)
def _target_navigation_status(root):
    request = st.session_state.get('_sent_target_menu_request')
    if request:
        if target_menu_acknowledged(root, request):
            st.success(ui_message('ui.8196ed4b581ce8'))
        else:
            st.info(ui_message('ui.9329a1f4fa1c06'))


def render_target_menu_button(root):
    st.caption(ui_message('ui.a5a79604da4aa2'))
    if browser_is_open(root, 'phaseeq'):
        st.caption(ui_message('ui.4003761327c5ba'))
        if st.button(ui_message('ui.a85fb6496b7f2d'), key='open_phaseeq_target_menu', width='stretch'):
            request = request_target_menu(root)
            if request:
                st.session_state['_sent_target_menu_request'] = request
            else:
                st.info(ui_message('ui.0e6e12a5d9628f'))
        _target_navigation_status(root)
    else:
        url = os.environ.get('PHASEEQ_URL', 'http://localhost:8501')
        st.link_button(ui_message('ui.a85fb6496b7f2d'), url.rstrip('/')+'/?page=Target', width='stretch')


def receive_target_menu_request(root):
    session = st.session_state.get('_phaseeq_browser_session_id', '')
    request = target_menu_request(root, session)
    if request and request != st.session_state.get('_target_menu_request_seen'):
        st.session_state['_target_menu_request_seen'] = request
        st.session_state['_open_registered_target_menu'] = True
        st.rerun()
    elif (request and not st.session_state.get('_open_registered_target_menu')
          and st.session_state.get('active_page') == 'Target'
          and st.session_state.get('_active_context_assignment') is None
          and not st.query_params.get('target_edit')):
        try:
            acknowledge_target_menu(root, session, request)
            st.session_state.pop('_target_menu_ack_error', None)
        except OSError as exc:
            st.session_state['_target_menu_ack_error'] = str(exc)
            st.warning('Target画面の受信確認を送れません。次の監視で再試行します：' + str(exc))
