"""Browser liveness and launch controls, independent of audio processing."""
import uuid
import hashlib

import streamlit as st
from utils.ui_localization import display_text

from utils.browser_presence import browser_is_open, publish_presence


def ensure_browser_presence(root, app):
    from streamlit.runtime import exists, get_instance
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from utils.browser_presence import start_browser_presence
    context = get_script_run_ctx()
    engine = get_instance() if exists() else None
    if context is not None and callable(getattr(engine, "is_active_session", None)):
        session_id = context.session_id
        if app == "phaseeq":
            st.session_state["_phaseeq_browser_session_id"] = session_id
        start_browser_presence(root, app, session_id, lambda: engine.is_active_session(session_id))
    else:
        session_id = st.session_state.setdefault(f"_presence_{app}", uuid.uuid4().hex)
        publish_presence(root, app, session_id)


def refresh_peer_presence(root, peer):
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    key = f"_peer_browser_open_{peer}"
    opened = browser_is_open(root, peer)
    previous = st.session_state.get(key)
    st.session_state[key] = opened
    context = get_script_run_ctx()
    # A full run already redraws the controls; restarting it would lose clicks.
    if previous is not None and previous != opened and context is not None and context.fragment_ids_this_run:
        st.rerun()


@st.fragment(run_every=2.0)
def browser_presence(root, app):
    ensure_browser_presence(root, app)
    refresh_peer_presence(root, "multiway" if app == "phaseeq" else "phaseeq")


def launch_button(root, app, label, url, *, disabled=False):
    opened = browser_is_open(root, app)
    st.link_button(
        display_text(label), url,
        key="launch_" + hashlib.sha256(f"{app}:{label}:{url}".encode()).hexdigest()[:16],
        disabled=disabled or opened, width="stretch", icon=":material/open_in_new:",
        help="既存のタブで操作してください。画面を閉じると約15秒後に再び開けます。" if opened else None,
    )
