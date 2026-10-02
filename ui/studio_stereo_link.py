"""Channel Link control beside Multiway's channel handoff controls."""
from __future__ import annotations

from utils.ui_localization import ui_message

import io

from utils.composite_exchange import latest_published_assignment_working_session
from utils.eq_links import transaction
from utils.export_bundle import config_payload_from_project_zip
from ui.stereo_link import render


def render_studio_link(state, root, system_id, sample_rate, rows):
    import streamlit as st

    if len(rows) < 2:
        return
    scope = f"{system_id}:{sample_rate}"
    options = {key: f"{row['group']} / {row['name']} ({row['way']})" for key, row in rows.items()}
    channel = st.selectbox(ui_message("ui.9267e2eb2d3ca4"), list(options), format_func=options.get,
                                 key=f"_stereo_studio_channel_{scope}")

    def load(channel_id):
        from phase_fir_designer import DesignConfig
        from utils.settings_io import build_config_payload
        previous = latest_published_assignment_working_session(
            root, channel_id=channel_id, system_id=system_id, exclude_assignment_id="")
        if previous is not None:
            return config_payload_from_project_zip(io.BytesIO(previous[1].read_bytes()))
        return build_config_payload(app_name="PhaseEQ", app_version="", app_author="",
                                    schema_version=3, config=DesignConfig(sample_rate, 511),
                                    ui_payload={}, ui_profile={})

    payload = load(channel)
    with transaction(root, scope) as model:
        model.ensure(channel, payload)
        state["_stereo_live_payload"] = model.effective(channel, payload)
    state["_stereo_context"] = (scope, channel)
    state["_stereo_options"] = options
    render(state, root, lambda payload: None, load)
