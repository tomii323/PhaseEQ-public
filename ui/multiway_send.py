"""Visible entry point to the existing Export publication workflow."""
from utils.ui_localization import ui_message
from collections.abc import Callable

import streamlit as st


def render_multiway_send_shortcut(assignment, navigate: Callable[..., None], *, compact: bool = False,
                                 blocked: bool = False) -> bool:
    if assignment is None:
        return False
    with st.container(border=not compact, gap="small"):
        if not compact:
            st.caption(ui_message('ui.23cf2335d0fe05', p0=f'{assignment.group}', p1=f'{assignment.channel_name}', p2=f'{assignment.way}'))
        return st.button(
            ui_message('ui.d219fe0c837005'),
            key=f"multiway_send_top_{assignment.assignment_id}",
            type="primary",
            icon=":material/send:",
            width="stretch",
            disabled=blocked,
            help=ui_message('ui.6c9587b866066b'),
            on_click=navigate,
            args=("Export",),
            kwargs={"export_edit_mode": "Downloads", "export_edit_mode_widget": "Downloads"},
        )
