"""Controls for the display-only reverse-null overlay."""
import streamlit as st

from composite_engine.display_projection import DisplayProjection
from composite_engine.display_projection.reverse_null import reverse_null_available
from utils.ui_localization import ui_message


def render_reverse_null_control(projection: DisplayProjection, graph_kind: str) -> bool:
    available = reverse_null_available(projection, graph_kind)
    requested = st.toggle(
        ui_message('ui.2b6d55b86615fa'),
        key="composite_studio_reverse_null_widget",
        disabled=not available,
        help=ui_message('ui.78992825ae50c0'),
    )
    render_reverse_null_guide()
    return available and requested


def render_reverse_null_guide() -> None:
    """Render shared instructions without reading or updating design state."""
    with st.expander(ui_message('ui.036d031dcb1e55'), expanded=False):
        st.markdown(ui_message('ui.2482b9e6a0865c'))
