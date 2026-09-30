"""Navigation policy for channels whose DSP has no FIR output."""


def available_page(page, fir_enabled):
    return 'IIR EQ' if page == 'FIR EQ' and not fir_enabled else page


def render_fir_output_toggle(enabled, *, inherited, on_change):
    """Mirror inherited capability without overwriting the standalone setting."""
    import streamlit as st
    widget_key = 'standalone_fir_enabled_widget'
    st.session_state[widget_key] = bool(enabled)

    def commit():
        st.session_state['standalone_fir_enabled'] = bool(st.session_state[widget_key])
        on_change()

    return st.toggle('FIR ON / OFF', key=widget_key, disabled=inherited, on_change=commit)


def render_iir_only_pages(pages, labels, current, navigate, *, disabled=False):
    """Use native disabled buttons: individual segmented options cannot disable."""
    import streamlit as st
    from utils.ui_localization import user_ui_label
    for column, page in zip(st.columns(len(pages), gap='small'), pages):
        column.button(user_ui_label(labels.get(page, page)), key=f'iir_only_page_{page}',
                      width='stretch', type='primary' if page == current else 'secondary',
                      disabled=disabled or page == 'FIR EQ', on_click=navigate, args=(page,))
    return current
