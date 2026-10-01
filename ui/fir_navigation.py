"""Navigation policy for channels whose DSP has no FIR output."""


DESIGN_PAGES = ('Input', 'Target', 'IIR EQ', 'FIR EQ', 'Linear FIR', 'Export')


def nearest_available_page(page, pages, unavailable=()):
    """Keep the current page, otherwise search left in the displayed menu order."""
    available = [candidate for candidate in pages if candidate not in unavailable]
    if page in available:
        return page
    if page in pages:
        for candidate in reversed(pages[:pages.index(page)]):
            if candidate in available:
                return candidate
    return available[0] if available else None


def available_page(page, fir_enabled):
    if page not in DESIGN_PAGES:
        return page
    return nearest_available_page(page, DESIGN_PAGES, () if fir_enabled else ('FIR EQ',))


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
