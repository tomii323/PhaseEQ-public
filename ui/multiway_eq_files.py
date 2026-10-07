"""Optional Multiway EQ file controls; persistence is owned by the caller."""
import streamlit as st

from utils.ui_localization import ui_message


def render_eq_files(bands, files, *, enabled, save_upload, persist):
    """Keep registered files visible and immutable while FIR is disabled."""
    with st.expander(ui_message('ui.bb5eeafcfdda50'), expanded=False):
        st.caption(ui_message('ui.10f314c79ba4c8'))
        for band in bands:
            st.markdown(ui_message('ui.50dbb2418f20a2', p0=band))
            current = files[band]
            uploaded = [name for name in current if name]
            if uploaded:
                st.caption(' → '.join(uploaded))
            for index, name in enumerate(current):
                if name:
                    continue
                uploaded_file = st.file_uploader(
                    ui_message('ui.ebb5e8b13899b0', p0=band, p1=str(index + 1)),
                    type=['bin', 'wav', 'txt', 'csv'], key=f'{band}_eq{index+1}',
                    label_visibility='collapsed', disabled=not enabled,
                )
                if enabled and uploaded_file:
                    try:
                        _, stored_name = save_upload(band, index, uploaded_file)
                        current[index] = stored_name
                        persist(files)
                        st.rerun()
                    except (ValueError, OSError) as exc:
                        st.error(ui_message('ui.ef3d2697f82aad', p0=str(exc)))
                break
            for index, name in enumerate(current):
                if name and st.button(
                    ui_message('ui.4bb4e0b09f6a70', p0=name),
                    key=f'{band}_eq{index+1}_clear',
                    help=ui_message('ui.de767b573ad94f'), type='secondary',
                    disabled=not enabled,
                ) and enabled:
                    current[index] = ''
                    persist(files)
                    st.rerun()
