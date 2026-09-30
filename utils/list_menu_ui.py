"""Shared Streamlit adapters; computation and persistence remain caller-owned."""
from utils.ui_localization import ui_message, display_text
from list_menu_state import resolve_choice, candidate_slice
from list_menu_state import list_policy


def single_row_table_selector(
    data,
    option_ids: list[str],
    *,
    selected_state_key: str,
    table_key: str,
    height: int,
    default_id: str = "",
    column_config: dict | None = None,
    required: bool = True,
) -> str:
    """Render one persistent checkmark-style row selector backed by session state."""
    import streamlit as st
    if not option_ids or len(data.index) != len(option_ids):
        return ""
    if len(option_ids) > list_policy(table_key).candidate_limit:
        indices = candidate_page(list(range(len(option_ids))), list_id=table_key)
        data = data.iloc[indices]
        option_ids = [option_ids[i] for i in indices]
    selected_id = str(st.session_state.get(selected_state_key, ""))
    preferred_id = str(default_id or "")
    if selected_id not in option_ids:
        selected_id = (preferred_id if preferred_id in option_ids else option_ids[0]) if required else ""
        st.session_state[selected_state_key] = selected_id

    nonce_key = f"{table_key}_selection_nonce"
    rendered_id_key = f"{table_key}_rendered_selected_id"
    nonce = int(st.session_state.get(nonce_key, 0) or 0)
    rendered_id = str(st.session_state.get(rendered_id_key, ""))
    ids_key = f"{table_key}_rendered_option_ids"
    previous_ids = st.session_state.get(ids_key)
    if (rendered_id and rendered_id != selected_id) or (previous_ids is not None and previous_ids != option_ids):
        nonce += 1
        st.session_state[nonce_key] = nonce
    st.session_state[rendered_id_key] = selected_id
    st.session_state[ids_key] = list(option_ids)

    default_rows = [option_ids.index(selected_id)] if selected_id in option_ids else []
    event = st.dataframe(
        data,
        hide_index=True,
        width="stretch",
        height=height,
        key=f"{table_key}_selection_{nonce}",
        on_select="rerun",
        selection_mode="single-row",
        selection_default={"selection": {"rows": default_rows}},
        column_config=column_config,
    )
    selected_rows = list(event.selection.rows)
    if selected_rows and 0 <= int(selected_rows[0]) < len(option_ids):
        selected_id = option_ids[int(selected_rows[0])]
        st.session_state[selected_state_key] = selected_id
        st.session_state[rendered_id_key] = selected_id
    elif not selected_rows:
        if required:
            st.session_state[nonce_key] = nonce + 1
            st.rerun()
        selected_id = ""
        st.session_state[selected_state_key] = ""
        st.session_state[rendered_id_key] = ""
    return selected_id


def choice_widget(label, options, *, key, current=None, default=None,
                  kind='selectbox', renderer=None, **kwargs):
    import streamlit as st
    options = list(options)
    if not options:
        return None
    if default is None and kwargs.get('index') is not None:
        index = kwargs['index']
        if isinstance(index, int) and 0 <= index < len(options):
            default = options[index]
    required = kwargs.get('required', True)
    incoming = st.session_state.get(key, current)
    value = resolve_choice(options, incoming, default, required=required)
    # A reset must be sent to the browser as a value change. Deleting the
    # widget state and changing only its default leaves the old selection visible.
    st.session_state[key] = value
    if kind == 'segmented_control':
        kwargs['required'] = required
        kwargs.pop('default', None)
    else:
        kwargs.pop('required', None)
        # Session State owns the selection; avoid a second, conflicting seed.
        kwargs['index'] = 0 if value in options else None
    return (renderer or getattr(st, kind))(display_text(label), options, key=key, **kwargs)


def selectbox(label, options, **kwargs):
    import streamlit as st
    options = list(options)
    list_id = kwargs['key']
    limit = list_policy(list_id).candidate_limit
    if len(options) > limit:
        selected = st.session_state.get(list_id)
        remaining = [item for item in options if item != selected]
        page = list(candidate_page(remaining, list_id=list_id, page_size=max(1, limit - 1)))
        options = ([selected] if selected in options else []) + page
        # Index belongs to the original candidates, not this page.
        kwargs.pop('index', None)
    if 'index' in kwargs and kwargs['index'] is None:
        kwargs.setdefault('required', False)
    return choice_widget(label, options, **kwargs)


def radio(label, options, **kwargs):
    return choice_widget(label, options, kind='radio', **kwargs)


def segmented_control(label, options, **kwargs):
    if kwargs.get('selection_mode') == 'multi':
        import streamlit as st
        return st.segmented_control(label, options, **kwargs)
    return choice_widget(label, options, kind='segmented_control', **kwargs)


def candidate_page(records, *, list_id, page_size=None):
    import streamlit as st
    _, _, pages = candidate_slice(records, list_id, page_size=page_size)
    page = 0
    if pages > 1:
        st.caption(ui_message('ui.a6bce8fca03c89', p0=f'{len(records):,}', p1=f'{pages}'))
        page = int(choice_widget(ui_message('ui.3d9273e24e69f8'), list(range(pages)), key=f'{list_id}_candidate_page',
            current=0, default=0, format_func=lambda p: f'{p + 1} / {pages}'))
    return candidate_slice(records, list_id, page, page_size=page_size)[0]


def database_candidates(loader, *args, list_id, **kwargs):
    """Fetch a bounded SQL page plus one overflow probe, never a whole DB."""
    import streamlit as st
    limit = list_policy(list_id).candidate_limit
    kwargs.pop('limit', None)
    signature = repr((args, kwargs))
    prefix = f'{list_id}_db_candidates'
    if st.session_state.get(prefix + '_query') != signature:
        st.session_state[prefix + '_query'] = signature
        st.session_state[prefix + '_page'] = 0
    page = int(st.session_state.get(prefix + '_page', 0))
    records = loader(*args, **kwargs, limit=limit + 1, offset=page * limit)
    if page or len(records) > limit:
        st.caption(ui_message('ui.b8321e01ad40ca', p0=f'{page * limit + 1:,}', p1=f'{limit:,}'))
        with st.container(horizontal=True):
            if st.button(ui_message('ui.3e6e0b01492726'), key=prefix + '_previous', disabled=page == 0):
                st.session_state[prefix + '_page'] = page - 1
                st.rerun()
            if st.button(ui_message('ui.2623e93a76cc01'), key=prefix + '_next', disabled=len(records) <= limit):
                st.session_state[prefix + '_page'] = page + 1
                st.rerun()
    return records[:limit]
