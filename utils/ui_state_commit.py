"""Commit scalar UI edits before a rerun can skip the originating control."""
from typing import Any, Callable


SETTING_WIDGET_PREFIXES = (
    '__num_val_', '__sel_val_', '__chk_val_', '__rad_val_',
    '__seg_val_', '__slider_val_', '__multi_val_', '__text_val_',
)


def project_setting_widget(state, settings_key, widget_key, default):
    """Project the committed setting; accept a mirror only for legacy initial state.

    Widget callbacks commit incoming edits before rendering. A surviving raw
    widget key can be older than that commit after a hidden-page cleanup.
    """
    from copy import deepcopy

    value = deepcopy(state.get(settings_key, state.get(widget_key, default)))
    state[settings_key] = value
    state[widget_key] = deepcopy(value)
    return value


def restore_setting(state, settings_key, widget_key, value):
    """Restore only the setting, invalidating the previous view projection."""
    state[settings_key] = value
    if widget_key != settings_key:
        state.pop(widget_key, None)


OUTPUT_POSTPROCESS_DEFAULTS = {
    'dc_gain_normalize': True,
    'output_fir_polarity_invert': False,
    'output_fir_cosine_taper_enabled': False,
    'fir_auto_gain_enabled': True,
}


def clear_input_context_for_restore(state) -> None:
    """Release input auxiliaries not reconstructed by DesignConfig.

    Invoke inside the caller's restore transaction, before projecting saved UI
    fields. Channel-owned input preservation remains the caller's responsibility.
    """
    for key in (
        '_restored_near_field_response_raw', '_restored_near_field_source_name',
        '_restored_port_response_raw', '_restored_port_source_name',
        '_speaker_calibration_state', '_speaker_auto_iir_metadata',
        '_current_speaker_timing_provenance', '_iir_auto_diagnostics',
        '_speaker_phase_center_search_start_ms', '_speaker_phase_center_search_end_ms',
        '_settings_asset_warning', '_response_asset_refs',
    ):
        state.pop(key, None)
    for source in ('speaker_measurement', 'mic_calibration', 'near_field_measurement', 'port_measurement'):
        for field in ('id', 'label'):
            state.pop(f'current_{source}_{field}', None)


def clear_restored_setting_widgets(state, setting_keys) -> None:
    """Discard source projections, including custom aliases and new settings.

    Restore boundaries never read an old widget back into destination settings.
    Rendered wrappers register aliases; conventional keys cover hidden controls.
    Unrelated application widgets and canonical values are left untouched.
    """
    keys = set(setting_keys)
    widgets = {prefix + key for key in keys for prefix in SETTING_WIDGET_PREFIXES}
    widgets.update(key + '_widget' for key in keys)
    for entry in state.get('__ui_index_registry', {}).values():
        if entry.get('settings_key') in keys:
            widget = entry.get('widget_key')
            if widget and widget not in keys:
                widgets.add(widget)
    for widget in widgets:
        state.pop(widget, None)


def sync_output_postprocess_state(state, *, inherited: bool) -> dict[str, bool]:
    """Project owner policy to both canonical settings and widget mirrors."""
    values = {key: False if inherited else bool(state.get(key, default))
              for key, default in OUTPUT_POSTPROCESS_DEFAULTS.items()}
    for key, value in values.items():
        state[key] = value
        state['__chk_val_' + key] = value
    state['_output_restore_snapshot'] = {
        'owner': 'assignment' if inherited else 'standalone',
        'settings': list(values),
        'consistent': all(state[key] == state['__chk_val_' + key] for key in values),
    }
    return values


def _commit(settings_key, widget_key, transform, callback, args, kwargs):
    import streamlit as st

    if widget_key not in st.session_state:
        return
    value = st.session_state[widget_key]
    st.session_state[settings_key] = transform(value) if transform else value
    if callback is not None:
        callback(*(args or ()), **(kwargs or {}))


def bind_setting_commit(widget_args: dict[str, Any], settings_key: str,
                        transform: Callable[[Any], Any] | None = None) -> None:
    """Chain the caller callback after domain commit, preserving its arguments."""
    callback = widget_args.pop('on_change', None)
    args = widget_args.pop('args', None)
    kwargs = widget_args.pop('kwargs', None)
    widget_args['on_change'] = _commit
    widget_args['args'] = (settings_key, widget_args['key'], transform, callback, args, kwargs)
