"""Ownership of measurement sources while editing a Multiway channel."""
from contextlib import contextmanager

CHANNEL_INPUT_HELP = 'マルチウェイのチャンネル編集中です。入力の変更は「スピーカー測定データ → マルチウェイに割り当て」で行ってください。'


def channel_input_locked(state):
    return state.get('_active_context_assignment') is not None


def _source_key(key):
    return str(key).startswith(('_restored_speaker_', '_restored_mic_', '_restored_near_field_',
        '_restored_port_', '_speaker_input_', 'current_speaker_measurement_', 'current_mic_calibration_',
        'current_near_field_measurement_', 'current_port_measurement_')) or key in {
            '_speaker_calibration_state', '_current_speaker_timing_provenance', '_speaker_auto_iir_metadata',
            'speaker_url', 'apply_mic_cal', '_speaker_source_name', '_speaker_phase_center_search_start_ms',
            '_speaker_phase_center_search_end_ms'}


@contextmanager
def preserve_channel_input(state):
    """Settings/EQ restore may not replace channel-owned measurement sources."""
    if not channel_input_locked(state):
        yield
        return
    before = {key: value for key, value in state.items() if _source_key(key)}
    refs = dict(state.get('_response_asset_refs') or {})
    try:
        yield
    finally:
        for key in list(state):
            if _source_key(key) and key not in before:
                state.pop(key, None)
        for key, value in before.items():
            state[key] = value
        current_refs = dict(state.get('_response_asset_refs') or {})
        for key in ('speaker_response_raw', 'mic_cal_response_raw'):
            if key in refs:
                current_refs[key] = refs[key]
            else:
                current_refs.pop(key, None)
        state['_response_asset_refs'] = current_refs
