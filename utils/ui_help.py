"""Shared hover-help timing, independent of DSP and project settings."""
import json
import math

from utils.ui_language import language_path, save_ui_preference, current_language

DEFAULT_DELAY_SECONDS = 2.5


def validate_delay(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 5:
        raise ValueError('Help delay must be between 0 and 5 seconds')
    return float(value)


def read_delay(path=None):
    path = path or language_path()
    payload = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(payload, dict):
        raise ValueError('UI preferences must be a JSON object')
    return validate_delay(payload.get('help_delay_seconds', DEFAULT_DELAY_SECONDS))


def save_delay(value, path=None):
    save_ui_preference('help_delay_seconds', validate_delay(value), path)


def help_css(delay):
    # Streamlit mounts the tooltip after its own short hover delay. Hiding the
    # content also removes it from hit testing while our shared delay elapses.
    seconds = validate_delay(delay)
    return f'''
:root {{ --phaseeq-help-delay: {seconds:g}s; }}
@keyframes phaseeq-help-reveal {{
  from {{ visibility: hidden; opacity: 0; pointer-events: none; }}
  to {{ visibility: visible; opacity: 1; pointer-events: auto; }}
}}
[data-testid="stTooltipContent"] {{
  animation: phaseeq-help-reveal 0s linear var(--phaseeq-help-delay) both !important;
}}
'''


def initialize_help():
    import streamlit as st
    try:
        delay = read_delay()
    except (OSError, ValueError) as exc:
        delay = DEFAULT_DELAY_SECONDS
        st.warning(f'ヘルプ設定を読み込めません / Cannot read help preferences: {exc}')
    st.markdown(f'<style>{help_css(delay)}</style>', unsafe_allow_html=True)


def render_help_setting(*, key):
    import streamlit as st
    japanese = current_language() != 'en'
    try:
        st.session_state[key] = read_delay()
    except (OSError, ValueError):
        st.session_state[key] = DEFAULT_DELAY_SECONDS

    def changed():
        try:
            save_delay(st.session_state[key])
        except (OSError, ValueError) as exc:
            st.session_state['_help_delay_save_error'] = str(exc)

    st.number_input('ヘルプ表示までの待ち時間（秒）' if japanese else 'Hover help delay (seconds)',
                    min_value=0.0, max_value=5.0, step=0.5, key=key, on_change=changed)
    st.caption('初期値は2.5秒。PhaseEQとマルチウェイで共通です。もう一方は次の画面更新で反映します。'
               if japanese else 'Default: 2.5 seconds. Shared by PhaseEQ and Multiway; the other app updates on its next rerun.')
    if error := st.session_state.pop('_help_delay_save_error', None):
        st.error(f'ヘルプ設定を保存できません / Cannot save help preferences: {error}')
