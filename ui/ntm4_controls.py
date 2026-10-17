"""Shared NTM4 control using the existing persistent numeric-input wrapper."""


def render_lobe_height(settings_key, *, default=-36.0, on_change=None, disabled=False):
    from utils.ui_index import number_input_stateful_safe
    from utils.ui_localization import display_text
    from crossover_engine.ntm4 import MIN_LOBE_DB, MAX_LOBE_DB
    return number_input_stateful_safe(
        display_text("Lobe Height (dB)"), settings_key=settings_key,
        min_value=MIN_LOBE_DB, max_value=MAX_LOBE_DB, default=default,
        step=0.1, format="%.1f", on_change=on_change, disabled=disabled,
    )
