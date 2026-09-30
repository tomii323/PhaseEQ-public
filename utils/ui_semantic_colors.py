"""Shared semantic color tokens for PhaseEQ and Multiway Streamlit UIs."""

from __future__ import annotations


SEMANTIC_ACTIVE_COLORS = {
    "gain": "#244B6B",
    "phase": "#503B65",
    "target": "#855626",
    "apply": "#3E6B59",
    "warning": "#806A38",
    "danger": "#984B45",
    "fir": "#6E583F",
    "export": "#505966",
}

SEMANTIC_ACCENT_COLORS = {
    "light": {
        "gain": "#365E80",
        "phase": "#68547C",
        "target": "#98652F",
        "apply": "#3E6B59",
        "warning": "#806A38",
        "danger": "#984B45",
        "fir": "#7A6245",
        "export": "#505966",
    },
    "dark": {
        "gain": "#7395B3",
        "phase": "#A997BC",
        "target": "#C69A67",
        "apply": "#79A991",
        "warning": "#BBA267",
        "danger": "#C9827B",
        "fir": "#B99B72",
        "export": "#A3ACB8",
    },
}


def semantic_chart_colors(theme_mode: str) -> dict[str, object]:
    """Return the shared chart palette used by PhaseEQ and Multiway.

    Entity colors remain stable between Gain and Phase views.  Line styles carry
    the second distinction, so the plots remain usable without color.
    """
    mode = "dark" if str(theme_mode).lower() == "dark" else "light"
    accent = SEMANTIC_ACCENT_COLORS[mode]
    if mode == "dark":
        return {
            "figure": "#0F1416",
            "axes": "#151C1E",
            "text": "#E8ECE9",
            "grid": "#3B494C",
            "spine": "#7F8E91",
            "sum": "#F2F3EF",
            "center": "#B39A64",
            "input": accent["gain"],
            "target": accent["target"],
            "result": accent["apply"],
            "correction": accent["phase"],
            "calibration": "#69A4A8",
            "delta": "#C2A46F",
            "gain_error": accent["danger"],
            "phase_error": accent["warning"],
            "baffle": accent["fir"],
            "bands": (
                accent["gain"], accent["apply"], accent["danger"],
                accent["phase"], "#69A4A8",
            ),
        }
    return {
        "figure": "#F2F3F1",
        "axes": "#FAFAF8",
        "text": "#182126",
        "grid": "#C3CAC7",
        "spine": "#68757A",
        "sum": "#182126",
        "center": "#8A7650",
        "input": accent["gain"],
        "target": accent["target"],
        "result": accent["apply"],
        "correction": accent["phase"],
        "calibration": "#2A7378",
        "delta": "#806943",
        "gain_error": accent["danger"],
        "phase_error": accent["warning"],
        "baffle": accent["fir"],
        "bands": (
            accent["gain"], accent["apply"], accent["danger"],
            accent["phase"], "#2A7378",
        ),
    }


def semantic_button_css(theme_mode: str) -> str:
    """Return shared accessible button styling for both Streamlit applications."""
    mode = "dark" if str(theme_mode).lower() == "dark" else "light"
    accent = SEMANTIC_ACCENT_COLORS[mode]
    panel = "#1B1F25" if mode == "dark" else "#FFFDF9"
    border = "#454B55" if mode == "dark" else "#D2C8BA"
    return f"""
    <style>
    :root {{
        --ui-gain: {accent['gain']}; --ui-gain-active: {SEMANTIC_ACTIVE_COLORS['gain']};
        --ui-phase: {accent['phase']}; --ui-phase-active: {SEMANTIC_ACTIVE_COLORS['phase']};
        --ui-target: {accent['target']}; --ui-target-active: {SEMANTIC_ACTIVE_COLORS['target']};
        --ui-apply: {accent['apply']}; --ui-apply-active: {SEMANTIC_ACTIVE_COLORS['apply']};
        --ui-warning: {accent['warning']}; --ui-warning-active: {SEMANTIC_ACTIVE_COLORS['warning']};
        --ui-danger: {accent['danger']}; --ui-danger-active: {SEMANTIC_ACTIVE_COLORS['danger']};
        --ui-fir: {accent['fir']}; --ui-fir-active: {SEMANTIC_ACTIVE_COLORS['fir']};
        --ui-export: {accent['export']}; --ui-export-active: {SEMANTIC_ACTIVE_COLORS['export']};
        --ui-semantic-panel: {panel}; --ui-semantic-border: {border};
    }}
    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {{
        background: var(--ui-apply-active) !important;
        border-color: var(--ui-apply) !important;
        color: #fff !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button[kind="primary"]:hover {{
        background: color-mix(in srgb, var(--ui-apply-active) 88%, #000) !important;
        border-color: var(--ui-apply) !important;
    }}
    div[class*="st-key-"][class*="gain"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) {{
        background: var(--ui-gain-active) !important; border-color: var(--ui-gain) !important; color: #fff !important;
    }}
    div[class*="st-key-"][class*="phase"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) {{
        background: var(--ui-phase-active) !important; border-color: var(--ui-phase) !important; color: #fff !important;
    }}
    div[class*="st-key-"][class*="target_source"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) {{
        background: var(--ui-target-active) !important; border-color: var(--ui-target) !important; color: #fff !important;
    }}
    div[class*="st-key-"][class*="crossover_method"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) {{
        background: var(--ui-fir-active) !important; border-color: var(--ui-fir) !important; color: #fff !important;
    }}
    :is(.st-key-target_edit_mode_widget, .st-key-iir_eq_view_widget, .st-key-fir_eq_view_widget)
    [data-testid="stButtonGroup"] button:nth-of-type(1):is([data-selected="true"], [aria-checked="true"]) {{
        background: var(--ui-gain-active) !important; border-color: var(--ui-gain) !important; color: #fff !important;
    }}
    :is(.st-key-target_edit_mode_widget, .st-key-iir_eq_view_widget, .st-key-fir_eq_view_widget)
    [data-testid="stButtonGroup"] button:nth-of-type(2):is([data-selected="true"], [aria-checked="true"]) {{
        background: var(--ui-phase-active) !important; border-color: var(--ui-phase) !important; color: #fff !important;
    }}
    div[class*="st-key-"][class*="gain"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) *,
    div[class*="st-key-"][class*="phase"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) *,
    div[class*="st-key-"][class*="target_source"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) *,
    div[class*="st-key-"][class*="crossover_method"] [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) * {{ color: #fff !important; font-weight: 760 !important; }}
    :is(.st-key-target_edit_mode_widget, .st-key-iir_eq_view_widget, .st-key-fir_eq_view_widget)
    [data-testid="stButtonGroup"] button:is([data-selected="true"], [aria-checked="true"]) * {{ color: #fff !important; font-weight: 760 !important; }}
    div[class*="st-key-"]:is([class*="apply"], [class*="save"], [class*="start"]) > .stButton button:not([kind="primary"]) {{
        border-color: var(--ui-apply) !important; color: var(--ui-apply) !important;
    }}
    div[class*="st-key-"]:is([class*="download"], [class*="export"], [class*="prepare"]) :is(.stButton, .stDownloadButton) button:not([kind="primary"]) {{
        border-color: var(--ui-export) !important; color: var(--ui-export) !important;
    }}
    div[class*="st-key-"]:is([class*="clear"], [class*="reset"], [class*="restore"], [class*="archive"], [class*="stop"]) > .stButton button:not([kind="primary"]) {{
        border-color: var(--ui-warning) !important; color: var(--ui-warning) !important;
    }}
    div[class*="st-key-"]:is([class*="delete"], [class*="abort"]) > .stButton button:not([kind="primary"]) {{
        border-color: var(--ui-danger) !important; color: var(--ui-danger) !important;
        background: color-mix(in srgb, var(--ui-danger) 9%, var(--ui-semantic-panel)) !important;
    }}
    :is(.stButton, .stDownloadButton) button:focus-visible,
    [data-testid="stButtonGroup"] button:focus-visible {{
        outline: 3px solid color-mix(in srgb, currentColor 58%, #fff) !important;
        outline-offset: 2px !important;
    }}
    a.ui-workspace-link {{
        align-items: center; background: var(--ui-apply-active); border: 1px solid var(--ui-apply);
        border-radius: 0.5rem; color: #fff !important; display: flex; font-weight: 700;
        justify-content: center; min-height: 2.5rem; padding: 0.45rem 0.8rem;
        text-decoration: none !important; width: 100%;
    }}
    a.ui-workspace-link:hover {{
        background: color-mix(in srgb, var(--ui-apply-active) 88%, #000);
        color: #fff !important;
    }}
    </style>
    """
