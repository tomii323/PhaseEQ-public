"""Dedicated export guidance in the existing Multiway DSP section."""
import streamlit as st

from composite_engine.dsp_export.cdsp_help import cdsp_guide_sections
from composite_engine.dsp_export.cdsp_dirac import CAUTION
from utils.ui_language import current_language
from ui.structured_help import render_help_sections


def cdsp_issue_message(issue):
    if current_language() != 'en':
        return issue.message
    messages = {
        'automatic_phase_required': 'Enable automatic structural phase compensation.',
        'design_pending': 'Recalculate to finalize the output for the current settings.',
        'unsupported_method': 'Set every boundary to LR4 or NTM4, then recalculate.',
        'acoustic_target': 'To use only Group Target for IIR Auto PEQ, turn off acoustic-target reflection at every crossover boundary and in phase/time alignment, then recalculate.',
        'no_channels': 'Enable output channels and recalculate.',
        'cdsp_rate': 'Set the sample rate to 48 kHz and recalculate.',
        'cdsp_fir': 'Turn DSP FIR output OFF and recalculate.',
        'output_limit': 'Assign outputs within CH01-CH12.',
        'delay_limit': 'Keep the common delay reference and set each delay within 0-20 ms.',
        'gain_limit': 'Set output gain within -72 to +12 dB.',
        'polarity': 'Set output polarity to +1 or -1.',
        'pre_dirac_configuration': 'Use direct LR4/NTM4 crossovers on every boundary and recalculate.',
        'xover_capacity': 'Xover exceeds 8 sections. Reduce the crossover/AP configuration.',
        'way_limit': 'Use at most four bands (3Way+SUB or 4Way), then recalculate.',
        'peq_capacity': 'PEQ exceeds 10 sections. Reduce the channel IIR configuration.',
        'peq_pending': 'Regenerate and return the IIR PEQ coefficients from PhaseEQ.',
        'dirac_target': 'Check the Group Target definition for every output group.',
    }
    return messages.get(issue.code, issue.message)


def render_cdsp_help(package):
    english = current_language() == 'en'
    fir = any(channel.final_fir is not None for channel in package.channels)
    automatic = bool(package.workspace.get('multiway_settings', {}).get('phase_alignment_auto_structure', False))
    st.caption(('Pre: measurement APF / Post: playback crossover. 48 kHz only.' if english else
                'Pre：測定用APF／Post：再生用クロスオーバー。48 kHz専用です。'))
    st.caption('Replace Pre with Post in the same 8 output Xover slots.' if english else
               '出力Xoverの同じ8段で、測定用Preから再生用Postへ置き換えます。')
    st.caption(f"FIR: {'ON' if fir else 'OFF'} / "
               + ('Automatic structural phase compensation' if english else '位相構造補償の自動適用')
               + f": {'ON' if automatic else 'OFF'}")
    st.warning(('CAUTION: Pre-Dirac APFs do not attenuate out-of-band signals. '
                'You are responsible for protecting the drivers during measurement.') if english else CAUTION)
    with st.expander('C-DSP Dirac/APF guide' if english else 'C-DSP Dirac/APFの操作手順'):
        sections = cdsp_guide_sections('en' if english else 'ja')
        render_help_sections(sections)
