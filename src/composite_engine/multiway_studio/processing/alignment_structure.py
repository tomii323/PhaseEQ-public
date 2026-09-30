"""Separate automatic structural phase correction from user-triggered delay fitting."""
import numpy as np

from composite_engine.phase_alignment import allpass_response, structural_allpass_sections
from .alignment_target import output_settings


def automatic_sections(settings, ways, crossovers, methods):
    if not settings.get("phase_alignment_auto_structure", False):
        return None
    return structural_allpass_sections(tuple(ways), tuple(crossovers), tuple(methods))


def analysis_response(row, response, frequency, sample_rate, *, time_only):
    result = np.asarray(response, dtype=complex).copy()
    # Undo only stages present in the direct Multiway response. Target-only
    # sections remain recipe data and must not be divided out of this signal.
    _, _, sections = output_settings(row)
    if sections and not time_only:
        result /= allpass_response(sections, frequency, sample_rate)
    delay = float(row.get("auto_alignment_delay_samples", 0.0))
    if delay:
        result *= np.exp(1j * 2.0 * np.pi * frequency * delay / float(sample_rate))
    return result


def previous_values(state, prefix, row, *, time_only):
    previous = {"delay": state.get(f"{prefix}_auto_alignment_delay", row.get("auto_alignment_delay_samples", 0.0))}
    if not time_only:
        previous["allpass"] = state.get(f"{prefix}_auto_alignment_allpass", [
            section.to_dict() for section in row.get("auto_alignment_allpass", ())])
    return previous


def apply_channel(state, prefix, proposal, *, time_only):
    state[f"{prefix}_auto_alignment_delay"] = proposal.dsp_delay_samples
    if not time_only:
        state[f"{prefix}_auto_alignment_allpass"] = [section.to_dict() for section in proposal.allpass_sections]


def undo_channel(state, prefix, previous, *, automatic):
    state[f"{prefix}_auto_alignment_delay"] = previous["delay"]
    if not automatic and "allpass" in previous:
        state[f"{prefix}_auto_alignment_allpass"] = previous["allpass"]
