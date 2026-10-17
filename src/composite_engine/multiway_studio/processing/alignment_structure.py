"""Separate automatic structural phase correction from user-triggered delay fitting."""
import numpy as np

from composite_engine.phase_alignment import allpass_response, structural_allpass_sections
from .alignment_target import output_settings

AUTOMATIC_OWNERS_KEY = '_composite_automatic_structure_owners'


def automatic_sections(settings, ways, crossovers, methods, *, lobe_heights_db=()):
    if not settings.get("phase_alignment_auto_structure", False):
        return None
    return structural_allpass_sections(tuple(ways), tuple(crossovers), tuple(methods),
                                      lobe_heights_db=lobe_heights_db)


def commit_automatic_mode(state, enabled, prefixes):
    """Release automatic phase stages on disable; keep independent timing intact."""
    settings = dict(state.get('settings', {}))
    was_enabled = bool(settings.get('phase_alignment_auto_structure', False))
    settings['phase_alignment_auto_structure'] = bool(enabled)
    if was_enabled and not enabled:
        # ON owns these stages. OFF must release them before rows and output
        # snapshots are rebuilt; stopping generation alone retains stale stages.
        owners = dict(state.get(AUTOMATIC_OWNERS_KEY, {}))
        owned_prefixes = set(prefixes)
        owned_prefixes.update(prefix for prefix, values in owners.items()
                              if state.get(f'{prefix}_auto_alignment_allpass') == values)
        for prefix in owned_prefixes:
            state[f'{prefix}_auto_alignment_allpass'] = []
        state.pop(AUTOMATIC_OWNERS_KEY, None)
    state['settings'] = settings
    state['_alignment_refresh_requested'] = True


def refresh_automatic_channel(state, prefix, sections):
    """Project calculated automatic sections into their owned channel state."""
    key = f'{prefix}_auto_alignment_allpass'
    state.setdefault(key, [])
    if sections is None:
        return
    values = [section.to_dict() for section in sections]
    owners = dict(state.get(AUTOMATIC_OWNERS_KEY, {}))
    owners[prefix] = [dict(value) for value in values]
    state[AUTOMATIC_OWNERS_KEY] = owners
    if state[key] != values:
        state[key] = values
        state['_alignment_refresh_requested'] = True


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
    previous["measurement_present"] = f"{prefix}_auto_alignment_measurement_delay_samples" in state
    previous["measurement"] = state.get(f"{prefix}_auto_alignment_measurement_delay_samples", row.get("auto_alignment_measurement_delay_samples"))
    if not time_only:
        previous["allpass"] = state.get(f"{prefix}_auto_alignment_allpass", [
            section.to_dict() for section in row.get("auto_alignment_allpass", ())])
    return previous


def apply_channel(state, prefix, proposal, *, time_only, timing_source="phase_fit"):
    state[f"{prefix}_auto_alignment_delay"] = proposal.dsp_delay_samples
    state[f"{prefix}_auto_alignment_measurement_delay_samples"] = (
        proposal.dsp_delay_samples if timing_source in ("phaseeq_tweeter_reference", "external_distance") else 0.0
    )
    if not time_only:
        state[f"{prefix}_auto_alignment_allpass"] = [section.to_dict() for section in proposal.allpass_sections]


def undo_channel(state, prefix, previous, *, automatic):
    state[f"{prefix}_auto_alignment_delay"] = previous["delay"]
    if previous.get("measurement_present", False):
        state[f"{prefix}_auto_alignment_measurement_delay_samples"] = previous.get("measurement")
    else:
        state.pop(f"{prefix}_auto_alignment_measurement_delay_samples", None)
    if not automatic and "allpass" in previous:
        state[f"{prefix}_auto_alignment_allpass"] = previous["allpass"]
