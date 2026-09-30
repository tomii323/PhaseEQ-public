"""One authoritative realization for inherited targets and selection previews."""
import io

from phase_fir_designer.speaker import load_speaker_response_text
from phase_fir_designer.config import SpeakerResponse
from .definition import target_definition_frd_bytes
from .response import apply_target_edit_to_response


def realize_target_definition(definition, sample_rate, analysis_fft_size=16384):
    base = load_speaker_response_text(io.StringIO(
        target_definition_frd_bytes(definition, sample_rate).decode()))
    if not isinstance(definition.get("materialized_response"), dict):
        from phase_fir_designer.target_extension import extend_target_response
        base = extend_target_response(base, sample_rate,
            lf_enabled=bool(definition.get("lf_extension_enabled", False)),
            hf_enabled=bool(definition.get("hf_extension_enabled", False)))
        base = apply_target_edit_to_response(base, sample_rate_hz=sample_rate,
            analysis_fft_size=analysis_fft_size, payload=definition.get("target_edit", {}),
            source_gain_only=definition.get("source_content_mode") == "Gain only")
    if base.phase_deg is None:
        base = SpeakerResponse(list(base.frequency), list(base.gain_db), [0.0]*len(base.frequency))
    return base


def inherited_target_locked(assignment, target_edit_session=None):
    return (assignment is not None and target_edit_session is None
            and getattr(assignment, "target_application", "") != "display_only"
            and (isinstance(getattr(assignment, "target_definition", None), dict)
                 or any(b.get("acoustic_target", False) for b in
                        (getattr(assignment, "band_split_recipe", None) or {}).get("boundaries", []))))


def realized_target_frd_bytes(definition, sample_rate):
    response = realize_target_definition(definition, sample_rate)
    return ("\n".join(f"{f:.16g} {g:.16g} {p:.16g}" for f, g, p in
            zip(response.frequency, response.gain_db, response.phase_deg, strict=True)) + "\n").encode()
