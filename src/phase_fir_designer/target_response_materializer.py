from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import (
    DesignConfig,
    GainLinearFilter,
    GainPEQ,
    GainShelf,
    GainTilt,
    PhaseAllPass,
    PhasePEQ,
    PhaseShelf,
    PhaseTilt,
    SpeakerResponse,
)
from .target_pipeline import TargetProcessingResult, TargetProcessingSettings, process_target


@dataclass(frozen=True)
class MaterializedTargetResponse:
    response: SpeakerResponse
    mode: str
    processing: TargetProcessingResult
    has_source_or_edits: bool


def materialize_target_response(
    payload: Mapping[str, Any],
    *,
    speaker_reference: SpeakerResponse | None,
    sample_rate: int,
    analysis_fft_size: int,
    source_response: SpeakerResponse | None = None,
) -> MaterializedTargetResponse:
    """Rebuild the effective Target from a persisted Target Package UI payload.

    Legacy Projects often saved Target edit parameters but no materialized
    response asset.  This module is deliberately UI-independent so migration,
    tests and DSP System generation can all rebuild the same response.
    """

    ui = dict(payload.get("ui", payload.get("target_ui", {})))
    edit = ui.get("target_response_edit", {})
    edit = dict(edit) if isinstance(edit, Mapping) else {}
    config = _edit_config(edit, int(sample_rate), int(analysis_fft_size))
    mode = str(ui.get("target_gain_shift_mode", "Off"))
    manual_shift = float(ui.get("target_gain_shift_db", 0.0) or 0.0)
    processing = process_target(
        source_response,
        TargetProcessingSettings(
            sample_rate=int(sample_rate),
            analysis_fft_size=int(analysis_fft_size),
            gain_shift_mode=mode,
            manual_gain_shift_db=manual_shift,
            source_gain_only=(str(ui.get("target_source_content_mode", "Gain + Phase")) == "Gain only"),
            phase_enabled=bool(ui.get("target_phase_enabled", True)),
        ),
        speaker_reference=speaker_reference,
        edit_config=config,
    )
    has_edits = any(
        bool(item.get("enabled", True))
        for key in (
            "gain_peq", "gain_shelf", "gain_linear", "gain_tilt",
            "phase_peq", "phase_shelf", "phase_tilt", "phase_allpass",
        )
        for item in edit.get(key, [])
        if isinstance(item, Mapping)
    )
    has_content = source_response is not None or has_edits or abs(manual_shift) > 1e-12
    return MaterializedTargetResponse(
        response=processing.effective_response,
        mode="Custom" if has_content else "Flat",
        processing=processing,
        has_source_or_edits=has_content,
    )


def _edit_config(edit: Mapping[str, Any], sample_rate: int, analysis_fft_size: int) -> DesignConfig:
    gain_peq = [
        GainPEQ(
            fc=float(item.get("fc", 1000.0)), q=float(item.get("q", 1.0)),
            gain_db=float(item.get("gain_db", 0.0)), enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("gain_peq", []) if isinstance(item, Mapping)
    ]
    gain_shelf = [
        GainShelf(
            mode=str(item.get("mode", "low")), fc=float(item.get("fc", 1000.0)),
            q=float(item.get("q", 1.0)), gain_db=float(item.get("gain_db", 0.0)),
            enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("gain_shelf", []) if isinstance(item, Mapping)
    ]
    gain_linear = [
        GainLinearFilter(
            mode="hp" if str(item.get("mode", "lp")) == "hp" else "lp",
            response="lr2", fc=float(item.get("fc", 1000.0)),
            cycles=float(item.get("cycles", 4.0)), beta=float(item.get("beta", 12.0)),
            enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("gain_linear", []) if isinstance(item, Mapping)
    ]
    gain_tilt = [
        GainTilt(
            mode="low" if str(item.get("mode", "high")) == "low" else "high",
            f_start=float(item.get("f_start", 100.0)),
            f_end=float(item.get("f_end", 20_000.0)),
            slope_db_per_oct=float(item.get("slope_db_per_oct", 0.0)),
            edge_smooth_oct=0.0, enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("gain_tilt", []) if isinstance(item, Mapping)
    ]
    phase_peq = [
        PhasePEQ(
            fc=float(item.get("fc", 1000.0)), q=float(item.get("q", 1.0)),
            phase_deg=float(item.get("phase_deg", 0.0)), enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("phase_peq", []) if isinstance(item, Mapping)
    ]
    phase_shelf = [
        PhaseShelf(
            mode=str(item.get("mode", "low")), fc=float(item.get("fc", 1000.0)),
            q=float(item.get("q", 1.0)), phase_deg=float(item.get("phase_deg", 0.0)),
            enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("phase_shelf", []) if isinstance(item, Mapping)
    ]
    phase_tilt = [
        PhaseTilt(
            f_start=float(item.get("f_start", 100.0)),
            f_end=float(item.get("f_end", 20_000.0)),
            slope_deg_per_oct=float(item.get("slope_deg_per_oct", 0.0)),
            edge_smooth_oct=0.0, enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("phase_tilt", []) if isinstance(item, Mapping)
    ]
    phase_allpass = [
        PhaseAllPass(
            fc=float(item.get("fc", 1000.0)), q=float(item.get("q", 1.0)),
            polarity=str(item.get("polarity", "positive")),
            enabled=bool(item.get("enabled", True)),
        )
        for item in edit.get("phase_allpass", []) if isinstance(item, Mapping)
    ]
    return DesignConfig(
        sample_rate=sample_rate,
        taps=1025,
        analysis_fft_size=analysis_fft_size,
        gain_peq_filters=gain_peq,
        gain_shelf_filters=gain_shelf,
        gain_linear_filters=gain_linear,
        gain_tilt_filters=gain_tilt,
        gain_tilt_smoothing_oct=float(edit.get("gain_tilt_smoothing_oct", 1.0 / 3.0)),
        peq_filters=phase_peq,
        shelf_filters=phase_shelf,
        tilt_filters=phase_tilt,
        phase_tilt_smoothing_oct=float(edit.get("phase_tilt_smoothing_oct", 1.0 / 3.0)),
        allpass_filters=phase_allpass,
    )
