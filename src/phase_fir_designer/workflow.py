from __future__ import annotations

from dataclasses import replace

from .auto_eq import auto_gain_eq_sections, auto_phase_eq_sections
from .config import DesignConfig
from .acoustic_target import enabled as acoustic_target_enabled


def stage_shows_linear_fir_eq_mask(stage: str) -> bool:
    """Keep FIR-only mask guides off pre-FIR and separate-IIR result stages."""

    return str(stage) not in {"Speaker/Input", "Target Response", "IIR EQ"}


def _target_for_auto_gain_stage(config: DesignConfig):
    return config.target_response if auto_gain_eq_sections(config.auto_eq) else None


def _target_for_auto_phase_or_later_stage(config: DesignConfig):
    if acoustic_target_enabled(config) or auto_gain_eq_sections(config.auto_eq) or auto_phase_eq_sections(config.auto_eq):
        return config.target_response
    return None


def config_for_preview_stage(config: DesignConfig, stage: str) -> DesignConfig:
    auto_off = replace(
        config.auto_eq,
        gain_enabled=False,
        phase_enabled=False,
        sections=[replace(section, gain_enabled=False, phase_enabled=False) for section in config.auto_eq.sections],
        gain_sections=[replace(section, enabled=False, gain_enabled=False) for section in config.auto_eq.gain_sections],
        phase_sections=[replace(section, enabled=False, phase_enabled=False) for section in config.auto_eq.phase_sections],
    )
    common = {
        "sample_rate": config.sample_rate,
        "taps": config.taps,
        "analysis_fft_size": config.analysis_fft_size,
        "window": config.window,
        "kaiser_beta": config.kaiser_beta,
        "chebyshev_attenuation_db": config.chebyshev_attenuation_db,
        "tukey_alpha": config.tukey_alpha,
        "phase_tilt_smoothing_oct": config.phase_tilt_smoothing_oct,
        "gain_tilt_smoothing_oct": config.gain_tilt_smoothing_oct,
        "dc_gain_normalize": config.dc_gain_normalize,
        "fir_fit_weighting": config.fir_fit_weighting,
        "limiter": config.limiter,
        "linear_fir_filters": config.linear_fir_filters,
        "band_split_recipe": config.band_split_recipe,
        "fir_generation_grid": config.fir_generation_grid,
        "linear_fir_eq_mask_smoothing_oct": config.linear_fir_eq_mask_smoothing_oct,
        "speaker_response": config.speaker_response,
        "acoustic_correction_enabled": False,
    }
    if stage == "Speaker/Input":
        return DesignConfig(auto_eq=auto_off, target_response=None, iir_filters=[], **common)
    if stage == "Target Response":
        return DesignConfig(auto_eq=auto_off, target_response=config.target_response, iir_filters=[], **common)
    if stage == "IIR EQ":
        return DesignConfig(
            auto_eq=auto_off,
            target_response=None,
            iir_filters=config.iir_filters,
            **common,
        )
    if stage == "Gain EQ":
        return DesignConfig(
            gain_peq_filters=config.gain_peq_filters,
            gain_shelf_filters=config.gain_shelf_filters,
            gain_linear_filters=config.gain_linear_filters,
            gain_tilt_filters=config.gain_tilt_filters,
            iir_filters=config.iir_filters,
            auto_eq=auto_off,
            target_response=None,
            **common,
        )
    if stage == "Phase EQ":
        return DesignConfig(
            peq_filters=config.peq_filters,
            shelf_filters=config.shelf_filters,
            tilt_filters=config.tilt_filters,
            allpass_filters=config.allpass_filters,
            gain_peq_filters=config.gain_peq_filters,
            gain_shelf_filters=config.gain_shelf_filters,
            gain_linear_filters=config.gain_linear_filters,
            gain_tilt_filters=config.gain_tilt_filters,
            iir_filters=config.iir_filters,
            auto_eq=auto_off,
            target_response=None,
            **common,
        )
    if stage == "Auto Gain EQ":
        return DesignConfig(
            peq_filters=config.peq_filters,
            shelf_filters=config.shelf_filters,
            tilt_filters=config.tilt_filters,
            allpass_filters=config.allpass_filters,
            gain_peq_filters=config.gain_peq_filters,
            gain_shelf_filters=config.gain_shelf_filters,
            gain_linear_filters=config.gain_linear_filters,
            gain_tilt_filters=config.gain_tilt_filters,
            iir_filters=config.iir_filters,
            auto_eq=replace(
                config.auto_eq,
                phase_enabled=False,
                sections=[replace(section, phase_enabled=False) for section in config.auto_eq.sections],
                phase_sections=[
                    replace(section, enabled=False, phase_enabled=False)
                    for section in config.auto_eq.phase_sections
                ],
            ),
            target_response=_target_for_auto_gain_stage(config),
            **common,
        )
    if stage == "Auto Phase EQ":
        return replace(config, target_response=_target_for_auto_phase_or_later_stage(config))
    if stage == "Linear FIR":
        return replace(config, target_response=_target_for_auto_phase_or_later_stage(config))
    return replace(config, target_response=_target_for_auto_phase_or_later_stage(config))


def config_for_composite_exchange(
    config: DesignConfig,
    *,
    assignment_target_is_reference: bool,
) -> DesignConfig:
    """Keep an Assignment Target out of the FIR returned to Composite.

    A Composite Group/Channel Target is a graph and editing reference.  It may
    populate PhaseEQ's Target page, but it is not an implicit FIR stage in the
    Multiway signal path.  Explicit PhaseEQ gain/phase/auto filter settings are
    preserved; only the direct Target response is removed here.
    """
    if (
        acoustic_target_enabled(config)
        or not assignment_target_is_reference
        or auto_gain_eq_sections(config.auto_eq)
        or auto_phase_eq_sections(config.auto_eq)
    ):
        return config
    return replace(config, target_response=None)


def previous_stage_for_edit_delta(stage: str) -> str | None:
    return {
        "Target Response": "Speaker/Input",
        "IIR EQ": "Target Response",
        "Gain EQ": "IIR EQ",
        "Phase EQ": "Gain EQ",
        "Auto Gain EQ": "Phase EQ",
        "Auto Phase EQ": "Auto Gain EQ",
        "Linear FIR": "Auto Phase EQ",
        "Export FIR": "Linear FIR",
    }.get(stage)
