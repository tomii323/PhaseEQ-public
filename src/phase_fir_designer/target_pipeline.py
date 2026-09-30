from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from .config import DesignConfig, SpeakerResponse
from .phase_curves import build_target_curves, frequency_axis
from .speaker import TargetGainShiftEstimate, estimate_target_auto_gain_shift
from .input_pipeline import shift_response_gain


@dataclass(frozen=True)
class TargetProcessingSettings:
    sample_rate: int
    analysis_fft_size: int = 16_384
    gain_shift_mode: str = "Off"
    manual_gain_shift_db: float = 0.0
    source_gain_only: bool = False
    phase_enabled: bool = True  # Source phase only; Phase EQ remains active.
    # PhaseEQ opts in; measurement provenance and other consumers stay unchanged.
    zero_missing_phase: bool = False
    lf_extension_enabled: bool = False
    hf_extension_enabled: bool = False


@dataclass(frozen=True)
class TargetProcessingResult:
    original_response: SpeakerResponse | None
    base_response: SpeakerResponse
    shifted_response: SpeakerResponse
    effective_response: SpeakerResponse
    applied_gain_shift_db: float
    gain_shift_estimate: TargetGainShiftEstimate
    generated_from_flat: bool
    messages: tuple[str, ...] = ()


def flat_target_response(sample_rate: int, analysis_fft_size: int) -> SpeakerResponse:
    frequency = frequency_axis(int(sample_rate), int(analysis_fft_size))
    zeros = np.zeros_like(frequency)
    return SpeakerResponse(frequency.tolist(), zeros.tolist(), None)


def process_target(
    source: SpeakerResponse | None,
    settings: TargetProcessingSettings,
    *,
    speaker_reference: SpeakerResponse | None = None,
    speaker_reference_range: tuple[float, float] | None = None,
    edit_config: DesignConfig | None = None,
    response_editor: Callable[[SpeakerResponse], SpeakerResponse | None] | None = None,
) -> TargetProcessingResult:
    if int(settings.sample_rate) <= 0:
        raise ValueError("sample_rate must be positive")
    generated = source is None
    base = source or flat_target_response(settings.sample_rate, settings.analysis_fft_size)
    from .target_extension import extend_target_response
    base = extend_target_response(base, settings.sample_rate,
        lf_enabled=settings.lf_extension_enabled, hf_enabled=settings.hf_extension_enabled)
    if (bool(settings.source_gain_only) or not bool(settings.phase_enabled)) and base.phase_deg is not None:
        base = SpeakerResponse(
            frequency=list(base.frequency),
            gain_db=list(base.gain_db),
            phase_deg=None,
        )
    mode = str(settings.gain_shift_mode).lower()
    if settings.zero_missing_phase and settings.phase_enabled and base.phase_deg is None:
        base = SpeakerResponse(list(base.frequency), list(base.gain_db), [0.0] * len(base.frequency))
    if mode in {"auto", "automatic"} or mode.startswith("auto:"):
        estimate = estimate_target_auto_gain_shift(
            base,
            speaker_reference,
            measured_range=speaker_reference_range,
            manual_shift_db=float(settings.manual_gain_shift_db),
        )
    elif mode in {"off", "none", "disabled"}:
        estimate = estimate_target_auto_gain_shift(None, None, manual_shift_db=0.0)
    else:
        estimate = estimate_target_auto_gain_shift(
            None,
            None,
            manual_shift_db=float(settings.manual_gain_shift_db),
        )
    shifted = shift_response_gain(base, estimate.shift_db)
    assert shifted is not None
    if response_editor is not None:
        effective = response_editor(shifted)
        if effective is None:
            effective = shifted
        return TargetProcessingResult(
            original_response=source,
            base_response=base,
            shifted_response=shifted,
            effective_response=effective,
            applied_gain_shift_db=float(estimate.shift_db),
            gain_shift_estimate=estimate,
            generated_from_flat=generated,
            messages=("Flat Target generated",) if generated else (),
        )
    config = edit_config or DesignConfig(
        sample_rate=int(settings.sample_rate),
        taps=1025,
        analysis_fft_size=int(settings.analysis_fft_size),
    )
    config = replace(
        config,
        sample_rate=int(settings.sample_rate),
        analysis_fft_size=int(settings.analysis_fft_size),
        speaker_response=None,
        target_response=shifted,
        iir_filters=[],
        linear_fir_filters=[],
        auto_eq=replace(config.auto_eq, gain_enabled=False, phase_enabled=False, sections=[], gain_sections=[], phase_sections=[]),
    )
    frequency = np.asarray(shifted.frequency, dtype=float)
    gain, phase = build_target_curves(config, frequency)
    effective = SpeakerResponse(frequency.tolist(), gain.tolist(), phase.tolist())
    return TargetProcessingResult(
        original_response=source,
        base_response=base,
        shifted_response=shifted,
        effective_response=effective,
        applied_gain_shift_db=float(estimate.shift_db),
        gain_shift_estimate=estimate,
        generated_from_flat=generated,
        messages=("Flat Target generated",) if generated else (),
    )
