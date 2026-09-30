from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Sequence

import numpy as np
from scipy import fft
from scipy.sparse.linalg import LinearOperator, lsqr

from ..config import DesignConfig
from ..design import DesignResult, calculate_group_delay_ms
from ..phase_curves import build_target_curves, frequency_axis, phase_error_deg, wrapped_phase_deg


WAY_FIR_SYNTHESIS_ENGINE = "single_projection_v1"


@dataclass(frozen=True)
class WayFIRTargetStage:
    """One FIR-shaped target contribution folded into the final Way FIR."""

    name: str
    coefficients: np.ndarray
    enabled: bool = True


@dataclass(frozen=True)
class WayFIRSynthesisResult:
    """A single fixed-length Way FIR and the target it was projected from."""

    design_result: DesignResult
    engine: str
    design_signature: str
    stage_names: tuple[str, ...]
    stage_taps: tuple[int, ...]
    natural_stage_taps: int
    output_taps: int
    bypassed: bool
    rms_complex_error: float
    max_complex_error: float

    @property
    def fir(self) -> np.ndarray:
        return self.design_result.fir


def synthesize_way_fir(
    config: DesignConfig,
    stages: Sequence[WayFIRTargetStage] = (),
    *,
    enabled: bool = True,
    compute_group_delay: bool = True,
) -> WayFIRSynthesisResult:
    """Project all enabled FIR targets directly into one device-length FIR.

    The Project Input, IIR EQ, FIR EQ and Auto EQ remain represented by
    ``DesignConfig``.  Crossover and alignment FIRs are supplied as target
    stages.  Their centered complex responses are multiplied in frequency,
    then solved once at the requested output tap count.  No intermediate FIR
    convolution or crop is performed.
    """

    normalized = config.normalized()
    normalized.validate()
    active = tuple(_validated_stage(stage, index) for index, stage in enumerate(stages) if stage.enabled)
    natural_taps = 1 + sum(coefficients.size - 1 for _, coefficients in active)

    if not enabled:
        design = _bypass_design_result(normalized, compute_group_delay=compute_group_delay)
        return _result(
            normalized,
            design,
            active,
            natural_taps=natural_taps,
            bypassed=True,
        )

    design_fft_size = fft.next_fast_len(
        max(
            int(normalized.taps) * 8,
            int(normalized.analysis_fft_size),
            max((coefficients.size for _, coefficients in active), default=1),
        ),
        real=True,
    )
    if design_fft_size % 2:
        design_fft_size += 1
    design_frequency = frequency_axis(normalized.sample_rate, design_fft_size)
    target = _combined_target_response(normalized, design_frequency, design_fft_size, active)
    fir = _project_complex_target(
        target,
        taps=int(normalized.taps),
        grid_size=design_fft_size,
    )

    analysis_fft_size = fft.next_fast_len(
        max(int(normalized.analysis_fft_size), int(normalized.taps)),
        real=True,
    )
    if analysis_fft_size % 2:
        analysis_fft_size += 1
    analysis_frequency = frequency_axis(normalized.sample_rate, analysis_fft_size)
    analysis_target = _combined_target_response(
        normalized,
        analysis_frequency,
        analysis_fft_size,
        active,
    )
    realized = _centered_fir_response(fir, normalized.sample_rate, analysis_fft_size)
    target_gain = 20.0 * np.log10(np.maximum(np.abs(analysis_target), 1e-12))
    realized_gain = 20.0 * np.log10(np.maximum(np.abs(realized), 1e-12))
    target_phase = np.rad2deg(np.unwrap(np.angle(analysis_target)))
    realized_phase = np.rad2deg(np.unwrap(np.angle(realized)))
    if compute_group_delay:
        group_delay = calculate_group_delay_ms(fir, normalized.sample_rate, analysis_frequency)
    else:
        group_delay = np.full_like(analysis_frequency, np.nan, dtype=float)
    design = DesignResult(
        fir=fir,
        frequency=analysis_frequency,
        target_gain_db=target_gain,
        realized_gain_db=realized_gain,
        gain_error_db=realized_gain - target_gain,
        target_phase_wrapped_deg=wrapped_phase_deg(target_phase),
        realized_phase_wrapped_deg=wrapped_phase_deg(realized_phase),
        realized_phase_unwrapped_deg=realized_phase,
        phase_error_deg=phase_error_deg(
            target_phase,
            realized_phase,
            np.minimum(realized_gain, target_gain),
        ),
        impulse_response=fir.copy(),
        group_delay_ms=group_delay,
        step_response=np.cumsum(fir),
        target_phase_unwrapped_deg=target_phase,
    )
    return _result(
        normalized,
        design,
        active,
        natural_taps=natural_taps,
        bypassed=False,
        target=analysis_target,
        realized=realized,
    )


def fixed_way_fir_result(
    config: DesignConfig,
    coefficients: np.ndarray,
    *,
    stage_name: str = "Retained FIR artifact",
    compute_group_delay: bool = True,
) -> WayFIRSynthesisResult:
    """Represent a content-addressed FIR already resident in a device."""

    normalized = config.normalized()
    normalized.validate()
    fir = np.asarray(coefficients, dtype=float).ravel()
    if fir.size < 1 or not np.all(np.isfinite(fir)):
        raise ValueError("retained FIR artifact contains invalid coefficients")
    analysis_fft_size = fft.next_fast_len(
        max(int(normalized.analysis_fft_size), fir.size), real=True,
    )
    if analysis_fft_size % 2:
        analysis_fft_size += 1
    frequency = frequency_axis(normalized.sample_rate, analysis_fft_size)
    realized = _centered_fir_response(fir, normalized.sample_rate, analysis_fft_size)
    gain = 20.0 * np.log10(np.maximum(np.abs(realized), 1e-12))
    phase = np.rad2deg(np.unwrap(np.angle(realized)))
    group_delay = (
        calculate_group_delay_ms(fir, normalized.sample_rate, frequency)
        if compute_group_delay
        else np.full_like(frequency, np.nan, dtype=float)
    )
    zeros = np.zeros_like(frequency, dtype=float)
    design = DesignResult(
        fir=fir.copy(),
        frequency=frequency,
        target_gain_db=gain.copy(),
        realized_gain_db=gain.copy(),
        gain_error_db=zeros.copy(),
        target_phase_wrapped_deg=wrapped_phase_deg(phase),
        realized_phase_wrapped_deg=wrapped_phase_deg(phase),
        realized_phase_unwrapped_deg=phase.copy(),
        phase_error_deg=zeros.copy(),
        impulse_response=fir.copy(),
        group_delay_ms=group_delay,
        step_response=np.cumsum(fir),
        target_phase_unwrapped_deg=phase.copy(),
    )
    return _result(
        normalized,
        design,
        ((stage_name, fir.copy()),),
        natural_taps=fir.size,
        bypassed=False,
        target=realized,
        realized=realized,
    )


def _combined_target_response(
    config: DesignConfig,
    frequency: np.ndarray,
    fft_size: int,
    stages: tuple[tuple[str, np.ndarray], ...],
) -> np.ndarray:
    target_gain, target_phase = build_target_curves(config, frequency)
    target = np.power(10.0, np.asarray(target_gain, dtype=float) / 20.0) * np.exp(
        1j * np.deg2rad(np.asarray(target_phase, dtype=float))
    )
    for _name, coefficients in stages:
        target *= _centered_fir_response(coefficients, config.sample_rate, fft_size)
    if target.size:
        target[0] = complex(float(np.real(target[0])), 0.0)
        if np.isclose(float(frequency[-1]), float(config.sample_rate) / 2.0):
            target[-1] = complex(float(np.real(target[-1])), 0.0)
    if not np.all(np.isfinite(target)):
        raise ValueError("Way FIR target contains non-finite values")
    return target


def _project_complex_target(target: np.ndarray, *, taps: int, grid_size: int) -> np.ndarray:
    target = np.asarray(target, dtype=complex)
    if target.shape != (grid_size // 2 + 1,):
        raise ValueError("Way FIR target must match the real FFT frequency grid")
    if taps < 1:
        raise ValueError("Way FIR taps must be positive")
    if _is_centered_unity_target(target):
        return _centered_delta(taps)

    delay_samples = (taps - 1) / 2.0
    frequency_bins = np.arange(target.size, dtype=float)
    delay = np.exp(1j * 2.0 * np.pi * frequency_bins * delay_samples / float(grid_size))
    rhs = np.concatenate((np.real(target), np.imag(target)))

    def matvec(coefficients: np.ndarray) -> np.ndarray:
        response = fft.rfft(np.asarray(coefficients, dtype=float), n=grid_size) * delay
        return np.concatenate((np.real(response), np.imag(response)))

    def rmatvec(residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(residual, dtype=float)
        size = target.size
        spectrum = np.conj(delay) * (residual[:size] + 1j * residual[size:])
        adjoint_spectrum = spectrum.copy()
        if adjoint_spectrum.size > 2:
            adjoint_spectrum[1:-1] *= 0.5
        adjoint = fft.irfft(adjoint_spectrum, n=grid_size) * grid_size
        return np.asarray(adjoint[:taps], dtype=float)

    operator = LinearOperator(
        shape=(rhs.size, taps),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=float,
    )
    solution = lsqr(
        operator,
        rhs,
        damp=1e-6,
        atol=1e-7,
        btol=1e-7,
        iter_lim=300,
        show=False,
    )[0]
    output = np.asarray(solution, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise ValueError("Way FIR projection produced non-finite coefficients")
    return output


def _centered_fir_response(coefficients: np.ndarray, sample_rate: int, fft_size: int) -> np.ndarray:
    values = np.asarray(coefficients, dtype=float)
    response = fft.rfft(values, n=fft_size)
    frequency = frequency_axis(sample_rate, fft_size)
    delay_samples = (values.size - 1) / 2.0
    return response * np.exp(1j * 2.0 * np.pi * frequency * delay_samples / float(sample_rate))


def _bypass_design_result(config: DesignConfig, *, compute_group_delay: bool) -> DesignResult:
    # OFF/Unavailable means the FIR block is removed.  A one-tap delta keeps
    # the simulated transfer at unity without inventing a device FIR delay.
    fir = _centered_delta(1)
    analysis_fft_size = fft.next_fast_len(max(int(config.analysis_fft_size), int(config.taps)), real=True)
    if analysis_fft_size % 2:
        analysis_fft_size += 1
    frequency = frequency_axis(config.sample_rate, analysis_fft_size)
    zeros = np.zeros_like(frequency, dtype=float)
    group_delay = (
        calculate_group_delay_ms(fir, config.sample_rate, frequency)
        if compute_group_delay
        else np.full_like(frequency, np.nan, dtype=float)
    )
    return DesignResult(
        fir=fir,
        frequency=frequency,
        target_gain_db=zeros.copy(),
        realized_gain_db=zeros.copy(),
        gain_error_db=zeros.copy(),
        target_phase_wrapped_deg=zeros.copy(),
        realized_phase_wrapped_deg=zeros.copy(),
        realized_phase_unwrapped_deg=zeros.copy(),
        phase_error_deg=zeros.copy(),
        impulse_response=fir.copy(),
        group_delay_ms=group_delay,
        step_response=np.cumsum(fir),
        target_phase_unwrapped_deg=zeros.copy(),
    )


def _result(
    config: DesignConfig,
    design: DesignResult,
    stages: tuple[tuple[str, np.ndarray], ...],
    *,
    natural_taps: int,
    bypassed: bool,
    target: np.ndarray | None = None,
    realized: np.ndarray | None = None,
) -> WayFIRSynthesisResult:
    signature = hashlib.sha256()
    signature.update(WAY_FIR_SYNTHESIS_ENGINE.encode("ascii"))
    signature.update(int(config.sample_rate).to_bytes(8, "little", signed=False))
    signature.update(int(config.taps).to_bytes(8, "little", signed=False))
    signature.update(b"1" if bypassed else b"0")
    for name, coefficients in stages:
        signature.update(name.encode("utf-8"))
        signature.update(np.asarray(coefficients, dtype="<f8").tobytes())
    signature.update(np.asarray(design.target_gain_db, dtype="<f8").tobytes())
    signature.update(np.asarray(design.target_phase_unwrapped_deg, dtype="<f8").tobytes())
    if target is None or realized is None:
        rms_error = 0.0
        max_error = 0.0
    else:
        difference = np.asarray(realized, dtype=complex) - np.asarray(target, dtype=complex)
        rms_error = float(np.sqrt(np.mean(np.abs(difference) ** 2)))
        max_error = float(np.max(np.abs(difference)))
    return WayFIRSynthesisResult(
        design_result=design,
        engine=WAY_FIR_SYNTHESIS_ENGINE,
        design_signature=signature.hexdigest(),
        stage_names=tuple(name for name, _coefficients in stages),
        stage_taps=tuple(int(coefficients.size) for _name, coefficients in stages),
        natural_stage_taps=int(natural_taps),
        output_taps=int(design.fir.size),
        bypassed=bool(bypassed),
        rms_complex_error=rms_error,
        max_complex_error=max_error,
    )


def _validated_stage(stage: WayFIRTargetStage, index: int) -> tuple[str, np.ndarray]:
    values = np.asarray(stage.coefficients, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Way FIR target stages must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(values)):
        raise ValueError("Way FIR target stages must contain only finite coefficients")
    return stage.name.strip() or f"stage-{index + 1}", values.copy()


def _centered_delta(taps: int) -> np.ndarray:
    values = np.zeros(max(1, int(taps)), dtype=np.float64)
    values[(values.size - 1) // 2] = 1.0
    return values


def _is_centered_unity_target(target: np.ndarray) -> bool:
    return bool(np.allclose(target, 1.0 + 0.0j, rtol=0.0, atol=1e-12))
