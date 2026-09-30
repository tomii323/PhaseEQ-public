from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FIRStage:
    name: str
    coefficients: np.ndarray
    enabled: bool = True


@dataclass(frozen=True)
class FIRChainComposition:
    natural_fir: np.ndarray
    output_fir: np.ndarray
    stage_names: tuple[str, ...]
    stage_taps: tuple[int, ...]
    natural_taps: int
    output_taps: int
    cropped: bool
    padded: bool
    retained_energy_ratio: float


def compose_fir_chain(
    stages: Sequence[FIRStage],
    *,
    target_taps: int | None = None,
) -> FIRChainComposition:
    """Convolve every enabled FIR stage and fit the tap budget only once."""

    active: list[tuple[str, np.ndarray]] = []
    for index, stage in enumerate(stages):
        if not stage.enabled:
            continue
        coefficients = _validated_fir(stage.coefficients, stage.name or f"stage-{index + 1}")
        active.append((stage.name or f"stage-{index + 1}", coefficients))

    natural = np.asarray([1.0], dtype=np.float64)
    for _, coefficients in active:
        natural = signal.fftconvolve(natural, coefficients, mode="full").astype(
            np.float64,
            copy=False,
        )
    if not np.all(np.isfinite(natural)):
        raise ValueError("composed FIR contains non-finite coefficients")

    requested_taps = natural.size if target_taps is None else int(target_taps)
    if requested_taps < 1:
        raise ValueError("target_taps must be a positive integer")
    output = center_fit_fir(natural, requested_taps)
    natural_energy = float(np.dot(natural, natural))
    output_energy = float(np.dot(output, output))
    retained = 1.0 if natural_energy <= 0.0 else min(1.0, output_energy / natural_energy)
    return FIRChainComposition(
        natural_fir=natural.copy(),
        output_fir=output,
        stage_names=tuple(name for name, _ in active),
        stage_taps=tuple(int(coefficients.size) for _, coefficients in active),
        natural_taps=int(natural.size),
        output_taps=int(output.size),
        cropped=output.size < natural.size,
        padded=output.size > natural.size,
        retained_energy_ratio=float(retained),
    )


def center_fit_fir(fir: np.ndarray, target_taps: int) -> np.ndarray:
    """Center crop or pad an FIR without changing its coefficient scale."""

    source = _validated_fir(fir, "FIR")
    size = int(target_taps)
    if size < 1:
        raise ValueError("target_taps must be a positive integer")
    if source.size == size:
        return source.copy()
    if source.size < size:
        padding = size - source.size
        left = padding // 2
        return np.pad(source, (left, padding - left))
    start = (source.size - size) // 2
    return source[start : start + size].copy()


def _validated_fir(coefficients: np.ndarray, name: str) -> np.ndarray:
    output = np.asarray(coefficients, dtype=np.float64)
    if output.ndim != 1 or output.size == 0:
        raise ValueError(f"{name} FIR must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} FIR must contain only finite coefficients")
    return output.copy()
