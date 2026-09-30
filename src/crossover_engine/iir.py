from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class IIRCrossoverConfig:
    enabled: bool = False
    family: str = "Linkwitz-Riley"
    order: int = 4
    highpass_hz: float = 0.0
    lowpass_hz: float = 0.0
    common_highpass_hz: float = 0.0
    common_lowpass_hz: float = 0.0
    overlap_oct: float = 0.0
    lr2_polarity: int = 1
    highpass_order: int = 0
    lowpass_order: int = 0

    def validate(self, sample_rate_hz: int) -> None:
        if self.family not in {"Linkwitz-Riley", "Butterworth", "Bessel"}:
            raise ValueError(f"unsupported IIR crossover family: {self.family}")
        if int(self.order) not in {2, 4, 6, 8}:
            raise ValueError("IIR crossover order must be 2, 4, 6 or 8")
        if self.family == "Linkwitz-Riley" and int(self.order) % 2:
            raise ValueError("Linkwitz-Riley order must be even")
        nyquist = float(sample_rate_hz) / 2.0
        for name, value in (("highpass", self.highpass_hz), ("lowpass", self.lowpass_hz)):
            if float(value) < 0.0 or float(value) >= nyquist:
                raise ValueError(f"{name} cutoff must be in 0..Nyquist")
        if self.highpass_hz > 0.0 and self.lowpass_hz > 0.0 and self.highpass_hz >= self.lowpass_hz:
            raise ValueError("IIR crossover highpass must be below lowpass")
        if not np.isfinite(float(self.overlap_oct)) or not -1.0 <= float(self.overlap_oct) <= 1.0:
            raise ValueError("IIR crossover overlap must be within -1..+1 oct")
        if int(self.lr2_polarity) not in {-1, 1}:
            raise ValueError("LR2 crossover polarity must be +1 or -1")
        if int(self.highpass_order) not in {0, 2, 4} or int(self.lowpass_order) not in {0, 2, 4}:
            raise ValueError("exclusive IIR boundary order must be 0, 2 or 4")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def definition_hash(self, sample_rate_hz: int) -> str:
        payload = {**self.to_dict(), "coefficient_sample_rate_hz": int(sample_rate_hz)}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def crossover_edges_hz(common_hz: float, overlap_oct: float) -> tuple[float, float]:
    """Return (LP, HP) edges symmetric on a logarithmic frequency axis."""
    frequency = float(common_hz)
    overlap = float(overlap_oct)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("common crossover frequency must be positive")
    if not np.isfinite(overlap) or not -1.0 <= overlap <= 1.0:
        raise ValueError("crossover overlap must be within -1..+1 oct")
    from crossover_engine.filters import kaiser_overlap_edges_hz
    return kaiser_overlap_edges_hz(frequency, overlap)


def realized_iir_crossover_config(
    *, way: str, ordered_ways: tuple[str, ...], crossover_frequencies_hz: tuple[float, ...],
    enabled: bool, order: int, overlap_oct: tuple[float, ...], lr2_auto_polarity: bool = True,
) -> IIRCrossoverConfig:
    """Resolve shared boundary controls into one immutable per-Way LR config."""
    if way not in ordered_ways:
        raise ValueError(f"unknown crossover Way: {way}")
    if len(crossover_frequencies_hz) != len(ordered_ways) - 1 or len(overlap_oct) != len(crossover_frequencies_hz):
        raise ValueError("crossover boundary count does not match Ways")
    index = ordered_ways.index(way)
    hp_common = 0.0 if index == 0 else float(crossover_frequencies_hz[index - 1])
    lp_common = 0.0 if index == len(ordered_ways) - 1 else float(crossover_frequencies_hz[index])
    hp = crossover_edges_hz(hp_common, overlap_oct[index - 1])[1] if hp_common else 0.0
    lp = crossover_edges_hz(lp_common, overlap_oct[index])[0] if lp_common else 0.0
    # LR2 adjacent outputs have opposite phase. Keep this compensation separate
    # from the user Channel polarity so it cannot be applied twice.
    lr2_polarity = -1 if enabled and int(order) == 2 and lr2_auto_polarity and index % 2 else 1
    return IIRCrossoverConfig(
        enabled=bool(enabled), family="Linkwitz-Riley", order=int(order),
        highpass_hz=hp, lowpass_hz=lp,
        common_highpass_hz=hp_common, common_lowpass_hz=lp_common,
        overlap_oct=float(overlap_oct[index - 1] if hp_common else overlap_oct[index] if lp_common else 0.0),
        lr2_polarity=lr2_polarity,
    )


def realized_exclusive_iir_config(
    *, way: str, ordered_ways: tuple[str, ...],
    crossover_frequencies_hz: tuple[float, ...], methods: tuple[str, ...],
    overlap_oct: tuple[float, ...], lr2_auto_polarity: bool = True,
) -> IIRCrossoverConfig:
    """Resolve exclusive per-boundary Kaiser/LR2/LR4 choices for one Way."""
    if way not in ordered_ways:
        raise ValueError(f"unknown crossover Way: {way}")
    if len(methods) != len(ordered_ways) - 1 or len(overlap_oct) != len(methods):
        raise ValueError("exclusive crossover boundary count does not match Ways")
    if any(method not in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR", "LR2", "LR4", "Through"} for method in methods):
        raise ValueError("unsupported crossover method")
    index = ordered_ways.index(way)
    hp_index = index - 1 if index > 0 else None
    lp_index = index if index < len(ordered_ways) - 1 else None
    hp_method = methods[hp_index] if hp_index is not None else "Kaiser FIR"
    lp_method = methods[lp_index] if lp_index is not None else "Kaiser FIR"
    hp_common = float(crossover_frequencies_hz[hp_index]) if hp_index is not None else 0.0
    lp_common = float(crossover_frequencies_hz[lp_index]) if lp_index is not None else 0.0
    hp_order = int(hp_method[2:]) if hp_method.startswith("LR") else 0
    lp_order = int(lp_method[2:]) if lp_method.startswith("LR") else 0
    hp = crossover_edges_hz(hp_common, overlap_oct[hp_index])[1] if hp_order else 0.0
    lp = crossover_edges_hz(lp_common, overlap_oct[lp_index])[0] if lp_order else 0.0
    return IIRCrossoverConfig(
        enabled=bool(hp_order or lp_order), family="Linkwitz-Riley",
        order=max(hp_order, lp_order, 2), highpass_hz=hp, lowpass_hz=lp,
        common_highpass_hz=hp_common, common_lowpass_hz=lp_common,
        overlap_oct=float(overlap_oct[hp_index] if hp_order else overlap_oct[lp_index] if lp_order else 0.0),
        lr2_polarity=(
            -1 if lr2_auto_polarity and sum(method == "LR2" for method in methods[:index]) % 2 else 1
        ),
        highpass_order=hp_order, lowpass_order=lp_order,
    )


def iir_crossover_sos(config: IIRCrossoverConfig, sample_rate_hz: int) -> np.ndarray:
    if not config.enabled:
        return np.empty((0, 6), dtype=float)
    config.validate(sample_rate_hz)
    sections: list[np.ndarray] = []
    for kind, cutoff, boundary_order in (
        ("highpass", config.highpass_hz, config.highpass_order),
        ("lowpass", config.lowpass_hz, config.lowpass_order),
    ):
        if float(cutoff) <= 0.0:
            continue
        design_order = int(boundary_order) or int(config.order)
        if config.family == "Linkwitz-Riley":
            base_order = max(1, design_order // 2)
            base = signal.butter(base_order, float(cutoff), btype=kind, fs=sample_rate_hz, output="sos")
            sections.extend((base, base.copy()))
        elif config.family == "Butterworth":
            sections.append(signal.butter(design_order, float(cutoff), btype=kind, fs=sample_rate_hz, output="sos"))
        else:
            sections.append(signal.bessel(design_order, float(cutoff), btype=kind, fs=sample_rate_hz, output="sos", norm="phase"))
    return np.vstack(sections) if sections else np.empty((0, 6), dtype=float)


def iir_crossover_response(
    config: IIRCrossoverConfig,
    frequency_hz: np.ndarray,
    sample_rate_hz: int,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=float)
    sos = iir_crossover_sos(config, sample_rate_hz)
    if sos.size == 0:
        return np.ones(frequency.shape, dtype=np.complex128)
    _frequency, response = signal.sosfreqz(sos, worN=frequency, fs=float(sample_rate_hz))
    return np.asarray(response, dtype=np.complex128)


def sos_is_stable(sos: np.ndarray, *, radius_limit: float = 1.0) -> bool:
    values = np.asarray(sos, dtype=float)
    for section in values:
        poles = np.roots(section[3:])
        if np.any(np.abs(poles) >= float(radius_limit)):
            return False
    return True
