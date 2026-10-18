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
    # Full topology is present only for NTM4 banks; old LR definitions and
    # their hashes keep the previous serialization unchanged.
    ntm4_topology: dict | None = None

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
        if self.ntm4_topology is not None:
            from .methods import METHOD_DEFINITIONS
            from .ntm4 import validate_lobe_db
            from .topology import ROUTING_VERSION
            topology = self.ntm4_topology
            required = {"way_index", "methods", "frequencies", "overlaps", "lobes"}
            if not isinstance(topology, dict) or set(topology) not in (required, required | {"routing_version"}):
                raise ValueError("Invalid NTM4 topology")
            if "routing_version" in topology and (type(topology["routing_version"]) is not int
                                                 or topology["routing_version"] != ROUTING_VERSION):
                raise ValueError("Unsupported IIR bank routing version")
            count = len(topology["methods"])
            if (not any(METHOD_DEFINITIONS.get(method) and METHOD_DEFINITIONS[method].requires_bank_definition
                        for method in topology["methods"]) or type(topology["way_index"]) is not int
                    or not 0 <= topology["way_index"] <= count
                    or any(len(topology[key]) != count for key in ("frequencies", "overlaps", "lobes"))):
                raise ValueError("NTM4 boundary count does not match Ways")
            from .recipe import METHODS
            if any(method not in METHODS for method in topology["methods"]):
                raise ValueError("Invalid NTM4 bank method")
            previous = 0.0
            for fc, overlap, lobe in zip(topology["frequencies"], topology["overlaps"], topology["lobes"]):
                if not previous < float(fc) < nyquist:
                    raise ValueError("Invalid NTM4 bank crossover frequency")
                lp, hp = crossover_edges_hz(fc, overlap)
                if not 0 < min(lp, hp) <= max(lp, hp) < nyquist:
                    raise ValueError("NTM4 bank cutoff must be within 0..Nyquist")
                validate_lobe_db(lobe)
                previous = float(fc)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.ntm4_topology is None:
            result.pop("ntm4_topology")
        return result

    def definition_hash(self, sample_rate_hz: int) -> str:
        return iir_definition_hash(self.to_dict(), sample_rate_hz)


def iir_definition_hash(definition: dict, sample_rate_hz: int) -> str:
    """Hash the shared definition fields for both publishers and receivers.

    Preserve omitted legacy fields instead of inserting current defaults.
    Transport metadata and SOS are checked by their separate contracts.
    """
    payload = {
        key: value for key, value in definition.items()
        if key in IIRCrossoverConfig.__dataclass_fields__
        and not (key == "ntm4_topology" and value is None)
    }
    payload["coefficient_sample_rate_hz"] = int(sample_rate_hz)
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
    lobe_heights_db: tuple[float, ...] = (),
) -> IIRCrossoverConfig:
    """Resolve exclusive boundary choices and optional NTM4 topology for one Way."""
    if way not in ordered_ways:
        raise ValueError(f"unknown crossover Way: {way}")
    if len(methods) != len(ordered_ways) - 1 or len(overlap_oct) != len(methods):
        raise ValueError("exclusive crossover boundary count does not match Ways")
    if any(method not in {"Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR", "LR2", "LR4", "NTM4", "Through"} for method in methods):
        raise ValueError("unsupported crossover method")
    index = ordered_ways.index(way)
    hp_index = index - 1 if index > 0 else None
    lp_index = index if index < len(ordered_ways) - 1 else None
    hp_method = methods[hp_index] if hp_index is not None else "Kaiser FIR"
    lp_method = methods[lp_index] if lp_index is not None else "Kaiser FIR"
    hp_common = float(crossover_frequencies_hz[hp_index]) if hp_index is not None else 0.0
    lp_common = float(crossover_frequencies_hz[lp_index]) if lp_index is not None else 0.0
    from .methods import boundary_method
    definitions = tuple(boundary_method(method) for method in methods)
    hp_order = boundary_method(hp_method).order
    lp_order = boundary_method(lp_method).order
    hp = crossover_edges_hz(hp_common, overlap_oct[hp_index])[1] if hp_order else 0.0
    lp = crossover_edges_hz(lp_common, overlap_oct[lp_index])[0] if lp_order else 0.0
    topology = None
    if any(method.requires_bank_definition for method in definitions):
        from .ntm4 import DEFAULT_LOBE_DB, validate_lobe_db
        from .topology import ROUTING_VERSION
        lobes = lobe_heights_db or (DEFAULT_LOBE_DB,) * len(methods)
        if len(lobes) != len(methods) or len(crossover_frequencies_hz) != len(methods):
            raise ValueError("NTM4 boundary count does not match Ways")
        lobes = tuple(validate_lobe_db(value) for value in lobes)
        topology = dict(routing_version=ROUTING_VERSION, way_index=index, methods=list(methods),
                        frequencies=list(crossover_frequencies_hz), overlaps=list(overlap_oct),
                        # Inactive controls remain in Studio settings, while
                        # the effective definition matches the portable Recipe.
                        lobes=[value if method == "NTM4" else DEFAULT_LOBE_DB
                               for method, value in zip(methods, lobes)])
    return IIRCrossoverConfig(
        enabled=bool(hp_order or lp_order or topology), family="Linkwitz-Riley",
        order=max(hp_order, lp_order, 2), highpass_hz=hp, lowpass_hz=lp,
        common_highpass_hz=hp_common, common_lowpass_hz=lp_common,
        overlap_oct=float(overlap_oct[hp_index] if hp_order else overlap_oct[lp_index] if lp_order else 0.0),
        lr2_polarity=(
            -1 if lr2_auto_polarity and sum(method == "LR2" for method in methods[:index]) % 2 else 1
        ),
        highpass_order=hp_order, lowpass_order=lp_order,
        ntm4_topology=topology,
    )


def iir_crossover_sos(config: IIRCrossoverConfig, sample_rate_hz: int) -> np.ndarray:
    if not config.enabled:
        return np.empty((0, 6), dtype=float)
    config.validate(sample_rate_hz)
    if config.ntm4_topology is not None:
        return bank_way_sos(**config.ntm4_topology, fs=sample_rate_hz)
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


def bank_way_sos(way_index, methods, frequencies, overlaps, lobes, fs, routing_version=None):
    """Generate direct stages from a routing plan and boundary-owned prototypes."""
    from .methods import boundary_method
    from .topology import ROUTING_VERSION, resolve_split_topology
    if routing_version is not None and routing_version != ROUTING_VERSION:
        raise ValueError('Unsupported IIR bank routing version')
    plan = resolve_split_topology(methods)
    rows = []
    for index in plan.split_boundaries(way_index):
        method = boundary_method(methods[index])
        if method.split_sos is None:
            continue
        side = 'highpass' if index < way_index else 'lowpass'
        cutoff = crossover_edges_hz(frequencies[index], overlaps[index])[1 if side == 'highpass' else 0]
        rows.append(method.split_sos(side, cutoff, fs, lobes[index]))
    return np.vstack(rows) if rows else np.empty((0, 6), dtype=float)


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
