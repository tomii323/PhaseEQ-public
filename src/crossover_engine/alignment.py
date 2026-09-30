"""Shared output-band phase alignment coefficients (no time delay)."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class AllPassSection:
    frequency_hz: float
    q: float
    order: int = 2
    polarity: int = 1

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def allpass_sos(section: AllPassSection, sample_rate_hz: int) -> np.ndarray:
    frequency = float(section.frequency_hz)
    q = float(section.q)
    if not 0.0 < frequency < float(sample_rate_hz) / 2.0:
        raise ValueError("All-pass frequency must be within 0..Nyquist")
    order = int(section.order)
    if order not in {1, 2}:
        raise ValueError("All-pass order must be 1 or 2")
    if int(section.polarity) not in {-1, 1}:
        raise ValueError("All-pass polarity must be +1 or -1")
    if order == 2 and not 0.35 <= q <= 2.0:
        raise ValueError("All-pass Q must be within 0.35..2.0")
    if order == 1:
        tangent = np.tan(np.pi * frequency / float(sample_rate_hz))
        coefficient = (tangent - 1.0) / (tangent + 1.0)
        values = np.asarray([[
            coefficient, 1.0, 0.0, 1.0, coefficient, 0.0,
        ]], dtype=float)
        values[0, :3] *= int(section.polarity)
        return values
    w0 = 2.0 * np.pi * frequency / float(sample_rate_hz)
    alpha = np.sin(w0) / (2.0 * q)
    cosine = np.cos(w0)
    a0 = 1.0 + alpha
    values = np.asarray([[
        (1.0 - alpha) / a0,
        (-2.0 * cosine) / a0,
        (1.0 + alpha) / a0,
        1.0,
        (-2.0 * cosine) / a0,
        (1.0 - alpha) / a0,
    ]], dtype=float)
    values[0, :3] *= int(section.polarity)
    return values


def allpass_response(
    sections: tuple[AllPassSection, ...], frequency_hz: np.ndarray, sample_rate_hz: int,
) -> np.ndarray:
    response = np.ones(np.asarray(frequency_hz).shape, dtype=np.complex128)
    for section in sections:
        _frequency, section_response = signal.sosfreqz(
            allpass_sos(section, sample_rate_hz), worN=frequency_hz, fs=float(sample_rate_hz),
        )
        response *= section_response
    return response


def validate_alignment(value, sample_rate_hz):
    if not isinstance(value, dict) or not isinstance(value.get("acoustic_target"), bool):
        raise ValueError("Phase alignment target selection must be boolean")
    if type(value.get("polarity")) is not int or value["polarity"] not in {-1, 1}:
        raise ValueError("Channel polarity must be +1 or -1")
    if not isinstance(value.get("allpass"), list):
        raise ValueError("Phase alignment sections must be a list")
    for item in value["allpass"]:
        if not isinstance(item, dict) or set(item) != {"frequency_hz", "q", "order", "polarity"}:
            raise ValueError("Invalid phase alignment section")
        if type(item["order"]) is not int or type(item["polarity"]) is not int:
            raise ValueError("All-pass order and polarity must be integers")
        if not np.isfinite(float(item["q"])):
            raise ValueError("All-pass Q must be finite")
        allpass_sos(AllPassSection(**item), sample_rate_hz)


def target_enabled(recipe):
    return bool((recipe or {}).get("phase_alignment", {}).get("acoustic_target", False))


def separate_polarity(recipe):
    from .recipe import POLARITY_ALGORITHM_VERSION
    return (recipe or {}).get("algorithm_version") == POLARITY_ALGORITHM_VERSION


def target_crossover_polarity(recipe):
    """Assign each enabled LR2 boundary sign to its own acoustic target route."""
    if not separate_polarity(recipe) or not recipe.get("lr2_auto_polarity", True):
        return 1
    index = recipe["ordered_ways"].index(recipe["way"])
    count = sum(b["method"] == "LR2" and b.get("acoustic_target", False)
                for b in recipe["boundaries"][:index])
    return -1 if count % 2 else 1


def manual_polarity_sos(recipe):
    """Manual channel polarity is a direct, independent setting in v5."""
    if separate_polarity(recipe) and recipe.get("phase_alignment", {}).get("polarity", 1) == -1:
        return np.array([[-1., 0., 0., 1., 0., 0.]])
    return np.empty((0, 6))


def alignment_sos(recipe, *, target=False):
    """All-pass route; v4 retains its historical manual-polarity behavior."""
    value = (recipe or {}).get("phase_alignment")
    if value is None:
        return np.empty((0, 6))
    validate_alignment(value, recipe["sample_rate_hz"])
    if target_enabled(recipe) != target:
        return np.empty((0, 6))
    rows = [allpass_sos(AllPassSection(**item), recipe["sample_rate_hz"])[0]
            for item in value["allpass"]]
    if not separate_polarity(recipe) and value["polarity"] == -1:
        rows.append(np.array([-1., 0., 0., 1., 0., 0.]))
    return np.asarray(rows, dtype=float).reshape(-1, 6)


def output_polarity(recipe):
    from .recipe import iir_config
    return (1 if target_enabled(recipe) and not separate_polarity(recipe)
            else iir_config(recipe).lr2_polarity)


def output_response(recipe, frequency):
    sos = output_sos(recipe)
    if not sos.size:
        return np.ones(np.asarray(frequency).shape, dtype=np.complex128)
    return signal.sosfreqz(sos, worN=frequency, fs=recipe["sample_rate_hz"])[1]


def output_auxiliary_sos(recipe):
    """Complete direct alignment, including LR2 polarity for IIR-only DSPs."""
    sos = np.concatenate((alignment_sos(recipe), manual_polarity_sos(recipe)))
    if recipe is not None and output_polarity(recipe) == -1:
        sos = np.concatenate((sos, np.array([[-1., 0., 0., 1., 0., 0.]])))
    return sos


def output_sos(recipe):
    """HP, LP, All Pass and all direct polarities; no FIR approximation."""
    from .recipe import iir_config
    from .iir import iir_crossover_sos
    return np.concatenate((iir_crossover_sos(iir_config(recipe), recipe["sample_rate_hz"]),
                           output_auxiliary_sos(recipe)))
