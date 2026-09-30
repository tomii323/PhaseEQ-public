"""Matched all-pass biquads for Linkwitz-Riley crossover measurements.

The generated cascade has unity magnitude and the same phase as the selected
electrical Linkwitz-Riley branch.  Coefficients use the normalized difference
equation convention used throughout PhaseEQ::

    H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)

The module is intentionally independent from the DSP System data model so it
can be used by measurement tools and standalone coefficient exporters.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal


MatchedAllPassWay = Literal["lo", "hi", "mid"]
SUPPORTED_LR_ORDERS = (2, 4, 8)


@dataclass(frozen=True)
class BiquadCoefficients:
    """One normalized DSP biquad section."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float

    def as_sos_row(self) -> np.ndarray:
        return np.asarray(
            [self.b0, self.b1, self.b2, 1.0, self.a1, self.a2],
            dtype=np.float64,
        )

    def validate(self) -> None:
        row = self.as_sos_row()
        if not np.all(np.isfinite(row)):
            raise ValueError("matched all-pass coefficients must be finite")
        poles = np.roots([1.0, self.a1, self.a2])
        if np.any(np.abs(poles) >= 1.0 - 1e-10):
            raise ValueError("matched all-pass biquad is unstable")


@dataclass(frozen=True)
class MatchedAllPassSpec:
    """Specification for one Lo, Hi, or Mid measurement path.

    ``lower_crossover_hz`` is the high-pass boundary of Hi/Mid.
    ``upper_crossover_hz`` is the low-pass boundary of Lo/Mid.
    ``polarity_inverted`` describes a final output polarity inversion that
    must also be present in the phase-matched measurement path.
    """

    way: MatchedAllPassWay
    sample_rate: int
    lower_crossover_hz: float | None = None
    upper_crossover_hz: float | None = None
    lower_order: int = 4
    upper_order: int = 4
    polarity_inverted: bool = False


@dataclass(frozen=True)
class MatchedAllPassDesign:
    """DSP-ready matched all-pass design and its provenance."""

    spec: MatchedAllPassSpec
    sections: tuple[BiquadCoefficients, ...]
    intrinsic_polarity: int
    output_polarity: int

    @property
    def sos(self) -> np.ndarray:
        if not self.sections:
            return np.empty((0, 6), dtype=np.float64)
        return np.vstack([section.as_sos_row() for section in self.sections])

    def frequency_response(self, frequency_hz: np.ndarray) -> np.ndarray:
        frequency = np.asarray(frequency_hz, dtype=float)
        if frequency.ndim != 1:
            raise ValueError("frequency_hz must be one-dimensional")
        if np.any(~np.isfinite(frequency)):
            raise ValueError("frequency_hz must contain finite values")
        if np.any((frequency < 0.0) | (frequency > self.spec.sample_rate / 2.0)):
            raise ValueError("frequency_hz must lie between 0 Hz and Nyquist")
        _frequency, response = signal.sosfreqz(
            self.sos,
            worN=frequency,
            fs=self.spec.sample_rate,
        )
        return np.asarray(response, dtype=complex)


def _validate_order(order: int, *, name: str) -> int:
    value = int(order)
    if value not in SUPPORTED_LR_ORDERS:
        raise ValueError(f"{name} must be one of {SUPPORTED_LR_ORDERS}")
    return value


def _validate_cutoff(cutoff_hz: float | None, sample_rate: int, *, name: str) -> float:
    if cutoff_hz is None:
        raise ValueError(f"{name} is required")
    value = float(cutoff_hz)
    if not np.isfinite(value) or not 0.0 < value < sample_rate / 2.0:
        raise ValueError(f"{name} must lie between 0 Hz and Nyquist")
    return value


def _normalize_way(way: str) -> MatchedAllPassWay:
    value = str(way).strip().lower()
    aliases = {"low": "lo", "high": "hi", "band": "mid"}
    value = aliases.get(value, value)
    if value not in {"lo", "hi", "mid"}:
        raise ValueError("way must be 'lo', 'hi', or 'mid'")
    return value  # type: ignore[return-value]


def _edge_sections(sample_rate: int, cutoff_hz: float, order: int) -> list[BiquadCoefficients]:
    """Return the reduced-order all-pass of a squared Butterworth LR edge."""

    prototype = signal.butter(
        order // 2,
        cutoff_hz,
        btype="lowpass",
        fs=sample_rate,
        output="sos",
    )
    sections: list[BiquadCoefficients] = []
    for row in np.asarray(prototype, dtype=float):
        a0 = float(row[3])
        if not np.isfinite(a0) or abs(a0) < 1e-15:
            raise ValueError("Linkwitz-Riley prototype has an invalid a0 coefficient")
        a1 = float(row[4] / a0)
        a2 = float(row[5] / a0)
        if abs(a2) < 1e-14:
            # First-order all-pass: (a1 + z^-1) / (1 + a1 z^-1).
            section = BiquadCoefficients(a1, 1.0, 0.0, a1, 0.0)
        else:
            # Second-order all-pass: reversed denominator over denominator.
            section = BiquadCoefficients(a2, a1, 1.0, a1, a2)
        section.validate()
        sections.append(section)
    return sections


def _apply_polarity(
    sections: list[BiquadCoefficients],
    polarity: int,
) -> list[BiquadCoefficients]:
    if polarity not in {-1, 1}:
        raise ValueError("polarity must be -1 or +1")
    if polarity > 0:
        return sections
    first, *remainder = sections
    inverted = BiquadCoefficients(
        -first.b0,
        -first.b1,
        -first.b2,
        first.a1,
        first.a2,
    )
    inverted.validate()
    return [inverted, *remainder]


def design_matched_allpass(spec: MatchedAllPassSpec) -> MatchedAllPassDesign:
    """Generate DSP biquads matching the selected LR2/LR4/LR8 branch phase."""

    way = _normalize_way(spec.way)
    sample_rate = int(spec.sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    sections: list[BiquadCoefficients] = []
    intrinsic_polarity = 1
    normalized_spec = MatchedAllPassSpec(
        way=way,
        sample_rate=sample_rate,
        lower_crossover_hz=spec.lower_crossover_hz,
        upper_crossover_hz=spec.upper_crossover_hz,
        lower_order=int(spec.lower_order),
        upper_order=int(spec.upper_order),
        polarity_inverted=bool(spec.polarity_inverted),
    )

    if way in {"hi", "mid"}:
        lower_frequency = _validate_cutoff(
            normalized_spec.lower_crossover_hz,
            sample_rate,
            name="lower_crossover_hz",
        )
        lower_order = _validate_order(normalized_spec.lower_order, name="lower_order")
        sections.extend(_edge_sections(sample_rate, lower_frequency, lower_order))
        # LR2 HP is 180 degrees from its LP-derived all-pass. LR4/LR8 are not.
        if lower_order == 2:
            intrinsic_polarity *= -1

    if way in {"lo", "mid"}:
        upper_frequency = _validate_cutoff(
            normalized_spec.upper_crossover_hz,
            sample_rate,
            name="upper_crossover_hz",
        )
        upper_order = _validate_order(normalized_spec.upper_order, name="upper_order")
        sections.extend(_edge_sections(sample_rate, upper_frequency, upper_order))

    if way == "mid":
        assert normalized_spec.lower_crossover_hz is not None
        assert normalized_spec.upper_crossover_hz is not None
        if float(normalized_spec.lower_crossover_hz) >= float(normalized_spec.upper_crossover_hz):
            raise ValueError("Mid lower_crossover_hz must be below upper_crossover_hz")

    output_polarity = intrinsic_polarity * (-1 if normalized_spec.polarity_inverted else 1)
    sections = _apply_polarity(sections, output_polarity)
    design = MatchedAllPassDesign(
        spec=normalized_spec,
        sections=tuple(sections),
        intrinsic_polarity=intrinsic_polarity,
        output_polarity=output_polarity,
    )
    magnitude = np.abs(
        design.frequency_response(
            np.linspace(0.0, sample_rate / 2.0, 2049, dtype=float),
        )
    )
    if float(np.max(np.abs(magnitude - 1.0))) > 1e-8:
        raise ValueError("generated matched all-pass response is not unity magnitude")
    return design


def design_lo_matched_allpass(
    sample_rate: int,
    cutoff_hz: float,
    order: int = 4,
    *,
    polarity_inverted: bool = False,
) -> MatchedAllPassDesign:
    return design_matched_allpass(
        MatchedAllPassSpec(
            way="lo",
            sample_rate=sample_rate,
            upper_crossover_hz=cutoff_hz,
            upper_order=order,
            polarity_inverted=polarity_inverted,
        )
    )


def design_hi_matched_allpass(
    sample_rate: int,
    cutoff_hz: float,
    order: int = 4,
    *,
    polarity_inverted: bool = False,
) -> MatchedAllPassDesign:
    return design_matched_allpass(
        MatchedAllPassSpec(
            way="hi",
            sample_rate=sample_rate,
            lower_crossover_hz=cutoff_hz,
            lower_order=order,
            polarity_inverted=polarity_inverted,
        )
    )


def design_mid_matched_allpass(
    sample_rate: int,
    lower_crossover_hz: float,
    upper_crossover_hz: float,
    *,
    lower_order: int = 4,
    upper_order: int = 4,
    polarity_inverted: bool = False,
) -> MatchedAllPassDesign:
    return design_matched_allpass(
        MatchedAllPassSpec(
            way="mid",
            sample_rate=sample_rate,
            lower_crossover_hz=lower_crossover_hz,
            upper_crossover_hz=upper_crossover_hz,
            lower_order=lower_order,
            upper_order=upper_order,
            polarity_inverted=polarity_inverted,
        )
    )


def biquad_rows(design: MatchedAllPassDesign) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for index, section in enumerate(design.sections, start=1):
        rows.append(
            {
                "section": index,
                "b0": section.b0,
                "b1": section.b1,
                "b2": section.b2,
                "a0": 1.0,
                "a1": section.a1,
                "a2": section.a2,
            }
        )
    return rows


def export_biquad_csv(design: MatchedAllPassDesign) -> str:
    output = io.StringIO()
    columns = ("section", "b0", "b1", "b2", "a0", "a1", "a2")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(biquad_rows(design))
    return output.getvalue()


def export_biquad_json(design: MatchedAllPassDesign) -> str:
    payload = {
        "filter": "Matched All Pass",
        "way": design.spec.way,
        "sample_rate": design.spec.sample_rate,
        "lower_crossover_hz": design.spec.lower_crossover_hz,
        "upper_crossover_hz": design.spec.upper_crossover_hz,
        "lower_order": design.spec.lower_order,
        "upper_order": design.spec.upper_order,
        "polarity_inverted": design.spec.polarity_inverted,
        "intrinsic_polarity": design.intrinsic_polarity,
        "output_polarity": design.output_polarity,
        "transfer_function": "(b0+b1*z^-1+b2*z^-2)/(1+a1*z^-1+a2*z^-2)",
        "sections": biquad_rows(design),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_minidsp_biquads(design: MatchedAllPassDesign) -> str:
    """Return miniDSP Advanced Biquad text with negated feedback signs."""

    lines: list[str] = []
    for row in biquad_rows(design):
        lines.extend(
            [
                f"biquad{row['section']},",
                f"b0={float(row['b0']):.16g},",
                f"b1={float(row['b1']):.16g},",
                f"b2={float(row['b2']):.16g},",
                f"a1={-float(row['a1']):.16g},",
                f"a2={-float(row['a2']):.16g}",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def export_sigmastudio_biquads(design: MatchedAllPassDesign) -> str:
    """Return normalized values for SigmaStudio's General 2nd-Order UI."""

    lines = [
        "# PhaseEQ Matched All Pass / SigmaStudio General 2nd-Order",
        "# Order per section: B0, B1, B2, A1, A2",
        "# A values use H(z)=B(z)/(1+A1*z^-1+A2*z^-2); not a raw DSP RAM image.",
    ]
    for row in biquad_rows(design):
        lines.append(f"# section {row['section']}")
        lines.extend(f"{float(row[name]):.16g}" for name in ("b0", "b1", "b2", "a1", "a2"))
    return "\n".join(lines) + "\n"


def export_camilladsp_biquads(design: MatchedAllPassDesign) -> str:
    """Return CamillaDSP DiffEq filters in cascade order."""

    lines = ["# PhaseEQ Matched All Pass / CamillaDSP", "filters:"]
    names: list[str] = []
    for row in biquad_rows(design):
        name = f"phaseeq_matched_ap_{int(row['section'])}"
        names.append(name)
        lines.extend(
            [
                f"  {name}:",
                "    type: DiffEq",
                "    parameters:",
                f"      a: [1.0, {float(row['a1']):.16g}, {float(row['a2']):.16g}]",
                f"      b: [{float(row['b0']):.16g}, {float(row['b1']):.16g}, {float(row['b2']):.16g}]",
            ]
        )
    lines.extend(["# Apply in this order:", f"# names: [{', '.join(names)}]"])
    return "\n".join(lines) + "\n"


__all__ = [
    "SUPPORTED_LR_ORDERS",
    "BiquadCoefficients",
    "MatchedAllPassDesign",
    "MatchedAllPassSpec",
    "MatchedAllPassWay",
    "biquad_rows",
    "design_hi_matched_allpass",
    "design_lo_matched_allpass",
    "design_matched_allpass",
    "design_mid_matched_allpass",
    "export_biquad_csv",
    "export_biquad_json",
    "export_camilladsp_biquads",
    "export_minidsp_biquads",
    "export_sigmastudio_biquads",
]
