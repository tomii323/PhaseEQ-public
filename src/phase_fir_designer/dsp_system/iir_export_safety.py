from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..config import IIRFilter, quantize_iir_q
from ..iir import design_iir_sos
from .models import DSPInput, DSPSystem, DSPWay
from .output_pipeline import DSPSystemOutput


IIR_EXPORT_POLE_LIMIT = 0.9999


@dataclass(frozen=True)
class IIRSafetyCorrection:
    location: str
    kind: str
    frequency_hz: float
    gain_db: float
    q_before: float
    q_after: float
    pole_before: float
    pole_after: float


@dataclass(frozen=True)
class IIRSafetyResult:
    system: DSPSystem
    corrections: tuple[IIRSafetyCorrection, ...]
    checked_filters: int
    max_pole_radius: float


def validate_output_iir_for_export(output: DSPSystemOutput) -> tuple[int, float]:
    """Final guard for generated crossover and alignment sections."""

    checked = 0
    maximum = 0.0
    device_rates: dict[str, int] = {}
    for way in output.ways:
        device_rates[way.dsp_id] = int(way.sample_rate)
        for index, item in enumerate(way.iir_filters, 1):
            if not item.enabled:
                continue
            pole = quantized_max_pole_radius(item, way.sample_rate)
            if pole >= IIR_EXPORT_POLE_LIMIT:
                raise ValueError(
                    f"{way.way_name} generated IIR #{index}: pole {pole:.6f} is not export-safe; Export stopped"
                )
            checked += 1
            maximum = max(maximum, pole)
    for source in output.inputs:
        rates = {
            device_rates[binding.dsp_id]
            for binding in output.input_bindings
            if binding.input_id == source.id and binding.dsp_id in device_rates
        }
        for index, item in enumerate(source.iir_filters, 1):
            if not item.enabled:
                continue
            pole = max(quantized_max_pole_radius(item, rate) for rate in rates)
            if pole >= IIR_EXPORT_POLE_LIMIT:
                raise ValueError(
                    f"Input {source.name} IIR #{index}: pole {pole:.6f} is not export-safe; Export stopped"
                )
            checked += 1
            maximum = max(maximum, pole)
    return checked, maximum


def quantized_max_pole_radius(item: IIRFilter, sample_rate: int) -> float:
    """Return the largest pole radius after float32 device-format rounding."""

    sos = np.asarray(design_iir_sos(item, int(sample_rate)), dtype=np.float32).astype(float)
    if not np.all(np.isfinite(sos)):
        raise ValueError("IIR coefficients contain non-finite values after quantization")
    maximum = 0.0
    for section in sos:
        a0 = float(section[3])
        if not np.isfinite(a0) or abs(a0) < 1e-30:
            raise ValueError("IIR denominator is invalid after quantization")
        poles = np.roots([1.0, float(section[4]) / a0, float(section[5]) / a0])
        maximum = max(maximum, float(np.max(np.abs(poles))))
    return maximum


def _stabilize_filter(
    item: IIRFilter,
    sample_rates: tuple[int, ...],
    location: str,
) -> tuple[IIRFilter, IIRSafetyCorrection | None, float]:
    if not item.enabled:
        return item, None, 0.0

    def radius(candidate: IIRFilter) -> float:
        return max(quantized_max_pole_radius(candidate, rate) for rate in sample_rates)

    before = radius(item)
    if before < IIR_EXPORT_POLE_LIMIT:
        return item, None, before
    if (
        item.kind in {"high_pass", "low_pass"}
        or item.allpass_preset != "manual"
        or (item.kind == "allpass" and item.order == 1)
    ):
        raise ValueError(
            f"{location}: {item.kind} pole {before:.6f} is not export-safe; Export stopped"
        )
    q = quantize_iir_q(item.kind, item.q)
    while q > 0.1 + 1e-9:
        q = round(q - 0.1, 1)
        candidate = replace(item, q=q)
        after = radius(candidate)
        if after < IIR_EXPORT_POLE_LIMIT:
            return candidate, IIRSafetyCorrection(
                location=location,
                kind=item.kind,
                frequency_hz=float(item.fc),
                gain_db=float(item.gain_db),
                q_before=float(item.q),
                q_after=float(candidate.q),
                pole_before=before,
                pole_after=after,
            ), after
    raise ValueError(
        f"{location}: lowering Q could not make pole {before:.6f} export-safe; Export stopped"
    )


def stabilize_system_iir_for_export(system: DSPSystem) -> IIRSafetyResult:
    """Correct only Q, preserving filter type, frequency and gain."""

    devices = {item.id: item for item in system.devices}
    checked = 0
    maximum = 0.0
    corrections: list[IIRSafetyCorrection] = []

    input_rates: dict[str, set[int]] = {item.id: set() for item in system.inputs}
    for binding in system.input_bindings:
        device = devices.get(binding.dsp_id)
        if device is not None:
            input_rates.setdefault(binding.input_id, set()).add(int(device.sample_rate))

    new_inputs: list[DSPInput] = []
    for source in system.inputs:
        rates = tuple(sorted(input_rates.get(source.id) or {int(item.sample_rate) for item in system.devices}))
        updated: list[IIRFilter] = []
        for index, item in enumerate(source.iir_filters, 1):
            corrected, report, pole = _stabilize_filter(item, rates, f"Input {source.name} #{index}")
            updated.append(corrected)
            if item.enabled:
                checked += 1
                maximum = max(maximum, pole)
            if report is not None:
                corrections.append(report)
        new_inputs.append(replace(source, iir_filters=tuple(updated)))

    new_ways: list[DSPWay] = []
    for way in system.ways:
        device = devices[way.dsp_id]
        rates = (int(device.sample_rate),)

        def update_group(filters: tuple[IIRFilter, ...], label: str) -> tuple[IIRFilter, ...]:
            nonlocal checked, maximum
            values: list[IIRFilter] = []
            for index, item in enumerate(filters, 1):
                corrected, report, pole = _stabilize_filter(item, rates, f"{way.name} {label} #{index}")
                values.append(corrected)
                if item.enabled:
                    checked += 1
                    maximum = max(maximum, pole)
                if report is not None:
                    corrections.append(report)
            return tuple(values)

        manual = update_group(way.manual_iir_filters, "Manual IIR")
        auto = update_group(way.auto_iir_filters, "Auto IIR")
        alignment = update_group(way.alignment_iir_filters, "Alignment IIR")
        config = way.correction_config
        if config is not None:
            config = replace(config, iir_filters=list(update_group(tuple(config.iir_filters), "Design IIR")))
        new_ways.append(replace(
            way,
            manual_iir_filters=manual,
            auto_iir_filters=auto,
            alignment_iir_filters=alignment,
            correction_config=config,
        ))

    return IIRSafetyResult(
        system=replace(system, inputs=tuple(new_inputs), ways=tuple(new_ways)),
        corrections=tuple(corrections),
        checked_filters=checked,
        max_pole_radius=maximum,
    )
