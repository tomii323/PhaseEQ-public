from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from phase_fir_designer.config import (
    GainLinearFilter,
    GainPEQ,
    GainShelf,
    GainTilt,
    IIRFilter,
    parse_iir_order,
    PhaseAllPass,
    PhasePEQ,
    PhaseShelf,
    PhaseTilt,
    SpeakerResponse,
)
from phase_fir_designer.iir import iir_frequency_response
from phase_fir_designer.phase_curves import (
    allpass_phase_deg,
    gain_linear_filter_db,
    gain_peq_db,
    gain_shelf_db,
    gain_tilt_db,
    peq_phase_deg,
    shelf_phase_deg,
    smooth_total_tilt_curve,
    tilt_phase_deg,
    frequency_axis,
)


TARGET_EDIT_LIST_KEYS = (
    "gain_peq", "gain_shelf", "gain_linear", "gain_tilt",
    "phase_peq", "phase_shelf", "phase_tilt", "phase_allpass", "iir_filters",
)


def normalize_target_edit_payload(payload: Any) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    normalized = {
        key: [dict(item) for item in source.get(key, []) if isinstance(item, dict)]
        for key in TARGET_EDIT_LIST_KEYS
    }
    normalized["gain_tilt_smoothing_oct"] = float(source.get("gain_tilt_smoothing_oct", 1.0 / 3.0))
    normalized["phase_tilt_smoothing_oct"] = float(source.get("phase_tilt_smoothing_oct", 1.0 / 3.0))
    return normalized


def apply_target_edit_payload(
    base_response: np.ndarray,
    frequency_hz: np.ndarray,
    sample_rate_hz: int,
    payload: Any,
) -> np.ndarray:
    """Apply the canonical PhaseEQ Target-edit payload to one complex response."""
    response = np.asarray(base_response, dtype=complex)
    frequency = np.asarray(frequency_hz, dtype=float)
    if response.shape != frequency.shape:
        raise ValueError("Target response and frequency axis must have the same shape")
    edit = normalize_target_edit_payload(payload)
    gain_edit = np.zeros_like(frequency)
    phase_edit = np.zeros_like(frequency)

    gain_peq = [GainPEQ(**_fields(item, "fc", "q", "gain_db", "enabled")) for item in edit["gain_peq"]]
    gain_shelf = [GainShelf(**_fields(item, "mode", "fc", "q", "gain_db", "enabled")) for item in edit["gain_shelf"]]
    gain_linear = [GainLinearFilter(**_fields(item, "mode", "response", "fc", "cycles", "beta", "enabled")) for item in edit["gain_linear"]]
    gain_tilt = [GainTilt(**_fields(item, "f_start", "f_end", "slope_db_per_oct", "mode", "enabled")) for item in edit["gain_tilt"]]
    phase_peq = [PhasePEQ(**_fields(item, "fc", "q", "phase_deg", "enabled")) for item in edit["phase_peq"]]
    phase_shelf = [PhaseShelf(**_fields(item, "mode", "fc", "q", "phase_deg", "enabled")) for item in edit["phase_shelf"]]
    phase_tilt = [PhaseTilt(**_fields(item, "f_start", "f_end", "slope_deg_per_oct", "enabled")) for item in edit["phase_tilt"]]
    phase_allpass = [PhaseAllPass(**_fields(item, "fc", "q", "polarity", "enabled")) for item in edit["phase_allpass"]]
    iir_filters = [
        IIRFilter(
            **_fields(item, "kind", "fc", "q", "gain_db", "family", "enabled", "origin", "allpass_preset", "ripple_db", "stop_db"),
            order=parse_iir_order(item.get("filter_order", item.get("order", 2))),
        )
        for item in edit["iir_filters"]
    ]

    for item in gain_peq:
        if item.enabled:
            gain_edit += gain_peq_db(frequency, item)
    for item in gain_shelf:
        if item.enabled:
            gain_edit += gain_shelf_db(frequency, item)
    for item in gain_linear:
        if item.enabled:
            gain_edit += gain_linear_filter_db(frequency, item)
    for mode in ("high", "low"):
        selected = [item for item in gain_tilt if item.enabled and item.mode == mode]
        if selected:
            total = sum((gain_tilt_db(frequency, replace(item, edge_smooth_oct=0.0)) for item in selected), np.zeros_like(frequency))
            open_high = mode == "high" and any(
                np.isclose(float(item.f_end), 0.0)
                or np.isclose(float(item.f_end), float(sample_rate_hz) / 2.0)
                for item in selected
            )
            gain_edit += smooth_total_tilt_curve(total, frequency, edit["gain_tilt_smoothing_oct"], open_high=open_high)

    for item in phase_peq:
        if item.enabled:
            phase_edit += peq_phase_deg(frequency, item)
    for item in phase_shelf:
        if item.enabled:
            phase_edit += shelf_phase_deg(frequency, item)
    selected_phase_tilt = [item for item in phase_tilt if item.enabled]
    if selected_phase_tilt:
        total = sum((tilt_phase_deg(frequency, replace(item, edge_smooth_oct=0.0)) for item in selected_phase_tilt), np.zeros_like(frequency))
        open_high = any(
            np.isclose(float(item.f_end), 0.0)
            or np.isclose(float(item.f_end), float(sample_rate_hz) / 2.0)
            for item in selected_phase_tilt
        )
        phase_edit += smooth_total_tilt_curve(total, frequency, edit["phase_tilt_smoothing_oct"], open_high=open_high)
    for item in phase_allpass:
        if item.enabled:
            phase_edit += allpass_phase_deg(frequency, item)
    enabled_iir = [item for item in iir_filters if item.enabled]
    if enabled_iir:
        realized = iir_frequency_response(enabled_iir, int(sample_rate_hz), frequency)
        gain_edit += 20.0 * np.log10(np.maximum(np.abs(realized), 1e-12))
        phase_edit += np.rad2deg(np.unwrap(np.angle(realized)))

    return response * 10.0 ** (gain_edit / 20.0) * np.exp(1j * np.deg2rad(phase_edit))


def apply_target_edit_to_response(
    base_target: SpeakerResponse | None,
    *,
    sample_rate_hz: int,
    analysis_fft_size: int,
    payload: Any,
    source_gain_only: bool = False,
) -> SpeakerResponse | None:
    """Materialize one editable Target while preserving its source definition."""
    edit = normalize_target_edit_payload(payload)
    enabled_items = [
        item
        for key in TARGET_EDIT_LIST_KEYS
        for item in edit[key]
        if bool(item.get("enabled", True))
    ]
    if base_target is None and not enabled_items:
        return None
    nyquist = float(sample_rate_hz) / 2.0
    open_high = any(
        key in {"gain_tilt", "phase_tilt"}
        and bool(item.get("enabled", True))
        and (
            np.isclose(float(item.get("f_end", 0.0)), 0.0)
            or np.isclose(float(item.get("f_end", 0.0)), nyquist)
        )
        for key in ("gain_tilt", "phase_tilt")
        for item in edit[key]
    )
    if base_target is None or open_high:
        frequency = frequency_axis(int(sample_rate_hz), int(analysis_fft_size))
    else:
        frequency = np.asarray(base_target.frequency, dtype=float)
        frequency = np.unique(np.clip(frequency[np.isfinite(frequency)], 0.0, nyquist))
        if frequency.size == 0:
            frequency = frequency_axis(int(sample_rate_hz), int(analysis_fft_size))
    if base_target is None:
        base_gain = np.zeros_like(frequency)
        base_phase = np.zeros_like(frequency)
        base_has_phase = False
    else:
        from response_completion import complete_response
        source_frequency = np.asarray(base_target.frequency, dtype=float)
        source_gain = np.asarray(base_target.gain_db, dtype=float)
        base_has_phase = base_target.phase_deg is not None and not source_gain_only
        source_phase = np.asarray(base_target.phase_deg, dtype=float) if base_has_phase else None
        base_gain, completed_phase, _ = complete_response(
            source_frequency, source_gain, source_phase, frequency)
        base_phase = np.zeros_like(frequency) if completed_phase is None else completed_phase
    base_complex = 10.0 ** (base_gain / 20.0) * np.exp(1j * np.deg2rad(base_phase))
    effective = apply_target_edit_payload(
        base_complex, frequency, int(sample_rate_hz), edit,
    )
    has_phase_edit = any(
        bool(item.get("enabled", True))
        for key in ("phase_peq", "phase_shelf", "phase_tilt", "phase_allpass", "iir_filters")
        for item in edit[key]
    )
    return SpeakerResponse(
        frequency=frequency.tolist(),
        gain_db=(20.0 * np.log10(np.maximum(np.abs(effective), 1e-12))).tolist(),
        phase_deg=(
            np.rad2deg(np.unwrap(np.angle(effective))).tolist()
            if base_has_phase or has_phase_edit else None
        ),
    )


def _fields(item: dict[str, Any], *names: str) -> dict[str, Any]:
    defaults = {
        "enabled": True, "fc": 1000.0, "q": 1.0, "gain_db": 0.0,
        "phase_deg": 0.0, "mode": "low", "response": "lr2", "cycles": 4.0,
        "beta": 12.0, "f_start": 100.0, "f_end": 20_000.0,
        "slope_db_per_oct": 0.0, "slope_deg_per_oct": 0.0,
        "polarity": "positive", "kind": "high_pass", "family": "linkwitz_riley",
        "origin": "manual", "allpass_preset": "manual",
        "ripple_db": 0.1, "stop_db": 80.0,
    }
    return {name: item.get(name, defaults[name]) for name in names}
