from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import CompositeResult
from ..fft import unwrapped_phase_on_reliable_branch


@dataclass(frozen=True)
class GraphSeries:
    name: str
    x: np.ndarray
    y: np.ndarray
    kind: str
    series_id: str = ""
    role: str = "Response"
    mask_response: np.ndarray | None = None


class GraphBuilder:
    """Pure projection of computed engine results; performs no DSP mutation."""

    def build(
        self,
        result: CompositeResult,
        kind: str,
        *,
        include_individual: bool = True,
        include_target: bool = True,
        wrapped_phase: bool = True,
    ) -> tuple[GraphSeries, ...]:
        normalized = kind.strip().lower().replace(" ", "_")
        if normalized == "magnitude":
            series = [self._sum_series(result, kind, result.magnitude_db)]
            if include_individual:
                series[:0] = [
                    self._channel_series(
                        name, result, kind,
                        20.0 * np.log10(np.maximum(np.abs(value), 1e-12)), value,
                    )
                    for name, value in result.channel_responses.items()
                ]
            if include_target and result.target_response is not None:
                series.append(self._target_series(
                    result, kind,
                    20.0 * np.log10(np.maximum(np.abs(result.target_response), 1e-12)),
                ))
            return tuple(series)
        if normalized == "phase":
            composite_phase = _sum_unwrapped_phase_deg(result)
            if wrapped_phase:
                composite_phase = _wrap_phase_deg(composite_phase)
            series = [self._sum_series(result, kind, composite_phase)]
            if include_individual:
                channel_series: list[GraphSeries] = []
                for name, value in result.channel_responses.items():
                    phase_values = (
                        np.asarray(result.channel_unwrapped_phase_deg[name], dtype=float)
                        if result.channel_unwrapped_phase_deg is not None
                        and name in result.channel_unwrapped_phase_deg
                        else np.rad2deg(unwrapped_phase_on_reliable_branch(value))
                    )
                    channel_series.append(self._channel_series(
                        name, result, kind,
                        _wrap_phase_deg(phase_values) if wrapped_phase else phase_values,
                        value,
                    ))
                series[:0] = channel_series
            if include_target and result.target_response is not None and result.target_phase_available:
                target_phase = np.rad2deg(
                    unwrapped_phase_on_reliable_branch(result.target_response)
                )
                if wrapped_phase:
                    target_phase = _wrap_phase_deg(target_phase)
                series.append(self._target_series(result, kind, target_phase))
            return tuple(series)
        if normalized == "group_delay":
            series = [self._sum_series(result, kind, result.group_delay_ms)]
            if include_individual and result.channel_group_delay_ms:
                series[:0] = [
                    self._channel_series(name, result, kind, value, result.channel_responses[name])
                    for name, value in result.channel_group_delay_ms.items()
                    if name in result.channel_responses
                ]
            return tuple(series)
        if normalized in {"impulse", "step"}:
            time_ms = (
                np.arange(result.impulse_response.size, dtype=float)
                - (result.impulse_response.size - 1) / 2.0
            ) / result.sample_rate_hz * 1000.0
            sum_values = result.impulse_response if normalized == "impulse" else np.cumsum(result.impulse_response)
            series = [self._sum_series(result, kind, sum_values, x=time_ms)]
            channel_values = result.channel_impulse_responses if normalized == "impulse" else result.channel_step_responses
            if include_individual and channel_values:
                series[:0] = [
                    self._channel_series(
                        name, result, kind, value, result.channel_responses[name], x=time_ms,
                    )
                    for name, value in channel_values.items()
                    if name in result.channel_responses
                ]
            return tuple(series)
        if normalized == "composite":
            return (self._sum_series(result, "Magnitude", result.magnitude_db),)
        if normalized == "individual":
            return tuple(
                self._channel_series(
                    name, result, "Magnitude",
                    20.0 * np.log10(np.maximum(np.abs(value), 1e-12)), value,
                )
                for name, value in result.channel_responses.items()
            )
        if normalized == "target" and result.target_response is not None:
            return (self._target_series(
                result, kind,
                20.0 * np.log10(np.maximum(np.abs(result.target_response), 1e-12)),
            ),)
        raise ValueError(f"unsupported graph kind: {kind}")

    @staticmethod
    def _channel_series(
        name: str, result: CompositeResult, kind: str, values: np.ndarray,
        response: np.ndarray, *, x: np.ndarray | None = None,
    ) -> GraphSeries:
        return GraphSeries(
            name, result.frequency_hz if x is None else x,
            np.asarray(values, dtype=float), kind,
            series_id=f"channel:{name}", role="Channel",
            mask_response=np.asarray(response, dtype=complex),
        )

    @staticmethod
    def _sum_series(
        result: CompositeResult, kind: str, values: np.ndarray,
        *, x: np.ndarray | None = None,
    ) -> GraphSeries:
        return GraphSeries(
            "Composite", result.frequency_hz if x is None else x,
            np.asarray(values, dtype=float), kind,
            series_id=f"sum:{result.group}", role="SystemSum",
            mask_response=np.asarray(result.complex_response, dtype=complex),
        )

    @staticmethod
    def _target_series(result: CompositeResult, kind: str, values: np.ndarray) -> GraphSeries:
        assert result.target_response is not None
        return GraphSeries(
            result.target_name or "Group Target", result.frequency_hz,
            np.asarray(values, dtype=float), kind,
            series_id=f"target:{result.group}", role="Target",
            mask_response=np.asarray(result.target_response, dtype=complex),
        )


def phase_gain_masked_values(
    kind: str,
    values: np.ndarray,
    response: np.ndarray | None,
    gain_mask_db: float,
) -> np.ndarray:
    """Mask phase-derived display values where response gain is too low."""
    output = np.asarray(values, dtype=float)
    normalized = kind.strip().casefold().replace("_", " ")
    if normalized not in {"phase", "group delay"} or response is None:
        return output
    gain_db = 20.0 * np.log10(
        np.maximum(np.abs(np.asarray(response, dtype=complex)), 1e-12)
    )
    return np.where(gain_db >= float(gain_mask_db), output, np.nan)


def _wrap_phase_deg(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _sum_unwrapped_phase_deg(result: CompositeResult) -> np.ndarray:
    """Place System Sum on the same display branch as its dominant Channel.

    The Complex Sum already contains every applied Delay.  This function only
    chooses an equivalent integer-360-degree display branch; it never changes
    the response or its phase slope.  Without this final branch join, Channel
    phases preserved from PhaseEQ FRD data can be shown several turns away
    from a Sum independently anchored by ``angle(sum)``.
    """
    phase = np.rad2deg(
        unwrapped_phase_on_reliable_branch(result.complex_response)
    )
    channel_phases = result.channel_unwrapped_phase_deg or {}
    candidates = [
        name for name in result.channel_responses
        if name in channel_phases
        and np.asarray(channel_phases[name]).shape == phase.shape
    ]
    if not candidates or phase.size == 0:
        return phase
    sum_magnitude = np.abs(np.asarray(result.complex_response, dtype=complex))
    finite = np.isfinite(phase) & np.isfinite(sum_magnitude)
    if not np.any(finite):
        return phase
    finite_indices = np.flatnonzero(finite)
    reference_index = int(finite_indices[np.argmax(sum_magnitude[finite])])
    dominant = max(
        candidates,
        key=lambda name: float(abs(result.channel_responses[name][reference_index])),
    )
    channel_phase = float(np.asarray(channel_phases[dominant])[reference_index])
    if not np.isfinite(channel_phase):
        return phase
    branch_shift = 360.0 * round(
        (channel_phase - float(phase[reference_index])) / 360.0
    )
    return phase + branch_shift


__all__ = ["GraphBuilder", "GraphSeries", "phase_gain_masked_values"]
