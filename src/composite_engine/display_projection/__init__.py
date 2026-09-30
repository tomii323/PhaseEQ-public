from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping

import numpy as np

from ..core import CompositeResult, MultichannelCompositeResult, result_from_responses
from ..fft import unwrapped_phase_on_reliable_branch
from ..graph import GraphBuilder, GraphSeries
from ..validation import CompositeValidationError, ValidationIssue


DISPLAY_MAIN = "Main"
DISPLAY_LEFT = "L"
DISPLAY_RIGHT = "R"
DISPLAY_STEREO = "L+R"


def normalize_display_value(value: object, *, stereo: bool) -> str:
    """Normalize current and legacy display-only values without creating a Group."""
    normalized = str(value or "").strip()
    aliases = {
        "Left": DISPLAY_LEFT,
        "Right": DISPLAY_RIGHT,
        "Stereo": DISPLAY_STEREO,
        "SUB": DISPLAY_STEREO if stereo else DISPLAY_MAIN,
        "Sub": DISPLAY_STEREO if stereo else DISPLAY_MAIN,
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {DISPLAY_LEFT, DISPLAY_RIGHT, DISPLAY_STEREO} if stereo else {DISPLAY_MAIN}
    return normalized if normalized in allowed else (DISPLAY_STEREO if stereo else DISPLAY_MAIN)


@dataclass(frozen=True)
class DisplayChannelSource:
    series_id: str
    display_name: str
    physical_group: str
    channel_id: str
    response: np.ndarray
    continuous_phase_deg: np.ndarray


@dataclass(frozen=True)
class DisplaySumSource:
    series_id: str
    display_name: str
    physical_group: str
    result: CompositeResult


@dataclass(frozen=True)
class DisplayTargetSource:
    series_id: str
    display_name: str
    physical_group: str
    response: np.ndarray
    phase_available: bool


@dataclass(frozen=True)
class DisplayWaveletSource:
    series_id: str
    display_name: str
    physical_group: str
    source_ir: np.ndarray
    target_ir: np.ndarray | None
    sample_rate_hz: int


@dataclass(frozen=True)
class DisplayProjection:
    view_id: str
    physical_groups: tuple[str, ...]
    channel_sources: tuple[DisplayChannelSource, ...]
    sum_sources: tuple[DisplaySumSource, ...]
    target_sources: tuple[DisplayTargetSource, ...]
    wavelet_sources: tuple[DisplayWaveletSource, ...]
    signature: str

    @property
    def primary_result(self) -> CompositeResult:
        return self.sum_sources[0].result


def resolve_display_projection(
    multichannel: MultichannelCompositeResult,
    configured_channels: Iterable[Mapping[str, object]],
    displayed_channel: object,
) -> DisplayProjection:
    """Resolve Main/L/R/L+R as a pure view over completed physical Groups."""
    stereo = "Left" in multichannel.groups and "Right" in multichannel.groups
    view_id = normalize_display_value(displayed_channel, stereo=stereo)
    physical_groups = {
        DISPLAY_MAIN: ("Main",),
        DISPLAY_LEFT: ("Left",),
        DISPLAY_RIGHT: ("Right",),
        DISPLAY_STEREO: ("Left", "Right"),
    }[view_id]
    missing = tuple(group for group in physical_groups if group not in multichannel.groups)
    if missing:
        raise CompositeValidationError(tuple(
            ValidationIssue("display_group", f"missing physical Group: {group}")
            for group in missing
        ))

    row_ids: dict[tuple[str, str], str] = {}
    for row in configured_channels:
        if not bool(row.get("enabled", True)):
            continue
        group = str(row.get("group", "")).strip()
        name = str(row.get("name", "")).strip()
        if group and name:
            row_ids[(group, name)] = str(row.get("channel_id") or f"{group}:{name}")

    shared_sub = multichannel.groups.get("Sub")
    effective_results: list[tuple[str, CompositeResult]] = []
    for group in physical_groups:
        base = multichannel.groups[group]
        if shared_sub is None:
            effective_results.append((group, base))
            continue
        responses = dict(base.channel_responses)
        responses.update(shared_sub.channel_responses)
        phases = dict(base.channel_unwrapped_phase_deg or {})
        phases.update(shared_sub.channel_unwrapped_phase_deg or {})
        effective_results.append((group, result_from_responses(
            group=group,
            sample_rate_hz=base.sample_rate_hz,
            fft_size=base.fft_size,
            frequency_hz=base.frequency_hz,
            channel_responses=responses,
            target_response=base.target_response,
            target_name=base.target_name,
            target_phase_available=base.target_phase_available,
            channel_unwrapped_phase_deg=phases or None,
        )))

    channel_sources: list[DisplayChannelSource] = []
    seen_channel_ids: set[str] = set()
    for group, result in effective_results:
        for name, response in result.channel_responses.items():
            source_group = "Sub" if shared_sub is not None and name in shared_sub.channel_responses else group
            channel_id = row_ids.get((source_group, name), row_ids.get((group, name), f"{source_group}:{name}"))
            if channel_id in seen_channel_ids:
                continue
            seen_channel_ids.add(channel_id)
            continuous = (
                np.asarray(result.channel_unwrapped_phase_deg[name], dtype=float)
                if result.channel_unwrapped_phase_deg is not None
                and name in result.channel_unwrapped_phase_deg
                else np.rad2deg(unwrapped_phase_on_reliable_branch(response))
            )
            channel_sources.append(DisplayChannelSource(
                series_id=f"channel:{channel_id}",
                display_name=name,
                physical_group=source_group,
                channel_id=channel_id,
                response=np.asarray(response, dtype=complex),
                continuous_phase_deg=continuous,
            ))

    sum_sources = tuple(DisplaySumSource(
        series_id=f"sum:{group}",
        display_name=("Main System Sum" if group == "Main" else f"{view} System Sum"),
        physical_group=group,
        result=result,
    ) for (group, result), view in zip(effective_results, (
        ("Main",) if view_id == DISPLAY_MAIN else
        ("L",) if view_id == DISPLAY_LEFT else
        ("R",) if view_id == DISPLAY_RIGHT else ("L", "R")
    )))

    targets: list[DisplayTargetSource] = []
    target_hashes: set[str] = set()
    for source in sum_sources:
        result = source.result
        if result.target_response is None:
            continue
        digest = _array_digest(np.asarray(result.target_response, dtype=complex))
        if digest in target_hashes:
            continue
        target_hashes.add(digest)
        targets.append(DisplayTargetSource(
            series_id=f"target:{source.physical_group}:{digest[:12]}",
            display_name=result.target_name or f"{source.physical_group} Target",
            physical_group=source.physical_group,
            response=np.asarray(result.target_response, dtype=complex),
            phase_available=bool(result.target_phase_available),
        ))

    wavelets = tuple(DisplayWaveletSource(
        series_id=f"wavelet:{source.physical_group}",
        display_name=source.display_name,
        physical_group=source.physical_group,
        source_ir=np.asarray(source.result.impulse_response, dtype=float),
        target_ir=(
            None if source.result.target_impulse_response is None
            else np.asarray(source.result.target_impulse_response, dtype=float)
        ),
        sample_rate_hz=int(source.result.sample_rate_hz),
    ) for source in sum_sources)

    signature = _projection_signature(
        view_id, physical_groups, channel_sources, sum_sources, targets,
    )
    return DisplayProjection(
        view_id=view_id,
        physical_groups=physical_groups,
        channel_sources=tuple(channel_sources),
        sum_sources=sum_sources,
        target_sources=tuple(targets),
        wavelet_sources=wavelets,
        signature=signature,
    )


def build_display_graph_series(
    projection: DisplayProjection,
    kind: str,
    *,
    include_target: bool = True,
    wrapped_phase: bool = True,
) -> tuple[GraphSeries, ...]:
    """Build one consistent line-series set for every display graph kind."""
    builder = GraphBuilder()
    output: list[GraphSeries] = []
    emitted_channels: set[str] = set()
    emitted_targets: set[str] = set()
    source_by_group_name = {
        (source.physical_group, source.display_name): source
        for source in projection.channel_sources
    }
    for sum_source in projection.sum_sources:
        for item in builder.build(
            sum_source.result,
            kind,
            include_individual=True,
            include_target=include_target,
            wrapped_phase=wrapped_phase,
        ):
            if item.role == "Channel":
                source = source_by_group_name.get(
                    (sum_source.physical_group, item.name)
                ) or source_by_group_name.get(("Sub", item.name))
                if source is None or source.series_id in emitted_channels:
                    continue
                emitted_channels.add(source.series_id)
                output.append(GraphSeries(
                    source.display_name, item.x, item.y, item.kind,
                    series_id=source.series_id, role="Channel",
                    mask_response=source.response,
                ))
            elif item.role == "SystemSum":
                output.append(GraphSeries(
                    sum_source.display_name, item.x, item.y, item.kind,
                    series_id=sum_source.series_id, role="SystemSum",
                    mask_response=sum_source.result.complex_response,
                ))
            elif item.role == "Target":
                digest = _array_digest(np.asarray(item.mask_response, dtype=complex))
                if digest in emitted_targets:
                    continue
                emitted_targets.add(digest)
                output.append(GraphSeries(
                    item.name, item.x, item.y, item.kind,
                    series_id=f"target:{sum_source.physical_group}:{digest[:12]}",
                    role="Target", mask_response=item.mask_response,
                ))
    return tuple(output)


def _projection_signature(
    view_id: str,
    physical_groups: tuple[str, ...],
    channels: Iterable[DisplayChannelSource],
    sums: Iterable[DisplaySumSource],
    targets: Iterable[DisplayTargetSource],
) -> str:
    digest = hashlib.sha256()
    digest.update(view_id.encode())
    digest.update("\0".join(physical_groups).encode())
    for source in channels:
        digest.update(source.series_id.encode())
        digest.update(_array_digest(source.response).encode())
    for source in sums:
        digest.update(source.series_id.encode())
        digest.update(_array_digest(source.result.complex_response).encode())
    for source in targets:
        digest.update(source.series_id.encode())
        digest.update(_array_digest(source.response).encode())
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


__all__ = [
    "DISPLAY_LEFT", "DISPLAY_MAIN", "DISPLAY_RIGHT", "DISPLAY_STEREO",
    "DisplayChannelSource", "DisplayProjection", "DisplaySumSource",
    "DisplayTargetSource", "DisplayWaveletSource", "build_display_graph_series",
    "normalize_display_value", "resolve_display_projection",
]
