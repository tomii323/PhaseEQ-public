"""Adapt completed display projections to the independent reverse-null engine."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from . import DisplayProjection, DisplaySumSource
from ..graph import GraphSeries
from ..reverse_null import ReverseNullChannel, calculate_reverse_null


def reverse_null_available(projection: DisplayProjection, graph_kind: str) -> bool:
    """Keep graph eligibility separate from the widget's requested state."""
    return graph_kind == "Magnitude" and any(
        len(source.result.channel_responses) > 1 for source in projection.sum_sources
    )


def build_reverse_null_series(
    projection: DisplayProjection,
    configured_channels: Iterable[Mapping[str, object]],
) -> tuple[GraphSeries, ...]:
    """Validate all inputs, then calculate and project display-only sums.

    Completed responses already include edited polarity, delay and processing.
    Only fresh display arrays are produced; engine results remain authoritative.
    """
    groups = _resolve_channels(projection, configured_channels)
    responses = calculate_reverse_null(groups)
    return tuple(
        _magnitude_series(source, responses[source.physical_group])
        for source in projection.sum_sources if source.physical_group in responses
    )


def _resolve_channels(
    projection: DisplayProjection,
    configured_channels: Iterable[Mapping[str, object]],
) -> dict[str, tuple[ReverseNullChannel, ...]]:
    """Translate display ownership into application-neutral channel inputs."""
    bands = {
        (str(row.get("group", "")), str(row.get("name", ""))):
        str(row.get("band") or row.get("way") or "")
        for row in configured_channels if bool(row.get("enabled", True))
    }
    sources = {
        (source.physical_group, source.display_name)
        for source in projection.channel_sources
    }
    groups = {}
    for summed in projection.sum_sources:
        channels = []
        for name, response in summed.result.channel_responses.items():
            key = (summed.physical_group, name)
            if key not in sources:
                key = ("Sub", name)
            channels.append(ReverseNullChannel(name, bands.get(key, ""), response))
        groups[summed.physical_group] = tuple(channels)
    return groups


def _magnitude_series(source: DisplaySumSource, response: np.ndarray) -> GraphSeries:
    return GraphSeries(
        source.display_name.replace("System Sum", "Reverse Null"),
        source.result.frequency_hz,
        20.0 * np.log10(np.maximum(np.abs(response), 1e-12)),
        "Magnitude", series_id=f"reverse-null:{source.physical_group}",
        role="ReverseNull", mask_response=response,
    )
