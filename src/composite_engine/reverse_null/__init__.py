"""Alternating-polarity complex sums, independent of UI and graph models."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np


# Structural band IDs, independent of editable names and input iteration order.
_BAND_ORDER = ("SUB", "Low", "Low-Mid", "Mid", "High-Mid", "High", "Fullrange")


@dataclass(frozen=True)
class ReverseNullChannel:
    name: str
    band: str
    response: np.ndarray


def calculate_reverse_null(
    groups: Mapping[str, Iterable[ReverseNullChannel]],
) -> dict[str, np.ndarray]:
    """Validate all groups before summation; return fresh arrays for 2+ Ways.

    Responses must share a one-dimensional finite frequency grid within each
    group. Signs are additional to the existing processing, from low to high.
    Single-channel groups are omitted; input arrays are never modified.
    """
    prepared = _prepare_groups(groups)
    return {group: _sum_responses(responses) for group, responses in prepared.items()}


def _prepare_groups(
    groups: Mapping[str, Iterable[ReverseNullChannel]],
) -> dict[str, tuple[np.ndarray, ...]]:
    prepared = {}
    for group, channels in groups.items():
        channels = tuple(channels)
        if len(channels) < 2:
            continue
        ordered = []
        shape = None
        for channel in channels:
            if channel.band not in _BAND_ORDER:
                raise ValueError(f"Reverse Null: missing band identity for {channel.name}")
            response = np.asarray(channel.response, dtype=complex)
            if response.ndim != 1 or not response.size or not np.all(np.isfinite(response)):
                raise ValueError(f"Reverse Null: invalid response for {channel.name}")
            if shape is not None and response.shape != shape:
                raise ValueError(f"Reverse Null: response length mismatch in {group}")
            shape = response.shape
            ordered.append((_BAND_ORDER.index(channel.band), response))
        if len({rank for rank, _ in ordered}) != len(ordered):
            raise ValueError("Reverse Null: duplicate bands in one output group")
        ordered.sort(key=lambda item: item[0])
        prepared[group] = tuple(response for _, response in ordered)
    return prepared


def _sum_responses(responses: tuple[np.ndarray, ...]) -> np.ndarray:
    response = np.zeros_like(responses[0], dtype=complex)
    for index, channel_response in enumerate(responses):
        response += channel_response * (-1 if index % 2 else 1)
    return response


__all__ = ["ReverseNullChannel", "calculate_reverse_null"]
