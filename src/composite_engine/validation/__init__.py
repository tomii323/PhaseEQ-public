from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..manifest import ChannelManifest, CompositeManifest


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    channel: str | None = None


class CompositeValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        super().__init__("; ".join(item.message for item in self.issues))


def validate_manifest(manifest: CompositeManifest) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if manifest.sample_rate_hz <= 0:
        issues.append(ValidationIssue("sample_rate", "sample_rate_hz must be positive"))
    if not manifest.channels:
        issues.append(ValidationIssue("empty", "manifest contains no channels"))
    names: set[str] = set()
    for channel in manifest.channels:
        if not channel.name.strip():
            issues.append(ValidationIssue("name", "channel name must not be empty"))
        if not channel.way.strip():
            issues.append(ValidationIssue("way", "way must not be empty", channel.name))
        if not channel.group.strip():
            issues.append(ValidationIssue("group", "composite group must not be empty", channel.name))
        if channel.name in names:
            issues.append(ValidationIssue("duplicate", f"duplicate channel name: {channel.name}", channel.name))
        names.add(channel.name)
        if bool(channel.wav) == bool(channel.frd):
            issues.append(ValidationIssue("source", "channel must specify exactly one of wav or frd", channel.name))
        if channel.polarity not in (-1, 1):
            issues.append(ValidationIssue("polarity", "polarity must be +1 or -1", channel.name))
        if not np.isfinite([channel.gain_db, channel.delay_samples]).all():
            issues.append(ValidationIssue("finite", "gain and delay must be finite", channel.name))
        if channel.tap_count is not None and channel.tap_count <= 0:
            issues.append(ValidationIssue("tap_count", "tap_count must be positive", channel.name))
        if channel.tap_count is not None and channel.center_position is not None:
            expected = (channel.tap_count - 1) / 2.0
            if not np.isclose(channel.center_position, expected, atol=1e-12):
                issues.append(ValidationIssue("center", f"center_position must equal (N-1)/2 ({expected:g})", channel.name))
        if channel.time_reference != "tap_center":
            issues.append(ValidationIssue("time_reference", "time_reference must be tap_center", channel.name))
        if channel.delay_application != "after_center_alignment":
            issues.append(ValidationIssue("delay_application", "delay_application must be after_center_alignment", channel.name))
    return tuple(issues)


def validate_coefficients(channel: ChannelManifest, coefficients: np.ndarray) -> tuple[ValidationIssue, ...]:
    values = np.asarray(coefficients)
    issues: list[ValidationIssue] = []
    if values.ndim != 1 or values.size == 0:
        issues.append(ValidationIssue("empty", "FIR must be a non-empty mono vector", channel.name))
        return tuple(issues)
    if not np.isfinite(values).all():
        issues.append(ValidationIssue("finite", "FIR contains NaN or Inf", channel.name))
    if channel.tap_count is not None and values.size != channel.tap_count:
        issues.append(ValidationIssue("tap_count", f"tap_count is {channel.tap_count}, file contains {values.size}", channel.name))
    expected_center = (values.size - 1) / 2.0
    if channel.center_position is not None and not np.isclose(channel.center_position, expected_center, atol=1e-12):
        issues.append(ValidationIssue("center", f"center_position does not match loaded FIR center {expected_center:g}", channel.name))
    return tuple(issues)


def resolve_source(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base / candidate
