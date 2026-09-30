from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from response_math import interpolate_values, interpolate_phase
from scipy.io import wavfile
from scipy.fft import next_fast_len

from ..complex_sum import complex_sum, transform_response
from ..fft import fft_axis, fir_complex_response, group_delay_ms
from ..manifest import ChannelManifest, CompositeManifest, load_manifest
from ..validation import CompositeValidationError, ValidationIssue, resolve_source, validate_coefficients, validate_manifest
from ..target import target_impulse_response


@dataclass(frozen=True)
class CompositeResult:
    group: str
    sample_rate_hz: int
    fft_size: int
    frequency_hz: np.ndarray
    channel_responses: dict[str, np.ndarray]
    complex_response: np.ndarray
    magnitude_db: np.ndarray
    phase_deg: np.ndarray
    group_delay_ms: np.ndarray
    impulse_response: np.ndarray
    channel_impulse_responses: dict[str, np.ndarray] | None = None
    channel_step_responses: dict[str, np.ndarray] | None = None
    channel_group_delay_ms: dict[str, np.ndarray] | None = None
    target_response: np.ndarray | None = None
    target_name: str | None = None
    target_phase_available: bool = False
    target_impulse_response: np.ndarray | None = None
    channel_unwrapped_phase_deg: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class MultichannelCompositeResult:
    """Independent acoustic output channels built from one Manifest."""

    sample_rate_hz: int
    groups: dict[str, CompositeResult]

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(self.groups)

    @property
    def channel_count(self) -> int:
        return sum(len(result.channel_responses) for result in self.groups.values())


class CompositeEngine:
    def __init__(self, manifest: CompositeManifest):
        self.manifest = manifest

    @classmethod
    def from_manifest(cls, source: str | Path | dict) -> "CompositeEngine":
        return cls(load_manifest(source))

    def build(self, group: str, *, fft_size: int | None = None) -> CompositeResult:
        issues = validate_manifest(self.manifest)
        if issues:
            raise CompositeValidationError(issues)
        channels = [item for item in self.manifest.channels if item.group == group]
        if not channels:
            raise CompositeValidationError((ValidationIssue("group", f"no channels for group: {group}"),))
        loaded = [(item, self._load_channel(item)) for item in channels]
        longest = max((len(value) for item, value in loaded if item.wav), default=2)
        requested_size = fft_size or self.manifest.fft_size
        size = (
            max(2, int(requested_size), longest)
            if requested_size is not None
            else max(2, longest, int(self.manifest.sample_rate_hz))
        )
        if size % 2:
            size += 1
        if requested_size is None:
            size = next_fast_len(size, real=True)
        frequency = fft_axis(self.manifest.sample_rate_hz, size)
        individual: dict[str, np.ndarray] = {}
        for channel, values in loaded:
            if channel.wav:
                center = channel.center_position if channel.center_position is not None else (len(values) - 1) / 2.0
                _, response = fir_complex_response(values, self.manifest.sample_rate_hz, size, center_position=center)
            else:
                response = _interpolate_frd(values, frequency)
            individual[channel.name] = transform_response(
                response, frequency, self.manifest.sample_rate_hz,
                gain_db=channel.gain_db, polarity=channel.polarity, delay_samples=channel.delay_samples,
            )
        combined = complex_sum(tuple(individual.values()))
        phase = np.unwrap(np.angle(combined))
        impulse_zero = np.fft.irfft(combined, n=size)
        impulse = np.roll(impulse_zero, int(size // 2))
        channel_impulses = {
            name: np.roll(np.fft.irfft(response, n=size), int(size // 2))
            for name, response in individual.items()
        }
        return CompositeResult(
            group=group, sample_rate_hz=self.manifest.sample_rate_hz, fft_size=size,
            frequency_hz=frequency, channel_responses=individual, complex_response=combined,
            magnitude_db=20.0 * np.log10(np.maximum(np.abs(combined), 1e-12)),
            phase_deg=_wrap_phase_deg(np.rad2deg(phase)),
            group_delay_ms=group_delay_ms(combined, frequency), impulse_response=impulse,
            channel_impulse_responses=channel_impulses,
            channel_step_responses={name: np.cumsum(value) for name, value in channel_impulses.items()},
            channel_group_delay_ms={name: group_delay_ms(value, frequency) for name, value in individual.items()},
        )

    def build_all(self, *, fft_size: int | None = None) -> MultichannelCompositeResult:
        """Build every Composite Group as an independent output channel."""
        issues = validate_manifest(self.manifest)
        if issues:
            raise CompositeValidationError(issues)
        requested_size = fft_size or self.manifest.fft_size
        initial = {
            group: self.build(group, fft_size=requested_size)
            for group in self.manifest.group_names
        }
        common_size = max(result.fft_size for result in initial.values())
        groups = initial if all(result.fft_size == common_size for result in initial.values()) else {
            group: self.build(group, fft_size=common_size)
            for group in self.manifest.group_names
        }
        return MultichannelCompositeResult(
            sample_rate_hz=self.manifest.sample_rate_hz,
            groups=groups,
        )

    def _load_channel(self, channel: ChannelManifest) -> np.ndarray:
        source = channel.wav or channel.frd
        assert source is not None
        path = resolve_source(self.manifest.base_directory, source)
        if channel.wav:
            sample_rate, values = wavfile.read(path)
            if int(sample_rate) != self.manifest.sample_rate_hz:
                raise CompositeValidationError((ValidationIssue("sample_rate", f"{channel.name}: WAV sample rate {sample_rate} differs from manifest {self.manifest.sample_rate_hz}", channel.name),))
            values = _pcm_to_float(values)
            issues = validate_coefficients(channel, values)
            if issues:
                raise CompositeValidationError(issues)
            return values
        is_csv = path.suffix.lower() == ".csv"
        values = np.genfromtxt(
            path, comments="#", delimiter="," if is_csv else None,
            skip_header=1 if is_csv else 0,
        )
        if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] == 0 or not np.isfinite(values).all():
            raise CompositeValidationError((ValidationIssue("frd", f"{channel.name}: FRD must contain finite frequency and dB columns", channel.name),))
        return values[:, :3]


def result_from_responses(
    *, group: str, sample_rate_hz: int, fft_size: int, frequency_hz: np.ndarray,
    channel_responses: dict[str, np.ndarray], target_response: np.ndarray | None = None,
    target_name: str | None = None, target_phase_available: bool = False,
    channel_unwrapped_phase_deg: dict[str, np.ndarray] | None = None,
) -> CompositeResult:
    """Build the canonical result from already-realized adapter responses."""
    frequency = np.asarray(frequency_hz, dtype=float)
    individual = {key: np.asarray(value, dtype=complex) for key, value in channel_responses.items()}
    combined = (
        complex_sum(tuple(individual.values()))
        if individual else np.zeros(frequency.shape, dtype=np.complex128)
    )
    phase = np.unwrap(np.angle(combined))
    impulse_zero = np.fft.irfft(combined, n=int(fft_size))
    channel_impulses = {
        name: np.roll(np.fft.irfft(response, n=int(fft_size)), int(fft_size) // 2)
        for name, response in individual.items()
    }
    channel_delay = {
        name: (
            -np.gradient(
                np.deg2rad(np.asarray(channel_unwrapped_phase_deg[name], dtype=float)),
                2.0 * np.pi * frequency,
                edge_order=1,
            ) * 1000.0
            if channel_unwrapped_phase_deg is not None
            and name in channel_unwrapped_phase_deg
            else group_delay_ms(value, frequency)
        )
        for name, value in individual.items()
    }
    return CompositeResult(
        group=group, sample_rate_hz=int(sample_rate_hz), fft_size=int(fft_size),
        frequency_hz=frequency, channel_responses=individual, complex_response=combined,
        magnitude_db=20.0 * np.log10(np.maximum(np.abs(combined), 1e-12)),
        phase_deg=_wrap_phase_deg(np.rad2deg(phase)),
        group_delay_ms=group_delay_ms(combined, frequency),
        impulse_response=np.roll(impulse_zero, int(fft_size) // 2),
        channel_impulse_responses=channel_impulses,
        channel_step_responses={name: np.cumsum(value) for name, value in channel_impulses.items()},
        channel_group_delay_ms=channel_delay,
        target_response=None if target_response is None else np.asarray(target_response, dtype=complex),
        target_name=None if target_response is None else (target_name or "Group Target"),
        target_phase_available=bool(target_response is not None and target_phase_available),
        target_impulse_response=(
            None if target_response is None
            else target_impulse_response(np.asarray(target_response, dtype=complex), int(fft_size))
        ),
        channel_unwrapped_phase_deg=channel_unwrapped_phase_deg,
    )


def _pcm_to_float(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1:
        return data
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        return data.astype(float) / scale
    return data.astype(float)


def _interpolate_frd(values: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    source_frequency = values[:, 0]
    magnitude = 10.0 ** (values[:, 1] / 20.0)
    # PhaseEQ/Speaker exchange FRD stores a continuous unwrapped phase trace.
    # Preserve that trace exactly; unwrapping it again can discard intentional
    # 360-degree rotations before the response reaches the DSP pipeline.
    phase = np.zeros_like(magnitude) if values.shape[1] < 3 else np.deg2rad(values[:, 2])
    projected_magnitude = interpolate_values(source_frequency, magnitude, frequency, axis="linear")
    projected_phase = interpolate_phase(source_frequency, phase, frequency)
    return projected_magnitude * np.exp(1j * projected_phase)


def _wrap_phase_deg(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0
