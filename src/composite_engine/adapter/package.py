from __future__ import annotations

from dataclasses import dataclass, replace
import json
import io
from pathlib import PurePosixPath
from pathlib import Path
from typing import Mapping
import zipfile

import numpy as np
from scipy.io import wavfile

from ..complex_sum import complex_sum, transform_response
from ..core import CompositeResult, MultichannelCompositeResult, _interpolate_frd, result_from_responses
from ..fft import fir_complex_response
from ..manifest import CompositeManifest
from ..validation import CompositeValidationError, ValidationIssue, validate_coefficients, validate_manifest


@dataclass(frozen=True)
class ChannelInput:
    name: str
    way: str
    group: str
    filename: str
    data: bytes
    gain_db: float = 0.0
    polarity: int = 1
    delay_samples: float = 0.0


@dataclass(frozen=True)
class CompositePackage:
    manifest: CompositeManifest
    assets: Mapping[str, bytes]

    @classmethod
    def from_uploads(cls, manifest_bytes: bytes, assets: Mapping[str, bytes]) -> "CompositePackage":
        payload = json.loads(manifest_bytes.decode("utf-8-sig"))
        manifest = CompositeManifest.from_dict(payload)
        normalized = {_asset_key(name): bytes(value) for name, value in assets.items()}
        return cls(manifest=manifest, assets=normalized)

    @classmethod
    def from_zip(cls, data: bytes) -> "CompositePackage":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [_asset_key(name) for name in archive.namelist() if not name.endswith("/")]
            if "manifest.json" not in names:
                raise ValueError("ZIP root must contain manifest.json")
            assets = {
                name: archive.read(original)
                for name, original in zip(names, (item for item in archive.namelist() if not item.endswith("/")), strict=True)
                if name != "manifest.json"
            }
            return cls.from_uploads(archive.read("manifest.json"), assets)

    @classmethod
    def from_channel_inputs(
        cls, sample_rate_hz: int, inputs: list[ChannelInput],
    ) -> "CompositePackage":
        if int(sample_rate_hz) <= 0:
            raise ValueError("sample rate must be positive")
        if not inputs:
            raise ValueError("at least one channel input is required")
        channels: list[dict[str, object]] = []
        assets: dict[str, bytes] = {}
        for item in inputs:
            filename = _asset_key(Path(item.filename).name)
            suffix = Path(filename).suffix.casefold()
            if suffix not in {".wav", ".frd", ".csv", ".txt"}:
                raise ValueError(f"unsupported channel input: {filename}")
            if filename in assets:
                raise ValueError(f"duplicate channel input filename: {filename}")
            channel: dict[str, object] = {
                "name": item.name.strip(), "way": item.way.strip(), "group": item.group.strip(),
                "gain": float(item.gain_db), "polarity": int(item.polarity),
                "delay": float(item.delay_samples),
                "wav": filename if suffix == ".wav" else None,
                "frd": filename if suffix != ".wav" else None,
                "time_reference": "tap_center",
                "delay_application": "after_center_alignment",
            }
            if suffix == ".wav":
                wav_rate, values = wavfile.read(io.BytesIO(item.data))
                if int(wav_rate) != int(sample_rate_hz):
                    raise ValueError(
                        f"{filename}: WAV sample rate {wav_rate} differs from {sample_rate_hz} Hz"
                    )
                tap_count = int(values.shape[0])
                channel["tap_count"] = tap_count
                channel["center_position"] = (tap_count - 1) / 2.0
            channels.append(channel)
            assets[filename] = bytes(item.data)
        manifest = {
            "format_version": 2,
            "sample_rate_hz": int(sample_rate_hz),
            "channels": channels,
            "composite_groups": list(dict.fromkeys(item.group.strip() for item in inputs)),
        }
        return cls.from_uploads(json.dumps(manifest, ensure_ascii=False).encode("utf-8"), assets)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "CompositePackage":
        root = Path(directory)
        manifest_path = root / "manifest.json"
        payload = manifest_path.read_bytes()
        manifest = CompositeManifest.from_dict(json.loads(payload.decode("utf-8-sig")))
        assets: dict[str, bytes] = {}
        for channel in manifest.channels:
            asset_name = channel.wav or channel.frd
            if asset_name is None:
                continue
            key = _asset_key(asset_name)
            assets[key] = (root / key).read_bytes()
        return cls.from_uploads(payload, assets)

    def build(self, group: str, *, fft_size: int | None = None) -> CompositeResult:
        frequency, responses, size = self.evaluate_responses(group, fft_size=fft_size)
        return result_from_responses(
            group=group, sample_rate_hz=self.manifest.sample_rate_hz,
            fft_size=size, frequency_hz=frequency, channel_responses=responses,
        )

    def evaluate_responses(
        self, group: str, *, fft_size: int | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
        """Evaluate validated stages without constructing unused IR/Step/GD."""
        issues = validate_manifest(self.manifest)
        if issues:
            raise CompositeValidationError(issues)
        channels = [channel for channel in self.manifest.channels if channel.group == group]
        if not channels:
            raise CompositeValidationError((ValidationIssue("group", f"no channels for group: {group}"),))
        loaded: list[tuple[object, np.ndarray]] = []
        for channel in channels:
            asset_name = channel.wav or channel.frd
            assert asset_name is not None
            key = _asset_key(asset_name)
            if key not in self.assets:
                raise CompositeValidationError((ValidationIssue("missing_asset", f"asset is missing: {asset_name}", channel.name),))
            if channel.wav:
                sample_rate, values = wavfile.read(io.BytesIO(self.assets[key]))
                if int(sample_rate) != self.manifest.sample_rate_hz:
                    raise CompositeValidationError((ValidationIssue("sample_rate", f"{channel.name}: WAV sample rate {sample_rate} differs from manifest {self.manifest.sample_rate_hz}", channel.name),))
                values = _pcm_to_float(values)
                coefficient_issues = validate_coefficients(channel, values)
                if coefficient_issues:
                    raise CompositeValidationError(coefficient_issues)
            else:
                values = _load_frd_bytes(self.assets[key], channel.name)
            loaded.append((channel, values))
        longest = max((len(values) for channel, values in loaded if channel.wav), default=2)
        size = max(2, int(fft_size or self.manifest.fft_size or 0), longest)
        if size % 2 == 0:
            size += 1
        frequency = np.fft.rfftfreq(size, 1.0 / self.manifest.sample_rate_hz)
        responses: dict[str, np.ndarray] = {}
        for channel, values in loaded:
            if channel.wav:
                center = channel.center_position if channel.center_position is not None else (len(values) - 1) / 2.0
                _, response = fir_complex_response(values, self.manifest.sample_rate_hz, size, center_position=center)
            else:
                response = _interpolate_frd(values, frequency)
            responses[channel.name] = transform_response(
                response, frequency, self.manifest.sample_rate_hz,
                gain_db=channel.gain_db, polarity=channel.polarity,
                delay_samples=channel.delay_samples,
            )
        return frequency, responses, size

    def with_channel_controls(
        self,
        controls: Mapping[str, Mapping[str, float | int]],
    ) -> "CompositePackage":
        """Return a package with explicit UI inputs applied to Manifest fields."""
        channels = []
        for channel in self.manifest.channels:
            source_key = channel.wav or channel.frd or channel.name
            values = controls.get(source_key, controls.get(channel.name))
            if values is None:
                channels.append(channel)
                continue
            channels.append(replace(
                channel,
                name=str(values.get("name", channel.name)).strip(),
                way=str(values.get("way", channel.way)).strip(),
                group=str(values.get("group", channel.group)).strip(),
                gain_db=float(values.get("gain_db", channel.gain_db)),
                polarity=int(values.get("polarity", channel.polarity)),
                delay_samples=float(values.get("delay_samples", channel.delay_samples)),
            ))
        return CompositePackage(
            manifest=replace(self.manifest, channels=tuple(channels)),
            assets=self.assets,
        )

    def write_manifest(self, directory: str | Path) -> Path:
        """Persist the current Manifest view without rewriting FIR/FRD assets."""
        root = Path(directory)
        path = root / "manifest.json"
        from utils.exchange_io import atomic_json, exchange_lock
        with exchange_lock(root):
            atomic_json(path, self.manifest.to_dict())
        return path

    def build_all(self, *, fft_size: int | None = None) -> MultichannelCompositeResult:
        """Build all Manifest groups without mixing channels between groups."""
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


def _asset_key(value: str) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe asset path: {value}")
    return str(path)


def _pcm_to_float(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    if data.ndim != 1:
        return data
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        return data.astype(float) / float(max(abs(info.min), info.max))
    return data.astype(float)


def _load_frd_bytes(data: bytes, channel_name: str) -> np.ndarray:
    text = data.decode("utf-8-sig")
    # FRD tools commonly use `*`, `#` or `;` for comment/header lines.
    # Normalize the legacy `*` form before NumPy parsing so already-published
    # PhaseEQ Speaker assets remain readable without republishing them.
    normalized_lines = [
        f"#{line.lstrip()[1:]}" if line.lstrip().startswith("*") else line
        for line in text.splitlines()
    ]
    text = "\n".join(normalized_lines)
    first_content = next((
        line for line in normalized_lines
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ), "")
    comma_separated = "," in first_content
    has_header = comma_separated and any(char.isalpha() for char in first_content)
    try:
        values = np.genfromtxt(
            io.StringIO(text), comments="#", delimiter="," if comma_separated else None,
            skip_header=1 if has_header else 0,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise CompositeValidationError((ValidationIssue("frd", f"{channel_name}: invalid FRD/CSV data", channel_name),)) from exc
    if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise CompositeValidationError((ValidationIssue("frd", f"{channel_name}: FRD/CSV must contain finite frequency and dB columns", channel_name),))
    return values[:, :3]
