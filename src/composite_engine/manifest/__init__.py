from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ChannelManifest:
    name: str
    way: str
    group: str
    gain_db: float = 0.0
    polarity: int = 1
    delay_samples: float = 0.0
    wav: str | None = None
    frd: str | None = None
    tap_count: int | None = None
    center_position: float | None = None
    time_reference: str = "tap_center"
    delay_application: str = "after_center_alignment"
    assignment_id: str = ""
    channel_id: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ChannelManifest":
        polarity = value.get("polarity", 1)
        if isinstance(polarity, bool):
            polarity = -1 if polarity else 1
        return cls(
            name=str(value["name"]),
            way=str(value["way"]),
            group=str(value.get("group", value.get("composite_group", "Main"))),
            gain_db=float(value.get("gain_db", value.get("gain", 0.0))),
            polarity=int(polarity),
            delay_samples=float(value.get("delay_samples", value.get("delay", 0.0))),
            wav=_optional_text(value.get("wav")),
            frd=_optional_text(value.get("frd")),
            tap_count=_optional_int(value.get("tap_count")),
            center_position=_optional_float(value.get("center_position")),
            time_reference=str(value.get("time_reference", "tap_center")),
            delay_application=str(value.get("delay_application", "after_center_alignment")),
            assignment_id=str(value.get("assignment_id", "")).strip(),
            channel_id=str(value.get("channel_id", "")).strip(),
        )


@dataclass(frozen=True)
class CompositeManifest:
    sample_rate_hz: int
    channels: tuple[ChannelManifest, ...]
    format_version: int = 2
    fft_size: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    base_directory: Path = Path(".")

    @property
    def group_names(self) -> tuple[str, ...]:
        """Return Composite Groups in their first manifest appearance order."""
        return tuple(dict.fromkeys(channel.group for channel in self.channels))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, base_directory: Path | str = ".") -> "CompositeManifest":
        known = {"format_version", "sample_rate_hz", "sample_rate", "fft_size", "channels", "composite_groups"}
        return cls(
            sample_rate_hz=int(value.get("sample_rate_hz", value.get("sample_rate", 0))),
            channels=tuple(ChannelManifest.from_dict(item) for item in value.get("channels", ())),
            format_version=int(value.get("format_version", 2)),
            fft_size=_optional_int(value.get("fft_size")),
            metadata={key: item for key, item in value.items() if key not in known},
            base_directory=Path(base_directory),
        )

    def to_dict(self) -> dict[str, Any]:
        channels = []
        for channel in self.channels:
            channels.append({
                "name": channel.name, "way": channel.way, "group": channel.group,
                "gain": channel.gain_db, "polarity": channel.polarity,
                "delay": channel.delay_samples, "wav": channel.wav, "frd": channel.frd,
                "tap_count": channel.tap_count, "center_position": channel.center_position,
                "time_reference": channel.time_reference,
                "delay_application": channel.delay_application,
                "assignment_id": channel.assignment_id,
                "channel_id": channel.channel_id,
            })
        return {
            "format_version": self.format_version,
            "sample_rate_hz": self.sample_rate_hz,
            "fft_size": self.fft_size,
            "channels": channels,
            "composite_groups": list(self.group_names),
            **dict(self.metadata),
        }


def load_manifest(source: str | Path | Mapping[str, Any]) -> CompositeManifest:
    if isinstance(source, Mapping):
        return CompositeManifest.from_dict(source)
    path = Path(source)
    return CompositeManifest.from_dict(json.loads(path.read_text(encoding="utf-8")), base_directory=path.parent)


def _optional_text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
