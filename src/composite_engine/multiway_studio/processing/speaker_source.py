from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResolvedSpeakerSource:
    label: str
    filename: str
    data: bytes
    revision: int = 0
    content_hash: str = ""


@dataclass(frozen=True)
class SpeakerAlignmentReadiness:
    """Describe whether one alignment set has a coherent Speaker basis."""

    mode: str
    configured_channels: tuple[str, ...]
    unity_channels: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.mode in {"speaker", "filter_only"}


def assess_speaker_alignment_readiness(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> SpeakerAlignmentReadiness:
    """Reject mixing measured Speaker phase with an implicit Unity source.

    An all-Unity set is a valid filter-only analysis.  Once one physical
    Speaker response participates, every Channel in the alignment set needs a
    Speaker response; otherwise the solver would interpret the missing
    acoustic phase as a real zero-delay reference.
    """

    configured: list[str] = []
    unity: list[str] = []
    for row in rows:
        name = str(row.get("name") or row.get("way") or "Channel")
        if resolve_speaker_source(row) is None:
            unity.append(name)
        else:
            configured.append(name)
    mode = "mixed" if configured and unity else "speaker" if configured else "filter_only"
    return SpeakerAlignmentReadiness(mode, tuple(configured), tuple(unity))


def resolve_speaker_source(row: dict[str, Any]) -> ResolvedSpeakerSource | None:
    """Select exactly one Speaker source using the Studio precedence contract."""
    returned = row.get("phaseeq_speaker_response")
    if isinstance(returned, dict) and returned.get("data"):
        return _from_mapping(
            "PhaseEQ Response Processing", returned, "speaker_response.frd"
        )

    package = row.get("speaker_library_response")
    if isinstance(package, dict) and package.get("data"):
        return _from_mapping(str(package.get("name") or "Speaker Package"), package, "speaker_package.frd")

    manual = row.get("speaker_response")
    if manual is not None:
        data = bytes(manual.getvalue())
        if data:
            return ResolvedSpeakerSource(
                "Manual Speaker",
                str(getattr(manual, "name", "speaker_response.frd")),
                data,
            )

    return None


def _from_mapping(
    label: str, value: dict[str, Any], default_filename: str,
) -> ResolvedSpeakerSource:
    return ResolvedSpeakerSource(
        label=label,
        filename=Path(str(value.get("filename", default_filename))).name,
        data=bytes(value["data"]),
        revision=int(value.get("revision", 0) or 0),
        content_hash=str(value.get("content_hash", "")),
    )
