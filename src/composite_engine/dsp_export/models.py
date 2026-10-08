from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from composite_engine.iir_crossover import IIRCrossoverConfig
from composite_engine.phase_alignment import AllPassSection
from composite_engine.fir_artifact import FinalFIRArtifact


@dataclass(frozen=True)
class CanonicalDSPChannel:
    output_index: int
    channel_id: str
    name: str
    group: str
    way: str
    sample_rate_hz: int
    final_fir: np.ndarray | None
    phaseeq_iir_parameters: tuple[dict[str, object], ...] = ()
    phaseeq_iir_sos: tuple[tuple[float, ...], ...] = ()
    baffle_iir_parameters: dict[str, object] | None = None
    baffle_iir_sos: tuple[tuple[float, ...], ...] = ()
    crossover: IIRCrossoverConfig = IIRCrossoverConfig()
    crossover_allpass: tuple[AllPassSection, ...] = ()
    gain_db: float = 0.0
    polarity: int = 1
    fir_alignment_delay_samples: float = 0.0
    manual_delay_samples: float = 0.0
    phase_alignment_delay_samples: float = 0.0
    timing_provenance: dict[str, object] | None = None
    final_fir_artifact: FinalFIRArtifact | None = field(default=None, repr=False, compare=False)
    adaptive_crop_metadata: dict[str, object] | None = None

    @property
    def total_delay_samples(self) -> float:
        return (
            float(self.fir_alignment_delay_samples)
            + float(self.manual_delay_samples)
            + float(self.phase_alignment_delay_samples)
        )


@dataclass(frozen=True)
class CanonicalDSPPackage:
    system_name: str
    generated_at: str
    mode: str
    channels: tuple[CanonicalDSPChannel, ...]
    source_signature: str
    workspace: dict[str, object] = field(default_factory=dict)
    system_id: str = ""
    management_no: str = ""
    system_revision_number: int | None = None
    system_content_hash: str = ""

    @property
    def sample_rate_hz(self) -> int:
        rates = {int(channel.sample_rate_hz) for channel in self.channels}
        if len(rates) != 1:
            raise ValueError("all DSP channels must share one sample rate")
        return rates.pop()


@dataclass(frozen=True)
class DSPProfile:
    profile_id: str
    adapter_id: str
    display_name: str
    sample_rates_hz: tuple[int, ...] = ()
    supports_fir: bool = True
    max_fir_taps: int = 0
    max_iir_sections: int = 0
    supports_allpass: bool = True
    supports_fractional_delay: bool = True
    supports_delay: bool = True
    fir_time_reference: Literal["tap_center", "sample_zero"] = "tap_center"
    options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityIssue:
    severity: Literal["error", "warning", "info"]
    code: str
    message: str
    channel_id: str | None = None
    stage: str | None = None


@dataclass(frozen=True)
class CompatibilityReport:
    adapter_id: str
    profile_id: str
    issues: tuple[CompatibilityIssue, ...]
    channel_rows: tuple[dict[str, object], ...]

    @property
    def compatible(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True)
class ExportedDSPTarget:
    adapter_id: str
    profile_id: str
    files: dict[str, bytes]
    report: CompatibilityReport
