from __future__ import annotations

from dataclasses import dataclass, field
import math

from ..config import DesignConfig, IIRFilter, LinearFIRFilter, SpeakerResponse
from .crossover_quality import DistortionProfile
from .crossover_models import CrossoverPlan


@dataclass(frozen=True)
class DSPSystemProcessing:
    smoothing_fraction: float = 0.0
    smoothing_mode: str = "Gain + Phase"
    gain_shift_db: float = 0.0
    fir_enabled: bool = True
    auto_gain: bool = True
    remove_nyquist: bool = False
    remove_nyquist_strength: float = 1.0


@dataclass(frozen=True)
class DSPSystemTarget:
    mode: str = "Flat"
    response: SpeakerResponse | None = None
    gain_shift_db: float = 0.0
    smoothing_fraction: float = 0.0


@dataclass(frozen=True)
class DSPInput:
    id: str
    name: str
    role: str = "Custom"
    muted: bool = False
    gain_db: float = 0.0
    polarity_invert: bool = False
    iir_filters: tuple[IIRFilter, ...] = ()

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("DSP input id must not be empty")
        if not self.name.strip():
            raise ValueError("DSP input name must not be empty")
        if not math.isfinite(self.gain_db):
            raise ValueError("DSP input gain must be finite")


@dataclass(frozen=True)
class DSPInputBinding:
    id: str
    input_id: str
    dsp_id: str
    input_channel: int = 0

    def validate(self) -> None:
        if not self.id.strip() or not self.input_id.strip() or not self.dsp_id.strip():
            raise ValueError("DSP input binding ids must not be empty")
        if self.input_channel < 0:
            raise ValueError("input_channel must be zero or greater")


@dataclass(frozen=True)
class DSPRoute:
    id: str
    input_id: str
    way_id: str
    enabled: bool = True
    gain_db: float = 0.0
    muted: bool = False

    def validate(self) -> None:
        if not self.id.strip() or not self.input_id.strip() or not self.way_id.strip():
            raise ValueError("DSP route ids must not be empty")
        if not math.isfinite(self.gain_db):
            raise ValueError("DSP route gain must be finite")


@dataclass(frozen=True)
class DSPDevice:
    id: str
    name: str
    target: str = "Generic"
    sample_rate: int = 48_000
    taps: int = 4097
    latency_offset_ms: float = 0.0
    max_output_channels: int = 8
    max_input_channels: int = 2
    fir_supported: bool = True
    fir_off_behavior: str = "removed"
    delay_resolution_ms: float = 0.01

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("DSP device id must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("DSP sample_rate must be positive")
        if self.taps < 1:
            raise ValueError("DSP taps must be positive")
        if self.max_output_channels < 1 or self.max_output_channels > 64:
            raise ValueError("DSP max_output_channels must be between 1 and 64")
        if self.max_input_channels < 1 or self.max_input_channels > 64:
            raise ValueError("DSP max_input_channels must be between 1 and 64")
        if self.fir_off_behavior not in {"removed", "retained", "delta_fir"}:
            raise ValueError("fir_off_behavior must be removed, retained, or delta_fir")
        if not math.isfinite(self.delay_resolution_ms) or self.delay_resolution_ms <= 0.0:
            raise ValueError("delay_resolution_ms must be positive")


@dataclass(frozen=True)
class DSPOutputChannel:
    """Physical DSP output settings, independent from an acoustic Way.

    A Way selects one output by ``DSPWay.output_id``. Source and correction
    settings stay with this physical output when the Way assignment changes;
    Gain, Delay, polarity and alignment remain owned by the acoustic Way.
    """

    id: str
    name: str
    dsp_id: str
    channel: int
    eq_mask_smoothing_oct: float = 0.5
    distortion_profile: DistortionProfile | None = None
    fir_enabled: bool = True
    speaker_package_id: str = ""
    manual_iir_filters: tuple[IIRFilter, ...] = ()
    auto_iir_filters: tuple[IIRFilter, ...] = ()
    retained_fir_artifact_id: str = ""
    correction_config: DesignConfig | None = None
    target_override: DSPSystemTarget | None = None
    target_override_enabled: bool = False

    @property
    def source_package_id(self) -> str:
        return self.speaker_package_id

    def validate(self) -> None:
        if not self.id.strip() or not self.dsp_id.strip():
            raise ValueError("DSP output channel ids must not be empty")
        if self.channel < 0:
            raise ValueError("DSP output channel must be zero or greater")
        if self.eq_mask_smoothing_oct < 0.0:
            raise ValueError("DSP output eq_mask_smoothing_oct must not be negative")
        if any(item.kind in {"high_pass", "low_pass"} for item in self.manual_iir_filters + self.auto_iir_filters):
            raise ValueError("DSP output EQ filters must not contain crossover filters")
        if self.distortion_profile is not None:
            self.distortion_profile.validate()
        if self.correction_config is not None and (
            self.correction_config.speaker_response is not None
            or self.correction_config.target_response is not None
        ):
            raise ValueError("DSP output correction_config must not embed Speaker or Target responses")


@dataclass(frozen=True)
class DSPWay:
    id: str
    name: str
    acoustic_group: str
    dsp_id: str
    output_channel: int
    linear_fir_filters: tuple[LinearFIRFilter, ...] = ()
    eq_mask_smoothing_oct: float = 0.5
    gain_db: float = 0.0
    delay_ms: float = 0.0
    polarity_invert: bool = False
    muted: bool = False
    solo: bool = False
    routing_role: str = "Custom"
    alignment_iir_filters: tuple[IIRFilter, ...] = ()
    alignment_fir: tuple[float, ...] = ()
    alignment_source: str = ""
    alignment_delay_ms: float = 0.0
    alignment_polarity_invert: bool = False
    distortion_profile: DistortionProfile | None = None
    fir_enabled: bool = True
    speaker_package_id: str = ""
    manual_iir_filters: tuple[IIRFilter, ...] = ()
    auto_iir_filters: tuple[IIRFilter, ...] = ()
    retained_fir_artifact_id: str = ""
    correction_config: DesignConfig | None = None
    output_id: str = ""
    output_target_override: DSPSystemTarget | None = None

    @property
    def source_package_id(self) -> str:
        """Return the Output-projected Speaker Package source id."""

        return self.speaker_package_id

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("DSP way id must not be empty")
        if not self.dsp_id.strip():
            raise ValueError("DSP way must reference a DSP device")
        if self.output_channel < 0:
            raise ValueError("output_channel must be zero or greater")
        if self.eq_mask_smoothing_oct < 0.0:
            raise ValueError("eq_mask_smoothing_oct must not be negative")
        if any(item.kind != "allpass" for item in self.alignment_iir_filters):
            raise ValueError("alignment IIR filters must be All Pass sections")
        if any(item.kind in {"high_pass", "low_pass"} for item in self.manual_iir_filters + self.auto_iir_filters):
            raise ValueError("Way EQ filters must not contain crossover filters")
        if self.alignment_fir and (
            len(self.alignment_fir) < 3
            or any(not math.isfinite(float(value)) for value in self.alignment_fir)
        ):
            raise ValueError("alignment FIR must contain at least three finite coefficients")
        if self.distortion_profile is not None:
            self.distortion_profile.validate()
        if self.correction_config is not None and (
            self.correction_config.speaker_response is not None
            or self.correction_config.target_response is not None
        ):
            raise ValueError("Way correction_config must not embed Speaker or Target responses")
        if not math.isfinite(self.alignment_delay_ms):
            raise ValueError("alignment delay must be finite")


@dataclass(frozen=True)
class DSPSystem:
    id: str
    name: str
    devices: tuple[DSPDevice, ...] = field(default_factory=tuple)
    ways: tuple[DSPWay, ...] = field(default_factory=tuple)
    inputs: tuple[DSPInput, ...] = field(default_factory=tuple)
    input_bindings: tuple[DSPInputBinding, ...] = field(default_factory=tuple)
    routes: tuple[DSPRoute, ...] = field(default_factory=tuple)
    processing: DSPSystemProcessing = field(default_factory=DSPSystemProcessing)
    targets: dict[str, DSPSystemTarget] = field(default_factory=dict)
    project_revisions: dict[str, str] = field(default_factory=dict)
    crossover_plan: CrossoverPlan | None = None
    revision: int = 1
    last_design_signature: str = ""
    way_design_signatures: dict[str, str] = field(default_factory=dict)
    last_alignment_signature: str = ""
    last_export_signature: str = ""
    output_channels: tuple[DSPOutputChannel, ...] = field(default_factory=tuple)
    applied_design_signature: str = ""
    live_result_signature: str = ""
    saved_design_signature: str = ""

    def validate(self) -> None:
        device_ids = [device.id for device in self.devices]
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("DSP device ids must be unique")
        way_ids = [way.id for way in self.ways]
        if len(way_ids) != len(set(way_ids)):
            raise ValueError("DSP way ids must be unique")
        for device in self.devices:
            device.validate()
        known_devices = set(device_ids)
        devices_by_id = {device.id: device for device in self.devices}
        output_ids = [output.id for output in self.output_channels]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("DSP output channel ids must be unique")
        output_slots: set[tuple[str, int]] = set()
        for output in self.output_channels:
            output.validate()
            if output.dsp_id not in known_devices:
                raise ValueError(f"unknown DSP output device: {output.dsp_id}")
            if output.channel >= devices_by_id[output.dsp_id].max_output_channels:
                raise ValueError(
                    f"{output.name}: output {output.channel + 1} exceeds "
                    f"{devices_by_id[output.dsp_id].name} Max outputs "
                    f"({devices_by_id[output.dsp_id].max_output_channels})"
                )
            slot = (output.dsp_id, output.channel)
            if slot in output_slots:
                raise ValueError(f"duplicate DSP output channel: {slot}")
            output_slots.add(slot)
        occupied: set[tuple[str, int]] = set()
        assigned_output_ids: set[str] = set()
        for way in self.ways:
            way.validate()
            if way.dsp_id not in known_devices:
                raise ValueError(f"unknown DSP device: {way.dsp_id}")
            if way.output_channel >= devices_by_id[way.dsp_id].max_output_channels:
                raise ValueError(
                    f"{way.name}: output {way.output_channel + 1} exceeds "
                    f"{devices_by_id[way.dsp_id].name} Max outputs "
                    f"({devices_by_id[way.dsp_id].max_output_channels})"
                )
            slot = (way.dsp_id, way.output_channel)
            if slot in occupied:
                raise ValueError(f"duplicate DSP output assignment: {slot}")
            occupied.add(slot)
            if self.output_channels:
                if way.output_id not in set(output_ids):
                    raise ValueError(f"{way.name}: unknown DSP output assignment")
                if way.output_id in assigned_output_ids:
                    raise ValueError(f"duplicate Way to output assignment: {way.output_id}")
                assigned_output_ids.add(way.output_id)
        input_ids = [item.id for item in self.inputs]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("DSP input ids must be unique")
        for item in self.inputs:
            item.validate()
        known_inputs = set(input_ids)
        binding_slots: set[tuple[str, str]] = set()
        physical_input_slots: set[tuple[str, int]] = set()
        for binding in self.input_bindings:
            binding.validate()
            if binding.input_id not in known_inputs or binding.dsp_id not in known_devices:
                raise ValueError("DSP input binding references an unknown input or DSP")
            if binding.input_channel >= devices_by_id[binding.dsp_id].max_input_channels:
                input_name = next(item.name for item in self.inputs if item.id == binding.input_id)
                raise ValueError(
                    f"{input_name}: input {binding.input_channel + 1} exceeds "
                    f"{devices_by_id[binding.dsp_id].name} Max inputs "
                    f"({devices_by_id[binding.dsp_id].max_input_channels})"
                )
            slot = (binding.input_id, binding.dsp_id)
            if slot in binding_slots:
                raise ValueError(f"duplicate DSP input binding: {slot}")
            binding_slots.add(slot)
            physical_slot = (binding.dsp_id, binding.input_channel)
            if physical_slot in physical_input_slots:
                raise ValueError(f"duplicate DSP physical input assignment: {physical_slot}")
            physical_input_slots.add(physical_slot)
        known_ways = set(way_ids)
        route_pairs: set[tuple[str, str]] = set()
        route_ids: set[str] = set()
        for route in self.routes:
            route.validate()
            if route.id in route_ids:
                raise ValueError("DSP route ids must be unique")
            route_ids.add(route.id)
            if route.input_id not in known_inputs or route.way_id not in known_ways:
                raise ValueError("DSP route references an unknown input or Way")
            pair = (route.input_id, route.way_id)
            if pair in route_pairs:
                raise ValueError(f"duplicate DSP route: {pair}")
            route_pairs.add(pair)
        if self.processing.smoothing_fraction < 0.0:
            raise ValueError("smoothing_fraction must not be negative")
        if not 0.0 <= self.processing.remove_nyquist_strength <= 1.0:
            raise ValueError("remove_nyquist_strength must be between zero and one")
        if self.crossover_plan is not None:
            self.crossover_plan.validate()
            crossover_ids = set(self.crossover_plan.main_way_ids) | {
                item.way_id for item in self.crossover_plan.sub_crossovers
            }
            if crossover_ids != known_ways:
                raise ValueError("crossover plan Ways must match DSP System Ways")

    def device(self, device_id: str) -> DSPDevice:
        for device in self.devices:
            if device.id == device_id:
                return device
        raise KeyError(device_id)
