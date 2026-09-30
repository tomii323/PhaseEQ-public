from __future__ import annotations

from dataclasses import replace

from .models import DSPOutputChannel, DSPSystem, DSPWay


def _channel_id(dsp_id: str, channel: int) -> str:
    return f"output-{dsp_id}-{channel + 1}"


def _blank_channel(system: DSPSystem, dsp_id: str, channel: int) -> DSPOutputChannel:
    device = system.device(dsp_id)
    return DSPOutputChannel(
        id=_channel_id(dsp_id, channel),
        name=f"{device.name} Out {channel + 1}",
        dsp_id=dsp_id,
        channel=channel,
    )


def apply_output_channel_to_way(way: DSPWay, output: DSPOutputChannel) -> DSPWay:
    """Build the runtime-compatible Way without moving Way trim/alignment."""

    return replace(
        way,
        output_id=output.id,
        dsp_id=output.dsp_id,
        output_channel=output.channel,
        eq_mask_smoothing_oct=output.eq_mask_smoothing_oct,
        distortion_profile=output.distortion_profile,
        fir_enabled=output.fir_enabled,
        speaker_package_id=output.speaker_package_id,
        manual_iir_filters=output.manual_iir_filters,
        auto_iir_filters=output.auto_iir_filters,
        retained_fir_artifact_id=output.retained_fir_artifact_id,
        correction_config=output.correction_config,
        output_target_override=(
            output.target_override if output.target_override_enabled else None
        ),
    )


def ensure_output_channel_assignments(
    system: DSPSystem,
) -> DSPSystem:
    """Normalize physical outputs and project them into runtime DSPWay objects.

    Output-owned settings are authoritative. Unused hardware outputs are
    created as Flat, filter-free channels.
    """

    if not system.devices:
        return system
    valid_slots = tuple(
        (device.id, channel)
        for device in system.devices
        for channel in range(device.max_output_channels)
    )
    valid_slot_set = set(valid_slots)
    existing_by_slot = {
        (output.dsp_id, output.channel): output
        for output in system.output_channels
        if (output.dsp_id, output.channel) in valid_slot_set
    }
    outputs: list[DSPOutputChannel] = []
    for dsp_id, channel in valid_slots:
        output = existing_by_slot.get((dsp_id, channel))
        if output is None:
            output = _blank_channel(system, dsp_id, channel)
        outputs.append(output)

    if len(system.ways) > len(outputs):
        raise ValueError("Way数がDSPの使用可能なOutput数を超えています。")

    outputs_by_id = {output.id: output for output in outputs}
    outputs_by_slot = {(output.dsp_id, output.channel): output for output in outputs}
    assigned: set[str] = set()
    next_ways: list[DSPWay] = []
    for way in system.ways:
        output = outputs_by_id.get(way.output_id)
        if output is None or output.id in assigned:
            output = outputs_by_slot.get((way.dsp_id, way.output_channel))
        if output is None or output.id in assigned:
            output = next(item for item in outputs if item.id not in assigned)
        assigned.add(output.id)
        next_ways.append(apply_output_channel_to_way(way, output))

    return replace(
        system,
        output_channels=tuple(outputs),
        ways=tuple(next_ways),
    )


def update_output_channel(system: DSPSystem, changed: DSPOutputChannel) -> DSPSystem:
    normalized = ensure_output_channel_assignments(system)
    if changed.id not in {output.id for output in normalized.output_channels}:
        raise ValueError("unknown DSP output channel")
    updated = replace(
        normalized,
        output_channels=tuple(
            changed if output.id == changed.id else output
            for output in normalized.output_channels
        ),
        revision=normalized.revision + 1,
    )
    return ensure_output_channel_assignments(updated)


def assign_output_channel_to_way(
    system: DSPSystem,
    *,
    way_id: str,
    output_id: str,
) -> DSPSystem:
    """Assign one physical output to a Way, swapping an occupied output.

    Output-owned Speaker Package/IIR/FIR stay on the physical channel.  Way
    Gain/Delay/Polarity/Mute/Solo and alignment stay on the acoustic Way.
    """

    normalized = ensure_output_channel_assignments(system)
    if output_id not in {output.id for output in normalized.output_channels}:
        raise ValueError("unknown DSP output channel")
    selected = next((way for way in normalized.ways if way.id == way_id), None)
    if selected is None:
        raise ValueError("unknown DSP Way")
    if selected.output_id == output_id:
        return normalized
    occupant = next(
        (way for way in normalized.ways if way.output_id == output_id),
        None,
    )
    next_ways: list[DSPWay] = []
    for way in normalized.ways:
        if way.id == selected.id:
            next_ways.append(replace(way, output_id=output_id))
        elif occupant is not None and way.id == occupant.id:
            next_ways.append(replace(way, output_id=selected.output_id))
        else:
            next_ways.append(way)
    updated = replace(
        normalized,
        ways=tuple(next_ways),
        revision=normalized.revision + 1,
    )
    return ensure_output_channel_assignments(updated)


def assigned_way(system: DSPSystem, output_id: str) -> DSPWay | None:
    normalized = ensure_output_channel_assignments(system)
    return next((way for way in normalized.ways if way.output_id == output_id), None)
