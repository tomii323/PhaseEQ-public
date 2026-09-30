from __future__ import annotations

from dataclasses import replace
import uuid

from ..config import LinearFIRFilter
from .models import DSPInput, DSPInputBinding, DSPRoute, DSPSystem, DSPSystemTarget, DSPWay
from .crossover_models import CrossoverBoundary, CrossoverPlan, SubCrossover
from .crossover_system import (
    crossover_plan_for_ways,
    ensure_system_crossover_plan,
    is_sub_way,
)
from .output_assignment import ensure_output_channel_assignments


def _filters(*items: tuple[str, float]) -> tuple[LinearFIRFilter, ...]:
    return tuple(
        LinearFIRFilter(mode=mode, response="linear_phase_lr2", fc=frequency)
        for mode, frequency in items
    )


def topology_specs(main_way_count: int, sub_enabled: bool) -> tuple[tuple[str, tuple[LinearFIRFilter, ...]], ...]:
    """Return standard main-way bands, with Sub as an independent optional output."""
    specs = {
        1: (("Main", ()),),
        2: (("Hi", _filters(("hp", 1800.0))), ("Low", _filters(("lp", 1800.0)))),
        3: (
            ("Hi", _filters(("hp", 3000.0))),
            ("Mid", _filters(("hp", 300.0), ("lp", 3000.0))),
            ("Low", _filters(("lp", 300.0))),
        ),
        4: (
            ("Hi", _filters(("hp", 5000.0))),
            ("Mid-Hi", _filters(("hp", 800.0), ("lp", 5000.0))),
            ("Mid-Low", _filters(("hp", 100.0), ("lp", 800.0))),
            ("Low", _filters(("lp", 100.0))),
        ),
    }
    if main_way_count not in specs:
        raise ValueError("main_way_count must be 1, 2, 3, or 4")
    selected = specs[main_way_count]
    if sub_enabled:
        selected += (("Sub", _filters(("lp", 80.0))),)
    return selected


def system_topology(system: DSPSystem) -> tuple[int, bool]:
    plan = system.crossover_plan
    sub_enabled = bool(plan.sub_crossovers) if plan is not None else any(
        way.name.strip().casefold() == "sub" for way in system.ways
    )
    if plan is not None:
        groups = plan.effective_main_way_groups
        group_sizes = {len(group) for group in groups}
        if len(group_sizes) == 1:
            return next(iter(group_sizes)), sub_enabled
    return len(system.ways) - int(sub_enabled), sub_enabled


def system_sub_count(system: DSPSystem) -> int:
    """Return the number of independent Sub output channels."""

    if system.crossover_plan is not None:
        return len(system.crossover_plan.sub_crossovers)
    return sum(is_sub_way(way) for way in system.ways)


def configure_stereo_topology(system: DSPSystem) -> DSPSystem:
    """Expand the current N-way main topology to two independent input channels."""
    system = ensure_output_channel_assignments(system)
    normalized = ensure_system_crossover_plan(system)
    assert normalized.crossover_plan is not None
    main_way_count, _sub_enabled = system_topology(normalized)
    if main_way_count not in {1, 2, 3, 4}:
        raise ValueError("Stereo Quick setup supports 1Way, 2Way, 3Way, or 4Way")

    devices = {device.id: device for device in normalized.devices}
    way_by_id = {way.id: way for way in normalized.ways}
    old_groups = normalized.crossover_plan.effective_main_way_groups
    channel_sources: list[list[DSPWay]] = []
    for group in old_groups[:2]:
        if len(group) == main_way_count and all(way_id in way_by_id for way_id in group):
            channel_sources.append([way_by_id[way_id] for way_id in reversed(group)])
    if not channel_sources:
        main_ways = [way for way in normalized.ways if way.id in normalized.crossover_plan.main_way_ids]
        channel_sources.append(main_ways[:main_way_count])
    while len(channel_sources) < 2:
        channel_sources.append(channel_sources[0])

    specs = topology_specs(main_way_count, False)
    next_ways: list[DSPWay] = []
    main_channel_ways: list[list[DSPWay]] = []
    grouped_ids: list[tuple[str, ...]] = []
    next_output_by_device: dict[str, int] = {device_id: 0 for device_id in devices}
    channel_roles = (("Ch1", "Left"), ("Ch2", "Right"))
    for channel_index, (channel_label, routing_role) in enumerate(channel_roles):
        source_ways = channel_sources[channel_index]
        channel_ways: list[DSPWay] = []
        for band_index, (band_name, _default_filters) in enumerate(specs):
            source = source_ways[band_index]
            device = devices[source.dsp_id]
            output_channel = next_output_by_device[device.id]
            if output_channel >= device.max_output_channels:
                raise ValueError(
                    f"{device.name}: {main_way_count}Way Stereoには"
                    f"{output_channel + 1}出力が必要です（Max output {device.max_output_channels}）。"
                )
            next_output_by_device[device.id] += 1
            reuse_source = channel_index < len(old_groups) and source.id in old_groups[channel_index]
            way_id = source.id if reuse_source else str(uuid.uuid4())
            channel_ways.append(replace(
                source,
                id=way_id,
                name=f"{channel_label} {band_name}",
                acoustic_group=channel_label,
                output_channel=output_channel,
                routing_role=routing_role,
            ))
        next_ways.extend(channel_ways)
        main_channel_ways.append(channel_ways)
        grouped_ids.append(tuple(way.id for way in reversed(channel_ways)))

    sub_ways = [way for way in normalized.ways if way.id not in normalized.crossover_plan.main_way_ids]
    next_subs: list[DSPWay] = []
    for sub in sub_ways:
        device = devices[sub.dsp_id]
        output_channel = next_output_by_device[device.id]
        if output_channel >= device.max_output_channels:
            raise ValueError(
                f"{device.name}: SUBを含めるには{output_channel + 1}出力が必要です"
                f"（Max output {device.max_output_channels}）。"
            )
        next_output_by_device[device.id] += 1
        next_subs.append(replace(sub, output_channel=output_channel, routing_role="LFE"))
    next_ways.extend(next_subs)

    if main_way_count == 1 and len(next_subs) == 2:
        ordered = [main_channel_ways[0][0], next_subs[0], main_channel_ways[1][0], next_subs[1]]
        next_output_by_device = {device_id: 0 for device_id in devices}
        next_ways = []
        for channel_index, way in enumerate(ordered):
            device = devices[way.dsp_id]
            output_channel = next_output_by_device[device.id]
            if output_channel >= device.max_output_channels:
                raise ValueError(
                    f"{device.name}: 1Way + SUB Stereoには4出力が必要です"
                    f"（Max output {device.max_output_channels}）。"
                )
            next_output_by_device[device.id] += 1
            channel_label = f"Ch{channel_index // 2 + 1}"
            name = f"{channel_label} {'Main' if channel_index % 2 == 0 else 'SUB'}"
            next_ways.append(replace(
                way,
                name=name,
                acoustic_group=channel_label,
                output_channel=output_channel,
            ))
        main_channel_ways = [[next_ways[0]], [next_ways[2]]]
        next_subs = [next_ways[1], next_ways[3]]
        grouped_ids = [(next_ways[0].id,), (next_ways[2].id,)]

    used_device_ids = {way.dsp_id for way in next_ways}
    for device_id in used_device_ids:
        device = devices[device_id]
        if device.max_input_channels < 2:
            raise ValueError(f"{device.name}: StereoにはMax input 2以上が必要です。")

    existing_boundaries = {
        (boundary.lower_way_id, boundary.upper_way_id): boundary
        for boundary in normalized.crossover_plan.boundaries
    }
    source_group = old_groups[0]
    source_boundaries = [
        existing_boundaries[(source_group[index], source_group[index + 1])]
        for index in range(len(source_group) - 1)
    ]
    next_boundaries: list[CrossoverBoundary] = []
    for group_index, group in enumerate(grouped_ids):
        for boundary_index in range(len(group) - 1):
            pair = (group[boundary_index], group[boundary_index + 1])
            existing = existing_boundaries.get(pair)
            source = existing or source_boundaries[boundary_index]
            lower = next(way for way in next_ways if way.id == pair[0])
            upper = next(way for way in next_ways if way.id == pair[1])
            next_boundaries.append(replace(
                source,
                id=source.id if existing is not None else f"crossover-{group_index + 1}-{boundary_index + 1}",
                name=f"{lower.name} / {upper.name}",
                lower_way_id=pair[0],
                upper_way_id=pair[1],
            ))

    inputs = list(normalized.inputs[:2])
    while len(inputs) < 2:
        next_index = len(inputs) + 1
        input_id = f"input-{next_index}"
        while any(item.id == input_id for item in inputs):
            next_index += 1
            input_id = f"input-{next_index}"
        inputs.append(DSPInput(input_id, f"Input {len(inputs) + 1}"))
    inputs = [
        replace(inputs[0], name="Input 1", role="Left"),
        replace(inputs[1], name="Input 2", role="Right"),
    ]
    bindings = tuple(
        DSPInputBinding(str(uuid.uuid4()), input_item.id, device_id, input_index)
        for input_index, input_item in enumerate(inputs)
        for device_id in devices
        if device_id in used_device_ids
    )
    routes = tuple(
        DSPRoute(str(uuid.uuid4()), inputs[channel_index].id, way.id)
        for channel_index, group in enumerate(grouped_ids)
        for way in next_ways
        if way.id in group
    ) + tuple(
        DSPRoute(str(uuid.uuid4()), inputs[channel_index].id, sub.id)
        for channel_index, sub in enumerate(next_subs)
        if len(next_subs) == 2
    ) + tuple(
        DSPRoute(
            str(uuid.uuid4()),
            input_item.id,
            next_subs[0].id,
            gain_db=-6.020599913279624,
        )
        for input_item in inputs
        if len(next_subs) == 1
    ) + tuple(
        DSPRoute(
            str(uuid.uuid4()),
            input_item.id,
            sub.id,
            gain_db=-6.020599913279624,
        )
        for sub in next_subs
        for input_item in inputs
        if len(next_subs) > 2
    )
    targets = dict(normalized.targets)
    base_target = targets.get("Main", DSPSystemTarget())
    targets.setdefault("Ch1", base_target)
    targets.setdefault("Ch2", base_target)
    existing_sub_crossovers = {
        item.way_id: item for item in normalized.crossover_plan.sub_crossovers
    }
    source_sub = next(iter(normalized.crossover_plan.sub_crossovers), None)
    next_sub_crossovers = tuple(
        replace(
            existing_sub_crossovers.get(sub.id, source_sub or SubCrossover(sub.id)),
            way_id=sub.id,
        )
        for sub in next_subs
    )
    plan = CrossoverPlan(
        main_way_ids=tuple(way_id for group in grouped_ids for way_id in group),
        boundaries=tuple(next_boundaries),
        sub_crossovers=next_sub_crossovers,
        revision=normalized.crossover_plan.revision + 1,
        overlap_convention=normalized.crossover_plan.overlap_convention,
        main_way_groups=tuple(grouped_ids),
    )
    updated = replace(
        normalized,
        ways=tuple(next_ways),
        inputs=tuple(inputs),
        input_bindings=bindings,
        routes=routes,
        targets=targets,
        crossover_plan=plan,
        revision=normalized.revision + 1,
    )
    updated = ensure_output_channel_assignments(updated)
    updated.validate()
    return updated


def resize_system_topology(
    system: DSPSystem,
    main_way_count: int,
    sub_enabled: bool | int,
) -> DSPSystem:
    """Resize every independent Main channel group without collapsing stereo.

    ``main_way_count`` is the number of bands per logical channel.  ``sub_enabled``
    remains bool-compatible and also accepts an integer number of independent
    Sub outputs.
    """

    system = ensure_output_channel_assignments(system)
    if not system.devices:
        raise ValueError("DSP Systemには1台以上のDeviceが必要です。")
    topology_specs(main_way_count, False)  # validates the supported count
    sub_count = int(sub_enabled) if not isinstance(sub_enabled, bool) else int(sub_enabled)
    if sub_count < 0:
        raise ValueError("Sub output count must not be negative")

    plan = system.crossover_plan
    way_by_id = {way.id: way for way in system.ways}
    old_group_ids = plan.effective_main_way_groups if plan is not None else ()
    old_groups: list[list[DSPWay]] = []
    for group in old_group_ids:
        selected = [way_by_id[way_id] for way_id in reversed(group) if way_id in way_by_id]
        if selected:
            old_groups.append(selected)
    if not old_groups:
        legacy_main = [way for way in system.ways if not is_sub_way(way)]
        old_groups = [legacy_main]
    group_count = max(1, len(old_groups))
    specs = topology_specs(main_way_count, False)
    next_main_groups: list[list[DSPWay]] = []

    def band_matches(way: DSPWay, band_name: str) -> bool:
        name = way.name.strip().casefold()
        band = band_name.casefold()
        return name == band or name.endswith(f" {band}")

    for group_index in range(group_count):
        sources = list(old_groups[group_index]) if group_index < len(old_groups) else []
        unused = list(sources)
        group_label = (
            sources[0].acoustic_group.strip()
            if sources and sources[0].acoustic_group.strip()
            else (f"Ch{group_index + 1}" if group_count > 1 else "Main")
        )
        if group_count > 1 and group_label == "Main":
            group_label = f"Ch{group_index + 1}"
        input_role = (
            system.inputs[group_index].role
            if group_index < len(system.inputs)
            else "Custom"
        )
        group_ways: list[DSPWay] = []
        for band_index, (band_name, _filters_spec) in enumerate(specs):
            reused = next(
                (way for way in unused if band_matches(way, band_name)),
                None,
            )
            if reused is None and unused:
                reused = unused[0]
            if reused is not None:
                unused.remove(reused)
                device = system.device(reused.dsp_id)
            else:
                device_index = 0 if len(system.devices) == 1 or band_index < 2 else 1
                device = system.devices[min(device_index, len(system.devices) - 1)]
                reused = DSPWay(
                    id=str(uuid.uuid4()),
                    name=band_name,
                    acoustic_group=group_label,
                    dsp_id=device.id,
                    output_channel=0,
                    routing_role=input_role,
                )
            display_name = (
                f"{group_label} {band_name}" if group_count > 1 else band_name
            )
            group_ways.append(
                replace(
                    reused,
                    name=display_name,
                    acoustic_group=group_label,
                    dsp_id=device.id,
                    routing_role=(
                        input_role
                        if input_role in {"Left", "Right", "Mono", "Center", "LFE"}
                        else ("Custom" if reused.routing_role == "LFE" else reused.routing_role)
                    ),
                )
            )
        next_main_groups.append(group_ways)

    old_subs = [way for way in system.ways if is_sub_way(way)]
    next_subs: list[DSPWay] = []
    for sub_index in range(sub_count):
        reused = old_subs[sub_index] if sub_index < len(old_subs) else None
        if reused is None:
            device = system.devices[min(sub_index, len(system.devices) - 1)]
            reused = DSPWay(
                id=str(uuid.uuid4()),
                name="Sub",
                acoustic_group="Sub",
                dsp_id=device.id,
                output_channel=0,
                routing_role="LFE",
            )
        group_label = (
            next_main_groups[sub_index][0].acoustic_group
            if group_count > 1 and sub_count == group_count and sub_index < group_count
            else (reused.acoustic_group if reused.acoustic_group.strip() else "Sub")
        )
        next_subs.append(
            replace(
                reused,
                name="Sub" if sub_count == 1 else f"Sub {sub_index + 1}",
                acoustic_group=group_label,
                routing_role="LFE",
            )
        )

    ordered_ways = [
        way for group in next_main_groups for way in group
    ] + next_subs
    next_output_by_device: dict[str, int] = {
        device.id: 0 for device in system.devices
    }
    assigned_ways: list[DSPWay] = []
    for way in ordered_ways:
        device = system.device(way.dsp_id)
        output_channel = next_output_by_device[device.id]
        if output_channel >= device.max_output_channels:
            raise ValueError(
                f"{device.name}: 必要なOutput数がMax outputs "
                f"{device.max_output_channels}を超えます。"
            )
        next_output_by_device[device.id] += 1
        assigned_ways.append(replace(way, output_channel=output_channel))
    ways = tuple(assigned_ways)
    assigned_by_id = {way.id: way for way in ways}
    grouped_ids = tuple(
        tuple(assigned_by_id[way.id].id for way in reversed(group))
        for group in next_main_groups
    )

    next_main_ids = {way.id for group in next_main_groups for way in group}
    # Main routes retain their per-route values. SUB routing is regenerated so
    # switching between shared and per-channel SUB cannot leave stale L/R sums.
    routes = [route for route in system.routes if route.way_id in next_main_ids]
    routed_pairs = {(route.input_id, route.way_id) for route in routes}
    for group_index, group in enumerate(next_main_groups):
        old_ids = {way.id for way in old_groups[group_index]} if group_index < len(old_groups) else set()
        input_ids = list(dict.fromkeys(
            route.input_id for route in system.routes if route.way_id in old_ids
        ))
        if not input_ids and group_index < len(system.inputs):
            input_ids = [system.inputs[group_index].id]
        for way in group:
            for input_id in input_ids:
                if (input_id, way.id) not in routed_pairs:
                    routes.append(DSPRoute(str(uuid.uuid4()), input_id, way.id))
                    routed_pairs.add((input_id, way.id))
    for sub_index, sub in enumerate(next_subs):
        if group_count > 1 and sub_count == group_count and sub_index < len(system.inputs):
            selected_inputs = (system.inputs[sub_index],)
            gain_db = 0.0
        else:
            selected_inputs = system.inputs
            gain_db = -6.020599913279624 if len(selected_inputs) > 1 else 0.0
        routes.extend(
            DSPRoute(str(uuid.uuid4()), input_item.id, sub.id, gain_db=gain_db)
            for input_item in selected_inputs
        )

    fallback_plan = crossover_plan_for_ways(
        ways,
        main_way_groups=grouped_ids,
    )
    if plan is not None:
        existing_boundaries = {
            (item.lower_way_id, item.upper_way_id): item for item in plan.boundaries
        }
        existing_subs = {item.way_id: item for item in plan.sub_crossovers}
        fallback_plan = replace(
            fallback_plan,
            boundaries=tuple(
                existing_boundaries.get(
                    (item.lower_way_id, item.upper_way_id),
                    item,
                )
                for item in fallback_plan.boundaries
            ),
            sub_crossovers=tuple(
                existing_subs.get(item.way_id, item)
                for item in fallback_plan.sub_crossovers
            ),
            revision=plan.revision + 1,
            overlap_convention=plan.overlap_convention,
        )
    updated = replace(
        system,
        ways=ways,
        routes=tuple(routes),
        crossover_plan=fallback_plan,
        revision=system.revision + 1,
    )
    updated = ensure_output_channel_assignments(updated)
    updated.validate()
    return updated
