from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal
import uuid

import numpy as np

from ..config import IIRFilter
from ..iir import iir_frequency_response
from .models import DSPInput, DSPInputBinding, DSPRoute, DSPSystem
from .topology import configure_stereo_topology


RoutingPathState = Literal[
    "ready",
    "off",
    "muted",
    "missing_binding",
    "invalid_input",
    "invalid_output",
]


@dataclass(frozen=True)
class RoutingPathStatus:
    """Resolved physical path for one logical Input-to-Way intersection."""

    input_id: str
    way_id: str
    dsp_id: str
    state: RoutingPathState
    route_id: str = ""
    input_channel: int | None = None
    output_channel: int | None = None
    package_connected: bool = False
    message: str = ""

    @property
    def route_ready(self) -> bool:
        return self.state == "ready"

    @property
    def end_to_end_ready(self) -> bool:
        return self.route_ready

    @property
    def source_mode(self) -> Literal["package", "flat"]:
        return "package" if self.package_connected else "flat"


def resolve_routing_path(
    system: DSPSystem,
    input_id: str,
    way_id: str,
) -> RoutingPathStatus:
    """Resolve logical routing and its physical DSP input/output endpoints."""

    input_item = next((item for item in system.inputs if item.id == input_id), None)
    way = next((item for item in system.ways if item.id == way_id), None)
    if input_item is None:
        return RoutingPathStatus(
            input_id,
            way_id,
            way.dsp_id if way is not None else "",
            "invalid_input",
            message="Logical Inputが見つかりません。",
        )
    if way is None:
        return RoutingPathStatus(
            input_id,
            way_id,
            "",
            "invalid_output",
            message="Output channelが見つかりません。",
        )
    route = next(
        (
            item
            for item in system.routes
            if item.input_id == input_id and item.way_id == way_id
        ),
        None,
    )
    common = {
        "input_id": input_id,
        "way_id": way_id,
        "dsp_id": way.dsp_id,
        "route_id": route.id if route is not None else "",
        "output_channel": int(way.output_channel),
        "package_connected": bool(way.source_package_id),
    }
    if route is None or not route.enabled:
        return RoutingPathStatus(**common, state="off", message="RouteはOFFです。")
    if input_item.muted or route.muted:
        return RoutingPathStatus(**common, state="muted", message="InputまたはRouteがMuteです。")
    binding = next(
        (
            item
            for item in system.input_bindings
            if item.input_id == input_id and item.dsp_id == way.dsp_id
        ),
        None,
    )
    if binding is None:
        return RoutingPathStatus(
            **common,
            state="missing_binding",
            message=f"{system.device(way.dsp_id).name}の物理Input割り当てがありません。",
        )
    device = system.device(way.dsp_id)
    bound_common = {**common, "input_channel": int(binding.input_channel)}
    if binding.input_channel < 0 or binding.input_channel >= device.max_input_channels:
        return RoutingPathStatus(
            **bound_common,
            state="invalid_input",
            message=f"物理Inputが{device.name}の範囲外です。",
        )
    conflicts = [
        item
        for item in system.ways
        if item.id != way.id
        and item.dsp_id == way.dsp_id
        and item.output_channel == way.output_channel
    ]
    if (
        way.output_channel < 0
        or way.output_channel >= device.max_output_channels
        or conflicts
    ):
        return RoutingPathStatus(
            **bound_common,
            state="invalid_output",
            message=f"物理Outputが{device.name}で範囲外または重複しています。",
        )
    return RoutingPathStatus(
        **bound_common,
        state="ready",
        message=(
            "接続完了"
            if way.source_package_id
            else "接続完了・Flat source"
        ),
    )


def initialize_default_routing(system: DSPSystem) -> DSPSystem:
    """Create the explicit default routing for a newly constructed system."""

    if system.inputs or system.input_bindings or system.routes:
        raise ValueError("default routing can only initialize an empty system")
    input_item = DSPInput("input-1", "Input 1")
    dsp_ids = tuple(dict.fromkeys(way.dsp_id for way in system.ways))
    bindings = tuple(
        DSPInputBinding(str(uuid.uuid4()), input_item.id, dsp_id, 0)
        for dsp_id in dsp_ids
    )
    routes = tuple(
        DSPRoute(str(uuid.uuid4()), input_item.id, way.id)
        for way in system.ways
    )
    return replace(
        system,
        inputs=(input_item,),
        input_bindings=bindings,
        routes=routes,
    )


def active_routes(system: DSPSystem, *, input_id: str | None = None) -> tuple[DSPRoute, ...]:
    """Return routes whose logical and physical endpoints are both usable."""

    return tuple(
        route for route in system.routes
        if resolve_routing_path(system, route.input_id, route.way_id).route_ready
        and (input_id is None or route.input_id == input_id)
    )


def common_input_iir_filters(system: DSPSystem, way_id: str) -> tuple[IIRFilter, ...]:
    """Return the single Input-IIR chain feeding one Way.

    A single Way FIR can compensate one upstream transfer function.  Multiple
    active Inputs are therefore allowed only when their enabled Input-IIR
    chains are identical.  Input gain/polarity/delay remain independent output
    controls and are deliberately not folded into FIR regeneration.
    """

    inputs = {item.id: item for item in system.inputs}
    chains = [
        tuple(item for item in inputs[route.input_id].iir_filters if item.enabled)
        for route in active_routes(system)
        if route.way_id == way_id and route.input_id in inputs
    ]
    if not chains:
        return ()
    reference = chains[0]
    if any(chain != reference for chain in chains[1:]):
        way = next((item for item in system.ways if item.id == way_id), None)
        name = way.name if way is not None else way_id
        raise ValueError(
            f"{name}: 同じWayへ異なるInput IIRが接続されています。"
            "1つのWay FIRでは同時補正できないため、Input IIRを揃えるかRouteを分けてください。"
        )
    return reference


def route_complex_gain(
    route: DSPRoute,
    frequency: np.ndarray,
    *,
    input_item: DSPInput | None = None,
    sample_rate: int | None = None,
) -> np.ndarray:
    gain_db = float(route.gain_db) + (float(input_item.gain_db) if input_item is not None else 0.0)
    gain = 10.0 ** (gain_db / 20.0)
    polarity_invert = bool(
        input_item.polarity_invert if input_item is not None else False
    )
    polarity = -1.0 if polarity_invert else 1.0
    response = np.full_like(frequency, gain * polarity, dtype=complex)
    if input_item is not None and input_item.iir_filters:
        rate = int(sample_rate or max(2, round(float(np.max(frequency))) * 2))
        response = response * iir_frequency_response(
            tuple(item for item in input_item.iir_filters if item.enabled),
            rate,
            np.asarray(frequency, dtype=float),
        )
    return response


def way_route_gain(system: DSPSystem, way_id: str, frequency: np.ndarray, *, input_id: str | None) -> np.ndarray:
    selected = [route for route in active_routes(system, input_id=input_id) if route.way_id == way_id]
    if not selected:
        return np.zeros_like(frequency, dtype=complex)
    inputs = {item.id: item for item in system.inputs}
    way = next(item for item in system.ways if item.id == way_id)
    sample_rate = system.device(way.dsp_id).sample_rate
    return sum(
        (
            route_complex_gain(
                route,
                frequency,
                input_item=inputs.get(route.input_id),
                sample_rate=sample_rate,
            )
            for route in selected
        ),
        np.zeros_like(frequency, dtype=complex),
    )


def routing_headroom(system: DSPSystem) -> tuple[float, float]:
    """Return coherent and uncorrelated attenuation required in dB."""
    totals: dict[str, list[float]] = {}
    inputs = {item.id: item for item in system.inputs}
    for route in active_routes(system):
        input_gain = float(inputs.get(route.input_id, DSPInput("fallback", "fallback")).gain_db)
        totals.setdefault(route.way_id, []).append(
            10.0 ** ((float(route.gain_db) + input_gain) / 20.0)
        )
    coherent = max((sum(values) for values in totals.values()), default=1.0)
    uncorrelated = max((math.sqrt(sum(value * value for value in values)) for values in totals.values()), default=1.0)
    return max(0.0, 20.0 * math.log10(coherent)), max(0.0, 20.0 * math.log10(uncorrelated))


def unconnected_way_ids(system: DSPSystem) -> tuple[str, ...]:
    connected = {route.way_id for route in active_routes(system)}
    return tuple(way.id for way in system.ways if way.id not in connected)


def toggle_route_connection(system: DSPSystem, input_id: str, way_id: str) -> DSPSystem:
    """Toggle one Input-to-Output matrix intersection without touching other Routes."""
    if input_id not in {item.id for item in system.inputs}:
        raise KeyError(input_id)
    if way_id not in {way.id for way in system.ways}:
        raise KeyError(way_id)
    existing = next(
        (
            route
            for route in system.routes
            if route.input_id == input_id and route.way_id == way_id
        ),
        None,
    )
    if existing is not None and existing.enabled:
        routes = tuple(route for route in system.routes if route.id != existing.id)
    elif existing is not None:
        routes = tuple(
            replace(route, enabled=True) if route.id == existing.id else route
            for route in system.routes
        )
    else:
        routes = system.routes + (DSPRoute(str(uuid.uuid4()), input_id, way_id),)
    return replace(system, routes=routes, revision=system.revision + 1)


def replace_routes_for_preset(system: DSPSystem, preset: str, *, mono_input_id: str | None = None) -> DSPSystem:
    if preset == "Stereo":
        return configure_stereo_topology(system)
    routes: list[DSPRoute] = []
    inputs = list(system.inputs)
    if not inputs:
        return system
    if preset == "1:1":
        pairs = zip(inputs, system.ways)
        routes = [DSPRoute(str(uuid.uuid4()), item.id, way.id) for item, way in pairs]
    elif preset == "Mono to all":
        selected = next((item for item in inputs if item.id == mono_input_id), inputs[0])
        routes = [DSPRoute(str(uuid.uuid4()), selected.id, way.id) for way in system.ways]
    elif preset == "L+R to mono":
        selected = [item for item in inputs if item.role in {"Left", "Right"}]
        routes = [DSPRoute(str(uuid.uuid4()), item.id, way.id, gain_db=-6.020599913279624) for way in system.ways for item in selected]
    elif preset != "Clear":
        raise ValueError(f"unknown routing preset: {preset}")
    return replace(system, routes=tuple(routes), revision=system.revision + 1)
