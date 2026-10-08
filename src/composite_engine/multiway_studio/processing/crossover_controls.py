"""Pure state contract shared by every Studio crossover boundary control."""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass

from crossover_engine.recipe import METHODS


TARGET_METHODS = frozenset({
    "Linear-phase LR2 FIR", "Linear-phase LR4 FIR", "LR2", "LR4",
})


@dataclass(frozen=True)
class BoundaryControlSnapshot:
    methods: tuple[str, ...]
    acoustic_targets: tuple[bool, ...]


def normalize_method(value: object, fallback: str = "Kaiser FIR") -> str:
    value = str(value).strip()
    return value if value in METHODS else fallback


def method_state_key(mode_key: str, index: int) -> str:
    return f"composite_studio_{mode_key}_crossover_method_{index}"


def target_state_key(mode_key: str, index: int) -> str:
    return f"composite_studio_{mode_key}_acoustic_target_{index}"


def acoustic_target_allowed(method: object) -> bool:
    return normalize_method(method) in TARGET_METHODS


def boundary_control_snapshot(
    state: Mapping[str, object],
    mode_key: str,
    boundary_count: int,
    *,
    fallback_methods: Sequence[object] = (),
    fallback_targets: Sequence[object] = (),
) -> BoundaryControlSnapshot:
    """Read and validate all boundary controls as one immutable snapshot."""
    methods: list[str] = []
    targets: list[bool] = []
    for index in range(max(0, int(boundary_count))):
        fallback_method = normalize_method(
            fallback_methods[index] if index < len(fallback_methods) else "Kaiser FIR"
        )
        method = normalize_method(
            state.get(method_state_key(mode_key, index), fallback_method),
            fallback_method,
        )
        fallback_target = bool(
            fallback_targets[index] if index < len(fallback_targets) else False
        )
        target = bool(
            state.get(target_state_key(mode_key, index), fallback_target)
        ) and acoustic_target_allowed(method)
        methods.append(method)
        targets.append(target)
    return BoundaryControlSnapshot(tuple(methods), tuple(targets))


def seed_boundary_controls(
    state: MutableMapping[str, object],
    mode_key: str,
    methods: Sequence[object],
    acoustic_targets: Sequence[object],
    *,
    overwrite: bool,
) -> BoundaryControlSnapshot:
    """Seed widget state from persisted state without mixing mode ownership."""
    normalized_methods = tuple(normalize_method(value) for value in methods)
    normalized_targets = tuple(
        bool(acoustic_targets[index])
        and acoustic_target_allowed(normalized_methods[index])
        if index < len(acoustic_targets) else False
        for index in range(len(normalized_methods))
    )
    for index, method in enumerate(normalized_methods):
        method_key = method_state_key(mode_key, index)
        target_key = target_state_key(mode_key, index)
        if overwrite or method_key not in state:
            state[method_key] = method
        if overwrite or target_key not in state:
            state[target_key] = normalized_targets[index]
    return boundary_control_snapshot(
        state, mode_key, len(normalized_methods),
        fallback_methods=normalized_methods,
        fallback_targets=normalized_targets,
    )
