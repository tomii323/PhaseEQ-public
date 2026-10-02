# -*- coding: utf-8 -*-
"""State-safe Streamlit widget helpers for APPY applications.

The helpers keep the persisted setting key separate from the widget key.
They store and return actual values instead of indexes, so option ordering
changes do not silently alter restored settings.
"""

from __future__ import annotations
from utils.ui_localization import ui_message, display_text

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any
import unicodedata
from list_menu import core as list_core
from list_menu_state import resolve_choice
from .list_menu_ui import choice_widget
from .ui_state_commit import bind_setting_commit


UI_REGISTRY_STATE_KEY = "__ui_index_registry"
UI_LIST_REGISTRY_STATE_KEY = "__ui_index_list_registry"
UI_DIRTY_BASELINES_STATE_KEY = "__ui_index_dirty_baselines"
UI_DEPENDENCIES_STATE_KEY = "__ui_index_dependencies"
UI_ACTIVE_SCOPE_STATE_KEY = "__ui_index_active_scope"
UI_SCOPE_REGISTRY_STATE_KEY = "__ui_index_scope_registry"
UI_ENTITY_REFS_STATE_KEY = "__ui_index_entity_refs"


def safe_widget_key(prefix: str, settings_key: str, key: str | None = None) -> str:
    """Return a widget key that avoids colliding with the persisted setting key."""
    if not key or key == settings_key:
        return f"{prefix}{settings_key}"
    return key


def label_to_name(label: str) -> str:
    """Extract a readable device name from a PortAudio-like label."""
    value = str(label)
    idx = value.find("]")
    if idx != -1:
        value = value[idx + 1 :]
    value = value.strip()
    if value.startswith("★"):
        value = value.lstrip("★").strip()
    split_at = value.find(" | ")
    if split_at != -1:
        value = value[:split_at]
    return unicodedata.normalize("NFKC", value)


def _coerce(value: Any, value_coerce: Callable[[Any], Any] | None) -> Any:
    return value_coerce(value) if value_coerce else value


def _normalize(value: Any, normalize: Callable[[Any], Any] | None) -> Any:
    return normalize(value) if normalize else value


def _labels_map(values: Sequence[Any], labels: Sequence[str] | None) -> dict[Any, str]:
    return {value: (labels[idx] if labels and idx < len(labels) else str(value)) for idx, value in enumerate(values)}


def _current_value(
    *,
    settings_key: str,
    values: Sequence[Any],
    default: Any | None,
    value_coerce: Callable[[Any], Any] | None,
) -> Any:
    fallback = default if default in values else values[0]
    current = _coerce(_session_state().get(settings_key, fallback), value_coerce)
    return resolve_choice(values, current, fallback)


def _session_state():
    import streamlit as st

    return st.session_state


def selectbox_indexed(
    label: str,
    values: Sequence[Any],
    *,
    settings_key: str,
    labels: Sequence[str] | None = None,
    default: Any | None = None,
    value_coerce: Callable[[Any], Any] | None = None,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> Any:
    """Selectbox that stores and returns the selected value."""
    import streamlit as st

    options = list(values)
    if not options:
        st.session_state[settings_key] = None
        return None

    widget_key = safe_widget_key("__sel_val_", settings_key, key)
    current = _current_value(settings_key=settings_key, values=options, default=default, value_coerce=value_coerce)
    widget_args = {
        "key": widget_key,
        "help": help,
        "disabled": disabled,
        "format_func": lambda value: _labels_map(options, labels).get(value, str(value)),
        **widget_kwargs,
    }
    bind_setting_commit(widget_args, settings_key, lambda value: _coerce(value, value_coerce))
    selected = _coerce(choice_widget(label, options, current=current, default=current, **widget_args), value_coerce)
    register_ui_entry(
        kind="selectbox",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        options=options,
        metadata={"disabled": disabled},
    )
    st.session_state[settings_key] = selected
    return selected


def radio_indexed(
    label: str,
    values: Sequence[Any],
    *,
    settings_key: str,
    labels: Sequence[str] | None = None,
    default: Any | None = None,
    value_coerce: Callable[[Any], Any] | None = None,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> Any:
    """Radio that stores and returns the selected value."""
    import streamlit as st

    options = list(values)
    if not options:
        st.session_state[settings_key] = None
        return None

    widget_key = safe_widget_key("__rad_val_", settings_key, key)
    current = _current_value(settings_key=settings_key, values=options, default=default, value_coerce=value_coerce)
    widget_args = {
        "key": widget_key,
        "help": help,
        "disabled": disabled,
        "format_func": lambda value: _labels_map(options, labels).get(value, str(value)),
        **widget_kwargs,
    }
    # Remounts must not fall back to the first option while saved state differs.
    widget_args["index"] = options.index(current)
    bind_setting_commit(widget_args, settings_key, lambda value: _coerce(value, value_coerce))
    selected = _coerce(choice_widget(label, options, kind="radio", current=current, default=current, **widget_args), value_coerce)
    register_ui_entry(
        kind="radio",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        options=options,
        metadata={"disabled": disabled},
    )
    st.session_state[settings_key] = selected
    return selected


def segmented_control_indexed(
    label: str,
    values: Sequence[Any],
    *,
    settings_key: str,
    labels: Sequence[str] | None = None,
    default: Any | None = None,
    value_coerce: Callable[[Any], Any] | None = None,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> Any:
    """Segmented control that stores and returns the selected value."""
    import streamlit as st

    from .ui_localization import localized_segmented_control

    options = list(values)
    if not options:
        st.session_state[settings_key] = None
        return None

    widget_key = safe_widget_key("__seg_val_", settings_key, key)
    current = _current_value(settings_key=settings_key, values=options, default=default, value_coerce=value_coerce)
    widget_args = {
        "key": widget_key,
        "help": help,
        "disabled": disabled,
        "format_func": lambda value: _labels_map(options, labels).get(value, str(value)),
        **widget_kwargs,
    }
    # Keep the browser remount default aligned with the persisted selection.
    widget_args["default"] = current
    widget_args.pop("default", None)
    bind_setting_commit(widget_args, settings_key,
                        lambda value: current if value is None else _coerce(value, value_coerce))
    selected = _coerce(choice_widget(label, options, kind="segmented_control", renderer=localized_segmented_control,
                                    current=current, default=current, **widget_args), value_coerce)
    if selected is None:
        selected = current
    register_ui_entry(
        kind="segmented_control",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        options=options,
        metadata={"disabled": disabled},
    )
    st.session_state[settings_key] = selected
    return selected


def multiselect_indexed(
    label: str,
    values: Sequence[Any],
    *,
    settings_key: str,
    labels: Sequence[str] | None = None,
    default_values: Iterable[Any] | None = None,
    value_coerce: Callable[[Any], Any] | None = None,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> list[Any]:
    """Multiselect that stores and returns selected values."""
    import streamlit as st

    options = list(values)
    if not options:
        st.session_state[settings_key] = []
        return []

    widget_key = safe_widget_key("__multi_val_", settings_key, key)
    current_values = list(st.session_state.get(widget_key, st.session_state.get(settings_key, list(default_values or []))) or [])
    if value_coerce:
        current_values = [_coerce(value, value_coerce) for value in current_values]
    selected_defaults: list[Any] = []
    for value in current_values:
        if value in options and value not in selected_defaults:
            selected_defaults.append(value)

    widget_args = {
        "key": widget_key,
        "help": help,
        "disabled": disabled,
        "format_func": lambda value: _labels_map(options, labels).get(value, str(value)),
        **widget_kwargs,
    }
    if widget_key in st.session_state and list(st.session_state[widget_key] or []) != selected_defaults:
        st.session_state.pop(widget_key, None)
    widget_args["default"] = selected_defaults
    bind_setting_commit(widget_args, settings_key,
                        lambda values: [_coerce(value, value_coerce) for value in (values or [])])
    selected = list(st.multiselect(label, options, **widget_args))
    if value_coerce:
        selected = [_coerce(value, value_coerce) for value in selected]
    register_ui_entry(
        kind="multiselect",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        options=options,
        metadata={"disabled": disabled},
    )
    st.session_state[settings_key] = selected
    return selected


def number_input_stateful_safe(
    label: str,
    *,
    settings_key: str,
    min_value: int | float,
    max_value: int | float,
    default: int | float,
    step: int | float = 1,
    normalize: Callable[[Any], Any] | None = None,
    rerun_on_normalize: bool = False,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> Any:
    """Number input that clamps existing values before widget creation."""
    import streamlit as st

    widget_key = safe_widget_key("__num_val_", settings_key, key)
    bounded_max = max(min_value, max_value)
    current = st.session_state.get(settings_key, default)
    try:
        current_number = float(current)
    except Exception:
        current_number = float(default)
    current_number = max(float(min_value), min(float(bounded_max), current_number))
    current_number = _normalize(current_number, normalize)
    current_number = max(float(min_value), min(float(bounded_max), float(current_number)))
    desired_widget_value = type(default)(current_number)
    if widget_key in st.session_state:
        try:
            widget_number = float(st.session_state.get(widget_key))
            if widget_number < float(min_value) or widget_number > float(bounded_max):
                normalized_widget_number = _normalize(widget_number, normalize)
                if float(min_value) <= float(normalized_widget_number) <= float(bounded_max):
                    st.session_state[widget_key] = type(default)(normalized_widget_number)
                else:
                    st.session_state[widget_key] = desired_widget_value
            elif (
                normalize is not None
                and _normalize(widget_number, normalize) != widget_number
                and _normalize(widget_number, normalize) == current_number
            ):
                st.session_state[widget_key] = desired_widget_value
        except Exception:
            st.session_state[widget_key] = desired_widget_value

    widget_args = {
        "min_value": min_value,
        "max_value": bounded_max,
        "step": step,
        "key": widget_key,
        "help": help,
        "disabled": disabled,
        **widget_kwargs,
    }
    # Explicitly send the current value even when the server key survives a
    # browser remount. Incoming edits take priority over a previously loaded
    # setting; state-only initialization avoids two competing value sources.
    st.session_state[widget_key] = type(default)(
        st.session_state.get(widget_key, desired_widget_value)
    )
    # Streamlit sends this explicit state as the browser value, including on
    # remount. Do not also provide a default after a programmatic restore.
    widget_args.pop("value", None)
    # Return-value assignment alone loses edits when another event skips this
    # renderer. Preserve caller callbacks, but commit domain state first.
    bind_setting_commit(widget_args, settings_key,
                        lambda value: _normalize(max(min_value, min(bounded_max, value)), normalize))
    raw_value = st.number_input(display_text(label), **widget_args)
    value = _normalize(raw_value, normalize)
    register_ui_entry(
        kind="number_input",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        metadata={
            "min_value": min_value,
            "max_value": bounded_max,
            "step": step,
            "disabled": disabled,
            "normalized": value != raw_value,
        },
    )
    st.session_state[settings_key] = value
    if rerun_on_normalize and value != raw_value:
        st.rerun()
    return value


def slider_stateful_safe(
    label: str,
    *,
    settings_key: str,
    min_value: int | float,
    max_value: int | float,
    default: int | float,
    step: int | float | None = None,
    normalize: Callable[[Any], Any] | None = None,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> Any:
    """Slider that clamps existing values before widget creation."""
    import streamlit as st

    widget_key = safe_widget_key("__slider_val_", settings_key, key)
    bounded_max = max(min_value, max_value)
    current = st.session_state.get(settings_key, default)
    try:
        current_number = float(current)
    except Exception:
        current_number = float(default)
    current_number = max(float(min_value), min(float(bounded_max), current_number))
    current_number = _normalize(current_number, normalize)
    current_number = max(float(min_value), min(float(bounded_max), float(current_number)))
    desired_widget_value = type(default)(current_number)
    if widget_key in st.session_state:
        try:
            widget_number = float(st.session_state.get(widget_key))
            if widget_number < float(min_value) or widget_number > float(bounded_max):
                st.session_state[widget_key] = desired_widget_value
        except Exception:
            st.session_state[widget_key] = desired_widget_value

    widget_args = {
        "min_value": min_value,
        "max_value": bounded_max,
        "key": widget_key,
        "help": help,
        "disabled": disabled,
        **widget_kwargs,
    }
    # Restore through state alone, including browser remounts and clamped values.
    # Providing a default as well triggers Streamlit's duplicate-value warning.
    st.session_state[widget_key] = type(default)(
        st.session_state.get(widget_key, desired_widget_value)
    )
    widget_args.pop("value", None)
    bind_setting_commit(widget_args, settings_key, lambda value: _normalize(value, normalize))
    if step is not None:
        widget_args["step"] = step
    value = st.slider(display_text(label), **widget_args)
    value = _normalize(value, normalize)
    register_ui_entry(
        kind="slider",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        metadata={
            "min_value": min_value,
            "max_value": bounded_max,
            "step": step,
            "disabled": disabled,
        },
    )
    st.session_state[settings_key] = value
    return value


def checkbox_stateful(
    label: str,
    *,
    settings_key: str,
    default: bool = False,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> bool:
    """Checkbox that stores and returns a boolean setting."""
    import streamlit as st

    widget_key = safe_widget_key("__chk_val_", settings_key, key)
    widget_args = {"key": widget_key, "help": help, "disabled": disabled, **widget_kwargs}
    # State is the sole initialization source. Streamlit sends this explicit
    # value to a remounted browser control without a competing default.
    st.session_state[widget_key] = bool(
        st.session_state.get(widget_key, st.session_state.get(settings_key, default))
    )
    widget_args.pop("value", None)
    bind_setting_commit(widget_args, settings_key, bool)
    value = bool(st.checkbox(display_text(label), **widget_args))
    register_ui_entry(
        kind="checkbox",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        metadata={"disabled": disabled},
    )
    st.session_state[settings_key] = value
    return value


def text_input_stateful(
    label: str,
    *,
    settings_key: str,
    default: str = "",
    normalize: Callable[[str], str] | None = None,
    key: str | None = None,
    help: str | None = None,
    disabled: bool = False,
    **widget_kwargs: Any,
) -> str:
    """Text input that stores and returns a string setting."""
    import streamlit as st

    widget_key = safe_widget_key("__text_val_", settings_key, key)
    widget_args = {"key": widget_key, "help": help, "disabled": disabled, **widget_kwargs}
    # One source for restored values. Writing before registration also sends
    # the current value to a remounted browser control (not an empty default).
    st.session_state[widget_key] = str(
        st.session_state.get(widget_key, st.session_state.get(settings_key, default))
    )
    widget_args.pop("value", None)
    bind_setting_commit(widget_args, settings_key, lambda value: _normalize(str(value), normalize))
    value = str(st.text_input(label, **widget_args))
    if normalize:
        value = normalize(value)
    register_ui_entry(
        kind="text_input",
        label=label,
        settings_key=settings_key,
        widget_key=widget_key,
        metadata={"disabled": disabled},
    )
    st.session_state[settings_key] = value
    return value


# ---------------------------------------------------------------------------
# UI registry and reverse lookup
# ---------------------------------------------------------------------------


def _json_friendly(value: Any) -> Any:
    """Return a session-state friendly copy for registry metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_friendly(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_friendly(item) for item in value]
    return str(value)


def _registry_key(kind: str, settings_key: str | None, widget_key: str | None, label: str) -> str:
    base = widget_key or settings_key or label
    return f"{kind}:{base}"


def _registry_state(name: str) -> dict[str, dict[str, Any]]:
    state = _session_state()
    raw_registry = state.get(name, {})
    if not isinstance(raw_registry, dict):
        raw_registry = {}
    state[name] = raw_registry
    return raw_registry


def clear_ui_registry(*, include_lists: bool = False) -> None:
    """Clear the runtime UI registry.

    Apps may call this once near the beginning of each rerun when they want the
    registry to represent only the currently rendered page.
    """
    state = _session_state()
    state[UI_REGISTRY_STATE_KEY] = {}
    if include_lists:
        state[UI_LIST_REGISTRY_STATE_KEY] = {}


def register_ui_entry(
    *,
    kind: str,
    label: str,
    settings_key: str | None = None,
    widget_key: str | None = None,
    path: Sequence[str] | None = None,
    options: Sequence[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register one rendered UI control for reverse lookup."""
    registry = _registry_state(UI_REGISTRY_STATE_KEY)
    entry_key = _registry_key(kind, settings_key, widget_key, label)
    entry = {
        "kind": kind,
        "label": str(label),
        "settings_key": settings_key,
        "widget_key": widget_key,
        "path": list(path or []),
        "options": _json_friendly(list(options)) if options is not None else None,
        "metadata": _json_friendly(metadata or {}),
    }
    registry[entry_key] = entry
    return deepcopy(entry)


def register_section_tree(sections: Sequence["UISection"], *, path: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Register navigation sections so they appear in reverse lookup results."""
    registered: list[dict[str, Any]] = []
    base_path = list(path or [])

    def walk(section: UISection, parent_path: list[str], level: int) -> None:
        section_path = [*parent_path, section.id]
        registered.append(
            register_ui_entry(
                kind="section",
                label=section.label,
                settings_key=section.id,
                widget_key=None,
                path=section_path,
                metadata={
                    "status": section.status,
                    "icon": section.icon,
                    "disabled": section.disabled,
                    "description": section.description,
                    "parent": parent_path[-1] if len(parent_path) > len(base_path) else None,
                    "level": level,
                    "child_count": len(section.children),
                },
            )
        )
        for child in section.children:
            walk(child, section_path, level + 1)

    for section in sections:
        walk(section, base_path, 0)
    return registered


def register_list_schema(
    *,
    settings_key: str,
    item_type: str = "item",
    label: str | None = None,
    fields: Sequence[str] | dict[str, Any] | None = None,
    path: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a stateful list definition for inspection."""
    registry = _registry_state(UI_LIST_REGISTRY_STATE_KEY)
    if isinstance(fields, dict):
        field_names = list(fields.keys())
        field_defaults = _json_friendly(fields)
    else:
        field_names = list(fields or [])
        field_defaults = None
    items = _state_list(settings_key)
    active_count = len([item for item in items if bool(item.get("enabled", True))])
    entry = {
        "kind": "list",
        "label": label or settings_key,
        "settings_key": settings_key,
        "item_type": item_type,
        "fields": field_names,
        "field_defaults": field_defaults,
        "path": list(path or []),
        "count": len(items),
        "active_count": active_count,
        "metadata": _json_friendly(metadata or {}),
    }
    registry[settings_key] = entry
    return deepcopy(entry)


def registered_ui_entries(
    *,
    kind: str | None = None,
    settings_key: str | None = None,
    widget_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return registered UI controls."""
    entries = list(_registry_state(UI_REGISTRY_STATE_KEY).values())
    if kind is not None:
        entries = [entry for entry in entries if entry.get("kind") == kind]
    if settings_key is not None:
        entries = [entry for entry in entries if entry.get("settings_key") == settings_key]
    if widget_key is not None:
        entries = [entry for entry in entries if entry.get("widget_key") == widget_key]
    return deepcopy(entries)


def registered_list_entries(*, settings_key: str | None = None, item_type: str | None = None) -> list[dict[str, Any]]:
    """Return registered stateful list definitions."""
    entries = list(_registry_state(UI_LIST_REGISTRY_STATE_KEY).values())
    if settings_key is not None:
        entries = [entry for entry in entries if entry.get("settings_key") == settings_key]
    if item_type is not None:
        entries = [entry for entry in entries if entry.get("item_type") == item_type]
    return deepcopy(entries)


def reverse_lookup_ui(query: str, *, exact: bool = False, include_lists: bool = True) -> list[dict[str, Any]]:
    """Find registered controls/lists by label, settings key, widget key, or path."""
    needle = str(query)
    needle_folded = needle.casefold()

    def matches(value: Any) -> bool:
        haystack = str(value or "")
        if exact:
            return haystack == needle
        return needle_folded in haystack.casefold()

    results: list[dict[str, Any]] = []
    for entry in registered_ui_entries():
        searchable = [
            entry.get("kind"),
            entry.get("label"),
            entry.get("settings_key"),
            entry.get("widget_key"),
            "/".join(str(part) for part in entry.get("path", [])),
        ]
        if any(matches(value) for value in searchable):
            found = deepcopy(entry)
            found["registry"] = "ui"
            results.append(found)
    if include_lists:
        for entry in registered_list_entries():
            searchable = [
                entry.get("kind"),
                entry.get("label"),
                entry.get("settings_key"),
                entry.get("item_type"),
                "/".join(str(part) for part in entry.get("path", [])),
                " ".join(str(field) for field in entry.get("fields", [])),
            ]
            if any(matches(value) for value in searchable):
                found = deepcopy(entry)
                found["registry"] = "list"
                results.append(found)
    return results


def reverse_lookup_state_key(settings_key: str) -> list[dict[str, Any]]:
    """Find UI/list registrations attached to a settings key."""
    return [
        *registered_ui_entries(settings_key=settings_key),
        *registered_list_entries(settings_key=settings_key),
    ]


def registered_settings_keys(*, include_lists: bool = True) -> list[str]:
    """Return known settings keys from the registry."""
    keys = {
        str(entry.get("settings_key"))
        for entry in registered_ui_entries()
        if entry.get("settings_key") is not None
    }
    if include_lists:
        keys.update(
            str(entry.get("settings_key"))
            for entry in registered_list_entries()
            if entry.get("settings_key") is not None
        )
    return sorted(keys)


def unregistered_state_keys(
    *,
    exclude_prefixes: Sequence[str] = ("_", "__", "$"),
    registered_prefixes: Sequence[str] = (),
) -> list[str]:
    """Return session_state keys that are not known to the registry."""
    state = _session_state()
    known = set(registered_settings_keys(include_lists=True))
    known.update(
        str(entry.get("widget_key"))
        for entry in registered_ui_entries()
        if entry.get("widget_key") is not None
    )
    ignored_prefixes = tuple(exclude_prefixes)
    allowed_prefixes = tuple(registered_prefixes)
    unknown: list[str] = []
    for key in state.keys():
        key_str = str(key)
        if key_str in (UI_REGISTRY_STATE_KEY, UI_LIST_REGISTRY_STATE_KEY):
            continue
        if key_str in known:
            continue
        if ignored_prefixes and key_str.startswith(ignored_prefixes):
            continue
        if allowed_prefixes and any(key_str.startswith(prefix) for prefix in allowed_prefixes):
            continue
        unknown.append(key_str)
    return sorted(unknown)


def registry_summary() -> dict[str, Any]:
    """Return counts useful for diagnostics screens."""
    ui_entries = registered_ui_entries()
    list_entries = registered_list_entries()
    by_kind: dict[str, int] = {}
    for entry in ui_entries:
        kind = str(entry.get("kind", "unknown"))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {
        "ui_entry_count": len(ui_entries),
        "list_entry_count": len(list_entries),
        "settings_key_count": len(registered_settings_keys(include_lists=True)),
        "by_kind": by_kind,
    }


# ---------------------------------------------------------------------------
# Scope and entity references
# ---------------------------------------------------------------------------


def normalize_scope(scope: str | Sequence[str] | None = None) -> str:
    """Return a stable scope id. Empty scope means global state."""
    if scope is None:
        return "global"
    if isinstance(scope, str):
        value = scope.strip()
    else:
        value = "/".join(str(part).strip() for part in scope if str(part).strip())
    return value or "global"


def set_active_scope(scope: str | Sequence[str] | None) -> str:
    """Set the current UI editing scope and return it."""
    scope_id = normalize_scope(scope)
    _session_state()[UI_ACTIVE_SCOPE_STATE_KEY] = scope_id
    register_scope(scope_id)
    return scope_id


def get_active_scope(default: str = "global") -> str:
    """Return the current UI editing scope."""
    return normalize_scope(_session_state().get(UI_ACTIVE_SCOPE_STATE_KEY, default))


def scoped_settings_key(key: str, scope: str | Sequence[str] | None = None) -> str:
    """Return a session key namespaced by scope."""
    scope_id = normalize_scope(scope if scope is not None else get_active_scope())
    if scope_id == "global":
        return str(key)
    safe_scope = scope_id.replace(":", ".").replace("/", ".")
    return f"{safe_scope}.{key}"


def register_scope(
    scope: str | Sequence[str] | None,
    *,
    label: str = "",
    parent: str | Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a project/system/slot/profile scope."""
    scope_id = normalize_scope(scope)
    registry = _registry_state(UI_SCOPE_REGISTRY_STATE_KEY)
    entry = {
        "scope": scope_id,
        "label": label or scope_id,
        "parent": normalize_scope(parent) if parent is not None else None,
        "metadata": _json_friendly(metadata or {}),
    }
    registry[scope_id] = entry
    return deepcopy(entry)


def registered_scopes() -> list[dict[str, Any]]:
    """Return registered scopes."""
    return deepcopy(list(_registry_state(UI_SCOPE_REGISTRY_STATE_KEY).values()))


def register_entity_ref(
    *,
    kind: str,
    entity_id: str,
    settings_key: str,
    scope: str | Sequence[str] | None = None,
    label: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register a DB entity reference used by the current UI/profile."""
    scope_id = normalize_scope(scope if scope is not None else get_active_scope())
    registry = _registry_state(UI_ENTITY_REFS_STATE_KEY)
    entry_key = f"{scope_id}:{settings_key}"
    entry = {
        "scope": scope_id,
        "kind": kind,
        "entity_id": entity_id,
        "settings_key": settings_key,
        "label": label or entity_id,
        "metadata": _json_friendly(metadata or {}),
    }
    registry[entry_key] = entry
    return deepcopy(entry)


def entity_refs(
    *,
    scope: str | Sequence[str] | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """Return registered entity references."""
    scope_id = normalize_scope(scope) if scope is not None else None
    refs = list(_registry_state(UI_ENTITY_REFS_STATE_KEY).values())
    if scope_id is not None:
        refs = [ref for ref in refs if ref.get("scope") == scope_id]
    if kind is not None:
        refs = [ref for ref in refs if ref.get("kind") == kind]
    return deepcopy(refs)


def clear_entity_refs(scope: str | Sequence[str] | None = None) -> None:
    """Clear entity references globally or for one scope."""
    registry = _registry_state(UI_ENTITY_REFS_STATE_KEY)
    if scope is None:
        registry.clear()
        return
    scope_id = normalize_scope(scope)
    for key in list(registry.keys()):
        if registry[key].get("scope") == scope_id:
            registry.pop(key, None)


# ---------------------------------------------------------------------------
# Registry snapshot, health check, validation
# ---------------------------------------------------------------------------


def _state_value_snapshot(
    *,
    keys: Sequence[str] | None = None,
    include_widget_keys: bool = False,
    exclude_prefixes: Sequence[str] = ("__",),
) -> dict[str, Any]:
    state = _session_state()
    if keys is None:
        selected_keys = registered_settings_keys(include_lists=True)
        if include_widget_keys:
            selected_keys.extend(
                str(entry.get("widget_key"))
                for entry in registered_ui_entries()
                if entry.get("widget_key") is not None
            )
    else:
        selected_keys = [str(key) for key in keys]
    ignored_prefixes = tuple(exclude_prefixes)
    snapshot: dict[str, Any] = {}
    for key in sorted(set(selected_keys)):
        if ignored_prefixes and key.startswith(ignored_prefixes):
            continue
        if key in state:
            snapshot[key] = _json_friendly(state.get(key))
    return snapshot


def export_ui_registry_snapshot(
    *,
    include_state_values: bool = True,
    include_widget_values: bool = False,
    include_unregistered_keys: bool = True,
) -> dict[str, Any]:
    """Return a portable diagnostic snapshot of the UI registry."""
    snapshot = {
        "schema": "appy.ui_index.snapshot.v1",
        "summary": registry_summary(),
        "active_scope": get_active_scope(),
        "scopes": registered_scopes(),
        "entity_refs": entity_refs(),
        "ui_entries": registered_ui_entries(),
        "list_entries": registered_list_entries(),
        "dependencies": dependency_map(),
    }
    if include_state_values:
        snapshot["state_values"] = _state_value_snapshot(include_widget_keys=include_widget_values)
    if include_unregistered_keys:
        snapshot["unregistered_state_keys"] = unregistered_state_keys()
    return deepcopy(snapshot)


def export_ui_registry_snapshot_json(**kwargs: Any) -> str:
    """Return the registry snapshot as formatted JSON."""
    return json.dumps(export_ui_registry_snapshot(**kwargs), ensure_ascii=False, indent=2, sort_keys=True)


def ui_health_check() -> dict[str, Any]:
    """Check registry consistency and return warnings/errors."""
    issues: list[dict[str, Any]] = []
    ui_entries = registered_ui_entries()
    list_entries = registered_list_entries()

    def add_issue(level: str, code: str, message: str, **extra: Any) -> None:
        issues.append({"level": level, "code": code, "message": message, **_json_friendly(extra)})

    settings_seen: dict[str, int] = {}
    widget_seen: dict[str, int] = {}
    for entry in ui_entries:
        settings_key = entry.get("settings_key")
        widget_key = entry.get("widget_key")
        if settings_key:
            settings_seen[str(settings_key)] = settings_seen.get(str(settings_key), 0) + 1
        if widget_key:
            widget_seen[str(widget_key)] = widget_seen.get(str(widget_key), 0) + 1
        if not entry.get("label"):
            add_issue("warning", "missing_label", "UI entry has no label.", entry=entry)
        if settings_key and widget_key and settings_key == widget_key:
            add_issue(
                "error",
                "settings_widget_key_collision",
                "settings_key and widget_key should be separated.",
                settings_key=settings_key,
            )
        if entry.get("kind") in {"selectbox", "radio", "segmented_control", "multiselect"} and not entry.get("options"):
            add_issue("warning", "missing_options", "Choice widget has no registered options.", entry=entry)

    for widget_key, count in widget_seen.items():
        if count > 1:
            add_issue("error", "duplicate_widget_key", "widget_key is registered multiple times.", widget_key=widget_key, count=count)
    for settings_key, count in settings_seen.items():
        if count > 1:
            add_issue(
                "info",
                "shared_settings_key",
                "settings_key is used by multiple controls.",
                settings_key=settings_key,
                count=count,
            )

    for entry in list_entries:
        if not entry.get("fields"):
            add_issue("info", "list_without_fields", "List schema has no registered fields.", settings_key=entry.get("settings_key"))
        if int(entry.get("active_count", 0)) > int(entry.get("count", 0)):
            add_issue("error", "invalid_list_count", "active_count is greater than count.", entry=entry)

    scope_ids = {entry.get("scope") for entry in registered_scopes()}
    for ref in entity_refs():
        if ref.get("scope") not in scope_ids and ref.get("scope") != "global":
            add_issue(
                "warning",
                "entity_ref_unknown_scope",
                "Entity reference points to an unregistered scope.",
                ref=ref,
            )

    unknown = unregistered_state_keys()
    if unknown:
        add_issue(
            "info",
            "unregistered_state_keys",
            "session_state contains keys that are not registered.",
            count=len(unknown),
            keys=unknown[:50],
        )

    return {
        "ok": not any(issue["level"] == "error" for issue in issues),
        "issue_count": len(issues),
        "issues": issues,
        "summary": registry_summary(),
    }


def validate_registered_ui_state() -> dict[str, Any]:
    """Validate current state values against registered UI metadata."""
    state = _session_state()
    issues: list[dict[str, Any]] = []

    def add_issue(level: str, code: str, message: str, **extra: Any) -> None:
        issues.append({"level": level, "code": code, "message": message, **_json_friendly(extra)})

    for entry in registered_ui_entries():
        settings_key = entry.get("settings_key")
        if not settings_key or settings_key not in state:
            continue
        value = state.get(settings_key)
        kind = entry.get("kind")
        options = entry.get("options")
        metadata = entry.get("metadata") or {}
        if kind in {"selectbox", "radio", "segmented_control"} and options is not None and value not in options:
            add_issue(
                "warning",
                "value_not_in_options",
                "Stored value is not in registered options.",
                settings_key=settings_key,
                value=value,
                options=options,
            )
        if kind == "multiselect" and options is not None:
            invalid_values = [item for item in list(value or []) if item not in options] if isinstance(value, list) else [value]
            if invalid_values:
                add_issue(
                    "warning",
                    "multiselect_value_not_in_options",
                    "Stored multiselect values contain unknown options.",
                    settings_key=settings_key,
                    values=invalid_values,
                )
        if kind in {"number_input", "slider"}:
            try:
                numeric_value = float(value)
                min_value = metadata.get("min_value")
                max_value = metadata.get("max_value")
                if min_value is not None and numeric_value < float(min_value):
                    add_issue("warning", "value_below_min", "Stored value is below min_value.", settings_key=settings_key, value=value, min_value=min_value)
                if max_value is not None and numeric_value > float(max_value):
                    add_issue("warning", "value_above_max", "Stored value is above max_value.", settings_key=settings_key, value=value, max_value=max_value)
            except Exception:
                add_issue("warning", "numeric_value_invalid", "Stored numeric value cannot be converted to float.", settings_key=settings_key, value=value)

    return {
        "ok": not any(issue["level"] == "error" for issue in issues),
        "issue_count": len(issues),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Dirty state and settings diff
# ---------------------------------------------------------------------------


def diff_settings(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return added/removed/changed settings between two dictionaries."""
    current_clean = _json_friendly(current)
    baseline_clean = _json_friendly(baseline)
    keys = sorted(set(current_clean.keys()) | set(baseline_clean.keys()))
    diff: dict[str, dict[str, Any]] = {}
    for key in keys:
        current_exists = key in current_clean
        baseline_exists = key in baseline_clean
        current_value = current_clean.get(key)
        baseline_value = baseline_clean.get(key)
        if not baseline_exists:
            diff[key] = {"status": "added", "current": current_value, "baseline": None}
        elif not current_exists:
            diff[key] = {"status": "removed", "current": None, "baseline": baseline_value}
        elif current_value != baseline_value:
            diff[key] = {"status": "changed", "current": current_value, "baseline": baseline_value}
    return diff


def settings_fingerprint(
    *,
    keys: Sequence[str] | None = None,
    include_widget_keys: bool = False,
) -> str:
    """Return a stable fingerprint for selected session_state settings."""
    payload = _state_value_snapshot(keys=keys, include_widget_keys=include_widget_keys)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mark_clean(
    name: str = "default",
    *,
    keys: Sequence[str] | None = None,
    include_widget_keys: bool = False,
    scope: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Store the current settings snapshot as a clean baseline."""
    scope_id = normalize_scope(scope if scope is not None else get_active_scope())
    baseline_name = f"{scope_id}:{name}"
    payload = _state_value_snapshot(keys=keys, include_widget_keys=include_widget_keys)
    baseline = {
        "scope": scope_id,
        "name": name,
        "fingerprint": settings_fingerprint(keys=keys, include_widget_keys=include_widget_keys),
        "payload": payload,
        "keys": list(keys) if keys is not None else None,
        "include_widget_keys": include_widget_keys,
    }
    baselines = _registry_state(UI_DIRTY_BASELINES_STATE_KEY)
    baselines[baseline_name] = baseline
    return deepcopy(baseline)


def changed_since_clean(name: str = "default", *, scope: str | Sequence[str] | None = None) -> dict[str, Any]:
    """Compare current settings against a stored clean baseline."""
    scope_id = normalize_scope(scope if scope is not None else get_active_scope())
    baseline_name = f"{scope_id}:{name}"
    baselines = _registry_state(UI_DIRTY_BASELINES_STATE_KEY)
    baseline = baselines.get(baseline_name)
    if baseline is None and scope is None:
        baseline = baselines.get(name)
    if not baseline:
        return {"dirty": True, "reason": "missing_baseline", "scope": scope_id, "changed": {}}
    keys = baseline.get("keys")
    include_widget_keys = bool(baseline.get("include_widget_keys", False))
    current_payload = _state_value_snapshot(keys=keys, include_widget_keys=include_widget_keys)
    changed = diff_settings(current_payload, baseline.get("payload", {}))
    current_fingerprint = settings_fingerprint(keys=keys, include_widget_keys=include_widget_keys)
    return {
        "dirty": current_fingerprint != baseline.get("fingerprint"),
        "scope": baseline.get("scope", scope_id),
        "fingerprint": current_fingerprint,
        "baseline_fingerprint": baseline.get("fingerprint"),
        "changed": changed,
    }


def is_dirty(name: str = "default", *, scope: str | Sequence[str] | None = None) -> bool:
    """Return True if settings changed since mark_clean()."""
    return bool(changed_since_clean(name, scope=scope).get("dirty", True))


def mark_scope_clean(
    scope: str | Sequence[str],
    name: str = "default",
    *,
    keys: Sequence[str] | None = None,
    include_widget_keys: bool = False,
) -> dict[str, Any]:
    """Store a clean baseline for one scope."""
    return mark_clean(name, keys=keys, include_widget_keys=include_widget_keys, scope=scope)


def changed_since_scope_clean(scope: str | Sequence[str], name: str = "default") -> dict[str, Any]:
    """Compare current state with a scope baseline."""
    return changed_since_clean(name, scope=scope)


# ---------------------------------------------------------------------------
# Dependency / impact map
# ---------------------------------------------------------------------------


def register_dependency(
    source: str,
    targets: str | Sequence[str],
    *,
    reason: str = "",
    stage: str = "",
    scope: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Register that source changes can affect target stages/settings."""
    scope_id = normalize_scope(scope if scope is not None else get_active_scope())
    registry = _registry_state(UI_DEPENDENCIES_STATE_KEY)
    target_list = [targets] if isinstance(targets, str) else list(targets)
    registry_key = f"{scope_id}:{source}"
    current = registry.setdefault(registry_key, {"source": source, "scope": scope_id, "targets": [], "metadata": {}})
    for target in target_list:
        if target not in current["targets"]:
            current["targets"].append(target)
    if reason:
        current.setdefault("metadata", {})["reason"] = reason
    if stage:
        current.setdefault("metadata", {})["stage"] = stage
    return deepcopy(current)


def dependency_map() -> dict[str, dict[str, Any]]:
    """Return the registered dependency map."""
    return deepcopy(_registry_state(UI_DEPENDENCIES_STATE_KEY))


def dependencies_for(source: str, *, scope: str | Sequence[str] | None = None) -> list[str]:
    """Return direct dependency targets for a source."""
    registry = _registry_state(UI_DEPENDENCIES_STATE_KEY)
    if scope is not None:
        scope_id = normalize_scope(scope)
        entry = registry.get(f"{scope_id}:{source}", {})
        return list(entry.get("targets", []))
    targets: list[str] = []
    for key in (source, f"{get_active_scope()}:{source}", f"global:{source}"):
        for target in registry.get(key, {}).get("targets", []):
            if target not in targets:
                targets.append(target)
    return targets


def affected_by(target: str, *, scope: str | Sequence[str] | None = None) -> list[str]:
    """Return sources that directly affect a target."""
    scope_id = normalize_scope(scope) if scope is not None else None
    result: list[str] = []
    for source, entry in _registry_state(UI_DEPENDENCIES_STATE_KEY).items():
        if scope_id is not None and entry.get("scope") != scope_id:
            continue
        if target in entry.get("targets", []):
            result.append(str(entry.get("source", source)))
    return sorted(result)


def dependency_tree(
    source: str,
    *,
    scope: str | Sequence[str] | None = None,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Return a recursive dependency tree for one source."""
    visited: set[str] = set()
    scope_id = normalize_scope(scope if scope is not None else get_active_scope())

    def walk(node: str, depth: int) -> dict[str, Any]:
        if depth > max_depth or node in visited:
            return {"source": node, "scope": scope_id, "targets": [], "truncated": True}
        visited.add(node)
        children = [walk(target, depth + 1) for target in dependencies_for(node, scope=scope_id)]
        return {"source": node, "scope": scope_id, "targets": children, "truncated": False}

    return walk(source, 0)


# ---------------------------------------------------------------------------
# Profile serialization and batch profile context
# ---------------------------------------------------------------------------


def serialize_profile_state(
    *,
    scope: str | Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
    include_registry: bool = False,
    include_entity_refs: bool = True,
) -> dict[str, Any]:
    """Serialize selected UI state for DB profile storage."""
    scope_id = normalize_scope(scope if scope is not None else get_active_scope())
    payload = {
        "schema": "appy.ui_index.profile.v1",
        "scope": scope_id,
        "state": _state_value_snapshot(keys=keys, include_widget_keys=False),
    }
    if include_entity_refs:
        payload["entity_refs"] = entity_refs(scope=scope_id)
    if include_registry:
        payload["registry_snapshot"] = export_ui_registry_snapshot(include_state_values=False)
    return deepcopy(payload)


def apply_profile_state(
    payload: dict[str, Any],
    *,
    scope: str | Sequence[str] | None = None,
    overwrite: bool = True,
    restore_entity_refs: bool = True,
) -> dict[str, Any]:
    """Apply serialized profile state to session_state."""
    state = _session_state()
    scope_id = normalize_scope(scope if scope is not None else payload.get("scope", get_active_scope()))
    set_active_scope(scope_id)
    values = payload.get("state", {})
    applied: list[str] = []
    skipped: list[str] = []
    if not isinstance(values, dict):
        return {"scope": scope_id, "applied": applied, "skipped": ["invalid_state"]}
    for key, value in values.items():
        if not overwrite and key in state:
            skipped.append(str(key))
            continue
        state[str(key)] = deepcopy(value)
        applied.append(str(key))
    if restore_entity_refs:
        clear_entity_refs(scope_id)
        for ref in payload.get("entity_refs", []) if isinstance(payload.get("entity_refs", []), list) else []:
            if isinstance(ref, dict) and ref.get("kind") and ref.get("entity_id") and ref.get("settings_key"):
                register_entity_ref(
                    kind=str(ref["kind"]),
                    entity_id=str(ref["entity_id"]),
                    settings_key=str(ref["settings_key"]),
                    scope=scope_id,
                    label=str(ref.get("label", "")),
                    metadata=ref.get("metadata", {}),
                )
    return {"scope": scope_id, "applied": applied, "skipped": skipped}


def profile_fingerprint(payload: dict[str, Any]) -> str:
    """Return a stable fingerprint for a serialized profile."""
    encoded = json.dumps(_json_friendly(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def profile_context(
    payload: dict[str, Any],
    *,
    scope: str | Sequence[str] | None = None,
    keys: Sequence[str] | None = None,
    restore_entity_refs: bool = True,
):
    """Temporarily apply a profile for batch export without permanently switching UI."""
    state = _session_state()
    previous_scope = get_active_scope()
    profile_state = payload.get("state", {}) if isinstance(payload.get("state", {}), dict) else {}
    target_keys = [str(key) for key in (keys or profile_state.keys())]
    previous_values = {key: deepcopy(state[key]) for key in target_keys if key in state}
    missing_keys = [key for key in target_keys if key not in state]
    previous_refs = entity_refs()
    try:
        apply_profile_state(
            {"scope": payload.get("scope", scope), "state": {key: profile_state.get(key) for key in target_keys if key in profile_state}, "entity_refs": payload.get("entity_refs", [])},
            scope=scope,
            overwrite=True,
            restore_entity_refs=restore_entity_refs,
        )
        yield {
            "scope": get_active_scope(),
            "keys": target_keys,
            "fingerprint": profile_fingerprint(payload),
        }
    finally:
        for key in target_keys:
            if key in previous_values:
                state[key] = previous_values[key]
            elif key in missing_keys:
                state.pop(key, None)
        set_active_scope(previous_scope)
        if restore_entity_refs:
            clear_entity_refs()
            for ref in previous_refs:
                register_entity_ref(
                    kind=str(ref["kind"]),
                    entity_id=str(ref["entity_id"]),
                    settings_key=str(ref["settings_key"]),
                    scope=str(ref.get("scope", "global")),
                    label=str(ref.get("label", "")),
                    metadata=ref.get("metadata", {}),
                )


# ---------------------------------------------------------------------------
# Structured UI model
# ---------------------------------------------------------------------------


StatusValue = str
ItemDict = dict[str, Any]


@dataclass(frozen=True)
class UIStatus:
    """Small status descriptor for page and section navigation."""

    value: StatusValue
    label: str
    symbol: str
    css_class: str


STATUS_UNSET = UIStatus("unset", "未設定", "", "ui-status-unset")
STATUS_PARTIAL = UIStatus("partial", "一部設定", "", "ui-status-partial")
STATUS_DONE = UIStatus("done", "完了", "", "ui-status-done")
STATUS_WARNING = UIStatus("warning", "確認", "!", "ui-status-warning")
STATUS_ERROR = UIStatus("error", "要修正", "×", "ui-status-error")

STATUS_BY_VALUE: dict[str, UIStatus] = {
    status.value: status
    for status in (STATUS_UNSET, STATUS_PARTIAL, STATUS_DONE, STATUS_WARNING, STATUS_ERROR)
}


@dataclass(frozen=True)
class UISection:
    """Navigation tree node for APPY-style settings pages."""

    id: str
    label: str
    status: StatusValue = "unset"
    icon: str = ""
    children: tuple["UISection", ...] = field(default_factory=tuple)
    disabled: bool = False
    description: str = ""


@dataclass(frozen=True)
class UIListAction:
    """Result of a list operation triggered by the UI."""

    action: str
    item_id: str | None = None
    changed: bool = False


def status_info(status: StatusValue | UIStatus | None) -> UIStatus:
    """Return a normalized status descriptor."""
    if isinstance(status, UIStatus):
        return status
    return STATUS_BY_VALUE.get(str(status or "unset"), STATUS_UNSET)


def status_badge_text(status: StatusValue | UIStatus | None, label: str = "") -> str:
    """Return compact text for a labeled status."""
    info = status_info(status)
    return f"{info.symbol} {label}".strip()


def flatten_sections(sections: Sequence[UISection], *, include_disabled: bool = True) -> list[UISection]:
    """Flatten a tree of UISection objects in display order."""
    flattened: list[UISection] = []
    for section in sections:
        if include_disabled or not section.disabled:
            flattened.append(section)
        flattened.extend(flatten_sections(section.children, include_disabled=include_disabled))
    return flattened


def section_options(sections: Sequence[UISection], *, include_disabled: bool = False) -> list[str]:
    """Return selectable section ids in display order."""
    return [section.id for section in flatten_sections(sections, include_disabled=include_disabled) if not section.disabled]


def section_label_map(sections: Sequence[UISection]) -> dict[str, str]:
    """Return plain labels for navigation widgets."""
    return {
        section.id: section.label
        for section in flatten_sections(sections, include_disabled=True)
    }


def find_section(sections: Sequence[UISection], section_id: str) -> UISection | None:
    """Find one UISection by id."""
    for section in flatten_sections(sections, include_disabled=True):
        if section.id == section_id:
            return section
    return None


def navigation_radio(
    label: str,
    sections: Sequence[UISection],
    *,
    settings_key: str,
    default: str | None = None,
    key: str | None = None,
    horizontal: bool = False,
    help: str | None = None,
) -> str:
    """Render a radio navigation control."""
    register_section_tree(sections, path=[settings_key])
    options = section_options(sections)
    labels_map = section_label_map(sections)
    return radio_indexed(
        label,
        options,
        settings_key=settings_key,
        labels=[labels_map.get(option, option) for option in options],
        default=default or (options[0] if options else None),
        key=key,
        horizontal=horizontal,
        help=help,
    )


def navigation_segmented(
    label: str,
    sections: Sequence[UISection],
    *,
    settings_key: str,
    default: str | None = None,
    key: str | None = None,
    help: str | None = None,
) -> str:
    """Render a segmented navigation control."""
    register_section_tree(sections, path=[settings_key])
    options = section_options(sections)
    labels_map = section_label_map(sections)
    return segmented_control_indexed(
        label,
        options,
        settings_key=settings_key,
        labels=[labels_map.get(option, option) for option in options],
        default=default or (options[0] if options else None),
        key=key,
        help=help,
    )


# ---------------------------------------------------------------------------
# Stateful list model
# ---------------------------------------------------------------------------


def new_item_id(prefix: str = "item") -> str:
    return list_core.new_item_id(prefix)


def make_list_item(
    item_type: str,
    values: dict[str, Any] | None = None,
    *,
    item_id: str | None = None,
    enabled: bool = True,
    order: int | float = 0,
) -> ItemDict:
    return list_core.make_list_item(item_type, values, item_id=item_id, enabled=enabled, order=order)


def _state_list(settings_key: str) -> list[ItemDict]:
    return list_core._state_list(_session_state(), settings_key)


def ensure_list_state(
    settings_key: str,
    *,
    default_items: Iterable[ItemDict] | None = None,
    item_type: str = "item",
) -> list[ItemDict]:
    register_list_schema(settings_key=settings_key, item_type=item_type)
    return list_core.ensure_list_state(_session_state(), settings_key, default_items=default_items, item_type=item_type)


def list_items(
    settings_key: str,
    *,
    include_disabled: bool = True,
    sort_by_order: bool = True,
) -> list[ItemDict]:
    return list_core.list_items(_session_state(), settings_key, include_disabled=include_disabled, sort_by_order=sort_by_order)


def active_list_items(settings_key: str, *, sort_by_order: bool = True) -> list[ItemDict]:
    return list_core.active_list_items(_session_state(), settings_key, sort_by_order=sort_by_order)


def _find_item_index(items: Sequence[ItemDict], item_id: str) -> int | None:
    return list_core._find_item_index(items, item_id)


def add_list_item(
    settings_key: str,
    item: ItemDict | None = None,
    *,
    item_type: str = "item",
    values: dict[str, Any] | None = None,
    enabled: bool = True,
) -> ItemDict:
    return list_core.add_list_item(_session_state(), settings_key, item, item_type=item_type, values=values, enabled=enabled)


def duplicate_list_item(settings_key: str, item_id: str, *, suffix: str = " copy") -> ItemDict | None:
    return list_core.duplicate_list_item(_session_state(), settings_key, item_id, suffix=suffix)


def set_list_item_enabled(settings_key: str, item_id: str, enabled: bool) -> bool:
    return list_core.set_list_item_enabled(_session_state(), settings_key, item_id, enabled)


def disable_list_item(settings_key: str, item_id: str) -> bool:
    return list_core.disable_list_item(_session_state(), settings_key, item_id)


def delete_list_item(settings_key: str, item_id: str) -> bool:
    return list_core.delete_list_item(_session_state(), settings_key, item_id)


def update_list_item(settings_key: str, item_id: str, values: dict[str, Any]) -> bool:
    return list_core.update_list_item(_session_state(), settings_key, item_id, values)


def move_list_item(settings_key: str, item_id: str, direction: int) -> bool:
    return list_core.move_list_item(_session_state(), settings_key, item_id, direction)


def normalize_list_order(settings_key: str) -> list[ItemDict]:
    return list_core.normalize_list_order(_session_state(), settings_key)


def sort_list_items(
    settings_key: str,
    key_func: Callable[[ItemDict], Any],
    *,
    reverse: bool = False,
) -> list[ItemDict]:
    return list_core.sort_list_items(_session_state(), settings_key, key_func, reverse=reverse)


def list_count(settings_key: str, *, include_disabled: bool = True) -> int:
    return list_core.list_count(_session_state(), settings_key, include_disabled=include_disabled)


def list_status(settings_key: str) -> UIStatus:
    """Return a simple status for a stateful list."""
    total = list_count(settings_key, include_disabled=True)
    active = list_count(settings_key, include_disabled=False)
    if total == 0:
        return STATUS_UNSET
    if active == 0:
        return STATUS_PARTIAL
    return STATUS_DONE


# ---------------------------------------------------------------------------
# Optional render helpers for list item controls
# ---------------------------------------------------------------------------


def render_item_actions(
    settings_key: str,
    item_id: str,
    *,
    allow_delete: bool = False,
    disabled: bool = False,
    key_prefix: str = "item_action",
) -> UIListAction:
    """Render compact list actions and apply them immediately.

    The default operation favors disabling over deletion. This is safer for
    DSP workflows where comparing a filter before/after matters.
    """
    import streamlit as st

    cols = st.columns([1, 1, 1, 1 if allow_delete else 0.01])
    action = UIListAction("none", item_id=item_id, changed=False)
    if cols[0].button("↑", key=f"{key_prefix}_{item_id}_up", disabled=disabled):
        action = UIListAction("move_up", item_id=item_id, changed=move_list_item(settings_key, item_id, -1))
    if cols[1].button("↓", key=f"{key_prefix}_{item_id}_down", disabled=disabled):
        action = UIListAction("move_down", item_id=item_id, changed=move_list_item(settings_key, item_id, 1))
    if cols[2].button(ui_message('ui.38cca6bea010af'), key=f"{key_prefix}_{item_id}_disable", disabled=disabled):
        action = UIListAction("disable", item_id=item_id, changed=disable_list_item(settings_key, item_id))
    if allow_delete and cols[3].button(ui_message('ui.e2d0a54968ead2'), key=f"{key_prefix}_{item_id}_delete", disabled=disabled):
        action = UIListAction("delete", item_id=item_id, changed=delete_list_item(settings_key, item_id))
    return action


def render_add_item_button(
    label: str,
    *,
    settings_key: str,
    item_type: str,
    values: dict[str, Any] | None = None,
    key: str | None = None,
) -> ItemDict | None:
    """Render an add button and append a list item when clicked."""
    import streamlit as st

    if st.button(label, key=key or f"add_{settings_key}_{item_type}"):
        return add_list_item(settings_key, item_type=item_type, values=values)
    return None
