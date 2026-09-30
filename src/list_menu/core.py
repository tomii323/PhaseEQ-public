"""State mapping list operations, independent of Streamlit, DB and DSP."""
from __future__ import annotations
from copy import deepcopy
from collections.abc import Callable, Iterable, Sequence
from typing import Any
import uuid
from list_menu_state import require_capacity, list_policy
ItemDict = dict[str, Any]

def new_item_id(prefix: str='item') -> str:
    """Return a stable id suitable for stateful list items."""
    safe_prefix = ''.join((ch if ch.isalnum() or ch == '_' else '_' for ch in str(prefix).strip())) or 'item'
    return f'{safe_prefix}_{uuid.uuid4().hex[:10]}'

def make_list_item(item_type: str, values: dict[str, Any] | None=None, *, item_id: str | None=None, enabled: bool=True, order: int | float=0) -> ItemDict:
    """Create a JSON-friendly list item with common metadata."""
    payload = deepcopy(values or {})
    payload.setdefault('id', item_id or new_item_id(item_type))
    payload.setdefault('type', item_type)
    payload.setdefault('enabled', bool(enabled))
    payload.setdefault('order', order)
    return payload

def _state_list(state, settings_key: str) -> list[ItemDict]:
    state = state
    raw_items = state.get(settings_key, [])
    if not isinstance(raw_items, list):
        raw_items = []
    items = [item for item in raw_items if isinstance(item, dict)]
    state[settings_key] = items
    return items

def ensure_list_state(state, settings_key: str, *, default_items: Iterable[ItemDict] | None=None, item_type: str='item') -> list[ItemDict]:
    """Ensure a session_state list exists and each item has metadata."""
    state = state
    if settings_key not in state:
        state[settings_key] = [deepcopy(item) for item in default_items or []]
    items = _state_list(state, settings_key)
    changed = False
    for idx, item in enumerate(items):
        if 'id' not in item:
            item['id'] = new_item_id(str(item.get('type', item_type)))
            changed = True
        if 'type' not in item:
            item['type'] = item_type
            changed = True
        if 'enabled' not in item:
            item['enabled'] = True
            changed = True
        if 'order' not in item:
            item['order'] = idx
            changed = True
    if changed:
        state[settings_key] = items
    return items

def list_items(state, settings_key: str, *, include_disabled: bool=True, sort_by_order: bool=True) -> list[ItemDict]:
    """Return list items without exposing the mutable state list."""
    items = [deepcopy(item) for item in _state_list(state, settings_key)]
    if not include_disabled:
        items = [item for item in items if bool(item.get('enabled', True))]
    if sort_by_order:
        items.sort(key=lambda item: (float(item.get('order', 0)), str(item.get('id', ''))))
    return items

def active_list_items(state, settings_key: str, *, sort_by_order: bool=True) -> list[ItemDict]:
    """Return enabled list items only."""
    return list_items(state, settings_key, include_disabled=False, sort_by_order=sort_by_order)

def _find_item_index(items: Sequence[ItemDict], item_id: str) -> int | None:
    for idx, item in enumerate(items):
        if str(item.get('id', '')) == str(item_id):
            return idx
    return None

def add_list_item(state, settings_key: str, item: ItemDict | None=None, *, item_type: str='item', values: dict[str, Any] | None=None, enabled: bool=True) -> ItemDict:
    """Append one item and return it."""
    items = ensure_list_state(state, settings_key, item_type=item_type)
    require_capacity(settings_key, len(items))
    next_order = max([float(existing.get('order', idx)) for idx, existing in enumerate(items)] or [-1.0]) + 1.0
    new_item = deepcopy(item) if item is not None else make_list_item(item_type, values, enabled=enabled)
    new_item.setdefault('id', new_item_id(str(new_item.get('type', item_type))))
    new_item.setdefault('type', item_type)
    new_item.setdefault('enabled', enabled)
    new_item['order'] = next_order
    items.append(new_item)
    state[settings_key] = items
    return deepcopy(new_item)

def duplicate_list_item(state, settings_key: str, item_id: str, *, suffix: str=' copy') -> ItemDict | None:
    """Duplicate one item while preserving type and values."""
    items = ensure_list_state(state, settings_key)
    idx = _find_item_index(items, item_id)
    if idx is None:
        return None
    if len(items) >= list_policy(settings_key).edit_limit:
        return None
    copied = deepcopy(items[idx])
    copied['id'] = new_item_id(str(copied.get('type', 'item')))
    if 'name' in copied:
        copied['name'] = f"{copied['name']}{suffix}"
    copied['enabled'] = bool(copied.get('enabled', True))
    copied['order'] = float(items[idx].get('order', idx)) + 0.5
    items.insert(idx + 1, copied)
    state[settings_key] = items
    normalize_list_order(state, settings_key)
    return deepcopy(copied)

def set_list_item_enabled(state, settings_key: str, item_id: str, enabled: bool) -> bool:
    """Enable or disable one item without deleting it."""
    items = ensure_list_state(state, settings_key)
    idx = _find_item_index(items, item_id)
    if idx is None:
        return False
    items[idx]['enabled'] = bool(enabled)
    state[settings_key] = items
    return True

def disable_list_item(state, settings_key: str, item_id: str) -> bool:
    """Disable one item."""
    return set_list_item_enabled(state, settings_key, item_id, False)

def delete_list_item(state, settings_key: str, item_id: str) -> bool:
    """Delete one item permanently."""
    items = ensure_list_state(state, settings_key)
    idx = _find_item_index(items, item_id)
    if idx is None:
        return False
    del items[idx]
    state[settings_key] = items
    normalize_list_order(state, settings_key)
    return True

def update_list_item(state, settings_key: str, item_id: str, values: dict[str, Any]) -> bool:
    """Update one item in place."""
    items = ensure_list_state(state, settings_key)
    idx = _find_item_index(items, item_id)
    if idx is None:
        return False
    items[idx].update(deepcopy(values))
    state[settings_key] = items
    return True

def move_list_item(state, settings_key: str, item_id: str, direction: int) -> bool:
    """Move one item up or down in display order."""
    if direction == 0:
        return False
    items = list_items(state, settings_key, include_disabled=True, sort_by_order=True)
    idx = _find_item_index(items, item_id)
    if idx is None:
        return False
    new_idx = max(0, min(len(items) - 1, idx + (-1 if direction < 0 else 1)))
    if new_idx == idx:
        return False
    items[idx], items[new_idx] = (items[new_idx], items[idx])
    for order, item in enumerate(items):
        item['order'] = order
    state[settings_key] = items
    return True

def normalize_list_order(state, settings_key: str) -> list[ItemDict]:
    """Compact item order values to 0..N-1."""
    items = list_items(state, settings_key, include_disabled=True, sort_by_order=True)
    for order, item in enumerate(items):
        item['order'] = order
    state[settings_key] = items
    return items

def sort_list_items(state, settings_key: str, key_func: Callable[[ItemDict], Any], *, reverse: bool=False) -> list[ItemDict]:
    """Sort items and rewrite order values."""
    items = list_items(state, settings_key, include_disabled=True, sort_by_order=False)
    items.sort(key=key_func, reverse=reverse)
    for order, item in enumerate(items):
        item['order'] = order
    state[settings_key] = items
    return items

def list_count(state, settings_key: str, *, include_disabled: bool=True) -> int:
    """Return the number of items in a stateful list."""
    return len(list_items(state, settings_key, include_disabled=include_disabled))
