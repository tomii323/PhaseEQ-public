"""Explicit editor switch requests, independent of a changing option list."""
from copy import deepcopy


def active_assignment_crossover(state):
    """Only a connected Assignment can own the Studio crossover stage."""
    if state.get('_active_context_assignment') is None:
        return None
    payload = state.get('_composite_iir_crossover')
    return payload if isinstance(payload, dict) else None


def detach_standalone_crossover(state):
    """Retain orphaned Studio data for recovery, never as an active filter."""
    if state.get('_active_context_assignment') is not None:
        return False
    payload = state.pop('_composite_iir_crossover', None)
    if payload is None:
        return False
    detached = deepcopy(state.get('_detached_channel_band_split') or {})
    detached['iir'] = deepcopy(payload)
    state['_detached_channel_band_split'] = detached
    return True


def queue_channel_switch(state):
    state["_pending_composite_context"] = {
        "assignment_id": str(state.get("composite_phaseeq_channel_switcher", "") or ""),
        "payload": state.get("_composite_context_payload"),
    }


def requested_context_id(query_id, current_id, pending):
    # An incoming Multiway URL wins over an older queued widget action.
    if query_id and query_id != current_id:
        return query_id
    return str(pending["assignment_id"]) if isinstance(pending, dict) else query_id


def same_channel_context(source, destination):
    return (source is not None and destination is not None
            and (source.system_id, source.channel_id) == (destination.system_id, destination.channel_id))
