"""Explicit editor switch requests, independent of a changing option list."""


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
