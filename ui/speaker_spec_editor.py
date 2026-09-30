"""Keep the specification editor identity tied to the visible table selection."""


def sync_spec_editor_selection(state, selected, load_record):
    selected_id = selected.id if selected is not None else ""
    previous_id = str(state.get("_speaker_db_editor_id", ""))
    if selected_id:
        if selected_id != previous_id or selected_id != state.get("_speaker_db_last_selected_id", ""):
            load_record(selected)
        state["_speaker_db_unselected_draft"] = False
    elif previous_id:
        # Keep typed values as a new draft, but never retain an invisible update ID.
        state["_speaker_db_unselected_draft"] = True
    state["_speaker_db_editor_id"] = selected_id
    state["_speaker_db_last_selected_id"] = selected_id
    return selected_id
