"""Streamlit projection of EQ ownership; DSP and persistence use EQLinks."""
from __future__ import annotations

from utils.ui_localization import ui_message, display_text

from contextlib import contextmanager
from copy import deepcopy
from uuid import uuid4

from utils.eq_links import EQLinks, extract_eq, transaction
from utils.link_change import plan_change, commit_change


def channel_context(state, root, assignment, assignments=()):
    if assignment is None:
        names = {"Left": "Left", "Right": "Right"}
        names.update({name: name for name in state.get("_stereo_links", EQLinks()).channels})
        return "standalone", str(state.get("_stereo_channel", "Left")), names
    options = {item.channel_id: f"{item.group} / {item.channel_name} ({item.way})" for item in assignments}
    options[assignment.channel_id] = f"{assignment.group} / {assignment.channel_name} ({assignment.way})"
    return f"{assignment.system_id}:{assignment.sample_rate_hz}", assignment.channel_id, options


@contextmanager
def model_for(state, root, scope):
    if scope == "standalone":
        model = state.setdefault("_stereo_links", EQLinks())
        yield model
    else:
        with transaction(root, scope) as model:
            yield model


def prepare(state, root, assignment, assignments, default_payload, apply):
    scope, channel, options = channel_context(state, root, assignment, assignments)
    loaded = state.pop("_stereo_loaded_payload", None)
    payload = loaded or state.get("_stereo_live_payload") or default_payload
    if loaded and assignment is None and "eq_links" in loaded:
        state["_stereo_links"] = EQLinks(loaded["eq_links"])
        state["_stereo_channel"] = loaded.get("eq_link_channel", "Left")
        channel = state["_stereo_channel"]
        state["_stereo_channel_selector"] = channel
        options.update({name: name for name in state["_stereo_links"].channels})
        state["_stereo_channel_payloads"] = deepcopy(loaded.get("eq_link_channels", {}))
    with model_for(state, root, scope) as model:
        model.ensure(channel, payload)
        if loaded and "eq_links" not in loaded and (
            state.get("_stereo_context") == (scope, channel)
            or not model.channels[channel]["group"]
        ):
            # Loading EQ is an edit to the currently resolved owner.
            model.update(channel, payload)
        effective = model.effective(channel, payload)
        expected = deepcopy(model.resolve_eq(channel))
    if extract_eq(effective) != extract_eq(payload) or state.get("_stereo_context") != (scope, channel):
        apply(effective)
    state["_stereo_context"] = (scope, channel)
    state["_stereo_expected"] = expected
    state["_stereo_live_payload"] = effective
    state["_stereo_options"] = options
    state.pop("_stereo_loaded_payload", None)


def finish(state, root, payload):
    scope, channel = state["_stereo_context"]
    with model_for(state, root, scope) as model:
        # Do not write an unchanged stale projection over a newer owner's edit.
        if extract_eq(payload) != extract_eq(state["_stereo_live_payload"]):
            model.update(channel, payload, expected=state["_stereo_expected"])
        effective = model.effective(channel, payload)
        state["_stereo_expected"] = deepcopy(model.resolve_eq(channel))
    state["_stereo_live_payload"] = effective
    if scope == "standalone":
        state.setdefault("_stereo_channel_payloads", {})[channel] = clean_payload(effective)
    return effective


def clean_payload(payload):
    return deepcopy({key: value for key, value in payload.items() if not key.startswith("eq_link")})


def saved_fields(state, root):
    context = state.get("_stereo_context")
    if context is None:
        return {}
    scope, channel = context
    with model_for(state, root, scope) as model:
        fields = {"eq_links": model.snapshot(), "eq_link_channel": channel}
    if scope == "standalone":
        fields["eq_link_channels"] = deepcopy(state.get("_stereo_channel_payloads", {}))
    return fields


def queue_toggle(state, enabled_key, scope, channel):
    state["_stereo_toggle_request"] = (scope, channel, bool(state[enabled_key]))


def render(state, root, load_channel, load_other=None):
    import streamlit as st

    scope, channel = state["_stereo_context"]
    options = state["_stereo_options"]
    with model_for(state, root, scope) as model:
        record = model.channels[channel]
        group_id = record["group"]
        members = [key for key, value in model.channels.items() if group_id and value["group"] == group_id]
        categories = model.groups[group_id]["categories"] if group_id else ["iir", "fir", "target"]
    selected_key = f"_stereo_members_{scope}_{channel}"
    source_key = f"_stereo_source_{scope}_{channel}"
    settings_key = f"_stereo_categories_{scope}_{channel}"
    enabled_key = f"_stereo_enabled_{scope}_{channel}"
    reset_key = "_stereo_reset_toggle"
    requested = state.pop("_stereo_toggle_request", None)
    state.pop(reset_key, None)
    state[enabled_key] = bool(group_id)
    controls_generation = (group_id, tuple(members), tuple(categories))
    generation_key = f"_stereo_controls_{scope}_{channel}"
    if state.get(generation_key) != controls_generation:
        state[selected_key] = [member for member in members if member != channel and member in options]
        state[settings_key] = list(categories)
        state[generation_key] = controls_generation
    state.setdefault(selected_key, [member for member in members if member != channel and member in options])
    state.setdefault(settings_key, list(categories))
    selected_channel = channel
    if scope == "standalone":
        state.setdefault("_stereo_channel_selector", channel)
        channel_host, toggle_host, detail_host = st.columns([0.48, 0.46, 0.06], gap="small", vertical_alignment="center")
        with channel_host:
            selected_channel = st.selectbox(ui_message("ui.9267e2eb2d3ca4"), list(options),
                                            key="_stereo_channel_selector", label_visibility="collapsed")
    else:
        toggle_host, detail_host = st.columns([0.90, 0.10], gap="small", vertical_alignment="center")
    with toggle_host:
        enabled = st.toggle("🔗 Stereo Link · Linked" if group_id else "⛓ Stereo Link · Unlinked",
                            key=enabled_key,
                            on_change=queue_toggle, args=(state, enabled_key, scope, channel))
    if requested and requested[:2] == (scope, channel):
        enabled = requested[2]
    with detail_host, st.popover("⚙", help=ui_message("ui.dcba367abef085"), use_container_width=False):
        selected = st.multiselect(ui_message("ui.b911e2e20811d2"), [key for key in options if key != channel],
            format_func=lambda key: options[key], key=selected_key)
        candidates = [channel, *selected]
        source = st.selectbox(ui_message("ui.c99fa954c2e1d0"), candidates,
            format_func=lambda key: ui_message("ui.695a8a6f8f3e4b") if key == channel else options[key], key=source_key)
        shared = st.multiselect(ui_message("ui.95eee0aefa9e15"), ["iir", "fir", "target"],
            format_func=lambda key: {"iir": "IIR EQ", "fir": ui_message("ui.ca46694f443358"), "target": ui_message("ui.d7b5054c74e1ea")}[key], key=settings_key)
        st.caption(ui_message("ui.3454bed137c4ea"))
        apply_clicked = st.button(ui_message("ui.2b22a780e4da52"), disabled=not selected or not shared)
        if scope == "standalone":
            new_name = st.text_input(ui_message("ui.e2d4a90ec7d2cf"), key="_stereo_new_name")
            if st.button(ui_message("ui.31aa8cc845b8ab"), disabled=not new_name.strip()):
                if new_name.strip() in options:
                    st.error(ui_message("ui.ffeb8315e37c62"))
                else:
                    with model_for(state, root, scope) as model:
                        model.ensure(new_name.strip(), state["_stereo_live_payload"])
                    st.rerun()
    if apply_clicked or enabled != bool(group_id):
        try:
            seeds = {}
            linking = apply_clicked or enabled
            if linking:
                if len(candidates) < 2 or not shared:
                    raise ValueError(ui_message("ui.1f3503aa4fdefb"))
                for target in candidates:
                    if target == channel:
                        seeds[target] = state["_stereo_live_payload"]
                    elif scope == "standalone":
                        seeds[target] = state.get("_stereo_channel_payloads", {}).get(target, state["_stereo_live_payload"])
                    elif load_other is not None:
                        seeds[target] = load_other(target)
                    if not seeds.get(target):
                        raise ValueError(ui_message("ui.3c13d23bdec3f3"))
            with model_for(state, root, scope) as model:
                change = plan_change(model, channel=channel, linking=linking,
                    candidates=candidates, source=source, categories=shared, seeds=seeds)
            state["_stereo_pending_change"] = (scope, change)
        except (OSError, ValueError) as exc:
            state["_stereo_notice"] = str(exc)
        state[reset_key] = True
        st.rerun()
    if state.get("_stereo_notice"):
        st.warning(display_text(state.pop("_stereo_notice")))
    pending = state.get("_stereo_pending_change")
    if pending and (pending[0], pending[1].channel) == (scope, channel):
        render_confirmation(state, root, scope, pending[1], options)
    if scope == "standalone":
        if selected_channel != channel:
            state.setdefault("_stereo_channel_payloads", {})[channel] = clean_payload(state["_stereo_live_payload"])
            target = state["_stereo_channel_payloads"].get(selected_channel)
            if target is None:
                target = clean_payload(state["_stereo_live_payload"])
                target["config"]["speaker_response"] = None
                target["io"] = {key: value for key, value in target.get("io", {}).items() if key.startswith("target_")}
                target["ui"].update({name: None for name in ("current_speaker_measurement", "current_mic_calibration", "current_near_field_measurement", "current_port_measurement")})
                target["ui_profile"] = {}
                target["ui"]["design_history_id"] = str(uuid4())
            state["_stereo_channel"] = selected_channel
            load_channel(target)
            st.rerun()


def render_confirmation(state, root, scope, change, options):
    """Display the prepared change; only the explicit OK branch can commit."""
    import streamlit as st

    @st.dialog(ui_message("ui.ec7446478d49c8"), dismissible=False)
    def confirm():
        st.write(ui_message("ui.6fadad233a13a4"))
        for member in change.affected:
            st.write("• " + options.get(member, member))
        if change.linking:
            detached = [member for member in change.affected if member not in change.candidates]
            if detached:
                st.write(ui_message("ui.e81bcd17291bb6") + ", ".join(options.get(member, member) for member in detached))
            st.write(ui_message("ui.1bf6bea7b1cfae") + options.get(change.source, change.source))
            st.write(ui_message("ui.aaa58681013d7c") + ", ".join({"iir": "IIR EQ", "fir": ui_message("ui.ca46694f443358"), "target": ui_message("ui.d7b5054c74e1ea")}[name] for name in change.categories))
        else:
            st.write(ui_message("ui.f284e4d4e94330"))
        st.caption(ui_message("ui.74a94d5556fc7e"))
        ok, cancel = st.columns(2)
        if ok.button(ui_message("ui.5071fb2c5f0141"), type="primary"):
            try:
                with model_for(state, root, scope) as model:
                    commit_change(model, change, approved=True)
                state.pop("_stereo_pending_change", None)
                state["_stereo_reset_toggle"] = True
                st.rerun()
            except (OSError, ValueError) as exc:
                st.error(display_text(str(exc)))
        if cancel.button(ui_message("ui.bca84ea5c65fee")):
            state.pop("_stereo_pending_change", None)
            state["_stereo_reset_toggle"] = True
            st.rerun()
    confirm()
