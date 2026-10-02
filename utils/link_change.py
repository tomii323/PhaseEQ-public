"""Pure two-phase channel-link changes: plan first, commit after approval."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from utils.eq_links import CONFIG_FIELDS, EQLinks


@dataclass(frozen=True)
class LinkChange:
    channel: str
    linking: bool
    candidates: tuple[str, ...]
    source: str
    categories: tuple[str, ...]
    seeds: dict
    baseline: dict
    affected: tuple[str, ...]


@dataclass(frozen=True)
class RestoreChange:
    channel: str
    baseline: dict
    successor: dict
    affected: tuple[str, ...]
    payload: dict


def plan_restore(model: EQLinks, *, channel, payload, available):
    """Restore only the selected saved owner, retaining unrelated groups."""
    successor = EQLinks(model.snapshot())
    saved = EQLinks(payload["eq_links"]) if "eq_links" in payload else None
    source = payload.get("eq_link_channel", channel)
    if saved is not None and source != channel:
        raise ValueError("保存データのチャンネルと編集対象が一致しません。")
    if saved is not None and source not in saved.channels:
        raise ValueError("保存データのチャンネルが見つかりません。")
    group = saved.channels[source]["group"] if saved else None
    members = [key for key, record in saved.channels.items() if record["group"] == group] if group else [channel]
    if not set(members) <= set(available):
        raise ValueError("保存されたリンク先チャンネルが現在のシステムに存在しません。")
    affected = set(members)
    for member in members:
        old_group = model.channels.get(member, {}).get("group")
        if old_group:
            affected.update(key for key, record in model.channels.items() if record["group"] == old_group)
    successor.unlink(members)
    for member in members:
        seed = saved.effective(member, {"config": {}, "ui": {}, "io": {}}) if saved else payload
        successor.ensure(member, seed)
        successor.update(member, seed)
    if group:
        successor.link(members, source, saved.groups[group]["categories"])
    return RestoreChange(channel, model.snapshot(), successor.snapshot(),
                         tuple(sorted(affected)), deepcopy(payload))


def plan_edit(model: EQLinks, *, channel, payload):
    successor = EQLinks(model.snapshot())
    successor.update(channel, payload)
    group = model.channels[channel]["group"]
    affected = tuple(key for key, record in model.channels.items()
                     if key == channel or (group and record["group"] == group))
    return RestoreChange(channel, model.snapshot(), successor.snapshot(), affected, {})


def plan_change(model: EQLinks, *, channel, linking, candidates, source, categories, seeds):
    """Compute a reviewable replacement without modifying any existing owner."""
    candidates = tuple(dict.fromkeys(candidates))
    categories = tuple(categories)
    if channel not in model.channels:
        raise ValueError("現在のチャンネルを取得できません。")
    if linking:
        if len(candidates) < 2 or channel not in candidates or source not in candidates:
            raise ValueError("リンク先チャンネルと採用元を選択してください。")
        if not categories or not set(categories) <= set(CONFIG_FIELDS):
            raise ValueError("共有する設定を選択してください。")
        if any(not seeds.get(target) for target in candidates if target not in model.channels):
            raise ValueError("リンク先のEQ設定を取得できません。先に対象チャンネルの設定を保存してください。")
    affected = set(candidates if linking else [channel])
    for member in tuple(affected):
        old_group = model.channels.get(member, {}).get("group")
        if old_group:
            affected.update(key for key, record in model.channels.items() if record["group"] == old_group)
    return LinkChange(channel, bool(linking), candidates, source, categories,
                      deepcopy(seeds), model.snapshot(), tuple(sorted(affected)))


def commit_change(model: EQLinks, change: LinkChange | RestoreChange, *, approved: bool):
    """The only ownership mutation path; cancellation and conflicts are inert."""
    if not approved:
        return False
    if model.snapshot() != change.baseline:
        raise ValueError("確認中に設定が変更されました。最新状態で再度確認してください。")
    # Build a complete successor first. Validation cannot leave a partial group.
    successor = EQLinks(model.snapshot())
    if isinstance(change, RestoreChange):
        successor = EQLinks(change.successor)
    elif change.linking:
        for target, seed in change.seeds.items():
            successor.ensure(target, seed)
        successor.link(change.candidates, change.source, change.categories)
    else:
        successor.unlink([change.channel])
    model.channels, model.groups = successor.channels, successor.groups
    return True
