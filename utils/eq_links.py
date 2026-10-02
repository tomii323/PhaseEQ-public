"""EQ ownership, portable snapshots and transactional Multiway persistence.

Channel-specific inputs never enter this model. resolve_eq is the sole owner
lookup; linked channels return the identical dictionary, not mirrored copies.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

CONFIG_FIELDS = {
    "iir": frozenset({"iir_filters"}),
    "fir": frozenset({"peq_filters", "shelf_filters", "tilt_filters", "allpass_filters",
        "gain_peq_filters", "gain_shelf_filters", "gain_tilt_filters", "gain_linear_filters",
        "auto_eq", "auto_gain_input_shaping", "linear_fir_eq_mask_smoothing_oct",
        "phase_tilt_smoothing_oct", "gain_tilt_smoothing_oct", "window", "kaiser_beta",
        "chebyshev_attenuation_db", "tukey_alpha"}),
    "target": frozenset({"target_response"}),
}
UI_PREFIXES = {"iir": ("auto_iir_",), "fir": ("auto_gain_", "auto_phase_",),
               "target": ("target_",)}
TARGET_UI_EXCLUDED = {"target_output_formats", "target_url"}


def extract_eq(payload, categories=("iir", "fir", "target")):
    fields = set().union(*(CONFIG_FIELDS[name] for name in categories))
    prefixes = tuple(prefix for name in categories for prefix in UI_PREFIXES[name])
    return deepcopy({
        "config": {key: value for key, value in payload.get("config", {}).items() if key in fields},
        "io": {key: value for key, value in payload.get("io", {}).items()
               if "target" in categories and key in {"target_response_raw", "target_source_name"}},
        "ui": {key: value for key, value in payload.get("ui", {}).items()
               if key.startswith(prefixes) and key not in TARGET_UI_EXCLUDED},
    })


def apply_eq(payload, eq):
    result = deepcopy(payload)
    for section in ("config", "ui", "io"):
        result.setdefault(section, {}).update(deepcopy(eq.get(section, {})))
    return result


class EQLinks:
    def __init__(self, snapshot=None):
        data = deepcopy(snapshot) if snapshot is not None else {"version": 1, "channels": {}, "groups": {}}
        if data.get("version") != 1 or not isinstance(data.get("channels"), dict) or not isinstance(data.get("groups"), dict):
            raise ValueError("Invalid Stereo Link settings")
        self.channels, self.groups = data["channels"], data["groups"]
        for channel, record in self.channels.items():
            if not isinstance(channel, str) or not isinstance(record, dict) or not isinstance(record.get("local"), dict):
                raise ValueError("Invalid Stereo Link channel")
            if record.get("group") is not None and record["group"] not in self.groups:
                raise ValueError("Missing Stereo Link shared settings")
            if extract_eq(record["local"]) != record["local"]:
                raise ValueError("Stereo Link contains channel-specific settings")
        for group in self.groups.values():
            if not isinstance(group.get("eq"), dict) or not set(group.get("categories", ())) <= set(CONFIG_FIELDS):
                raise ValueError("Invalid Stereo Link group")
            if extract_eq(group["eq"], group["categories"]) != group["eq"]:
                raise ValueError("Stereo Link contains channel-specific settings")

    def snapshot(self):
        return deepcopy({"version": 1, "channels": self.channels, "groups": self.groups})

    def ensure(self, channel, payload):
        self.channels.setdefault(channel, {"group": None, "local": extract_eq(payload)})

    def resolve_eq(self, channel):
        record = self.channels[channel]
        return self.groups[record["group"]]["eq"] if record["group"] else record["local"]

    def effective(self, channel, payload):
        record = self.channels[channel]
        eq = record["local"]
        result = apply_eq({key: value for key, value in payload.items() if not key.startswith("eq_link")}, eq)
        if record["group"]:
            result = apply_eq(result, self.resolve_eq(channel))
        return result

    def update(self, channel, payload, expected=None):
        record = self.channels[channel]
        if expected is not None and self.resolve_eq(channel) != expected:
            raise ValueError("別の画面でStereo Link設定が変更されました。再読込して編集してください。")
        # Preserve private categories while replacing only the selected group's fields.
        record["local"] = extract_eq(payload)
        owner = self.resolve_eq(channel)
        categories = self.groups[record["group"]]["categories"] if record["group"] else tuple(CONFIG_FIELDS)
        updated = extract_eq(payload, categories)
        owner.clear()
        owner.update(updated)

    def linked(self, channels):
        owners = [self.channels.get(channel, {}).get("group") for channel in channels]
        return bool(owners and owners[0] and all(owner == owners[0] for owner in owners))

    def link(self, channels, source, categories=("iir", "fir", "target")):
        channels = tuple(channels)
        if len(set(channels)) < 2 or source not in channels or not categories:
            raise ValueError("Stereo Link requires channels, an explicit source and shared settings")
        if not set(categories) <= set(CONFIG_FIELDS):
            raise ValueError("Invalid shared settings category")
        source_eq = self.effective(source, {"config": {}, "ui": {}, "io": {}})
        self.unlink(channels)
        group_id = uuid4().hex
        self.groups[group_id] = {"categories": list(categories), "eq": extract_eq(source_eq, categories)}
        for channel in channels:
            self.channels[channel]["group"] = group_id

    def unlink(self, channels):
        group_ids = {self.channels[channel]["group"] for channel in channels if channel in self.channels}
        # Group transitions are atomic: every member retains the current effective EQ.
        for group_id in group_ids - {None}:
            members = [channel for channel, record in self.channels.items() if record["group"] == group_id]
            locals_by_channel = {channel: extract_eq(self.effective(channel, {"config": {}, "ui": {}, "io": {}})) for channel in members}
            for channel in members:
                self.channels[channel] = {"group": None, "local": deepcopy(locals_by_channel[channel])}
            del self.groups[group_id]


@contextmanager
def transaction(root: Path, scope: str):
    """Serialize ownership changes across independent Streamlit processes."""
    path = Path(root) / "eq_links.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=10) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS eq_links (scope TEXT PRIMARY KEY, snapshot TEXT NOT NULL)")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT snapshot FROM eq_links WHERE scope=?", (scope,)).fetchone()
        model = EQLinks(json.loads(row[0]) if row else None)
        yield model
        encoded = json.dumps(model.snapshot(), ensure_ascii=False, sort_keys=True)
        if not row or encoded != row[0]:
            connection.execute("INSERT INTO eq_links VALUES (?,?) ON CONFLICT(scope) DO UPDATE SET snapshot=excluded.snapshot", (scope, encoded))


def standalone_payload(payload):
    """Resolve the selected legacy EQ once, preserving channel-specific inputs."""
    if "eq_links" in payload:
        model = EQLinks(payload["eq_links"])
        channel = payload.get("eq_link_channel")
        if channel not in model.channels:
            raise ValueError("Missing Stereo Link channel")
        payload = model.effective(channel, payload)
    return deepcopy({key: value for key, value in payload.items() if not key.startswith("eq_link")})
