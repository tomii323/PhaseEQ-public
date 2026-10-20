"""Reject realized Multiway artifacts whose EQ owner has since changed."""
from __future__ import annotations

import io
from pathlib import Path

from utils.eq_links import extract_eq, transaction
from utils.export_bundle import config_payload_from_project_zip


def returned_settings_match(model, channel, payload):
    if channel not in model.channels:
        return True
    current = model.effective(channel, {"config": {}, "ui": {}, "io": {}})
    return extract_eq(payload) == extract_eq(current)


def invalidate_stale_returns(rows, root, scope):
    decisions = []
    with transaction(root, scope) as model:
        for row in rows:
            channel = str(row.get("channel_id", ""))
            path = row.get("phaseeq_working_session")
            has_result = bool(row.get("phaseeq_assignment_id"))
            if channel not in model.channels or not has_result:
                pending = False
            elif path and Path(path).is_file():
                payload = config_payload_from_project_zip(io.BytesIO(Path(path).read_bytes()))
                pending = not returned_settings_match(model, channel, payload)
            else:
                pending = bool(model.channels[channel]["group"])
            decisions.append((row, pending))
    # Publish one completed decision set. A failed read or migration must not
    # leave earlier rows marked fresh while later rows remain unverified.
    for row, pending in decisions:
        row["phaseeq_link_pending"] = pending
    return tuple(str(row.get("name", row.get("channel_id", "")))
                 for row, pending in decisions if pending)
