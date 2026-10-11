from __future__ import annotations

import pandas as pd

from utils.ui_localization import display_text, ui_message


def taper_limit_messages(channels) -> list[str]:
    """Explain recorded source limits without making a new crop decision."""
    messages = []
    for channel in channels:
        metadata = channel.adaptive_crop_metadata or {}
        decision = metadata.get("final_crop_range", {})
        if not channel.cosine_taper_enabled or not decision.get("edge_limited", False):
            continue
        core = int(metadata.get("base_candidate_taps") or
                   metadata.get("cosine_taper_reference_taps", 0))
        final = int(metadata["final_taps"])
        messages.append(ui_message(
            "ui.48b7ca78705bec" if final == core else "ui.388ef763d6f483",
            channel=channel.name, core=core, final=final,
            requested=int(decision["requested_taps"]), input=int(metadata["original_taps"]),
        ))
    return messages


def adaptive_crop_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build an Arrow-safe display projection without changing domain rows."""
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if "[taps]" in str(column):
            frame[column] = frame[column].map(str)
    return frame.rename(columns=display_text)
