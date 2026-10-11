"""Presentation policy for Standalone final-output tap status."""
from __future__ import annotations

from dataclasses import dataclass

from utils.ui_localization import ui_message


@dataclass(frozen=True)
class StandaloneOutputTapView:
    """UI-only view of configured, draft, resolved, and final tap counts."""

    draft_taps: int
    applied_taps: int
    final_taps: int | None
    sample_rate: int

    @property
    def has_unapplied_draft(self) -> bool:
        return self.draft_taps != self.applied_taps

    @property
    def final_duration_ms(self) -> float | None:
        if self.final_taps is None:
            return None
        return self.final_taps / self.sample_rate * 1000.0


def standalone_output_tap_view(
    draft_taps: int,
    applied_taps: int,
    final_taps: int | None,
    sample_rate: int,
) -> StandaloneOutputTapView:
    """Validate and build the tap-count presentation state."""
    draft = int(draft_taps)
    applied = int(applied_taps)
    final = None if final_taps is None or int(final_taps) <= 0 else int(final_taps)
    rate = int(sample_rate)
    if draft < 0 or applied < 0:
        raise ValueError("tap settings must be >= 0")
    if rate < 1:
        raise ValueError("sample rate must be positive")
    return StandaloneOutputTapView(draft, applied, final, rate)


def _setting_label(taps: int) -> str:
    if taps == 0:
        return ui_message("ui.3e891ae6fd9fb3")
    return ui_message("ui.b91338c8b8fd0a", p0=f"{taps:,}")


def tap_setting_caption(view: StandaloneOutputTapView) -> str:
    """Describe saved/draft settings without treating zero as a zero-length FIR."""
    if view.has_unapplied_draft:
        return ui_message(
            "ui.3c0c0d5ba2bc0d",
            p0=_setting_label(view.draft_taps),
            p1=_setting_label(view.applied_taps),
        )
    if view.applied_taps == 0:
        return ui_message("ui.d922fbc497528d")
    return ui_message(
        "ui.34ed421896036f",
        p0=f"{view.applied_taps:,}",
        p1=f"{view.applied_taps / view.sample_rate * 1000.0:.2f}",
    )


def final_tap_caption(view: StandaloneOutputTapView) -> str:
    """Describe the realized FIR length separately from the saved setting."""
    if view.final_taps is None or view.final_duration_ms is None:
        return ui_message("ui.529e33bb932695")
    return ui_message(
        "ui.38575ed42cebea",
        p0=f"{view.final_taps:,}",
        p1=f"{view.final_duration_ms:.2f}",
    )


def output_tap_setting_label(
    *,
    applied_taps: int,
    resolved_taps: int,
    adaptive: bool,
) -> str:
    """Label the setting and its runtime resolution in the output summary."""
    applied = int(applied_taps)
    resolved = int(resolved_taps)
    if applied < 0 or resolved < 1:
        raise ValueError("invalid output tap state")
    if applied == 0:
        message_id = "ui.37dd20ce86ae46" if adaptive else "ui.49638bc0d93a1e"
        return ui_message(message_id, p0=f"{resolved:,}")
    if adaptive:
        return ui_message("ui.907ac2c611ac50", p0=f"{resolved:,}")
    return ui_message("ui.b91338c8b8fd0a", p0=f"{resolved:,}")
