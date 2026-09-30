"""Nonblocking, per-session scheduling of the latest numeric edit."""
from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any

DEADLINE = "_numeric_edit_deadline"
QUIET_SECONDS = 0.2


def mark_numeric_edit(state: MutableMapping[str, Any], *, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    state[DEADLINE] = now + QUIET_SECONDS


def numeric_edit_ready(
    state: MutableMapping[str, Any],
    *,
    now: float | None = None,
) -> bool:
    deadline = state.get(DEADLINE)
    return deadline is not None and (time.monotonic() if now is None else now) >= float(deadline)
