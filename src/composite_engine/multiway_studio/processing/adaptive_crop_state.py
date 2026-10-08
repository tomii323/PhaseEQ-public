"""Session-boundary contract for Adaptive crop result rows."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


ADAPTIVE_AUTO_CROP_ROWS_KEY = "adaptive_crop_rows"

_ADAPTIVE_REQUIRED_KEYS = frozenset({
    "チャンネル", "元 [taps]", "直接クロップ後 [taps]", "採用 [taps]",
    "左削除", "右削除", "左zero", "右zero", "時間補償", "採用",
})


def _rows_matching(value: object, required_keys: frozenset[str]) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [
        dict(row) for row in value
        if isinstance(row, Mapping) and required_keys.issubset(row)
    ]


def adaptive_auto_crop_rows(value: object) -> list[dict[str, object]]:
    """Return only rows produced by the final-FIR Adaptive evaluator."""
    return _rows_matching(value, _ADAPTIVE_REQUIRED_KEYS)
