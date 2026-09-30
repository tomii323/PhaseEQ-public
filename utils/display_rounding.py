from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
from typing import Any


def round_for_display(value: Any, digits: int = 0) -> float:
    """Round a finite display value with conventional decimal half-up rounding."""
    number = float(value)
    if not math.isfinite(number):
        return number
    places = max(0, int(digits))
    quantum = Decimal(1).scaleb(-places)
    try:
        return float(Decimal(str(number)).quantize(quantum, rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return number


def format_for_display(
    value: Any,
    digits: int = 0,
    *,
    signed: bool = False,
    grouping: bool = False,
) -> str:
    """Format with conventional half-up rounding and fixed decimal places."""
    rounded = round_for_display(value, digits)
    sign = "+" if signed else ""
    group = "," if grouping else ""
    return format(rounded, f"{sign}{group}.{max(0, int(digits))}f")
