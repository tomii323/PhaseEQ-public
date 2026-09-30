from __future__ import annotations


def eq_filter_card_title(
    family: str,
    index: int,
    filter_type: str,
    *,
    detail: str = "",
    enabled: bool = True,
) -> str:
    parts = [f"{family} EQ{int(index) + 1}", str(filter_type)]
    if detail:
        parts.append(str(detail))
    label = "  |  ".join(parts)
    return label if enabled else f"{label} / OFF"


def allpass_polarity_detail(polarity: str) -> str:
    return "Min / 正" if str(polarity) == "positive" else "Max / 逆"
