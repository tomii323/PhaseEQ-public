from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .regeneration import DSPWayResult


DelayGuideState = Literal["eligible", "reference", "review"]


@dataclass(frozen=True)
class FixedDelayGuide:
    way_id: str
    setting_ms: float
    processing_ms: float
    total_ms: float
    suggested_add_ms: float
    suggested_total_ms: float
    state: DelayGuideState
    reason: str


def fixed_delay_guides(results: tuple[DSPWayResult, ...] | list[DSPWayResult]) -> tuple[FixedDelayGuide, ...]:
    """Suggest only deterministic delay equalisation; never optimise measured arrival."""

    if not results:
        return ()
    totals = {
        result.way.id: float(result.way.delay_ms) + float(result.latency.total_delay_ms)
        for result in results
    }
    reference = max(totals.values())
    guides: list[FixedDelayGuide] = []
    for result in results:
        current = totals[result.way.id]
        add = max(0.0, reference - current)
        deterministic = (
            result.latency.fir.confidence == "high"
            and result.latency.iir.confidence in {"high", "medium"}
        )
        state: DelayGuideState = "reference" if add <= 1e-9 else "eligible" if deterministic else "review"
        reason = {
            "reference": "最大の理論処理遅延を基準にします。",
            "eligible": "FIR係数とIIR係数から再現可能な固定差だけを補います。",
            "review": "非対称FIRなど推定要素があるため表示のみです。",
        }[state]
        guides.append(FixedDelayGuide(
            way_id=result.way.id,
            setting_ms=float(result.way.delay_ms),
            processing_ms=float(result.latency.total_delay_ms),
            total_ms=current,
            suggested_add_ms=add,
            suggested_total_ms=float(result.way.delay_ms) + add,
            state=state,
            reason=reason,
        ))
    return tuple(guides)
