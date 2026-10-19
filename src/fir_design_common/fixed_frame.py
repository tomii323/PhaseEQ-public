"""Plan and execute a manual coefficient frame without Adaptive decisions."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FixedFIRFramePlan:
    input_taps: int
    output_taps: int
    source_start: int
    source_stop: int
    padding_left: int
    padding_right: int

    @property
    def origin_samples(self) -> int:
        return self.source_start - self.padding_left


def plan_fixed_fir_frame(input_taps: int, output_taps: int) -> FixedFIRFramePlan:
    if input_taps < 1 or output_taps < 1:
        raise ValueError("FIR frame lengths must be positive")
    retained = min(input_taps, output_taps)
    start = (input_taps - retained) // 2
    left = (output_taps - retained) // 2
    return FixedFIRFramePlan(input_taps, output_taps, start, start + retained,
                             left, output_taps - retained - left)


def execute_fixed_fir_frame(source, plan: FixedFIRFramePlan) -> np.ndarray:
    if len(source) != plan.input_taps:
        raise ValueError("FIR source length differs from the frame plan")
    output = np.zeros(plan.output_taps, dtype=float)
    output[plan.padding_left:plan.output_taps-plan.padding_right] = source[
        plan.source_start:plan.source_stop]
    return output
