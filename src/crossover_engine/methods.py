"""Boundary method capabilities and coefficient adapters, without app state."""
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Callable


def _lr_sos(side, frequency, sample_rate, lobe, *, order):
    from .iir import IIRCrossoverConfig, iir_crossover_sos
    return iir_crossover_sos(IIRCrossoverConfig(
        enabled=True, order=order,
        highpass_hz=frequency if side == 'highpass' else 0.,
        lowpass_hz=frequency if side == 'lowpass' else 0.), sample_rate)


def _ntm4_sos(side, frequency, sample_rate, lobe):
    from .ntm4 import ntm4_sos
    return ntm4_sos(side, frequency, sample_rate, lobe)


def _ntm4_q(lobe):
    from .ntm4 import allpass_q
    return allpass_q(lobe)


def _lr2_q(lobe):
    return 0.


def _lr4_q(lobe):
    return 1. / sqrt(2.)


@dataclass(frozen=True)
class BoundaryMethod:
    kind: str
    response_key: str
    order: int = 0
    complementary_iir: bool = False
    requires_cascaded_bus: bool = False
    requires_bank_definition: bool = False
    split_sos: Callable | None = None
    sum_ap_order: int = 0
    sum_ap_q: Callable | None = None
    above_ap_polarity: int = 1


# The table owns method identity and routing capabilities. Numeric adapters
# reuse each prototype; UI labels, persistence and transfer remain separate.
METHOD_DEFINITIONS = {
    'Kaiser FIR': BoundaryMethod('fir', 'kaiser'),
    'Linear-phase LR2 FIR': BoundaryMethod('fir', 'linear_phase_lr2'),
    'Linear-phase LR4 FIR': BoundaryMethod('fir', 'linear_phase_lr4'),
    'LR2': BoundaryMethod('iir', 'iir_lr2', order=2, complementary_iir=True,
        split_sos=partial(_lr_sos, order=2), sum_ap_order=1, sum_ap_q=_lr2_q, above_ap_polarity=-1),
    'LR4': BoundaryMethod('iir', 'iir_lr4', order=4, complementary_iir=True,
        split_sos=partial(_lr_sos, order=4), sum_ap_order=2, sum_ap_q=_lr4_q),
    'NTM4': BoundaryMethod('iir', 'iir_ntm4', order=4, complementary_iir=True,
        requires_cascaded_bus=True, requires_bank_definition=True,
        split_sos=_ntm4_sos, sum_ap_order=2, sum_ap_q=_ntm4_q),
    'Through': BoundaryMethod('through', 'through'),
}


def boundary_method(name):
    try:
        return METHOD_DEFINITIONS[name]
    except KeyError:
        raise ValueError(f'Unsupported crossover method: {name}') from None
