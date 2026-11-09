"""Assemble existing equivalent and structural AP stages without application state."""
from __future__ import annotations

import numpy as np

from crossover_engine.alignment import AllPassSection, allpass_sos
from crossover_engine.iir import crossover_edges_hz
from crossover_engine.methods import boundary_method
from crossover_engine.topology import resolve_split_topology

from .common import allpass_sos_rows, normalized_sos
from .models import CanonicalDSPChannel


def equivalent_split_sections(channel: CanonicalDSPChannel) -> tuple[AllPassSection, ...]:
    """Replace only the direct split stages; structural APs have a separate owner."""
    config = channel.crossover
    if not config.enabled:
        raise ValueError("LR4またはNTM4の直接帯域分割を設定してください。")
    sections = []
    if config.ntm4_topology is not None:
        topology = config.ntm4_topology
        methods = tuple(topology['methods'])
        if not methods or any(method not in {'LR4', 'NTM4'} for method in methods):
            raise ValueError("すべての境界をLR4またはNTM4に設定してください。")
        plan = resolve_split_topology(methods)
        way_index = int(topology['way_index'])
        for index in plan.split_boundaries(way_index):
            method = boundary_method(methods[index])
            side = 1 if index < way_index else 0
            cutoff = crossover_edges_hz(topology['frequencies'][index], topology['overlaps'][index])[side]
            sections.append(AllPassSection(cutoff, method.sum_ap_q(topology['lobes'][index]),
                                           order=method.sum_ap_order))
    else:
        if config.family != 'Linkwitz-Riley' or config.order != 4:
            raise ValueError("直接帯域分割をLR4またはNTM4に設定してください。")
        method = boundary_method('LR4')
        for cutoff, order in ((config.highpass_hz, config.highpass_order),
                              (config.lowpass_hz, config.lowpass_order)):
            if cutoff > 0:
                if order not in (0, 4):
                    raise ValueError("すべての境界をLR4またはNTM4に設定してください。")
                sections.append(AllPassSection(cutoff, method.sum_ap_q(0.), order=method.sum_ap_order))
    return tuple(sections)


def pre_dirac_sos(channel: CanonicalDSPChannel) -> np.ndarray:
    equivalents = [allpass_sos(section, channel.sample_rate_hz)
                   for section in equivalent_split_sections(channel)]
    # Canonical crossover_allpass already contains the applied automatic APs.
    # Do not regenerate structural compensation or deduplicate by coefficient.
    return normalized_sos(np.vstack([*equivalents, allpass_sos_rows(channel)]))
