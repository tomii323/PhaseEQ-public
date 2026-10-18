"""Shared routing policy for direct split stages and structural all-pass plans."""
from dataclasses import dataclass

from .methods import boundary_method

ROUTING_VERSION = 2


@dataclass(frozen=True)
class SplitTopology:
    boundary_count: int
    cascaded: bool

    def split_boundaries(self, way_index):
        if not 0 <= way_index <= self.boundary_count:
            raise ValueError('Way index does not match boundaries')
        if self.cascaded:
            return tuple(range(min(way_index + 1, self.boundary_count)))
        return tuple(index for index in (way_index - 1, way_index)
                     if 0 <= index < self.boundary_count)

    def ap_recipients(self, boundary_index):
        if not 0 <= boundary_index < self.boundary_count:
            raise ValueError('Boundary index does not match topology')
        if self.cascaded:
            return tuple(range(boundary_index))
        return tuple(index for index in range(self.boundary_count + 1)
                     if index not in {boundary_index, boundary_index + 1})


def resolve_split_topology(methods, *, acoustic_targets=()):
    """Only a complete direct complementary IIR bank owns the remaining HP bus.

    FIR, Through and target-owned boundaries use the adjacent-boundary model;
    structural compensation in that model may add APs, never remote HPs.
    """
    methods = tuple(methods)
    targets = tuple(acoustic_targets)
    if targets and len(targets) != len(methods):
        raise ValueError('Acoustic target count does not match boundaries')
    definitions = tuple(boundary_method(method) for method in methods)
    return SplitTopology(len(methods), cascaded=(
        any(item.requires_cascaded_bus for item in definitions)
        and all(item.complementary_iir for item in definitions) and not any(targets)))
