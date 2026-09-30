from __future__ import annotations

from .base import DSPExportAdapter
from .camilladsp import CamillaDSPAdapter
from .generic import GenericParametersAdapter, GenericSOSAdapter
from .minidsp import MiniDSPAdapter
from .sigmastudio import SigmaStudioAdapter


_ADAPTERS: tuple[DSPExportAdapter, ...] = (
    MiniDSPAdapter(), CamillaDSPAdapter(), SigmaStudioAdapter(),
    GenericParametersAdapter(), GenericSOSAdapter(),
)


def available_adapters() -> tuple[DSPExportAdapter, ...]:
    return _ADAPTERS


def adapter_by_id(adapter_id: str) -> DSPExportAdapter:
    for adapter in _ADAPTERS:
        if adapter.adapter_id == adapter_id:
            return adapter
    raise KeyError(f"unknown DSP export adapter: {adapter_id}")
