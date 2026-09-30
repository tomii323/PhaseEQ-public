from __future__ import annotations

from abc import ABC, abstractmethod

from ..common import validate_package
from ..models import CanonicalDSPPackage, CompatibilityReport, DSPProfile, ExportedDSPTarget


class DSPExportAdapter(ABC):
    adapter_id: str
    display_name: str

    @abstractmethod
    def profiles(self) -> tuple[DSPProfile, ...]: ...

    def validate(self, package: CanonicalDSPPackage, profile: DSPProfile) -> CompatibilityReport:
        return validate_package(package, profile)

    @abstractmethod
    def export(self, package: CanonicalDSPPackage, profile: DSPProfile) -> ExportedDSPTarget: ...
