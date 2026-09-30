"""Application-neutral FIR/FRD composite engine."""

from .core import CompositeEngine, CompositeResult, MultichannelCompositeResult
from .manifest import ChannelManifest, CompositeManifest, load_manifest
from .validation import CompositeValidationError, ValidationIssue
from .features import NEW_COMPOSITE_ENGINE, selected_engine

__all__ = [
    "ChannelManifest",
    "CompositeEngine",
    "CompositeManifest",
    "CompositeResult",
    "MultichannelCompositeResult",
    "CompositeValidationError",
    "ValidationIssue",
    "load_manifest",
    "NEW_COMPOSITE_ENGINE",
    "selected_engine",
]
