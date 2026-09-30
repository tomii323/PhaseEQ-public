from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Any, MutableMapping


CACHE_COORDINATOR_SCHEMA_VERSION = 1
CACHE_COORDINATOR_STATE_KEY = "_cache_coordinator_state"


class CacheDomain(StrEnum):
    DESIGN_RESULT = "design_result"
    WAVELET = "wavelet"
    MEASUREMENT_RESULT = "measurement_result"
    AUDIO_CAPABILITY = "audio_capability"
    ESS_ANALYSIS = "ess_analysis"
    EXTERNAL_DOWNLOAD = "external_download"


class CacheEvent(StrEnum):
    DESIGN_CHANGED = "design_changed"
    INPUT_CHANGED = "input_changed"
    TARGET_CHANGED = "target_changed"
    EQ_CHANGED = "eq_changed"
    FIR_CHANGED = "fir_changed"
    PAGE_STAGE_CHANGED = "page_stage_changed"
    MEASUREMENT_CHANGED = "measurement_changed"
    AUDIO_DEVICES_REFRESHED = "audio_devices_refreshed"
    ESS_SOURCE_CHANGED = "ess_source_changed"
    EXTERNAL_CACHE_CHANGED = "external_cache_changed"
    ALGORITHM_CHANGED = "algorithm_changed"


EVENT_DOMAINS: dict[CacheEvent, frozenset[CacheDomain]] = {
    CacheEvent.DESIGN_CHANGED: frozenset({CacheDomain.DESIGN_RESULT, CacheDomain.WAVELET}),
    CacheEvent.INPUT_CHANGED: frozenset({CacheDomain.DESIGN_RESULT, CacheDomain.WAVELET}),
    CacheEvent.TARGET_CHANGED: frozenset({CacheDomain.DESIGN_RESULT, CacheDomain.WAVELET}),
    CacheEvent.EQ_CHANGED: frozenset({CacheDomain.DESIGN_RESULT, CacheDomain.WAVELET}),
    CacheEvent.FIR_CHANGED: frozenset({CacheDomain.DESIGN_RESULT, CacheDomain.WAVELET}),
    # Stage is already part of the design key. Navigation must not discard
    # unchanged stage results; source/EQ/algorithm events still invalidate them.
    CacheEvent.PAGE_STAGE_CHANGED: frozenset({CacheDomain.WAVELET}),
    CacheEvent.MEASUREMENT_CHANGED: frozenset({CacheDomain.MEASUREMENT_RESULT, CacheDomain.WAVELET}),
    CacheEvent.AUDIO_DEVICES_REFRESHED: frozenset({CacheDomain.AUDIO_CAPABILITY}),
    CacheEvent.ESS_SOURCE_CHANGED: frozenset({CacheDomain.ESS_ANALYSIS, CacheDomain.MEASUREMENT_RESULT}),
    CacheEvent.EXTERNAL_CACHE_CHANGED: frozenset({CacheDomain.EXTERNAL_DOWNLOAD}),
    CacheEvent.ALGORITHM_CHANGED: frozenset(CacheDomain),
}


DOMAIN_SESSION_KEYS: dict[CacheDomain, tuple[str, ...]] = {
    CacheDomain.DESIGN_RESULT: (
        "_cached_result_context",
        "_last_result_signature",
        "_design_stage_result_cache",
        "_response_chart_data_cache",
    ),
    CacheDomain.WAVELET: ("_wavelet_last_result",),
    CacheDomain.MEASUREMENT_RESULT: ("_ess_high_precision_cache",),
    CacheDomain.AUDIO_CAPABILITY: ("_ess_measure_device_format_check",),
    CacheDomain.ESS_ANALYSIS: (),
    CacheDomain.EXTERNAL_DOWNLOAD: (),
}


@dataclass(frozen=True)
class CacheKey:
    domain: CacheDomain
    revision: int
    algorithm_version: str = ""
    source_digest: str = ""
    settings_digest: str = ""
    stage: str = ""
    schema_version: int = CACHE_COORDINATOR_SCHEMA_VERSION

    @property
    def digest(self) -> str:
        payload = {
            "algorithm_version": self.algorithm_version,
            "domain": str(self.domain),
            "revision": int(self.revision),
            "schema_version": int(self.schema_version),
            "settings_digest": self.settings_digest,
            "source_digest": self.source_digest,
            "stage": self.stage,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def content_digest(*parts: Any) -> str:
    """Return a deterministic digest for already-serialized cache inputs."""

    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            data = part
        elif isinstance(part, bytearray):
            data = bytes(part)
        elif isinstance(part, memoryview):
            data = part.tobytes()
        else:
            data = str(part).encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


class CacheCoordinator:
    """Coordinate per-session derived caches through revisions and events.

    The coordinator stores only revision metadata in the supplied mapping.
    Cached values remain in their appropriate Streamlit session/process cache;
    persistent settings, databases, and measurement recovery files are never
    treated as cache entries.
    """

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self._state = state
        self._ensure_state()

    def _ensure_state(self) -> dict[str, Any]:
        current = self._state.get(CACHE_COORDINATOR_STATE_KEY)
        if not isinstance(current, dict) or current.get("schema_version") != CACHE_COORDINATOR_SCHEMA_VERSION:
            current = {
                "schema_version": CACHE_COORDINATOR_SCHEMA_VERSION,
                "revisions": {str(domain): 0 for domain in CacheDomain},
                "last_event": "",
            }
            self._state[CACHE_COORDINATOR_STATE_KEY] = current
        else:
            revisions = current.setdefault("revisions", {})
            for domain in CacheDomain:
                revisions.setdefault(str(domain), 0)
        return current

    def revision(self, domain: CacheDomain) -> int:
        current = self._ensure_state()
        return int(current["revisions"].get(str(domain), 0))

    def notify(self, event: CacheEvent) -> dict[CacheDomain, int]:
        domains = EVENT_DOMAINS[event]
        current = self._ensure_state()
        revisions = dict(current["revisions"])
        changed: dict[CacheDomain, int] = {}
        for domain in domains:
            next_revision = int(revisions.get(str(domain), 0)) + 1
            revisions[str(domain)] = next_revision
            changed[domain] = next_revision
            for key in DOMAIN_SESSION_KEYS[domain]:
                self._state.pop(key, None)
        self._state[CACHE_COORDINATOR_STATE_KEY] = {
            "schema_version": CACHE_COORDINATOR_SCHEMA_VERSION,
            "revisions": revisions,
            "last_event": str(event),
        }
        return changed

    def key(
        self,
        domain: CacheDomain,
        *,
        algorithm_version: str = "",
        source_digest: str = "",
        settings_digest: str = "",
        stage: str = "",
        revision: int | None = None,
    ) -> CacheKey:
        return CacheKey(
            domain=domain,
            revision=self.revision(domain) if revision is None else int(revision),
            algorithm_version=str(algorithm_version),
            source_digest=str(source_digest),
            settings_digest=str(settings_digest),
            stage=str(stage),
        )

    def status(self) -> dict[str, Any]:
        current = self._ensure_state()
        return {
            "schema_version": int(current["schema_version"]),
            "revisions": dict(current["revisions"]),
            "last_event": str(current.get("last_event", "")),
        }
