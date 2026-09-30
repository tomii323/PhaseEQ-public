from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from utils.exchange_io import atomic_json, locked_exchange

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

from ..iir_crossover import IIRCrossoverConfig, sos_is_stable


@dataclass(frozen=True)
class PhaseEQAssignmentRequest:
    sample_rate_hz: int
    tap_count: int
    channel_name: str
    way: str
    group: str
    channel_id: str = ""
    return_url: str = ""
    multiway_mode: str = ""
    highpass_hz: float = 0.0
    lowpass_hz: float = 0.0
    linear_fir_filters: tuple[dict[str, object], ...] = ()
    iir_crossover: dict[str, object] | None = None
    band_split_recipe: dict[str, object] | None = None
    target_definition: dict[str, object] | None = None
    target_application: str = "phaseeq_target"
    system_id: str = ""
    management_no: str = ""
    system_revision_number: int | None = None
    system_content_hash: str = ""
    target_id: str = ""
    target_revision_number: int | None = None
    target_content_hash: str = ""
    assignment_signature: str = ""

    @property
    def fir_enabled(self) -> bool:
        """Return whether this Assignment may return a FIR coefficient set."""
        return int(self.tap_count) > 0


@dataclass(frozen=True)
class PhaseEQAssignmentStatus:
    assignment_id: str
    channel_id: str
    state: str
    revision: int
    updated_at: str
    message: str = ""
    result: dict[str, object] | None = None
    published_at: str = ""


def validate_phaseeq_assignment_request(request: PhaseEQAssignmentRequest) -> None:
    if int(request.sample_rate_hz) <= 0:
        raise ValueError("sample rate must be positive")
    if request.band_split_recipe is not None:
        from crossover_engine.recipe import validate
        validate(request.band_split_recipe)
        if (request.band_split_recipe["sample_rate_hz"] != request.sample_rate_hz
                or request.band_split_recipe["way"] != request.way):
            raise ValueError("Assignment band split recipe mismatch")
        if request.band_split_recipe.get("fir_enabled", True) != (request.tap_count > 0):
            raise ValueError("Assignment band split FIR state mismatch")
        from crossover_engine.recipe import verify_iir_payload
        verify_iir_payload(request.band_split_recipe, request.iir_crossover)
    if int(request.tap_count) < 0:
        raise ValueError("tap count must be zero (FIR OFF) or positive")
    if not request.channel_name.strip() or not request.way.strip() or not request.group.strip():
        raise ValueError("Channel, Way and Composite Group are required")
    if request.iir_crossover is not None:
        payload_iir = request.iir_crossover
        if str(payload_iir.get("type", "")) not in {"OFF", "LR2", "LR4", "Mixed"}:
            raise ValueError("Assignment IIR crossover must be OFF, LR2, LR4 or Mixed")
        if str(payload_iir.get("application", "")) != "exclusive_crossover_stage" or not bool(payload_iir.get("exclusive", False)):
            raise ValueError("Assignment crossover must use the exclusive stage contract")
        coefficient_fs = int(payload_iir.get("coefficient_sample_rate_hz", 0))
        if coefficient_fs != int(request.sample_rate_hz):
            raise ValueError("Assignment IIR crossover sample rate mismatch")
        sos = np.asarray(payload_iir.get("sos", []), dtype=float)
        if sos.size and (sos.ndim != 2 or sos.shape[1] != 6 or not np.isfinite(sos).all() or not sos_is_stable(sos)):
            raise ValueError("Assignment IIR crossover SOS is invalid or unstable")
        config_keys = set(IIRCrossoverConfig.__dataclass_fields__)
        config = IIRCrossoverConfig(**{
            key: payload_iir[key] for key in config_keys if key in payload_iir
        })
        config.validate(int(request.sample_rate_hz))
        fir_modes = {
            str(item.get("mode", "")) for item in request.linear_fir_filters
            if isinstance(item, dict) and bool(item.get("enabled", True))
        }
        if (config.highpass_order and "hp" in fir_modes) or (config.lowpass_order and "lp" in fir_modes):
            raise ValueError("Kaiser FIR and LR crossover are mutually exclusive per boundary")
        if str(payload_iir.get("definition_hash", "")) != config.definition_hash(coefficient_fs):
            raise ValueError("Assignment IIR crossover definition hash mismatch")
    if request.target_definition is not None:
        if not isinstance(request.target_definition.get("target_edit", {}), dict):
            raise ValueError("Assignment Target edit payload must be an object")
        if not str(request.target_definition.get("revision", "")).strip():
            raise ValueError("Assignment Target definition requires a revision")
    if request.target_application not in {"phaseeq_target", "display_only"}:
        raise ValueError("Assignment Target application is invalid")
    if request.system_id:
        if int(request.system_revision_number or 0) < 1 or not request.system_content_hash:
            raise ValueError("Assignment System provenance is incomplete")
    if request.target_id:
        if int(request.target_revision_number or 0) < 1 or not request.target_content_hash:
            raise ValueError("Assignment Target provenance is incomplete")


@locked_exchange
def write_phaseeq_assignment(
    exchange_root: str | Path,
    request: PhaseEQAssignmentRequest,
) -> Path:
    validate_phaseeq_assignment_request(request)
    root = Path(exchange_root)
    root.mkdir(parents=True, exist_ok=True)
    channel_id = request.channel_id.strip() or phaseeq_channel_id(
        request.channel_name, request.way, request.group,
    )
    assignment_id = uuid.uuid4().hex
    created_at = _utc_now()
    request_payload = asdict(request)
    signature_payload = {
        key: value for key, value in request_payload.items()
        if key != "assignment_signature"
    }
    calculated_signature = hashlib.sha256(
        json.dumps(
            signature_payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=str,
        ).encode("utf-8")
    ).hexdigest()
    if request.assignment_signature and request.assignment_signature != calculated_signature:
        raise ValueError("Assignment signature mismatch")
    assignment_signature = request.assignment_signature or calculated_signature
    payload = {
        **request_payload,
        "format": "phaseeq-composite-assignment",
        "format_version": 3,
        "assignment_id": assignment_id,
        "channel_id": channel_id,
        "created_at": created_at,
        "revision": 0,
        "assignment_state_revision": 0,
        "assignment_signature": assignment_signature,
        "system": (
            {
                "system_id": request.system_id,
                "management_no": request.management_no,
                "revision_number": request.system_revision_number,
                "content_hash": request.system_content_hash,
            }
            if request.system_id else None
        ),
        "target": (
            {
                "target_id": request.target_id,
                "revision_number": request.target_revision_number,
                "content_hash": request.target_content_hash,
                "application": request.target_application,
                "definition_snapshot": request.target_definition,
            }
            if request.target_id else None
        ),
    }
    assignment_directory = root / "assignments" / assignment_id
    assignment_directory.mkdir(parents=True, exist_ok=False)
    path = assignment_directory / "assignment.json"
    _atomic_json(path, payload)
    _atomic_json(assignment_directory / "workspace.json", {
        "format_version": 2,
        "assignment_id": assignment_id,
        "channel_id": channel_id,
        "revision": 0,
        "assignment_state_revision": 0,
        "assignment_signature": assignment_signature,
        "updated_at": created_at,
    })
    _atomic_json(assignment_directory / "status.json", {
        "assignment_id": assignment_id,
        "channel_id": channel_id,
        "state": "Assigned",
        "revision": 0,
        "assignment_state_revision": 0,
        "assignment_signature": assignment_signature,
        "updated_at": created_at,
        "message": "",
    })
    latest_directory = root / "channels"
    latest_directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(latest_directory / f"{channel_id}.json", {
        "assignment_id": assignment_id,
        "channel_id": channel_id,
        "assignment_path": str(path.relative_to(root)),
        "updated_at": created_at,
    })
    # Compatibility latest pointer for older PhaseEQ builds.
    _atomic_json(root / "phaseeq_assignment.json", payload)
    return path


def read_phaseeq_assignment_status(
    exchange_root: str | Path,
    *,
    assignment_id: str,
) -> PhaseEQAssignmentStatus | None:
    normalized = _safe_id(assignment_id)
    if not normalized:
        return None
    path = Path(exchange_root) / "assignments" / normalized / "status.json"
    if not path.is_file():
        return None
    from utils.exchange_io import read_status_payload
    payload = read_status_payload(path)
    result = payload.get('result')
    published_at = str(payload.get('published_at', ''))
    return PhaseEQAssignmentStatus(
        assignment_id=str(payload.get("assignment_id", "")),
        channel_id=str(payload.get("channel_id", "")),
        state=str(payload.get("state", "Assigned")),
        revision=int(payload.get("revision", 0)),
        updated_at=str(payload.get("updated_at", "")),
        message=str(payload.get("message", "")),
        result=dict(result) if isinstance(result, dict) else None,
        published_at=published_at,
    )


def latest_phaseeq_assignment_id(exchange_root: str | Path, *, channel_id: str) -> str | None:
    normalized = _safe_id(channel_id)
    if not normalized:
        return None
    path = Path(exchange_root) / "channels" / f"{normalized}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignment_id = _safe_id(str(payload.get("assignment_id", "")))
    return assignment_id or None


def latest_published_phaseeq_assignment_id(
    exchange_root: str | Path, *, channel_id: str,
) -> str | None:
    """Return the newest same-channel Assignment with a retained published result."""
    normalized = _safe_id(channel_id)
    assignments_root = Path(exchange_root) / "assignments"
    if not normalized or not assignments_root.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for status_path in assignments_root.glob("*/status.json"):
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        assignment_id = _safe_id(str(payload.get("assignment_id", "")))
        try:
            status = (
                read_phaseeq_assignment_status(
                    exchange_root, assignment_id=assignment_id,
                )
                if assignment_id else None
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            status is None
            or status.channel_id != normalized
            or status.state not in {"Ready", "Editing"}
            or not isinstance(status.result, dict)
        ):
            continue
        working_session = (
            assignments_root / assignment_id / "results" / "working_session.zip"
        )
        try:
            published_order = datetime.fromisoformat(status.published_at).timestamp()
        except (TypeError, ValueError):
            try:
                published_order = working_session.stat().st_mtime
            except OSError:
                try:
                    published_order = datetime.fromisoformat(status.updated_at).timestamp()
                except (TypeError, ValueError):
                    published_order = 0.0
        candidates.append((published_order, assignment_id))
    return max(candidates)[1] if candidates else None


def phaseeq_channel_id(channel_name: str, way: str, group: str) -> str:
    source = "\0".join((group.strip(), way.strip(), channel_name.strip())).encode("utf-8")
    return f"ch_{hashlib.sha256(source).hexdigest()[:20]}"


def _safe_id(value: str) -> str:
    normalized = str(value).strip()
    return normalized if normalized and all(character.isalnum() or character in "_-" for character in normalized) else ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    atomic_json(path, payload)
