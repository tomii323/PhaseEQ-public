"""Validate an Assignment speaker/time pair before changing live rows."""
import hashlib
import io
from pathlib import Path

import numpy as np


def validated_speaker_asset(result, root, revision):
    sos = np.asarray(result.get("phaseeq_iir_sos", ()), dtype=float)
    if sos.size:
        from composite_engine.iir_crossover import sos_is_stable
        if sos.ndim != 2 or sos.shape[1] != 6 or not np.isfinite(sos).all() or not sos_is_stable(sos):
            raise ValueError("IIR係数が不正です（前回結果を維持）。")
    asset = result.get("speaker_response")
    timing = result.get("timing_provenance")
    if not isinstance(asset, dict):
        if isinstance(timing, dict):
            raise ValueError("時間情報に対応するSpeaker応答がありません。")
        return None
    path = Path(str(asset.get("path", ""))).resolve(strict=True)
    path.relative_to(Path(root).resolve())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if not asset.get("content_hash") or digest != asset["content_hash"]:
        raise ValueError("Speaker応答のhashが一致しません。")
    if isinstance(timing, dict) and timing.get("response_hash") and timing["response_hash"] != digest:
        raise ValueError("Speaker応答と時間情報の世代が一致しません。")
    if isinstance(timing, dict) and int(timing.get("schema_version", 1)) >= 2:
        if timing.get("response_hash") != digest:
            raise ValueError("時間契約にSpeaker応答のhashがありません。")
        if timing.get("phase_origin") not in {"measured", "missing", "estimated_minimum_phase", "unknown"}:
            raise ValueError("時間契約の位相の由来が不正です。")
    values = np.genfromtxt(io.BytesIO(data), comments="#")
    if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] not in (2, 3)
            or not np.isfinite(values).all() or np.any(values[:, 0] <= 0)
            or np.any(np.diff(values[:, 0]) <= 0)):
        raise ValueError("Speaker応答の形式が不正です。")
    available = values.shape[1] == 3 and asset.get("phase_available") is not False
    if not available and values.shape[1] == 3:
        # Legacy senders wrote zero phase for missing measurements.
        data = ("\n".join(f"{f:.12g} {g:.12g}" for f, g in values[:, :2]) + "\n").encode()
    return dict(filename=path.name, data=data, revision=revision, content_hash=digest,
                phase_available=available, source_stage=asset.get("source_stage", "Response Processing"))


def timing_for_asset(result, asset):
    timing = result.get("timing_provenance")
    if not isinstance(timing, dict):
        return None
    timing = dict(timing)
    if asset is not None and not asset["phase_available"]:
        timing["phase_origin"] = "missing"
    return timing
