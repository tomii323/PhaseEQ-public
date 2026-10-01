from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import re
import zipfile

import numpy as np
from fir_output_window import apply_output_window, COSINE_TAPER_ALPHA, remove_nyquist_component
from scipy.io import wavfile
from ..fir_artifact import FinalFIRArtifact
from fir_design_common import generation_fft_size, generation_frequency_axis, project_centered_fir, compose_fir_stages

from ..core import CompositeResult, MultichannelCompositeResult
from ..fft import fir_complex_response, group_delay_ms
from ..iir_crossover import IIRCrossoverConfig, iir_crossover_sos, sos_is_stable
from ..phase_alignment import AllPassSection, allpass_sos


DSP_RESUME_FORMAT = "phaseeq-multiway-dsp-package"
DSP_RESUME_FORMAT_VERSION = 2
DSP_RESUME_MIN_SUPPORTED_VERSION = 1
DSP_DELIVERY_FORMAT = "phaseeq-multiway-dsp-export"
DSP_DELIVERY_FORMAT_VERSION = 3


def _workspace_json_default(value: object) -> object:
    """Serialize normalized Studio state without retaining live UI objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict(orient="records")
        except TypeError:
            return value.to_dict()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


@dataclass(frozen=True)
class CompositeExport:
    files: dict[str, bytes]


@dataclass(frozen=True)
class DSPChannelExportInput:
    channel_id: str
    name: str
    way: str
    group: str
    sample_rate_hz: int
    tap_count: int | None
    fir_stages: tuple[tuple[str, np.ndarray], ...]
    phaseeq_iir: tuple[dict[str, object], ...] = ()
    phaseeq_iir_sos: tuple[tuple[float, ...], ...] = ()
    baffle_iir_parameters: dict[str, object] | None = None
    baffle_iir_sos: tuple[tuple[float, ...], ...] = ()
    iir_crossover: IIRCrossoverConfig = IIRCrossoverConfig()
    gain_db: float = 0.0
    polarity: int = 1
    dsp_additional_delay_samples: float = 0.0
    channel_relative_delay_samples: float = 0.0
    working_session_zip: bytes | None = None
    auto_alignment_delay_samples: float = 0.0
    auto_alignment_allpass: tuple[AllPassSection, ...] = ()
    timing_provenance: dict[str, object] | None = None
    cosine_taper_enabled: bool = False
    remove_nyquist_enabled: bool = False
    remove_nyquist_strength: float = 1.0
    pre_alignment_tap_count: int | None = None


def compose_channel_fir(channel: DSPChannelExportInput) -> np.ndarray:
    """Finish the assigned FIR frame before zero-padding it for output alignment."""
    output_taps = int(channel.tap_count or 0)
    design_taps = int(
        channel.pre_alignment_tap_count
        if channel.pre_alignment_tap_count is not None else output_taps
    )
    if not 0 < design_taps <= output_taps:
        raise ValueError("FIR alignment cannot shorten the assigned frame")
    fir = postprocess_channel_fir(compose_fir_stages(
        (values for _, values in channel.fir_stages), design_taps,
    ), channel)
    padding = output_taps - design_taps
    return np.pad(fir, (padding // 2, padding - padding // 2))


def postprocess_channel_fir(fir: np.ndarray, channel: DSPChannelExportInput) -> np.ndarray:
    """Match PhaseEQ order: remove Nyquist, then final taper; no Auto Gain."""
    values = np.asarray(fir, dtype=float)
    if channel.remove_nyquist_enabled:
        if not np.isfinite(channel.remove_nyquist_strength):
            raise ValueError("Nyquist removal strength must be finite")
        values, _ = remove_nyquist_component(values, channel.remove_nyquist_strength)
    return apply_output_window(values, channel.cosine_taper_enabled)


def fit_centered_fir_to_complex_response(response: np.ndarray, tap_count: int) -> np.ndarray:
    """Uniform-grid least-squares projection onto a centered real FIR support."""
    return project_centered_fir(response, tap_count)


def build_multichannel_export_zip(result: MultichannelCompositeResult) -> bytes:
    """Prepare analysis files and compress only on download."""
    exported = build_multichannel_export(result)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        from phase_fir_designer.output_terms import write_output_terms
        write_output_terms(archive)
        for path, data in exported.files.items():
            archive.writestr(path, data)
    return buffer.getvalue()


def build_dsp_resume_zip(
    channels: tuple[DSPChannelExportInput, ...],
    *,
    workspace: dict[str, object],
    composite: MultichannelCompositeResult | None = None,
    final_fir_artifacts: dict[str, FinalFIRArtifact] | None = None,
) -> bytes:
    if not channels:
        raise ValueError("at least one DSP channel is required")
    if final_fir_artifacts is not None:
        expected_ids = [c.channel_id for c in channels if c.fir_stages]
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError('duplicate FIR channel IDs')
        if set(final_fir_artifacts).difference(expected_ids):
            raise ValueError('unexpected final FIR artifact channel')
    sample_rates = {int(channel.sample_rate_hz) for channel in channels}
    if len(sample_rates) != 1:
        raise ValueError("all DSP channels must share one sample rate")
    sample_rate = sample_rates.pop()
    if any(channel.fir_stages and int(channel.tap_count or 0) < 1 for channel in channels):
        raise ValueError("FIR-enabled DSP channels require a positive tap_count")
    fft_size = generation_fft_size(
        sample_rate, *(int(channel.tap_count or 1) for channel in channels),
        *(1 + sum(np.asarray(values).size - 1 for _, values in channel.fir_stages)
          for channel in channels),
    )
    frequency = generation_frequency_axis(sample_rate, fft_size)
    files: dict[str, bytes] = {}
    manifest_channels: list[dict[str, object]] = []
    for channel in channels:
        desired = np.ones(frequency.shape, dtype=np.complex128)
        contains: list[str] = []
        for stage_name, coefficients in channel.fir_stages:
            values = np.asarray(coefficients, dtype=float)
            _frequency, stage_response = fir_complex_response(
                values, sample_rate, fft_size, center_position=(values.size - 1) / 2.0,
            )
            desired *= stage_response
            contains.append(str(stage_name))
        fir = None
        if channel.fir_stages:
            if final_fir_artifacts is not None:
                if channel.channel_id not in final_fir_artifacts:
                    raise ValueError(f'missing final FIR artifact: {channel.channel_id}')
                fir = final_fir_artifacts[channel.channel_id].validated_coefficients(channel)
            else:
                fir = compose_channel_fir(channel)
        if fir is not None:
            _frequency, realized = fir_complex_response(
                fir, sample_rate, fft_size, center_position=(fir.size - 1) / 2.0,
            )
        else:
            realized = np.ones(frequency.shape, dtype=np.complex128)
        stem = _safe_stem(channel.channel_id) or _safe_stem(channel.name) or "channel"
        root = f"channels/{stem}"
        if fir is not None:
            wav_buffer = io.BytesIO()
            wavfile.write(wav_buffer, sample_rate, fir.astype(np.float32))
            files[f"{root}/fir.wav"] = wav_buffer.getvalue()
            files[f"{root}/fir.csv"] = ("tap,coefficient\n" + "\n".join(
                f"{index},{value:.17g}" for index, value in enumerate(fir)
            ) + "\n").encode()
        sos = iir_crossover_sos(channel.iir_crossover, sample_rate)
        if sos.size and not sos_is_stable(sos):
            raise ValueError(f"unstable IIR crossover: {channel.name}")
        baffle_sos = np.asarray(channel.baffle_iir_sos, dtype=float)
        if baffle_sos.size:
            baffle_sos = baffle_sos.reshape(-1, 6)
            if not sos_is_stable(baffle_sos):
                raise ValueError(f"unstable IIR baffle compensation: {channel.name}")
        requested_dsp_delay = float(channel.dsp_additional_delay_samples)
        automatic_dsp_delay = 0.0
        effective_dsp_delay = requested_dsp_delay if requested_dsp_delay > 0.0 else automatic_dsp_delay
        alignment_sos = np.vstack([
            allpass_sos(section, sample_rate) for section in channel.auto_alignment_allpass
        ]) if channel.auto_alignment_allpass else np.empty((0, 6), dtype=float)
        if alignment_sos.size and not sos_is_stable(alignment_sos):
            raise ValueError(f"unstable phase alignment All-pass: {channel.name}")
        total_delay = (
            effective_dsp_delay
            + float(channel.channel_relative_delay_samples)
            + float(channel.auto_alignment_delay_samples)
        )
        if requested_dsp_delay < 0.0 or total_delay < 0.0:
            raise ValueError(f"negative DSP delay: {channel.name}")
        config = {
            "channel": channel.name,
            "channel_id": channel.channel_id,
            "way": channel.way,
            "group": channel.group,
            "sample_rate_hz": sample_rate,
            "processing_order": ["channel_fir", "phaseeq_iir", "baffle_iir", "iir_crossover", "phase_alignment", "gain", "polarity", "delay"],
            "fir_output_window": {"enabled": channel.cosine_taper_enabled,
                                  "name": "Cosine Tapered", "alpha": COSINE_TAPER_ALPHA},
            "fir_nyquist_removal": {"enabled": channel.remove_nyquist_enabled,
                                    "strength": channel.remove_nyquist_strength,
                                    "application": "before_output_window"},
            "channel_fir": (
                {"enabled": True, "file": "fir.wav", "tap_count": int(fir.size), "center_position": (fir.size - 1) / 2.0, "contains": contains}
                if fir is not None else
                {"enabled": False, "file": None, "tap_count": None, "center_position": None, "contains": []}
            ),
            "phaseeq_iir": list(channel.phaseeq_iir),
            "baffle_iir": {
                "enabled": bool(baffle_sos.size),
                "parameters": channel.baffle_iir_parameters,
                "sos": baffle_sos.tolist() if baffle_sos.size else [],
            },
            "iir_crossover": {**channel.iir_crossover.to_dict(), "sos": sos.tolist()},
            "phase_alignment": {
                "reference": "High",
                "allpass": [
                    {**section.to_dict(), "sos": allpass_sos(section, sample_rate).tolist()}
                    for section in channel.auto_alignment_allpass
                ],
                "delay_samples": float(channel.auto_alignment_delay_samples),
            },
            "gain_db": float(channel.gain_db),
            "polarity": int(channel.polarity),
            "effective_polarity": int(channel.polarity) * int(channel.iir_crossover.lr2_polarity),
            "delay": {
                "dsp_additional_samples": effective_dsp_delay,
                "dsp_additional_automatic": False,
                "channel_relative_samples": float(channel.channel_relative_delay_samples),
                "auto_alignment_samples": float(channel.auto_alignment_delay_samples),
                "total_samples": total_delay,
                "application": "frequency_domain_phase_rotation",
            },
        }
        files[f"{root}/dsp_config.json"] = json.dumps(config, ensure_ascii=False, indent=2).encode()
        phase = np.unwrap(np.angle(realized))
        error = realized - desired
        if fir is not None:
            report = {
                "requested_taps": int(channel.tap_count or 0),
                "actual_taps": int(fir.size),
                "center_position": (fir.size - 1) / 2.0,
                "fft_size": fft_size,
                "frequency_resolution_hz": sample_rate / fft_size,
                "max_complex_error": float(np.max(np.abs(error))),
                "max_magnitude_error_db": float(np.max(np.abs(
                    20.0 * np.log10(np.maximum(np.abs(realized), 1e-15))
                    - 20.0 * np.log10(np.maximum(np.abs(desired), 1e-15))
                ))),
                "phase_unwrapped_deg": np.rad2deg(phase).tolist(),
                "group_delay_ms": group_delay_ms(realized, frequency).tolist(),
                "normalization": "none",
            }
            files[f"{root}/fir_report.json"] = json.dumps(
                report, ensure_ascii=False,
            ).encode()
        if channel.working_session_zip is not None:
            files[f"workspace/phaseeq/{stem}.zip"] = bytes(channel.working_session_zip)
        manifest_channels.append({
            "id": channel.channel_id, "name": channel.name, "way": channel.way,
            "group": channel.group,
            "tap_count": int(fir.size) if fir is not None else None,
            "center_position": (fir.size - 1) / 2.0 if fir is not None else None,
            "fir": f"{root}/fir.wav" if fir is not None else None,
            "settings": f"{root}/dsp_config.json",
        })
    if composite is not None:
        files.update(build_multichannel_export(composite, root="groups").files)
    workspace_payload = dict(workspace)
    workspace_payload["format_version"] = DSP_RESUME_FORMAT_VERSION
    files["workspace.json"] = json.dumps(
        workspace_payload, ensure_ascii=False, indent=2,
        default=_workspace_json_default,
    ).encode()
    files["manifest.json"] = json.dumps({
        "format": DSP_RESUME_FORMAT, "format_version": DSP_RESUME_FORMAT_VERSION,
        "sample_rate_hz": sample_rate, "fft_size": fft_size,
        "frequency_resolution_hz": sample_rate / fft_size,
        "time_reference": "tap_center", "normalization": "none",
        "channels": manifest_channels,
    }, ensure_ascii=False, indent=2).encode()
    from phase_fir_designer.output_terms import OUTPUT_TERMS_FILENAME, OUTPUT_TERMS_BYTES
    files[OUTPUT_TERMS_FILENAME] = OUTPUT_TERMS_BYTES
    checksum_lines = [
        f"{hashlib.sha256(data).hexdigest()}  {path}"
        for path, data in sorted(files.items())
    ]
    files["checksums.sha256"] = ("\n".join(checksum_lines) + "\n").encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path, data in sorted(files.items()):
            output.writestr(path, data)
    return archive.getvalue()


def load_dsp_resume_zip(data: bytes) -> tuple[dict[str, object], dict[str, object]]:
    """Validate and read workspace metadata from resumable DSP packages."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if any(
            not name or name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/")
            for name in names
        ):
            raise ValueError("unsafe DSP package path")
        required = {"manifest.json", "checksums.sha256"}
        if not required.issubset(names):
            raise ValueError("DSP package is missing resume metadata")
        checksums = archive.read("checksums.sha256").decode("utf-8").splitlines()
        for line in checksums:
            digest, separator, path = line.partition("  ")
            if not separator or path not in names:
                raise ValueError("invalid DSP package checksum index")
            if hashlib.sha256(archive.read(path)).hexdigest() != digest:
                raise ValueError(f"DSP package checksum mismatch: {path}")
        manifest = json.loads(archive.read("manifest.json"))
        if not isinstance(manifest, dict):
            raise ValueError("invalid DSP package manifest")
        package_format = manifest.get("format")
        if package_format == DSP_RESUME_FORMAT:
            workspace_path = "workspace.json"
        elif package_format == DSP_DELIVERY_FORMAT:
            workspace_path = "configuration/workspace.json"
        else:
            raise ValueError("unsupported DSP package format")
        if workspace_path not in names:
            raise ValueError("DSP package is missing resume metadata")
        workspace = json.loads(archive.read(workspace_path))
    if not isinstance(workspace, dict):
        raise ValueError("invalid DSP package workspace")
    version = _validated_format_version(manifest, label="package")
    workspace_version = _validated_format_version(workspace, label="workspace")
    if package_format == DSP_RESUME_FORMAT:
        if not DSP_RESUME_MIN_SUPPORTED_VERSION <= version <= DSP_RESUME_FORMAT_VERSION:
            raise ValueError(f"unsupported DSP package format version: {version}")
        if workspace_version > DSP_RESUME_FORMAT_VERSION:
            raise ValueError(f"unsupported DSP workspace format version: {workspace_version}")
        if version >= 2 and workspace_version != version:
            raise ValueError("DSP package manifest/workspace version mismatch")
    else:
        if version != DSP_DELIVERY_FORMAT_VERSION:
            raise ValueError(f"unsupported DSP export format version: {version}")
        if workspace_version > DSP_RESUME_FORMAT_VERSION:
            raise ValueError(f"unsupported DSP workspace format version: {workspace_version}")
    try:
        sample_rate_hz = int(manifest.get("sample_rate_hz", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid DSP package sample rate") from exc
    if sample_rate_hz <= 0:
        raise ValueError("invalid DSP package sample rate")
    return manifest, workspace


def _validated_format_version(payload: dict[str, object], *, label: str) -> int:
    try:
        version = int(payload.get("format_version", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid DSP {label} format version") from exc
    if version < 1:
        raise ValueError(f"invalid DSP {label} format version")
    return version


def build_export(result: CompositeResult, *, stem: str = "composite") -> CompositeExport:
    wav_buffer = io.BytesIO()
    wavfile.write(wav_buffer, result.sample_rate_hz, np.asarray(result.impulse_response, dtype=np.float32))
    frd = "\n".join(f"{f:.16g} {m:.16g} {p:.16g}" for f, m, p in zip(result.frequency_hz, result.magnitude_db, result.phase_deg, strict=True)) + "\n"
    csv = "frequency_hz,magnitude_db,phase_deg,group_delay_ms\n" + "\n".join(
        f"{f:.16g},{m:.16g},{p:.16g},{g:.16g}" for f, m, p, g in zip(result.frequency_hz, result.magnitude_db, result.phase_deg, result.group_delay_ms, strict=True)
    ) + "\n"
    report = json.dumps({
        "engine": "Multiway FIR Composite Engine", "group": result.group,
        "sample_rate_hz": result.sample_rate_hz, "fft_size": result.fft_size,
        "channels": list(result.channel_responses), "normalization": "none",
    }, ensure_ascii=False, indent=2)
    return CompositeExport({f"{stem}.wav": wav_buffer.getvalue(), f"{stem}.frd": frd.encode(), f"{stem}.csv": csv.encode(), f"{stem}_report.json": report.encode()})


def build_multichannel_export(
    result: MultichannelCompositeResult,
    *, root: str = "composite",
) -> CompositeExport:
    """Export one complete artifact set per independent Composite Group."""
    files: dict[str, bytes] = {}
    group_reports: list[dict[str, object]] = []
    used_stems: set[str] = set()
    for index, (group, channel_result) in enumerate(result.groups.items(), start=1):
        base_stem = _safe_stem(group) or f"group_{index}"
        stem = base_stem
        suffix = 2
        while stem.casefold() in used_stems:
            stem = f"{base_stem}_{suffix}"
            suffix += 1
        used_stems.add(stem.casefold())
        exported = build_export(channel_result, stem=stem)
        files.update({f"{root}/{stem}/{name}": data for name, data in exported.files.items()})
        group_reports.append({
            "group": group,
            "directory": f"{root}/{stem}",
            "channels": list(channel_result.channel_responses),
            "fft_size": channel_result.fft_size,
        })
    files[f"{root}/multichannel_report.json"] = json.dumps({
        "engine": "Multiway FIR Composite Engine",
        "sample_rate_hz": result.sample_rate_hz,
        "composite_groups": group_reports,
        "output_channel_count": len(result.groups),
        "normalization": "none",
    }, ensure_ascii=False, indent=2).encode()
    return CompositeExport(files)


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
