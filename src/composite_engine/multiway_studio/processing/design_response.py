from __future__ import annotations

import io
import numpy as np
from scipy import signal
from scipy.io import wavfile

from composite_engine.adapter.pipeline import ResponseStage, _realize_stage_response
from composite_engine.iir_crossover import (
    IIRCrossoverConfig,
    iir_crossover_sos,
    sos_is_stable,
)
from composite_engine.phase_alignment import AllPassSection, allpass_response
from composite_engine.multiway_studio.processing.speaker_source import resolve_speaker_source
from composite_engine.speaker_timing import (
    SpeakerTimingProjection,
    resolve_speaker_timing_projection,
    speaker_timing_samples_for_row,
)


def ir_length_for_frequency_resolution(
    sample_rate_hz: int, minimum_frequency_hz: float = 2.0,
) -> int:
    """Return an odd IR/FFT length whose bin spacing reaches the frequency."""
    if int(sample_rate_hz) <= 0 or float(minimum_frequency_hz) <= 0.0:
        raise ValueError("sample rate and minimum frequency must be positive")
    length = int(np.ceil(float(sample_rate_hz) / float(minimum_frequency_hz)))
    return length if length % 2 == 1 else length + 1


def response_points_for_frequency_resolution(
    sample_rate_hz: int, minimum_frequency_hz: float = 2.0,
) -> int:
    """Return the RFFT point count matching the requested IR resolution."""
    return (
        ir_length_for_frequency_resolution(sample_rate_hz, minimum_frequency_hz) + 1
    ) // 2


def phaseeq_fir_tap_count(response: object) -> int:
    """Return declared taps, with a WAV fallback for pre-metadata caches."""
    if not isinstance(response, dict):
        return 0
    declared = int(response.get("tap_count", 0) or 0)
    if declared > 0:
        return declared
    filename = str(response.get("filename", ""))
    data = response.get("data")
    if not filename.casefold().endswith(".wav") or not isinstance(data, bytes):
        return 0
    try:
        _, values = wavfile.read(io.BytesIO(data))
    except (OSError, ValueError):
        return 0
    return int(np.asarray(values).shape[0]) if np.asarray(values).ndim else 0


def select_crossover_design_inputs(
    firs: dict[str, np.ndarray],
    channel_rows: list[dict[str, object]],
    group: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """Select enabled generated Ways belonging to the displayed Group."""
    eligible = [
        row for row in channel_rows
        if row.get("source") == "generated_band"
        and bool(row.get("enabled", True))
        and str(row.get("way", "")) in firs
    ]
    selected_group = str(group)
    selected_firs: dict[str, np.ndarray] = {}
    way_settings: dict[str, dict[str, object]] = {}
    for row in eligible:
        row_group = str(row.get("group", ""))
        way = str(row.get("way", ""))
        shared_sub = (
            way == "SUB" and row_group == "Sub"
            and selected_group in {"Left", "Right"}
        )
        if row_group != selected_group and not shared_sub:
            continue
        if way not in firs:
            continue
        selected_firs[way] = firs[way]
        from .alignment_target import output_settings
        _output_iir, output_polarity, output_allpass = output_settings(row)
        phaseeq_fir = row.get("phaseeq_fir_response")
        phaseeq_fir_taps = phaseeq_fir_tap_count(phaseeq_fir)
        way_settings[way] = {
            "group": row_group,
            "channel_id": str(row.get("channel_id", "")),
            "gain_db": row.get("gain_db", 0.0),
            "polarity": output_polarity,
            "delay_samples": row.get("delay_samples", 0.0),
            "alignment_delay_samples": row.get("auto_alignment_delay_samples", 0.0),
            "alignment_allpass": output_allpass,
            "speaker_applied": bool(
                row.get("speaker_response") is not None
                or row.get("speaker_library_response")
                or row.get("phaseeq_speaker_response")
            ),
            "speaker_source": (
                resolve_speaker_source(row).label
                if resolve_speaker_source(row) is not None else "Unity"
            ),
            "phaseeq_fir_applied": bool(phaseeq_fir),
            "phaseeq_fir_taps": phaseeq_fir_taps,
            "phaseeq_iir_count": len(tuple(row.get("phaseeq_iir_sos", ()))),
            "baffle_correction_mode": str(row.get("baffle_correction_mode", "OFF")),
            "baffle_iir_count": len(tuple(row.get("baffle_iir_sos", ()))),
        }
    return selected_firs, way_settings


def build_speaker_phaseeq_responses(
    channel_rows: list[dict[str, object]],
    ways: tuple[str, ...],
    group: str,
    sample_rate_hz: int,
    *,
    points: int,
    phaseeq_only: bool = False,
    speaker_only: bool = False,
    fir_enabled_by_way: dict[str, bool] | None = None,
    speaker_timing_projection: SpeakerTimingProjection | None = None,
) -> dict[str, np.ndarray]:
    """Realize assigned response stages for selected generated Ways."""
    fft_size = 2 * int(points) - 1
    frequency_hz = np.fft.rfftfreq(fft_size, 1.0 / int(sample_rate_hz))
    if speaker_timing_projection is None:
        speaker_timing_projection = resolve_speaker_timing_projection(
            channel_rows, {}, sample_rate_hz=int(sample_rate_hz),
        )
    selected = set(ways)
    responses: dict[str, np.ndarray] = {}
    for row in channel_rows:
        way = str(row.get("way", ""))
        row_group = str(row.get("group", ""))
        shared_sub = (
            way == "SUB" and row_group == "Sub"
            and str(group) in {"Left", "Right"}
        )
        if (
            row.get("source") != "generated_band"
            or not bool(row.get("enabled", True))
            or (row_group != str(group) and not shared_sub)
            or way not in selected
        ):
            continue
        has_phaseeq = bool(
            row.get("iir_response") is not None
            or (
                bool((fir_enabled_by_way or {}).get(way, True))
                and
                isinstance(row.get("phaseeq_fir_response"), dict)
                and row["phaseeq_fir_response"].get("data")
            )
            or tuple(row.get("phaseeq_iir_sos", ()))
        )
        speaker_source = resolve_speaker_source(row)
        if phaseeq_only and not has_phaseeq:
            continue
        if speaker_only and speaker_source is None:
            continue
        response = np.ones(frequency_hz.shape, dtype=np.complex128)
        stages: list[ResponseStage] = []
        upload_stages = (
            () if speaker_only else (("PhaseEQ IIR response", "iir_response"),)
        )
        for stage_name, upload_key in upload_stages:
            upload = row.get(upload_key)
            if upload is not None:
                stages.append(ResponseStage(
                    stage_name,
                    str(getattr(upload, "name", f"{way}_{upload_key}.frd")),
                    bytes(upload.getvalue()),
                ))
        speaker = None if phaseeq_only else speaker_source
        if speaker is not None:
            stages.append(ResponseStage(
                speaker.label, speaker.filename, speaker.data,
            ))
        phaseeq_fir = (
            None
            if speaker_only or not bool((fir_enabled_by_way or {}).get(way, True))
            else row.get("phaseeq_fir_response")
        )
        if isinstance(phaseeq_fir, dict) and phaseeq_fir.get("data"):
            stages.append(ResponseStage(
                "PhaseEQ FIR EQ",
                str(phaseeq_fir.get("filename", "phaseeq_fir.wav")),
                bytes(phaseeq_fir["data"]),
            ))
        for stage in stages:
            response *= _realize_stage_response(stage, int(sample_rate_hz), fft_size)
        if speaker is not None and speaker_timing_projection is not None:
            restored_timing_samples = speaker_timing_samples_for_row(
                speaker_timing_projection, row,
            )
            if restored_timing_samples:
                response *= np.exp(
                    -1j * 2.0 * np.pi * frequency_hz
                    * float(restored_timing_samples) / float(sample_rate_hz)
                )
        iir_rows = tuple(row.get("phaseeq_iir_sos", ()))
        if not speaker_only and not phaseeq_only:
            iir_rows = (*iir_rows, *tuple(row.get("baffle_iir_sos", ())))
        iir_sos = np.asarray(() if speaker_only else iir_rows, dtype=float)
        if iir_sos.size:
            if (
                iir_sos.ndim != 2 or iir_sos.shape[1] != 6
                or not np.isfinite(iir_sos).all() or not sos_is_stable(iir_sos)
            ):
                raise ValueError(f"{way}: PhaseEQ IIR SOS is invalid")
            _, iir_response = signal.sosfreqz(
                iir_sos, worN=frequency_hz, fs=float(sample_rate_hz),
            )
            response *= np.asarray(iir_response, dtype=np.complex128)
        responses[way] = response
    return responses


def build_crossover_design_responses(
    firs: dict[str, np.ndarray],
    iir_configs: dict[str, IIRCrossoverConfig],
    sample_rate_hz: int,
    *,
    points: int = 4096,
    way_settings: dict[str, dict[str, object]] | None = None,
    fir_enabled_by_way: dict[str, bool] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Return realized per-Way crossover responses and their complex sum.

    FIR boundaries are evaluated from the generated coefficient arrays. IIR
    boundaries are evaluated from the realized SOS. Speaker and PhaseEQ EQ are
    included when supplied through ``way_settings``; other Composite stages
    remain excluded. Studio Channel
    gain/polarity/manual delay and applied phase/time alignment are included
    through ``way_settings``.
    """
    fft_size = 2 * int(points) - 1
    frequency_hz = np.fft.rfftfreq(fft_size, 1.0 / float(sample_rate_hz))
    responses: dict[str, np.ndarray] = {}
    for way, coefficients in firs.items():
        values = np.asarray(coefficients, dtype=float)
        angular_frequency = 2.0 * np.pi * frequency_hz / float(sample_rate_hz)
        fir_enabled = bool(
            (fir_enabled_by_way or {}).get(way, True)
        )
        if fir_enabled:
            _, response = signal.freqz(values, worN=frequency_hz, fs=float(sample_rate_hz))
            # Compare FIR and IIR boundaries at the crossover design reference,
            # removing only the FIR's intrinsic linear-phase center delay.
            response = response * np.exp(1j * angular_frequency * ((values.size - 1) / 2.0))
        else:
            response = np.ones(frequency_hz.shape, dtype=np.complex128)
        settings = (way_settings or {}).get(way, {})
        upstream_response = settings.get("speaker_phaseeq_response")
        if upstream_response is not None:
            upstream = np.asarray(upstream_response, dtype=np.complex128)
            if upstream.shape != response.shape:
                raise ValueError(f"{way}: Speaker/PhaseEQ response grid mismatch")
            response = response * upstream
        config = iir_configs.get(way, IIRCrossoverConfig())
        sos = iir_crossover_sos(config, int(sample_rate_hz))
        if sos.size:
            _, iir_response = signal.sosfreqz(
                sos, worN=frequency_hz, fs=float(sample_rate_hz)
            )
            response = response * iir_response
        response *= int(config.lr2_polarity)
        gain_db = float(settings.get("gain_db", 0.0))
        polarity = int(settings.get("polarity", 1))
        delay_samples = float(settings.get("delay_samples", 0.0))
        alignment_delay_samples = float(settings.get("alignment_delay_samples", 0.0))
        sections = tuple(
            item for item in settings.get("alignment_allpass", ())
            if isinstance(item, AllPassSection)
        )
        response = response * (10.0 ** (gain_db / 20.0)) * polarity
        response = response * allpass_response(
            sections, frequency_hz, int(sample_rate_hz),
        )
        response = response * np.exp(
            -1j * angular_frequency * (delay_samples + alignment_delay_samples)
        )
        responses[way] = np.asarray(response, dtype=np.complex128)
    way_sum = (
        np.sum(tuple(responses.values()), axis=0)
        if responses else np.zeros_like(frequency_hz, dtype=np.complex128)
    )
    return frequency_hz, responses, np.asarray(way_sum, dtype=np.complex128)


def sum_crossover_comparison_response(
    responses: dict[str, np.ndarray], *, output_gain_linear: float,
    fir_enabled_by_way: dict[str, bool],
) -> np.ndarray:
    """Sum an adjacent pair at the pre-output-gain reference using one buffer.

    Reuse existing spectra; no FFT or per-Way corrected copies are needed.
    IIR-only Ways never consumed normalized FIR coefficients, so add them
    after undoing the common FIR gain. Channel adjustments remain intact.
    """
    gain = float(output_gain_linear)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("output gain must be finite and positive")
    if not responses:
        return np.array([], dtype=np.complex128)
    total = np.zeros_like(next(iter(responses.values())), dtype=np.complex128)
    for way, response in responses.items():
        if fir_enabled_by_way.get(way, False):
            np.add(total, response, out=total)
    if gain != 1.0:
        total /= gain
    for way, response in responses.items():
        if not fir_enabled_by_way.get(way, False):
            np.add(total, response, out=total)
    return total


def centered_impulses_from_responses(
    responses: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Convert one-sided realized responses to centered display impulses.

    The common FFT shift preserves relative manual/alignment delays and the
    realized IIR/all-pass phase while placing time zero at the chart center.
    """
    centered: dict[str, np.ndarray] = {}
    for way, response in responses.items():
        values = np.asarray(response, dtype=np.complex128)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("response must be a one-dimensional RFFT spectrum")
        impulse = np.fft.irfft(values, n=2 * values.size - 1)
        centered[way] = np.fft.fftshift(impulse)
    return centered
