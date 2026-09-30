from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np

from phase_fir_designer import SpeakerResponse
from phase_fir_designer.speaker import load_impulse_response_wav_with_ir, load_speaker_response_text


@dataclass(frozen=True)
class UploadedSpeakerResponse:
    speaker_response: SpeakerResponse
    source_name: str
    source_type: str = "reconstructed_from_response"
    impulse: np.ndarray | None = None
    impulse_sample_rate: int | None = None


def decode_uploaded_text(uploaded_file: Any) -> str:
    data = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "cp932", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_dirac_live_target_text(text: str) -> SpeakerResponse:
    """Parse the BREAKPOINTS section used by Dirac Live targetcurve files."""
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").splitlines()
    start = next((idx + 1 for idx, line in enumerate(lines) if line.strip().upper() == "BREAKPOINTS"), None)
    if start is None:
        raise ValueError("Dirac Live TargetにBREAKPOINTSがありません。")
    frequency: list[float] = []
    gain_db: list[float] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            break
        try:
            current_frequency = float(parts[0])
            current_gain = float(parts[1])
        except ValueError:
            break
        if np.isfinite(current_frequency) and np.isfinite(current_gain) and current_frequency > 0.0:
            frequency.append(current_frequency)
            gain_db.append(current_gain)
    if len(frequency) < 2:
        raise ValueError("Dirac Live TargetのBREAKPOINTSに2点以上の数値が必要です。")
    order = np.argsort(np.asarray(frequency, dtype=float))
    return SpeakerResponse(
        frequency=np.asarray(frequency, dtype=float)[order].tolist(),
        gain_db=np.asarray(gain_db, dtype=float)[order].tolist(),
        phase_deg=None,
    )


def load_uploaded_target_response(uploaded_file: Any) -> UploadedSpeakerResponse:
    file_name = str(getattr(uploaded_file, "name", ""))
    if file_name.lower().endswith(".wav"):
        loaded = load_uploaded_speaker_response(uploaded_file)
        impulse = loaded.impulse
        sample_rate = loaded.impulse_sample_rate
        assert impulse is not None and sample_rate is not None
        # A target describes a curve, not the WAV's recording/export delay.
        # Zero padding resolves phase before unwrapping even for a centred IR.
        n_fft = 2 * len(impulse)
        frequency = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        spectrum = np.fft.rfft(impulse, n=n_fft)
        peak = int(np.argmax(np.abs(impulse)))
        spectrum *= np.exp(2j * np.pi * frequency * peak / sample_rate)
        return UploadedSpeakerResponse(
            speaker_response=SpeakerResponse(
                frequency=frequency.tolist(),
                gain_db=(20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))).tolist(),
                phase_deg=np.rad2deg(np.unwrap(np.angle(spectrum))).tolist(),
            ),
            source_name=loaded.source_name,
            source_type=loaded.source_type,
            impulse=impulse,
            impulse_sample_rate=sample_rate,
        )
    text = decode_uploaded_text(uploaded_file)
    if file_name.lower().endswith(".targetcurve") or any(
        line.strip().upper() == "BREAKPOINTS" for line in text.splitlines()
    ):
        return UploadedSpeakerResponse(
            speaker_response=load_dirac_live_target_text(text),
            source_name=file_name,
        )
    return load_uploaded_speaker_response(uploaded_file)


def load_uploaded_speaker_response(uploaded_file: Any) -> UploadedSpeakerResponse:
    file_name = str(getattr(uploaded_file, "name", ""))
    if file_name.lower().endswith(".wav"):
        impulse_input = load_impulse_response_wav_with_ir(io.BytesIO(uploaded_file.getvalue()))
        return UploadedSpeakerResponse(
            speaker_response=impulse_input.speaker_response,
            source_name=f"{file_name} (Impulse WAV mono, {float(impulse_input.sample_rate):.0f} Hz)",
            source_type=impulse_input.source_type,
            impulse=impulse_input.impulse,
            impulse_sample_rate=int(impulse_input.sample_rate),
        )
    delimiter = "," if file_name.lower().endswith(".csv") else None
    return UploadedSpeakerResponse(
        speaker_response=load_speaker_response_text(io.StringIO(decode_uploaded_text(uploaded_file)), delimiter=delimiter),
        source_name=file_name,
    )
