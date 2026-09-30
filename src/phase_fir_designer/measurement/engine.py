from __future__ import annotations

from dataclasses import replace
import queue
from pathlib import Path
import threading
import time
import uuid
import warnings
from typing import Any

import numpy as np

from .analysis import analyze_shot, build_standard_result, channel_quality, conservative_denoised_result, detect_ess_start, detect_first_ess_start, generate_ess_reference, reanalyze_gate, wavelet_from_ir
from .input_gain import InputGainSnapshot, input_gain_is_unchanged, read_input_gain_snapshot, set_input_gain_adjustment
from .models import HighPrecisionResult, MeasurementSettings, MeasurementState, SessionSnapshot, ShotResult, ShotStatus, readonly, utc_now
from .noise_preflight import AmbientNoiseProfile, assess_pilot, measure_ambient_noise, pilot_snr_thresholds
from .persistence import autosave_denoised, autosave_manifest, autosave_shot, autosave_standard, load_autosaved_shot
from .sounddevice_compat import import_sounddevice


_STREAM_RESET = object()
_STREAM_STALL_TIMEOUT_S = 3.0
_STREAM_WATCHDOG_INTERVAL_S = 0.5
_STREAM_RECOVERY_DELAYS_S = (0.25, 1.0, 2.0)
_STREAM_RECOVERY_SYNC_GRACE_S = 12.0


def configure_sounddevice_warning_filters() -> None:
    """Hide the sounddevice 0.5.5 array-shape warning on NumPy 2.5."""
    warnings.filterwarnings(
        "ignore",
        message=r"Setting the shape on a NumPy array has been deprecated.*",
        category=DeprecationWarning,
        module=r"sounddevice",
    )


def preferred_umik1_gain_db(
    settings: MeasurementSettings,
    snapshot: InputGainSnapshot,
    device_name: str = "",
) -> float | None:
    """Return the safe startup target for a writable UMIK-1 CoreAudio input."""

    identity = f"{settings.microphone_profile_id} {device_name} {snapshot.device_name}".lower()
    if settings.guided_measurement:
        return None
    if "umik-1" not in identity and "umik 1" not in identity:
        return None
    if not snapshot.available or not snapshot.writable or not snapshot.channel_gain_db:
        return None
    maximum = min(snapshot.channel_max_gain_db) if snapshot.channel_max_gain_db else 24.0
    minimum = max(snapshot.channel_min_gain_db) if snapshot.channel_min_gain_db else 0.0
    return float(np.clip(24.0, minimum, maximum))


def recovery_input_device(
    devices: list[dict[str, Any]],
    previous_name: str,
    minimum_channels: int,
) -> dict[str, Any] | None:
    """Find a re-enumerated input by stable name instead of stale index."""

    def token(value: str) -> str:
        return "".join(character for character in str(value).casefold() if character.isalnum())

    wanted = token(previous_name)
    compatible = [
        device for device in devices
        if int(device.get("max_input_channels", 0)) >= int(minimum_channels)
    ]
    if not compatible:
        return None
    exact = [device for device in compatible if token(device.get("name", "")) == wanted]
    if exact:
        return exact[0]
    partial = [
        device for device in compatible
        if wanted and (wanted in token(device.get("name", "")) or token(device.get("name", "")) in wanted)
    ]
    return partial[0] if len(partial) == 1 else None


def refine_marker_ess_start(
    recording: np.ndarray,
    reference: np.ndarray,
    predicted_start: int,
    *,
    radius_samples: int,
) -> tuple[int | None, float]:
    """Validate a timing-marker prediction against the full ESS waveform."""
    values = np.asarray(recording)
    ess = np.asarray(reference).reshape(-1)
    radius = max(0, int(radius_samples))
    predicted = int(predicted_start)
    lower = max(0, predicted - radius)
    upper = min(values.shape[0], predicted + radius + ess.size)
    if ess.size < 2 or upper - lower < ess.size:
        return None, 0.0
    local_start, correlation = detect_ess_start(values[lower:upper], ess)
    if local_start is None:
        return None, float(correlation)
    return lower + int(local_start), float(correlation)


def ess_band_detection_window(
    reference: np.ndarray,
    settings: MeasurementSettings,
    active_start_hz: float,
    active_end_hz: float,
) -> tuple[np.ndarray, int]:
    """Return the ESS time slice reproduced by the current speaker.

    This mask is used only for synchronization.  It never limits the measured
    response or the frequency range saved in a shot.
    """

    values = np.asarray(reference, dtype=np.float64).reshape(-1)
    sweep_samples = min(values.size, int(round(settings.sweep_duration_s * settings.sample_rate)))
    start_hz = max(float(settings.start_frequency_hz), 1e-6)
    end_hz = min(float(settings.end_frequency_hz), settings.sample_rate / 2.0)
    low = max(start_hz, float(active_start_hz))
    high = min(end_hz, float(active_end_hz))
    if sweep_samples < 256 or high <= low or end_hz <= start_hz:
        return readonly(values), 0
    log_span = np.log(end_hz / start_hz)
    if not np.isfinite(log_span) or log_span <= 0.0:
        return readonly(values), 0
    lower = int(np.floor(sweep_samples * np.log(low / start_hz) / log_span))
    upper = int(np.ceil(sweep_samples * np.log(high / start_hz) / log_span))
    minimum = min(sweep_samples, max(2048, int(round(0.060 * settings.sample_rate))))
    if upper - lower < minimum:
        center = (lower + upper) // 2
        lower = center - minimum // 2
        upper = lower + minimum
    lower = max(0, min(lower, sweep_samples - minimum))
    upper = min(sweep_samples, max(upper, lower + minimum))
    return readonly(values[lower:upper]), int(lower)


def circular_impulse_peak_delta_ms(
    source: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
) -> float:
    """Return repeat-to-repeat direct-IR peak movement on a circular FFT axis."""

    left = np.asarray(source).reshape(-1)
    right = np.asarray(reference).reshape(-1)
    size = min(left.size, right.size)
    if size < 2 or sample_rate <= 0:
        return float("inf")

    def signed_peak(values: np.ndarray) -> int:
        index = int(np.argmax(np.abs(values[:size])))
        return index - size if index > size // 2 else index

    return abs(signed_peak(left) - signed_peak(right)) / int(sample_rate) * 1000.0


def centered_impulse_similarity(
    source: np.ndarray,
    source_center: float,
    reference: np.ndarray,
    reference_center: float,
    sample_rate: int,
) -> float:
    """Compare IR shape around independently selected direct centers."""

    if sample_rate <= 0 or not np.isfinite(source_center) or not np.isfinite(reference_center):
        return 0.0
    radius = max(16, int(round(0.005 * sample_rate)))

    def window(values: np.ndarray, center: float) -> np.ndarray:
        data = np.asarray(values, dtype=np.float64).reshape(-1)
        index = int(round(center)) % data.size
        offsets = np.arange(-radius, radius + 1)
        result = data[(index + offsets) % data.size].copy()
        result -= np.mean(result)
        return result

    left = window(source, source_center)
    right = window(reference, reference_center)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def refresh_audio_devices() -> None:
    """Reinitialize PortAudio so devices connected after launch are visible.

    python-sounddevice initializes PortAudio when the module is first imported.
    Some host APIs keep that device snapshot until PortAudio is reinitialized.
    Call this only while PhaseEQ has no active input stream.
    """
    try:
        sd = import_sounddevice()
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"python-sounddevice is unavailable: {exc}") from exc
    terminate = getattr(sd, "_terminate", None)
    initialize = getattr(sd, "_initialize", None)
    if not callable(terminate) or not callable(initialize):
        raise RuntimeError("This sounddevice version cannot refresh the PortAudio device list")
    terminated = False
    try:
        terminate()
        terminated = True
        initialize()
    except Exception as exc:
        if terminated:
            try:
                initialize()
            except Exception:
                pass
        raise RuntimeError(f"PortAudio device refresh failed: {exc}") from exc


def list_input_devices(min_channels: int = 2) -> list[dict[str, Any]]:
    try:
        sd = import_sounddevice()
    except (ImportError, OSError):
        return []
    try:
        queried = sd.query_devices()
    except Exception:
        return []
    devices: list[dict[str, Any]] = []
    for index, device in enumerate(queried):
        if int(device["max_input_channels"]) < int(min_channels):
            continue
        devices.append({**dict(device), "index": index})
    return devices


def input_settings_error(
    device_id: int | None,
    sample_rate: int,
    channels: int,
) -> str:
    """Return PortAudio's reason when an input stream format is unsupported."""
    if device_id is None:
        return "No input device is selected."
    try:
        sd = import_sounddevice()
    except (ImportError, OSError) as exc:
        return f"python-sounddevice is unavailable: {exc}"
    check = getattr(sd, "check_input_settings", None)
    if not callable(check):
        return ""
    try:
        check(
            device=int(device_id),
            samplerate=int(sample_rate),
            channels=int(channels),
            dtype="float32",
        )
    except Exception as exc:
        return str(exc)
    return ""


def classify_repeat_timing(
    previous_start: int | None,
    current_start: int,
    *,
    expected_period_samples: int,
    group_position: int,
    group_repeats: int,
    variable_loop_gap: bool,
    tolerance_samples: int,
) -> tuple[bool, int, int]:
    """Classify one detected ESS without treating a player loop as missed shots.

    ``group_position`` is the 1-based position reached by the previous
    detection, including inferred missing repeats.  The returned position is
    for the current detection.  A new player-file loop always starts at 1.
    """

    if previous_start is None:
        return False, 0, 1
    if expected_period_samples <= 0:
        return False, 0, max(1, group_position + 1)
    if variable_loop_gap and group_repeats > 0 and group_position >= group_repeats:
        return True, 0, 1

    elapsed = max(0, int(current_start) - int(previous_start))
    period_multiple = max(1, int(round(elapsed / expected_period_samples)))
    missed = max(0, period_multiple - 1)
    next_position = max(1, group_position) + missed + 1
    if variable_loop_gap:
        crosses_group_end = group_repeats > 0 and next_position > group_repeats
        if crosses_group_end:
            return True, 0, 1
    return False, missed, next_position


def protected_shot_capture_samples(
    settings: MeasurementSettings,
    observed_period_samples: int | None = None,
    *,
    guard_s: float = 0.002,
) -> tuple[int, int]:
    """End capture before the next timing marker or ESS can enter the shot.

    Returns ``(capture_samples, predicted_overlap_samples)``.  The overlap is
    measured before applying the guard and is useful for diagnostics.
    """

    sample_rate = int(settings.sample_rate)
    configured_s = settings.shot_capture_duration_s or settings.shot_period_s
    configured = max(1, int(round(configured_s * sample_rate)))
    nominal_period = int(round(settings.shot_period_s * sample_rate))
    observed_period = (
        int(observed_period_samples)
        if observed_period_samples is not None and int(observed_period_samples) > 0
        else nominal_period
    )
    # A missed detection can make the observed interval two or more periods
    # long.  Never let that observation relax the boundary beyond the nominal
    # track period; a genuinely shorter player period must tighten it.
    period = min(nominal_period, observed_period)
    precursor = (
        int(round(settings.timing_marker_to_ess_s * sample_rate))
        if settings.reference_has_timing_marker else 0
    )
    next_content_start = max(1, period - precursor)
    overlap = max(0, configured - next_content_start)
    guard = max(1, int(round(max(0.0, float(guard_s)) * sample_rate)))
    protected = max(1, next_content_start - guard)
    return min(configured, protected), overlap


def _audio_start_error_message(
    exc: Exception,
    *,
    device: dict[str, Any] | None,
    sample_rate: int,
    channels: int,
) -> str:
    detail = str(exc)
    name = str((device or {}).get("name", "selected input"))
    context = f'{name} / {int(channels)} ch / {int(sample_rate) / 1000:g} kHz'
    mac_core_error = (
        "Internal PortAudio error" in detail
        or "PaErrorCode -9986" in detail
        or "-10851" in detail
    )
    if mac_core_error:
        return (
            f"{detail} ({context}). CoreAudio rejected the input stream. "
            "Select a sample rate supported by the device, confirm the same rate in Audio MIDI Setup, "
            "allow microphone access for the app/Terminal in System Settings, then Refresh input devices."
        )
    return f"{detail} ({context})"


class ContinuousMeasurementEngine:
    """Thread-safe continuous recorder with shot-level immutable results.

    The PortAudio callback only copies blocks. Detection, deconvolution and
    persistence run on a worker thread so a slow Wavelet update cannot stop
    capture.
    """

    def __init__(
        self,
        settings: MeasurementSettings,
        autosave_root: Path,
        *,
        reference: np.ndarray | None = None,
        timing_marker: np.ndarray | None = None,
        original_only: bool = False,
    ) -> None:
        self._lock = threading.RLock()
        self._settings = settings
        if (
            settings.timing_reference_kind == "phaseeq_tweeter_reference"
            and not settings.tweeter_reference_setup_ready
        ):
            raise ValueError(
                "Tweeter reference measurement requires verified separate outputs and completed external setup checks"
            )
        if original_only:
            reference = np.empty(0, dtype=np.float64)
        if reference is None:
            if settings.input_mode.startswith("known_wav"):
                raise ValueError("ESS Library reference mode requires an extracted ESS waveform")
            reference = generate_ess_reference(settings)
        reference_array = np.asarray(reference, dtype=np.float64).reshape(-1)
        if not original_only and (reference_array.size < 2 or not np.all(np.isfinite(reference_array))):
            raise ValueError("ESS reference is empty or invalid")
        self._reference = readonly(reference_array)
        marker_array = (
            np.asarray(timing_marker, dtype=np.float64).reshape(-1)
            if timing_marker is not None else np.empty(0, dtype=np.float64)
        )
        if marker_array.size and (marker_array.size < 2 or not np.all(np.isfinite(marker_array))):
            raise ValueError("timing marker is empty or invalid")
        self._timing_marker = readonly(marker_array) if marker_array.size else None
        self._sync_reference = self._reference
        self._sync_reference_offset = 0
        self._timing_marker_to_ess_samples = (
            int(round(settings.timing_marker_to_ess_s * settings.sample_rate))
            if self._timing_marker is not None else 0
        )
        self._session_id = uuid.uuid4().hex
        self._directory = autosave_root / self._session_id
        self._state = MeasurementState.IDLE
        self._shots: list[ShotResult] = []
        self._started_at = ""
        self._stopped_at = ""
        self._error = ""
        self._device_info: dict[str, Any] = {}
        self._blocks: queue.Queue[Any] = queue.Queue(maxsize=256)
        self._level_blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None
        self._level_worker: threading.Thread | None = None
        self._stop_finalizer: threading.Thread | None = None
        self._stream_watchdog: threading.Thread | None = None
        self._stream_recovery_thread: threading.Thread | None = None
        self._stream: Any = None
        self._stream_recovery_in_progress = False
        self._stream_recovery_sync_grace_until = 0.0
        self._stream_started_monotonic = 0.0
        self._last_audio_callback_monotonic = 0.0
        self._stop_requested = threading.Event()
        self._abort_requested = threading.Event()
        self._dropped_blocks = 0
        self._standard_result: HighPrecisionResult | None = None
        self._saved_original = None
        self._denoised_result: HighPrecisionResult | None = None
        self._measurement_revision = 0
        self._sweep_count = 0
        self._rejected_count = 0
        self._rejection_reasons: dict[str, int] = {}
        self._last_rejection_reason = ""
        self._ambient_noise: AmbientNoiseProfile | None = None
        self._preflight_ready = not settings.noise_preflight_enabled
        self._gain_recheck_required = False
        self._pilot_history: list[dict[str, Any]] = []
        self._live_input_level_updated_monotonic = 0.0

    @property
    def settings(self) -> MeasurementSettings:
        return self._settings

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            live_age_s = (
                max(0.0, time.monotonic() - self._live_input_level_updated_monotonic)
                if self._live_input_level_updated_monotonic > 0.0 else None
            )
            return SessionSnapshot(
                session_id=self._session_id,
                state=self._state,
                settings=self._settings,
                revision=self._measurement_revision,
                saved_original=self._saved_original,
                shots=tuple(self._shots),
                started_at=self._started_at,
                stopped_at=self._stopped_at,
                error_message=self._error,
                device_info=dict(self._device_info) | {
                    "dropped_blocks": self._dropped_blocks,
                    "detected_sweeps": self._sweep_count,
                    "accepted_shots": len(self._shots),
                    "rejected_shots": self._rejected_count,
                    "last_rejection_reason": self._last_rejection_reason,
                    "rejection_reasons": dict(self._rejection_reasons),
                    "preflight_ready": self._preflight_ready,
                    "gain_recheck_required": self._gain_recheck_required,
                    "pilot_history": list(self._pilot_history),
                    "live_input_level_age_s": live_age_s,
                },
                autosave_directory=self._directory,
                standard_result=self._standard_result,
                denoised_result=self._denoised_result,
            )

    @classmethod
    def from_snapshot(cls, snapshot: SessionSnapshot, autosave_root: Path | None = None) -> "ContinuousMeasurementEngine":
        root = autosave_root or (snapshot.autosave_directory.parent if snapshot.autosave_directory else Path("tmp/measurement_sessions"))
        reference = None
        timing_marker = None
        reference_path = snapshot.autosave_directory / "ess_reference.wav" if snapshot.autosave_directory else None
        if reference_path is not None and reference_path.exists():
            from scipy.io import wavfile

            _, stored_reference = wavfile.read(reference_path)
            reference = np.asarray(stored_reference, dtype=np.float64)
        marker_path = snapshot.autosave_directory / "timing_marker.wav" if snapshot.autosave_directory else None
        if marker_path is not None and marker_path.exists():
            from scipy.io import wavfile

            _, stored_marker = wavfile.read(marker_path)
            timing_marker = np.asarray(stored_marker, dtype=np.float64)
        engine = cls(snapshot.settings, root, reference=reference, timing_marker=timing_marker,
                     original_only=snapshot.saved_original is not None)
        engine._session_id = snapshot.session_id
        engine._directory = snapshot.autosave_directory or (root / snapshot.session_id)
        engine._state = snapshot.state
        engine._shots = list(snapshot.shots)
        engine._started_at = snapshot.started_at
        engine._stopped_at = snapshot.stopped_at
        engine._error = snapshot.error_message
        engine._device_info = dict(snapshot.device_info)
        engine._sweep_count = int(snapshot.device_info.get("detected_sweeps", len(snapshot.shots)))
        engine._rejected_count = int(snapshot.device_info.get("rejected_shots", 0))
        engine._rejection_reasons = dict(snapshot.device_info.get("rejection_reasons", {}))
        engine._last_rejection_reason = str(snapshot.device_info.get("last_rejection_reason", ""))
        engine._preflight_ready = bool(snapshot.device_info.get("preflight_ready", not snapshot.settings.noise_preflight_enabled))
        engine._gain_recheck_required = bool(snapshot.device_info.get("gain_recheck_required", False))
        engine._pilot_history = list(snapshot.device_info.get("pilot_history", []))
        engine._standard_result = snapshot.standard_result
        engine._saved_original = snapshot.saved_original
        engine._denoised_result = snapshot.denoised_result
        engine._measurement_revision = max(0, int(snapshot.revision))
        return engine

    def start(self) -> None:
        if self._reference.size < 2:
            raise RuntimeError('保存済みIRは録音を再開できません。新しい測定を開始してください。')
        with self._lock:
            if self._state not in {MeasurementState.IDLE, MeasurementState.STOPPED_WITH_RESULTS, MeasurementState.ERROR_RECOVERABLE, MeasurementState.ERROR_FATAL}:
                return
            if self._shots:
                raise RuntimeError("create a new engine before starting a new measurement session")
            self._state = MeasurementState.ARMING
            self._started_at = utc_now()
            self._directory.mkdir(parents=True, exist_ok=True)
            self._save_reference()
        try:
            sd = import_sounddevice()
        except (ImportError, OSError) as exc:
            self._fail(f"python-sounddevice is unavailable: {exc}", fatal=True)
            return
        configure_sounddevice_warning_filters()
        device: dict[str, Any] | None = None
        stream_channels = self._stream_channel_count()
        try:
            device_id = self._settings.input_device
            if device_id is None:
                compatible = list_input_devices(stream_channels)
                if not compatible:
                    raise RuntimeError(
                        f"No input device with channel {stream_channels} is available."
                    )
                device_id = int(compatible[0]["index"])
                self._settings = replace(self._settings, input_device=device_id)
            device = sd.query_devices(device_id, "input")
            if int(device["max_input_channels"]) < stream_channels:
                raise RuntimeError(
                    f"Selected input device provides only {int(device['max_input_channels'])} channel(s); "
                    f"channel {stream_channels} is required."
                )
            settings_error = input_settings_error(
                int(device_id),
                self._settings.sample_rate,
                stream_channels,
            )
            if settings_error:
                raise RuntimeError(
                    f"Selected input format is unavailable: {settings_error}"
                )
            self._device_info = dict(device) | {
                "measurement_channels": self._settings.channels,
                "stream_channels": stream_channels,
                "selected_input_channel": self._settings.input_channel,
                "input_mode": self._settings.input_mode,
                "reference_source_name": self._settings.reference_source_name,
            }
            gain_snapshot = read_input_gain_snapshot(str(device.get("name", "")), self._settings.channels)
            if gain_snapshot.available:
                preferred_gain = preferred_umik1_gain_db(
                    self._settings,
                    gain_snapshot,
                    str(device.get("name", "")),
                )
                if preferred_gain is not None and gain_snapshot.channel_gain_db:
                    current_gain = float(np.mean(gain_snapshot.channel_gain_db))
                    if abs(preferred_gain - current_gain) >= 0.05:
                        initialized = set_input_gain_adjustment(
                            gain_snapshot.device_name or str(device.get("name", "")),
                            self._settings.channels,
                            preferred_gain - current_gain,
                        )
                        if initialized.available and initialized.channel_gain_db:
                            gain_snapshot = initialized
                            self._device_info.update(
                                initial_input_gain_target_db=preferred_gain,
                                initial_input_gain_applied=True,
                            )
                        else:
                            self._device_info.update(
                                initial_input_gain_target_db=preferred_gain,
                                initial_input_gain_applied=False,
                                initial_input_gain_warning=initialized.error or "UMIK-1 Gainを+24 dBへ設定できませんでした。",
                            )
                self._settings = self._settings_with_input_gain_snapshot(gain_snapshot)
                self._device_info.update(
                    input_gain_source="CoreAudio",
                    input_gain_db_channels=list(gain_snapshot.channel_gain_db),
                    input_gain_min_db_channels=list(gain_snapshot.channel_min_gain_db),
                    input_gain_max_db_channels=list(gain_snapshot.channel_max_gain_db),
                    input_gain_device_name=gain_snapshot.device_name,
                    input_gain_writable=gain_snapshot.writable,
                )
            elif self._settings.input_gain_db_channels:
                self._device_info.update(
                    input_gain_source="CoreAudio",
                    input_gain_db_channels=list(self._settings.input_gain_db_channels),
                    input_gain_max_db_channels=list(self._settings.input_gain_max_db_channels),
                    input_gain_device_name=self._settings.input_gain_device_name,
                    input_gain_writable=False,
                    input_gain_error=gain_snapshot.error,
                )
            self._worker = threading.Thread(target=self._run_worker, name=f"ess-analysis-{self._session_id[:8]}", daemon=True)
            self._worker.start()
            self._stream = sd.InputStream(
                samplerate=self._settings.sample_rate,
                channels=stream_channels,
                dtype="float32",
                blocksize=0,
                device=device_id,
                callback=self._audio_callback,
                finished_callback=self._stream_finished,
            )
            self._stream.start()
            self._start_live_level_worker()
            now = time.monotonic()
            with self._lock:
                self._stream_started_monotonic = now
                self._last_audio_callback_monotonic = now
                self._state = (
                    MeasurementState.MEASURING_NOISE
                    if self._settings.noise_preflight_enabled else MeasurementState.SEARCHING
                )
            self._start_stream_watchdog()
        except Exception as exc:
            message = _audio_start_error_message(
                exc,
                device=device,
                sample_rate=self._settings.sample_rate,
                channels=stream_channels,
            )
            self._fail(f"audio input could not start: {message}", fatal=True)
            try:
                self._blocks.put_nowait(None)
            except queue.Full:
                pass
            self._close_stream()

    def feed_audio(self, block: np.ndarray) -> None:
        """Feed recorded audio in tests or an offline capture adapter."""
        if self._state == MeasurementState.IDLE:
            with self._lock:
                self._state = MeasurementState.SEARCHING
                self._started_at = utc_now()
                self._directory.mkdir(parents=True, exist_ok=True)
                self._save_reference()
            self._worker = threading.Thread(target=self._run_worker, name=f"ess-analysis-{self._session_id[:8]}", daemon=True)
            self._worker.start()
            self._start_live_level_worker()
        self._put_block(np.asarray(block, dtype=np.float32))

    def request_stop(self) -> None:
        """Accept a Stop request immediately and finish the session off-thread."""
        with self._lock:
            if self._state in {MeasurementState.IDLE, MeasurementState.STOPPED_WITH_RESULTS}:
                return
            self._state = MeasurementState.STOPPING
        self._stop_requested.set()

        with self._lock:
            if self._stop_finalizer is not None and self._stop_finalizer.is_alive():
                return
            finalizer = threading.Thread(
                target=self._complete_stop,
                name=f"ess-stop-{self._session_id[:8]}",
                daemon=True,
            )
            self._stop_finalizer = finalizer
        finalizer.start()

    def stop(self, timeout: float = 15.0) -> None:
        """Stop capture and wait up to ``timeout`` for session finalization."""
        self.request_stop()
        finalizer = self._stop_finalizer
        if finalizer is not None and finalizer is not threading.current_thread():
            finalizer.join(timeout=timeout)

    def _complete_stop(self) -> None:
        # Preserve every block already accepted by the audio callback.  The
        # queue may be full, so enqueue the sentinel here instead of blocking
        # the Streamlit Stop callback.
        self._blocks.put(None)
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        self._join_live_level_worker()
        self._close_stream()
        with self._lock:
            if self._state == MeasurementState.STOPPING:
                self._state = MeasurementState.STOPPED_WITH_RESULTS
            self._stopped_at = utc_now()
        if self._settings.autosave_enabled:
            autosave_manifest(self._directory, self.snapshot())
        if not self._abort_requested.is_set():
            try:
                self.finalize_standard()
            except Exception as exc:
                if not self._settings.guided_measurement:
                    raise
                self._fail(f"統合結果を作成できませんでした。一時データを保持しています: {exc}", fatal=False)
            finally:
                if self._settings.guided_measurement:
                    with self._lock:
                        self._device_info["guide_finalized"] = True

    def abort(self) -> None:
        self._abort_requested.set()
        self._stop_requested.set()
        try:
            self._blocks.put_nowait(None)
        except queue.Full:
            pass
        self._join_live_level_worker()
        self._close_stream()
        with self._lock:
            self._state = MeasurementState.STOPPED_WITH_RESULTS
            self._stopped_at = utc_now()
        if self._settings.autosave_enabled:
            autosave_manifest(self._directory, self.snapshot())

    def set_shot_included(self, shot_index: int, included: bool) -> None:
        if self._saved_original is not None:
            raise ValueError('保存済み統合IRの採用ショットは変更できません。')
        with self._lock:
            changed = any(
                shot.shot_index == shot_index and shot.included_in_average != bool(included)
                for shot in self._shots
            )
            self._shots = [replace(shot, included_in_average=bool(included)) if shot.shot_index == shot_index else shot for shot in self._shots]
            self._standard_result = None
            self._denoised_result = None
            if changed:
                self._measurement_revision += 1

    def set_include_warnings(self, included: bool) -> None:
        if self._saved_original is not None:
            return
        with self._lock:
            changed = self._settings.include_warnings_in_average != bool(included)
            self._settings = replace(self._settings, include_warnings_in_average=bool(included))
            self._standard_result = None
            self._denoised_result = None
            if changed:
                self._measurement_revision += 1

    def set_microphone_gain_db(self, target_gain_db: float) -> InputGainSnapshot:
        """Set the common CoreAudio input gain reference during preflight.

        The UI presents an absolute dB value. CoreAudio adjustment remains a
        common channel delta so an intentional dual-gain channel difference is
        preserved. Any real change invalidates READY until a separate Pilot
        confirms S/N, clipping, and headroom with the new gain.
        """
        target = float(target_gain_db)
        if not np.isfinite(target):
            raise ValueError("マイクGainには有限のdB値を入力してください。")
        with self._lock:
            if self._ambient_noise is None or self._state not in {MeasurementState.PREFLIGHT, MeasurementState.SEARCHING}:
                raise RuntimeError("マイクGainは暗騒音測定後、ESS探索中に変更してください。")
            device_name = str(self._device_info.get("name", "") or self._settings.input_gain_device_name)
            channels = self._settings.channels

        before = read_input_gain_snapshot(device_name, channels)
        if not before.available or not before.channel_gain_db:
            raise RuntimeError(before.error or "現在のマイクGainを取得できません。")
        if not before.writable:
            raise RuntimeError("この入力機器のマイクGainはPhaseEQから変更できません。")
        current_reference = float(np.mean(before.channel_gain_db))
        with self._lock:
            if self._settings.response_gain_reference_db is None:
                self._settings = replace(
                    self._settings,
                    response_gain_reference_db=current_reference,
                    minidsp_gain_adjustment_db=float(before.gain_adjustment_from_max_db or 0.0),
                )
        requested_delta = target - current_reference
        if abs(requested_delta) < 0.05:
            return before
        with self._lock:
            # Invalidate READY before touching CoreAudio so an ESS arriving at
            # the same time cannot become a formal shot with changing Gain.
            self._preflight_ready = False
            self._gain_recheck_required = True
            self._device_info.update(
                preflight_ready=False,
                gain_recheck_required=True,
                preflight_message="マイクGainを変更中です。ESSを再生せず、そのままお待ちください。",
            )
        changed = set_input_gain_adjustment(device_name, channels, requested_delta)
        if not changed.available or not changed.channel_gain_db:
            with self._lock:
                self._device_info["preflight_message"] = (
                    f"マイクGainを変更できません（{changed.error or 'CoreAudio error'}）。"
                    "Gainを確認してPilot ESSを再測定してください。"
                )
            raise RuntimeError(changed.error or "マイクGainを変更できません。")
        actual_delta = float(np.mean(changed.channel_gain_db) - current_reference)
        if abs(actual_delta) < 0.05:
            with self._lock:
                self._device_info["preflight_message"] = (
                    "指定値はマイクGainの制御範囲外で、現在値から変更されませんでした。"
                    "Gainを確認してPilot ESSを再測定してください。"
                )
            raise RuntimeError("指定値はマイクGainの制御範囲外です。現在値から変更されませんでした。")

        with self._lock:
            if self._ambient_noise is not None:
                self._ambient_noise = self._ambient_noise.shifted_by_gain(actual_delta)
            self._settings = self._settings_with_input_gain_snapshot(changed)
            self._preflight_ready = False
            self._gain_recheck_required = True
            self._device_info.update(
                input_gain_source="CoreAudio",
                input_gain_db_channels=list(changed.channel_gain_db),
                input_gain_min_db_channels=list(changed.channel_min_gain_db),
                input_gain_max_db_channels=list(changed.channel_max_gain_db),
                input_gain_device_name=changed.device_name,
                input_gain_writable=changed.writable,
                preflight_ready=False,
                gain_recheck_required=True,
                preflight_message=(
                    f"マイクGainを手動で{actual_delta:+.1f} dB変更し、"
                    f"{float(np.mean(changed.channel_gain_db)):+.1f} dBに設定しました。"
                    "次のPilot ESSを再測定してGainを確認します。"
                ),
            )
        if self._settings.autosave_enabled:
            autosave_manifest(self._directory, self.snapshot())
        return changed

    def update_gate(self, start_ms: float, end_ms: float, scope: str = "all") -> None:
        self.update_processing(
            gate_start_ms=start_ms,
            gate_end_ms=end_ms,
            merge_crossover_hz=self._settings.merge_crossover_hz,
            scope=scope,
        )

    def update_center_polarity(self, sync_polarity: str, scope: str = "all") -> None:
        if sync_polarity not in {"auto", "positive", "negative"}:
            raise ValueError("unsupported synchronization polarity")
        self.update_processing(
            gate_start_ms=self._settings.gate_start_ms,
            gate_end_ms=self._settings.gate_end_ms,
            merge_crossover_hz=self._settings.merge_crossover_hz,
            scope=scope,
            sync_polarity=sync_polarity,
            force_recenter=True,
        )

    def update_processing(
        self,
        *,
        gate_start_ms: float,
        gate_end_ms: float,
        merge_crossover_hz: float,
        scope: str = "all",
        sync_polarity: str | None = None,
        force_recenter: bool = False,
    ) -> None:
        if sync_polarity is not None and sync_polarity not in {"auto", "positive", "negative"}:
            raise ValueError("unsupported synchronization polarity")
        settings = replace(
            self._settings,
            gate_start_ms=float(gate_start_ms),
            gate_end_ms=float(gate_end_ms),
            merge_crossover_hz=max(0.0, float(merge_crossover_hz)),
            **({"sync_polarity": sync_polarity} if sync_polarity is not None else {}),
        )
        if self._saved_original is not None:
            from .original import result_from_original, shot_from_original
            # A saved aggregate has an established time origin. Re-gating must
            # not turn it into a newly detected shot or replace its arrival.
            recipe = {**self._saved_original.recipe, 'gate_start_ms': settings.gate_start_ms,
                      'gate_end_ms': settings.gate_end_ms, 'merge_crossover_hz': settings.merge_crossover_hz}
            original = replace(self._saved_original, recipe=recipe)
            with self._lock:
                self._saved_original = original
                self._settings = original.settings
                self._shots = [shot_from_original(original)]
                self._standard_result = result_from_original(original)
                self._denoised_result = None
                self._measurement_revision += 1
            return
        with self._lock:
            original_shots = list(self._shots)
            terminal = self._state in {
                MeasurementState.STOPPED_WITH_RESULTS,
                MeasurementState.ERROR_RECOVERABLE,
                MeasurementState.ERROR_FATAL,
            }
        indices = (
            range(len(original_shots))
            if scope == "all"
            else range(max(0, len(original_shots) - 1), len(original_shots))
        )
        durable_replacements: dict[int, ShotResult] = {}
        memory_replacements: dict[int, ShotResult] = {}
        for index in indices:
            original = original_shots[index]
            source = self._hydrate_shot_for_reprocessing(original)
            processing_settings = replace(
                settings,
                input_gain_db=source.settings_snapshot.input_gain_db,
                input_gain_db_channels=source.settings_snapshot.input_gain_db_channels,
                input_gain_max_db_channels=source.settings_snapshot.input_gain_max_db_channels,
                input_analog_gain_db=source.settings_snapshot.input_analog_gain_db,
                minidsp_gain_adjustment_db=source.settings_snapshot.minidsp_gain_adjustment_db,
                response_gain_reference_db=source.settings_snapshot.response_gain_reference_db,
            )
            updated = reanalyze_gate(source, processing_settings, force_recenter=force_recenter)
            durable_replacements[index] = updated
            memory_replacements[index] = replace(
                updated,
                raw_stereo_audio=original.raw_stereo_audio,
                combined_raw_ir=original.combined_raw_ir,
            )
        with self._lock:
            self._settings = settings
            self._shots = [memory_replacements.get(index, shot) for index, shot in enumerate(original_shots)]
            self._standard_result = None
            self._denoised_result = None
            self._measurement_revision += 1
        if self._settings.autosave_enabled:
            snapshot = self.snapshot()
            for shot in durable_replacements.values():
                autosave_shot(self._directory, snapshot, shot)
            autosave_manifest(self._directory, self.snapshot())
        if terminal:
            self.finalize_standard(force=True)

    def _hydrate_shot_for_reprocessing(self, shot: ShotResult) -> ShotResult:
        if shot.raw_stereo_audio is not None:
            return shot
        path = self._directory / f"shot_{shot.shot_index:04d}.npz"
        if not path.exists():
            return shot
        try:
            stored = load_autosaved_shot(path, self._settings)
        except (OSError, ValueError):
            return shot
        return replace(stored, included_in_average=shot.included_in_average)

    def finalize_standard(self, *, clock_mode: str | None = None, force: bool = False) -> HighPrecisionResult | None:
        if self._saved_original is not None:
            return self._standard_result
        with self._lock:
            if self._standard_result is not None and not force:
                return self._standard_result
            shots = list(self._shots)
        hydrated: list[ShotResult] = []
        for shot in shots:
            if shot.raw_stereo_audio is not None:
                hydrated.append(shot)
                continue
            path = self._directory / f"shot_{shot.shot_index:04d}.npz"
            if path.exists():
                try:
                    stored = load_autosaved_shot(path, self._settings)
                except (OSError, ValueError):
                    stored = shot
                hydrated.append(replace(stored, included_in_average=shot.included_in_average))
            else:
                hydrated.append(shot)
        selected_clock_mode = clock_mode or self._settings.clock_correction_mode
        result = build_standard_result(hydrated, self._reference, self._settings, clock_mode=selected_clock_mode)
        with self._lock:
            self._standard_result = result
            self._measurement_revision += 1
        if result is not None and self._settings.autosave_enabled:
            autosave_standard(self._directory, result)
            autosave_manifest(self._directory, self.snapshot())
        return result

    def create_denoised(self, *, max_attenuation_db: float = 3.0) -> HighPrecisionResult | None:
        standard = self.finalize_standard()
        if standard is None:
            return None
        result = conservative_denoised_result(standard, max_attenuation_db=max_attenuation_db)
        with self._lock:
            self._denoised_result = result
            self._measurement_revision += 1
        if self._settings.autosave_enabled:
            autosave_denoised(self._directory, result)
            autosave_manifest(self._directory, self.snapshot())
        return result

    def update_wavelet(self, shot_index: int) -> None:
        updated: ShotResult | None = None
        with self._lock:
            replacements = []
            for shot in self._shots:
                if shot.shot_index == shot_index and shot.combined_raw_ir is not None:
                    wavelet = wavelet_from_ir(
                        shot.combined_raw_ir,
                        shot.settings_snapshot,
                        label=f"Shot {shot.shot_index}",
                        center_sample=shot.center_sample,
                    )
                    shot = replace(shot, wavelet_map=wavelet)
                    updated = shot
                replacements.append(shot)
            self._shots = replacements
            if updated is not None:
                self._measurement_revision += 1
        if updated is not None and self._settings.autosave_enabled:
            snapshot = self.snapshot()
            autosave_shot(self._directory, snapshot, updated)
            autosave_manifest(self._directory, snapshot)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        with self._lock:
            self._last_audio_callback_monotonic = time.monotonic()
        if status:
            with self._lock:
                self._dropped_blocks += 1
                self._device_info["last_audio_status"] = str(status)
                self._device_info["audio_status_events"] = int(
                    self._device_info.get("audio_status_events", 0)
                ) + 1
            # Callback flags are transient input under/overflow notifications.
            # Keep recording the delivered block. Actual stream termination is
            # handled separately by _stream_finished().
            if frames <= 0 or np.asarray(indata).size == 0:
                return
        block = np.array(indata, copy=True)
        if self._settings.channels == 1:
            channel_index = self._settings.input_channel - 1
            block = block[:, channel_index : channel_index + 1]
        else:
            block = block[:, : self._settings.channels]
        self._put_block(block)

    def guide_begin_ess(self) -> None:
        """Release the noise-only stage after the external player has stopped."""
        with self._lock:
            if not self._settings.guided_measurement or self._ambient_noise is None:
                raise ValueError("暗騒音の取得が完了していません")
            if self._stop_requested.is_set():
                raise ValueError("取得を中断しています。音量確認から再開してください")
            self._device_info["guide_stage"] = "pilot"

    def guide_recheck(self, reason: str) -> None:
        with self._lock:
            self._device_info["guide_stage"] = "recheck"
            self._device_info["guide_reason"] = reason
        self.request_stop()

    def _stream_channel_count(self) -> int:
        if self._settings.channels == 1:
            return int(self._settings.input_channel)
        return int(self._settings.channels)

    def _put_block(self, block: np.ndarray) -> None:
        if self._stop_requested.is_set():
            return
        # Keep only the newest meter block. Audio capture must never wait for
        # UI metering or for the ESS analysis worker.
        try:
            self._level_blocks.put_nowait(block)
        except queue.Full:
            try:
                self._level_blocks.get_nowait()
            except queue.Empty:
                pass
            try:
                self._level_blocks.put_nowait(block)
            except queue.Full:
                pass
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            self._dropped_blocks += 1

    def _update_live_input_level(self, block: np.ndarray) -> None:
        """Publish cheap per-block input levels for the Streamlit meter."""

        values = np.asarray(block, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.size == 0:
            return
        finite = np.where(np.isfinite(values), values, 0.0)
        peak = np.max(np.abs(finite), axis=0)
        rms = np.sqrt(np.mean(finite * finite, axis=0))
        peak_dbfs = [float(20.0 * np.log10(max(value, 1e-15))) for value in peak]
        rms_dbfs = [float(20.0 * np.log10(max(value, 1e-15))) for value in rms]
        with self._lock:
            self._live_input_level_updated_monotonic = time.monotonic()
            self._device_info.update(
                live_input_peak_dbfs=peak_dbfs,
                live_input_rms_dbfs=rms_dbfs,
                live_input_headroom_db=[max(0.0, -value) for value in peak_dbfs],
            )

    def _start_live_level_worker(self) -> None:
        with self._lock:
            if self._level_worker is not None and self._level_worker.is_alive():
                return
            worker = threading.Thread(
                target=self._run_live_level_worker,
                name=f"ess-level-{self._session_id[:8]}",
                daemon=True,
            )
            self._level_worker = worker
        worker.start()

    def _run_live_level_worker(self) -> None:
        while not self._stop_requested.is_set() and not self._abort_requested.is_set():
            try:
                block = self._level_blocks.get(timeout=0.1)
            except queue.Empty:
                continue
            self._update_live_input_level(block)

    def _join_live_level_worker(self) -> None:
        worker = self._level_worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=0.5)

    def _stream_finished(self) -> None:
        if not self._stop_requested.is_set() and not self._stream_recovery_in_progress:
            self._schedule_stream_recovery("CoreAudio入力ストリームが予期せず終了しました")

    def _start_stream_watchdog(self) -> None:
        with self._lock:
            if self._stream_watchdog is not None and self._stream_watchdog.is_alive():
                return
            watchdog = threading.Thread(
                target=self._watch_stream_liveness,
                name=f"ess-audio-watchdog-{self._session_id[:8]}",
                daemon=True,
            )
            self._stream_watchdog = watchdog
        watchdog.start()

    def _watch_stream_liveness(self) -> None:
        while not self._stop_requested.wait(_STREAM_WATCHDOG_INTERVAL_S):
            with self._lock:
                active = self._state in {
                    MeasurementState.MEASURING_NOISE,
                    MeasurementState.PREFLIGHT,
                    MeasurementState.SEARCHING,
                }
                last_callback = max(
                    self._stream_started_monotonic,
                    self._last_audio_callback_monotonic,
                )
                recovering = self._stream_recovery_in_progress
            if (
                active
                and not recovering
                and last_callback > 0.0
                and time.monotonic() - last_callback >= _STREAM_STALL_TIMEOUT_S
            ):
                self._schedule_stream_recovery(
                    f"音声callbackが{_STREAM_STALL_TIMEOUT_S:.0f}秒以上停止しました"
                )

    def _schedule_stream_recovery(self, reason: str) -> None:
        if self._settings.guided_measurement:
            if not self._stop_requested.is_set():
                self.guide_recheck("マイク入力が切断されました。再生を停止し、接続・配置から再確認してください。")
            return
        with self._lock:
            if self._stop_requested.is_set() or self._stream_recovery_in_progress:
                return
            self._stream_recovery_in_progress = True
            self._device_info.update(
                audio_recovery_state="reconnecting",
                audio_recovery_reason=str(reason),
                last_recovery_status="CoreAudio入力を自動再接続しています。完了Shotは保持します。",
            )
            recovery = threading.Thread(
                target=self._recover_input_stream,
                args=(str(reason),),
                name=f"ess-audio-recovery-{self._session_id[:8]}",
                daemon=True,
            )
            self._stream_recovery_thread = recovery
        recovery.start()

    def _recover_input_stream(self, reason: str) -> None:
        errors: list[str] = []
        with self._lock:
            old_stream, self._stream = self._stream, None
            previous_name = str(self._device_info.get("name", ""))
            stream_channels = self._stream_channel_count()
        if old_stream is not None:
            try:
                old_stream.stop()
            except Exception:
                pass
            try:
                old_stream.close()
            except Exception:
                pass

        for attempt, delay_s in enumerate(_STREAM_RECOVERY_DELAYS_S, start=1):
            if self._stop_requested.wait(delay_s):
                with self._lock:
                    self._stream_recovery_in_progress = False
                return
            candidate_stream = None
            try:
                sd = import_sounddevice()

                refresh_audio_devices()
                selected = recovery_input_device(
                    list_input_devices(stream_channels),
                    previous_name,
                    stream_channels,
                )
                if selected is None:
                    raise RuntimeError(f"入力機器 {previous_name or '(unknown)'} を再検出できません")
                device_id = int(selected["index"])
                settings_error = input_settings_error(
                    device_id,
                    self._settings.sample_rate,
                    stream_channels,
                )
                if settings_error:
                    raise RuntimeError(settings_error)
                candidate_stream = sd.InputStream(
                    samplerate=self._settings.sample_rate,
                    channels=stream_channels,
                    dtype="float32",
                    blocksize=0,
                    device=device_id,
                    callback=self._audio_callback,
                    finished_callback=self._stream_finished,
                )
                # Put the boundary ahead of every callback from the new
                # stream. The worker may finish queued old blocks first, then
                # drops only the incomplete sweep that crosses this marker.
                self._blocks.put(_STREAM_RESET, timeout=2.0)
                candidate_stream.start()
                if self._stop_requested.is_set():
                    candidate_stream.stop()
                    candidate_stream.close()
                    with self._lock:
                        self._stream_recovery_in_progress = False
                    return
                now = time.monotonic()
                with self._lock:
                    self._stream = candidate_stream
                    self._settings = replace(self._settings, input_device=device_id)
                    self._stream_started_monotonic = now
                    self._last_audio_callback_monotonic = now
                    self._stream_recovery_sync_grace_until = now + _STREAM_RECOVERY_SYNC_GRACE_S
                    self._stream_recovery_in_progress = False
                    self._device_info.update(
                        dict(selected),
                        audio_recovery_state="recovered",
                        audio_recovery_reason=str(reason),
                        audio_recovery_attempt=attempt,
                        audio_recovery_count=int(self._device_info.get("audio_recovery_count", 0)) + 1,
                        last_recovery_status=(
                            f"CoreAudio入力を自動再接続しました（{attempt}回目）。"
                            "同期を取り直し、同じセッションで測定を継続します。"
                        ),
                    )
                if self._settings.autosave_enabled:
                    autosave_manifest(self._directory, self.snapshot())
                return
            except Exception as exc:
                errors.append(str(exc))
                if candidate_stream is not None:
                    try:
                        candidate_stream.close()
                    except Exception:
                        pass

        with self._lock:
            self._stream_recovery_in_progress = False
            self._device_info.update(
                audio_recovery_state="failed",
                audio_recovery_attempts=len(_STREAM_RECOVERY_DELAYS_S),
                last_recovery_status="CoreAudio入力の自動再接続に失敗しました。",
            )
        detail = errors[-1] if errors else "unknown error"
        self._fail(
            f"CoreAudio input recovery failed after {len(_STREAM_RECOVERY_DELAYS_S)} attempts: {detail}",
            fatal=False,
        )
        try:
            self._blocks.put_nowait(None)
        except queue.Full:
            pass

    def _run_worker(self) -> None:
        buffer = np.empty((0, self._settings.channels), dtype=np.float32)
        buffer_origin = 0
        previous_start: int | None = None
        timing_segment = 0
        timing_index = -1
        group_position = 0
        timing_reference_valid = False
        strict_tweeter_reference = (
            self._settings.timing_reference_kind == "phaseeq_tweeter_reference"
        )
        capture_duration_s = self._settings.shot_capture_duration_s or self._settings.shot_period_s
        shot_samples = int(round(capture_duration_s * self._settings.sample_rate))
        expected_period_samples = int(round(self._settings.shot_period_s * self._settings.sample_rate))
        use_timing_marker = self._settings.reference_mode == "timing_marker" and self._timing_marker is not None
        detection_reference = self._timing_marker if use_timing_marker else self._reference
        detection_offset = self._timing_marker_to_ess_samples if use_timing_marker else 0
        search_tail = int(round(self._settings.search_margin_after_s * self._settings.sample_rate))
        min_search = max(
            len(detection_reference) + detection_offset + search_tail,
            len(self._reference) + search_tail,
        )
        raw_writer: Any = None
        ambient_chunks: list[np.ndarray] = []
        ambient_remaining = int(round(self._settings.ambient_noise_duration_s * self._settings.sample_rate))
        try:
            if self._settings.raw_audio_storage == "session_wav":
                try:
                    import soundfile as sf

                    raw_writer = sf.SoundFile(
                        self._directory / self._raw_recording_name(),
                        mode="w",
                        samplerate=self._settings.sample_rate,
                        channels=self._settings.channels,
                        subtype="FLOAT",
                    )
                except (ImportError, OSError, RuntimeError) as exc:
                    self._error = f"raw session WAV disabled: {exc}"
            while not self._abort_requested.is_set():
                block = self._blocks.get()
                if block is None:
                    if self._stop_requested.is_set():
                        break
                    continue
                if block is _STREAM_RESET:
                    buffer_origin += buffer.shape[0]
                    buffer = np.empty((0, self._settings.channels), dtype=np.float32)
                    previous_start = None
                    timing_segment += 1
                    timing_index = -1
                    group_position = 0
                    timing_reference_valid = False
                    if self._settings.noise_preflight_enabled and self._ambient_noise is None:
                        ambient_chunks = []
                        ambient_remaining = int(round(
                            self._settings.ambient_noise_duration_s * self._settings.sample_rate
                        ))
                    with self._lock:
                        self._device_info["timing_resynchronizations"] = int(
                            self._device_info.get("timing_resynchronizations", 0)
                        ) + 1
                        self._device_info["last_synchronization_status"] = (
                            "CoreAudio再接続後のESS同期を探索中"
                        )
                        if self._settings.noise_preflight_enabled and self._ambient_noise is None:
                            self._state = MeasurementState.MEASURING_NOISE
                            self._device_info["preflight_message"] = (
                                "CoreAudio再接続後の暗騒音を最初から再測定しています。"
                            )
                    continue
                if self._settings.noise_preflight_enabled and self._ambient_noise is None:
                    take = min(ambient_remaining, block.shape[0])
                    if take:
                        ambient_chunks.append(np.asarray(block[:take], dtype=np.float32))
                        ambient_remaining -= take
                    block = block[take:]
                    if ambient_remaining > 0:
                        continue
                    ambient = np.concatenate(ambient_chunks, axis=0)
                    self._ambient_noise = measure_ambient_noise(ambient, self._settings)
                    with self._lock:
                        self._device_info.update(
                            ambient_noise_duration_s=self._ambient_noise.duration_s,
                            ambient_noise_p90_dbfs=[
                                20.0 * np.log10(max(value, 1e-15))
                                for value in self._ambient_noise.channel_p90_rms
                            ],
                            preflight_message="暗騒音を取得しました。Pilot ESSを再生してください。",
                        )
                        self._state = MeasurementState.PREFLIGHT
                        if self._settings.guided_measurement:
                            self._device_info["guide_stage"] = "noise"
                    if not block.size:
                        continue
                if self._settings.guided_measurement and self._device_info.get("guide_stage") not in {"pilot", "measuring"}:
                    buffer_origin += len(block)
                    continue
                if raw_writer is not None and self._preflight_ready:
                    raw_writer.write(block)
                buffer = np.concatenate((buffer, block), axis=0)
                while buffer.shape[0] >= min_search:
                    if self._settings.guided_measurement and self._stop_requested.is_set():
                        return
                    mono_energy = float(np.max(np.abs(buffer)))
                    if mono_energy < 1e-5:
                        discard = max(1, buffer.shape[0] - min_search + 1)
                        buffer, buffer_origin = buffer[discard:], buffer_origin + discard
                        break
                    start: int | None = None
                    correlation = 0.0
                    marker_detected = False
                    expected_reference_locked = False
                    sync_reference = self._sync_reference if self._preflight_ready else self._reference
                    sync_offset = self._sync_reference_offset if self._preflight_ready else 0
                    formal_correlation_threshold = (
                        self._adaptive_sync_correlation_threshold()
                        if self._preflight_ready else self._settings.correlation_threshold
                    )
                    recovery_sync_grace = (
                        self._preflight_ready
                        and bool(self._shots)
                        and self._stream_recovery_sync_grace_until > time.monotonic()
                    )
                    recovery_correlation_threshold = (
                        max(0.08, formal_correlation_threshold * 0.45)
                        if recovery_sync_grace else formal_correlation_threshold
                    )
                    expected_correlation_threshold = self._expected_sync_correlation_threshold(
                        formal_correlation_threshold
                    )
                    # After the first verified sweep, the full ESS is the most
                    # reliable clock.  Search it at the expected period before
                    # consulting the much shorter acoustic marker, which can
                    # resemble a fragment of a wide-band sweep in a room.
                    if previous_start is not None and not strict_tweeter_reference:
                        expected_slice = previous_start + expected_period_samples + sync_offset
                        expected_in_buffer = expected_slice - buffer_origin
                        before = int(round(self._settings.search_margin_before_s * self._settings.sample_rate))
                        after = int(round(self._settings.search_margin_after_s * self._settings.sample_rate))
                        tight_margin = int(round(0.020 * self._settings.sample_rate))
                        tight_start = max(0, expected_in_buffer - tight_margin)
                        tight_required_end = expected_in_buffer + tight_margin + len(sync_reference)
                        # Timing is the strongest prior after the first valid
                        # shot. Search a narrow window with a relaxed acoustic
                        # threshold before attempting broad re-synchronization.
                        if tight_required_end > buffer.shape[0]:
                            break
                        tight_end = min(buffer.shape[0], tight_required_end)
                        if tight_end - tight_start >= len(sync_reference):
                            local_start, ess_correlation = detect_ess_start(
                                buffer[tight_start:tight_end],
                                sync_reference,
                            )
                            if (
                                local_start is not None
                                and ess_correlation >= expected_correlation_threshold
                            ):
                                start = tight_start + local_start - sync_offset
                                correlation = ess_correlation
                                expected_reference_locked = True

                        if not expected_reference_locked:
                            search_start = max(0, expected_in_buffer - before)
                            required_search_end = expected_in_buffer + after + len(sync_reference)
                            # Do not search a truncated recovery window. A
                            # partial window can trigger an early false peak.
                            if required_search_end > buffer.shape[0]:
                                break
                            search_end = min(buffer.shape[0], required_search_end)
                            if search_end - search_start >= len(sync_reference):
                                local_start, ess_correlation = detect_ess_start(
                                    buffer[search_start:search_end],
                                    sync_reference,
                                )
                                if (
                                    local_start is not None
                                    and ess_correlation >= recovery_correlation_threshold
                                ):
                                    start = search_start + local_start - sync_offset
                                    correlation = ess_correlation
                                    expected_reference_locked = True

                    if previous_start is not None and not expected_reference_locked and (
                        start is None or correlation < recovery_correlation_threshold
                    ):
                        expected = previous_start + int(round(self._settings.shot_period_s * self._settings.sample_rate))
                        expected_in_buffer = expected - buffer_origin - detection_offset
                        before = int(round(self._settings.search_margin_before_s * self._settings.sample_rate))
                        after = int(round(self._settings.search_margin_after_s * self._settings.sample_rate))
                        search_start = max(0, expected_in_buffer - before)
                        search_end = min(buffer.shape[0], expected_in_buffer + after + len(detection_reference))
                        if search_end - search_start >= len(detection_reference):
                            local_start, correlation = detect_ess_start(buffer[search_start:search_end], detection_reference)
                            if local_start is not None:
                                start = search_start + local_start + detection_offset
                                marker_detected = use_timing_marker
                    # Staged expansion recovers after a missed or drifting
                    # external-player shot without ending the session.
                    if (
                        (start is None or correlation < recovery_correlation_threshold)
                        and (previous_start is None or not use_timing_marker)
                    ):
                        expanded_reference = (
                            sync_reference if not use_timing_marker else detection_reference
                        )
                        expanded_offset = sync_offset if not use_timing_marker else detection_offset
                        if not use_timing_marker and self._preflight_ready:
                            detected_start, correlation = detect_ess_start(buffer, expanded_reference)
                            if correlation < recovery_correlation_threshold:
                                detected_start = None
                        else:
                            detected_start, correlation = detect_first_ess_start(
                                buffer,
                                expanded_reference,
                                formal_correlation_threshold,
                            )
                        if detected_start is not None:
                            marker_detected = use_timing_marker
                            start = (
                                detected_start + expanded_offset
                                if marker_detected else detected_start - expanded_offset
                            )
                    if marker_detected and start is not None:
                        marker_defined_start = int(start)
                        validation_reference = sync_reference if self._preflight_ready else self._reference
                        validation_offset = sync_offset if self._preflight_ready else 0
                        predicted_validation_start = start + validation_offset
                        if buffer.shape[0] < predicted_validation_start + len(validation_reference):
                            break
                        refined_slice_start, ess_correlation = refine_marker_ess_start(
                            buffer,
                            validation_reference,
                            predicted_validation_start,
                            radius_samples=int(round(0.050 * self._settings.sample_rate)),
                        )
                        if refined_slice_start is not None and ess_correlation >= formal_correlation_threshold:
                            if strict_tweeter_reference:
                                # Keep the fixed-Tweeter marker as the common
                                # time plane. The target ESS correlation only
                                # proves that the expected measurement arrived.
                                start = marker_defined_start
                            else:
                                start = refined_slice_start - validation_offset
                            correlation = ess_correlation
                        else:
                            start = None
                            correlation = ess_correlation
                            marker_detected = False
                            with self._lock:
                                self._device_info["timing_marker_validation_failures"] = int(
                                    self._device_info.get("timing_marker_validation_failures", 0)
                                ) + 1
                    # A damaged or inaudible marker must not break an otherwise
                    # valid measurement. Fall back to legacy ESS correlation.
                    if (
                        use_timing_marker
                        and not strict_tweeter_reference
                        and not expected_reference_locked
                        and (start is None or correlation < recovery_correlation_threshold)
                    ):
                        slice_start, correlation = detect_ess_start(buffer, sync_reference)
                        if correlation < recovery_correlation_threshold:
                            slice_start = None
                        candidate_start = None if slice_start is None else slice_start - sync_offset
                        start = candidate_start if candidate_start is not None and candidate_start >= 0 else None
                        marker_detected = False
                    required_correlation_threshold = (
                        (
                            min(expected_correlation_threshold, recovery_correlation_threshold)
                            if recovery_sync_grace else expected_correlation_threshold
                        )
                        if expected_reference_locked
                        else recovery_correlation_threshold
                    )
                    if start is None or correlation < required_correlation_threshold:
                        # A sub-threshold search peak is not a detected shot.
                        # Keep it as a diagnostic instead of inflating the
                        # rejected-shot count on every buffer scan.
                        if self._preflight_ready and correlation > 0.0:
                            with self._lock:
                                self._device_info["last_weak_sync_candidate"] = {
                                    "correlation": float(correlation),
                                    "required": float(recovery_correlation_threshold),
                                    "nominal_required": float(formal_correlation_threshold),
                                    "recovery_sync_grace": bool(recovery_sync_grace),
                                }
                        discard = max(1, buffer.shape[0] - min_search + 1)
                        buffer, buffer_origin = buffer[discard:], buffer_origin + discard
                        break
                    absolute_start = buffer_origin + start
                    observed_period_samples = (
                        None if previous_start is None else absolute_start - previous_start
                    )
                    capture_samples, predicted_overlap_samples = protected_shot_capture_samples(
                        self._settings,
                        observed_period_samples,
                    )
                    reference_plane_offset_samples = (
                        min(
                            start,
                            int(round(
                                self._settings.timing_reference_capture_preroll_s
                                * self._settings.sample_rate
                            )),
                        )
                        if strict_tweeter_reference else 0
                    )
                    capture_start = start - reference_plane_offset_samples
                    capture_samples += reference_plane_offset_samples
                    if buffer.shape[0] < capture_start + capture_samples:
                        break
                    if predicted_overlap_samples > 0:
                        with self._lock:
                            self._device_info["next_shot_protection_events"] = int(
                                self._device_info.get("next_shot_protection_events", 0)
                            ) + 1
                            self._device_info["last_next_shot_protection"] = {
                                "predicted_overlap_ms": predicted_overlap_samples
                                / self._settings.sample_rate * 1000.0,
                                "protected_capture_ms": capture_samples
                                / self._settings.sample_rate * 1000.0,
                                "observed_period_ms": (
                                    None if observed_period_samples is None else
                                    observed_period_samples / self._settings.sample_rate * 1000.0
                                ),
                                "protected_content": (
                                    "timing marker" if self._settings.reference_has_timing_marker else "next ESS"
                                ),
                            }
                    if marker_detected:
                        timing_reference_valid = True
                        with self._lock:
                            self._device_info["timing_marker_detections"] = int(
                                self._device_info.get("timing_marker_detections", 0)
                            ) + 1
                    raw = buffer[capture_start : capture_start + capture_samples]
                    period = None if previous_start is None else (absolute_start - previous_start) / self._settings.sample_rate
                    is_pilot = not self._preflight_ready and self._ambient_noise is not None
                    loop_boundary = False
                    missed = 0
                    if not is_pilot:
                        loop_boundary, missed, group_position = classify_repeat_timing(
                            previous_start,
                            absolute_start,
                            expected_period_samples=expected_period_samples,
                            group_position=group_position,
                            group_repeats=self._settings.reference_repeat_count,
                            variable_loop_gap=self._settings.reference_loop_gap_variable,
                            tolerance_samples=int(round(0.020 * self._settings.sample_rate)),
                        )
                        if previous_start is None:
                            timing_index = 0
                        elif loop_boundary:
                            timing_segment += 1
                            timing_index = 0
                            with self._lock:
                                self._device_info["player_loop_boundaries"] = int(
                                    self._device_info.get("player_loop_boundaries", 0)
                                ) + 1
                        else:
                            timing_index += missed + 1
                    expected = (
                        absolute_start
                        if previous_start is None or loop_boundary
                        else previous_start + (missed + 1) * expected_period_samples
                    )
                    timing_error = (absolute_start - expected) / self._settings.sample_rate * 1000
                    verified_boundary = marker_detected or correlation >= max(
                        0.8,
                        self._settings.correlation_threshold,
                    )
                    marker_resynchronized = (
                        not is_pilot
                        and previous_start is not None
                        and not loop_boundary
                        and verified_boundary
                        and self._settings.reference_loop_gap_variable
                        and abs(timing_error) > 20.0
                    )
                    if marker_resynchronized:
                        loop_boundary = True
                        missed = 0
                        group_position = 1
                        timing_segment += 1
                        timing_index = 0
                        timing_error = 0.0
                        with self._lock:
                            self._device_info["player_loop_boundaries"] = int(
                                self._device_info.get("player_loop_boundaries", 0)
                            ) + 1
                            self._device_info["timing_resynchronizations"] = int(
                                self._device_info.get("timing_resynchronizations", 0)
                            ) + 1
                            self._device_info["last_recovery_status"] = "Recovered at player file boundary"
                    if missed and not is_pilot and self._settings.reference_mode == "timing_marker":
                        for offset in range(missed):
                            with self._lock:
                                self._sweep_count += 1
                            self._record_rejection("ESS not detected at the expected repeat")
                    with self._lock:
                        self._state = MeasurementState.ANALYZING_SHOT
                        shot_index = len(self._shots) + 1
                        if is_pilot:
                            sweep_index = 0
                        else:
                            self._sweep_count += 1
                            sweep_index = self._sweep_count
                    if is_pilot:
                        if self._settings.guided_measurement:
                            pilot_rms = float(np.sqrt(np.mean(np.asarray(raw[:len(self._reference)], dtype=float)**2)))
                            self._device_info["guide_pilot_rms"] = pilot_rms
                        pilot_quality = channel_quality(raw, self._settings)
                        assessment = assess_pilot(raw, pilot_quality, self._ambient_noise, self._settings)
                        self._handle_pilot(
                            assessment,
                            str(self._device_info.get("name", "")),
                            correlation=correlation,
                            marker_detected=marker_detected,
                        )
                        with self._lock:
                            self._state = MeasurementState.PREFLIGHT if not self._stop_requested.is_set() else MeasurementState.STOPPING
                        # Playback may be paused while the user changes volume.
                        # Start formal sweep timing afresh after every Pilot so
                        # that adjustment gaps are not recorded as missed shots.
                        previous_start = None
                        timing_index = -1
                        group_position = 0
                        consumed = capture_start + capture_samples
                        buffer, buffer_origin = buffer[consumed:], buffer_origin + consumed
                        continue
                    if self._settings.guided_measurement and period is not None:
                        other_cycle = any(abs(period-value) < .1 for value in (5., 7., 17., 23.) if abs(value-self._settings.shot_period_s) > .1)
                        if other_cycle:
                            self.guide_recheck("選択した音源と反復周期が異なります。再生を停止して曲番号を確認してください。")
                            return
                    period_rejection = (
                        f"ESS period error {timing_error:+.1f} ms exceeds 20 ms; shot rejected"
                        if (
                            ((self._settings.reference_mode == "timing_marker" and marker_detected) or self._settings.guided_measurement)
                            and previous_start is not None
                            and not loop_boundary
                            and abs(timing_error) > 20.0
                        )
                        else ""
                    )
                    pre_rejection = period_rejection or self._pre_analysis_rejection(raw, correlation)
                    if pre_rejection:
                        self._record_rejection(pre_rejection)
                        with self._lock:
                            self._state = MeasurementState.SEARCHING if not self._stop_requested.is_set() else MeasurementState.STOPPING
                        if period_rejection:
                            # A manually restarted/looped player may return to
                            # the file head before the declared repeat group is
                            # exhausted. Drop this ambiguous hit, then let the
                            # next verified ESS establish a fresh segment.
                            previous_start = None
                            timing_reference_valid = False
                            timing_index = -1
                            group_position = 0
                            timing_segment += 1
                            with self._lock:
                                self._device_info["timing_resynchronizations"] = int(
                                    self._device_info.get("timing_resynchronizations", 0)
                                ) + 1
                                self._device_info["last_recovery_status"] = (
                                    "Period error discarded; waiting for a fresh ESS anchor"
                                )
                        else:
                            previous_start = absolute_start
                        consumed = capture_start + capture_samples
                        buffer, buffer_origin = buffer[consumed:], buffer_origin + consumed
                        continue
                    mode = self._settings.wavelet_update_mode
                    do_wavelet = mode == "every_shot" or (mode == "every_n_shots" and shot_index % max(1, self._settings.wavelet_every_n) == 0)
                    if self._blocks.qsize() > 4:
                        do_wavelet = False
                    shot = analyze_shot(
                        raw,
                        self._reference,
                        self._settings,
                        shot_index=shot_index,
                        detected_start_sample=absolute_start,
                        correlation=correlation,
                        timing_error_ms=timing_error,
                        detected_period_s=period,
                        calculate_wavelet=False,
                        preferred_polarity=self._preferred_center_polarity(),
                        preferred_center_sample=self._preferred_center_sample(),
                        marker_detected=marker_detected,
                        marker_verified=bool(
                            marker_detected
                            if strict_tweeter_reference else
                            marker_detected or (expected_reference_locked and timing_reference_valid)
                        ),
                        reference_plane_offset_samples=reference_plane_offset_samples,
                    )
                    shot = replace(
                        shot,
                        sweep_index=sweep_index,
                        timing_segment=timing_segment,
                        timing_index=timing_index,
                        loop_boundary_before=loop_boundary,
                    )
                    if predicted_overlap_samples > 0:
                        shot = replace(
                            shot,
                            quality=replace(
                                shot.quality,
                                messages=shot.quality.messages + (
                                    "Capture shortened by "
                                    f"{predicted_overlap_samples / self._settings.sample_rate * 1000.0:.1f} ms "
                                    "to exclude the next timing marker or ESS.",
                                ),
                            ),
                        )
                    if shot.center_confidence < 0.15:
                        quality = replace(
                            shot.quality,
                            status=ShotStatus.WARNING,
                            messages=shot.quality.messages + (
                                f"Direct-IR center confidence is too low ({shot.center_confidence:.2f}); relative phase excluded.",
                            ),
                        )
                        shot = replace(
                            shot,
                            status=ShotStatus.WARNING,
                            quality=quality,
                            relative_phase_valid=False,
                            included_in_average=False,
                        )
                    anchors = [
                        candidate for candidate in self._shots
                        if candidate.combined_raw_ir is not None and candidate.relative_phase_valid
                    ]
                    if anchors and shot.combined_raw_ir is not None:
                        anchor = max(anchors, key=lambda candidate: candidate.correlation)
                        similarity = centered_impulse_similarity(
                            shot.combined_raw_ir,
                            shot.center_sample,
                            anchor.combined_raw_ir,
                            anchor.center_sample,
                            self._settings.sample_rate,
                        )
                        if similarity < 0.60:
                            quality = replace(
                                shot.quality,
                                status=ShotStatus.WARNING,
                                messages=shot.quality.messages + (
                                    f"Centered IR similarity is low ({similarity:.2f}); review this shot.",
                                ),
                            )
                            shot = replace(
                                shot,
                                status=ShotStatus.WARNING,
                                quality=quality,
                                centered_ir_similarity=float(similarity),
                                included_in_average=bool(similarity >= 0.35),
                            )
                        else:
                            shot = replace(shot, centered_ir_similarity=float(similarity))
                    post_rejection = self._post_analysis_rejection(shot)
                    if post_rejection:
                        self._record_rejection(post_rejection)
                        with self._lock:
                            self._state = MeasurementState.SEARCHING if not self._stop_requested.is_set() else MeasurementState.STOPPING
                        previous_start = absolute_start
                        consumed = capture_start + capture_samples
                        buffer, buffer_origin = buffer[consumed:], buffer_origin + consumed
                        continue
                    if do_wavelet and shot.combined_raw_ir is not None:
                        shot = replace(
                            shot,
                            wavelet_map=wavelet_from_ir(
                                shot.combined_raw_ir,
                                self._settings,
                                label=f"Shot {shot.shot_index}",
                                center_sample=shot.center_sample,
                            ),
                        )
                    with self._lock:
                        self._shots.append(shot)
                        self._measurement_revision += 1
                        self._stream_recovery_sync_grace_until = 0.0
                        self._device_info["adaptive_correlation_threshold"] = (
                            self._adaptive_sync_correlation_threshold()
                        )
                        if expected_reference_locked:
                            self._device_info["expected_ess_locks"] = int(
                                self._device_info.get("expected_ess_locks", 0)
                            ) + 1
                        self._last_rejection_reason = ""
                        self._device_info["last_synchronization_status"] = (
                            "Recovered at player file boundary"
                            if marker_resynchronized else
                            "Synchronized"
                        )
                        self._state = MeasurementState.SEARCHING if not self._stop_requested.is_set() else MeasurementState.STOPPING
                        snapshot = self.snapshot()
                    if self._settings.autosave_enabled:
                        autosave_shot(self._directory, snapshot, shot)
                        # Accepted Raw is durable in its shot NPZ. Keep
                        # analysis products in RAM for Gate
                        # edits and averaging without growing by another
                        # ~1 MiB/shot at 192 kHz during a long capture.
                        with self._lock:
                            self._shots[-1] = replace(self._shots[-1], raw_stereo_audio=None)
                    if self._settings.guided_measurement and len(self.snapshot().valid_shots) >= 3:
                        with self._lock:
                            self._device_info["guide_stage"] = "finalizing"
                        self.request_stop()
                        return
                    previous_start = absolute_start
                    consumed = capture_start + capture_samples
                    buffer, buffer_origin = buffer[consumed:], buffer_origin + consumed
        except Exception as exc:
            self._fail(f"analysis worker failed: {exc}", fatal=False)
        finally:
            if raw_writer is not None:
                raw_writer.close()
            with self._lock:
                if self._state == MeasurementState.STOPPING:
                    self._state = MeasurementState.STOPPED_WITH_RESULTS
                if self._state == MeasurementState.STOPPED_WITH_RESULTS and not self._stopped_at:
                    self._stopped_at = utc_now()
            if self._settings.autosave_enabled:
                try:
                    autosave_manifest(self._directory, self.snapshot())
                except OSError:
                    pass

    def _pre_analysis_rejection(self, raw: np.ndarray, correlation: float) -> str:
        gain_error = self._input_gain_rejection(str(self._device_info.get("name", "")))
        if gain_error:
            return gain_error
        values = np.asarray(raw)
        if values.ndim != 2 or values.shape[0] < len(self._reference) or values.shape[1] != self._settings.channels:
            return "Incomplete or mismatched shot data"
        if not np.all(np.isfinite(values)):
            return "Shot contains NaN or Inf"
        if self._settings.guided_measurement and self._preflight_ready:
            baseline = float(self._device_info.get("guide_pilot_rms", 0.0))
            current = float(np.sqrt(np.mean(values[:len(self._reference)].astype(float)**2)))
            if baseline > 0 and abs(20*np.log10(max(current, 1e-15)/baseline)) > 3.0:
                return "ESS gain changed by more than 3 dB; repeat noise level check"
        if float(np.max(np.abs(values))) < 1e-7:
            return "No usable microphone signal"
        if correlation < self._settings.correlation_threshold:
            return "ESS correlation below minimum"
        quality = channel_quality(values, self._settings)
        # The capture tail can contain legitimate room/speaker decay. Only the
        # explicitly strict gate treats a rise over the initial ambient sample
        # as a hard rejection; Standard relies on the shot S/N gate below.
        ambient_error = (
            self._ambient_noise_rejection(values, quality)
            if self._settings.quality_gate_mode == "strict"
            else ""
        )
        if ambient_error:
            return ambient_error
        if quality.left_clipped and quality.right_clipped:
            return "All usable input channels clipped"
        minimum_snr = {"strict": 12.0, "standard": 6.0, "lenient": 0.0}[self._settings.quality_gate_mode]
        if quality.snr_db < minimum_snr:
            return f"S/N below {minimum_snr:g} dB quality-gate minimum"
        if self._settings.quality_gate_mode == "strict" and quality.status != ShotStatus.VALID:
            return "Strict quality gate rejected a warning shot"
        return ""

    def _ambient_noise_rejection(self, raw: np.ndarray, quality) -> str:
        if self._ambient_noise is None:
            return ""
        count = max(8, min(raw.shape[0] // 8, int(round(0.03 * self._settings.sample_rate))))
        tail = np.asarray(raw[-count:], dtype=np.float64)
        current_rms = np.sqrt(np.mean(tail * tail, axis=0))
        if tail.shape[1] == 1 or quality.selected_channel == "mono":
            channel = 0
        elif str(quality.selected_channel).startswith("R"):
            channel = 1
        else:
            channel = 0
        baseline = self._ambient_noise.channel_p90_rms[min(channel, len(self._ambient_noise.channel_p90_rms) - 1)]
        increase_db = 20.0 * np.log10(max(float(current_rms[channel]), 1e-15) / max(float(baseline), 1e-15))
        if increase_db > 3.0:
            return f"Ambient noise rose by {increase_db:.1f} dB; shot rejected"
        return ""

    def _input_gain_rejection(self, device_name: str) -> str:
        if not self._settings.input_gain_db_channels:
            return ""
        expected = InputGainSnapshot(
            available=True,
            device_name=self._settings.input_gain_device_name,
            channel_gain_db=self._settings.input_gain_db_channels,
            channel_max_gain_db=self._settings.input_gain_max_db_channels,
            analog_gain_db=self._settings.input_analog_gain_db,
            source="CoreAudio",
        )
        current = read_input_gain_snapshot(device_name, self._settings.channels)
        if not current.available:
            return (
                "Input gain can no longer be verified; manufacturer SPL calibration is suspended"
                if self._settings.require_stable_input_gain else ""
            )
        if not input_gain_is_unchanged(expected, current):
            old_gain = float(np.mean(self._settings.input_gain_db_channels)) if self._settings.input_gain_db_channels else 0.0
            new_gain = float(np.mean(current.channel_gain_db)) if current.channel_gain_db else old_gain
            delta = new_gain - old_gain
            if self._ambient_noise is not None:
                self._ambient_noise = self._ambient_noise.shifted_by_gain(delta)
            self._settings = self._settings_with_input_gain_snapshot(current)
            self._preflight_ready = False
            self._gain_recheck_required = True
            self._device_info.update(
                input_gain_db_channels=list(current.channel_gain_db),
                input_gain_min_db_channels=list(current.channel_min_gain_db),
                input_gain_max_db_channels=list(current.channel_max_gain_db),
                input_gain_device_name=current.device_name,
                input_gain_writable=current.writable,
                preflight_ready=False,
                gain_recheck_required=True,
                preflight_message=(
                    f"マイクGainの変更 {delta:+.1f} dBを検出しました。"
                    "このSweepはGain切替境界として除外し、次のPilot ESSで再確認します。"
                ),
            )
            return "Input gain changed; transition sweep excluded and Pilot recheck required"
        return ""

    def _settings_with_input_gain_snapshot(self, snapshot: InputGainSnapshot) -> MeasurementSettings:
        gain_from_max = snapshot.gain_adjustment_from_max_db
        initialize_response_reference = self._settings.response_gain_reference_db is None
        current_reference = (
            float(np.mean(snapshot.channel_gain_db))
            if snapshot.channel_gain_db else float(self._settings.input_gain_db)
        )
        return replace(
            self._settings,
            input_gain_db_channels=snapshot.channel_gain_db,
            input_gain_max_db_channels=snapshot.channel_max_gain_db,
            input_gain_device_name=snapshot.device_name,
            input_analog_gain_db=snapshot.analog_gain_db,
            minidsp_gain_adjustment_db=(
                float(gain_from_max or 0.0)
                if initialize_response_reference
                else self._settings.minidsp_gain_adjustment_db
            ),
            response_gain_reference_db=(
                current_reference
                if self._settings.response_gain_reference_db is None
                else self._settings.response_gain_reference_db
            ),
        )

    def _post_analysis_rejection(self, shot: ShotResult) -> str:
        if shot.status in {ShotStatus.INVALID, ShotStatus.CLIPPED, ShotStatus.DETECTION_FAILED}:
            return f"Shot analysis status: {shot.status}"
        if shot.quality.selected_channel == "none":
            return "No usable gain channel"
        arrays = (
            shot.combined_raw_ir,
            shot.gated_ir,
            shot.frequency_hz,
            shot.ungated_complex_response,
            shot.gated_complex_response,
            shot.merged_complex_response,
        )
        if any(value is None or not np.all(np.isfinite(value)) for value in arrays):
            return "Deconvolution produced invalid data"
        if shot.combined_raw_ir is None or float(np.max(np.abs(shot.combined_raw_ir))) < 1e-12:
            return "Direct impulse could not be resolved"
        if self._settings.quality_gate_mode == "strict" and shot.status != ShotStatus.VALID:
            return "Strict quality gate rejected a warning shot"
        return ""

    def _record_rejection(self, reason: str) -> None:
        if self._settings.guided_measurement and any(token in reason.lower() for token in ("clipped", "gain", "s/n below")):
            self.guide_recheck(reason)
        with self._lock:
            self._rejected_count += 1
            self._last_rejection_reason = str(reason)
            self._rejection_reasons[str(reason)] = self._rejection_reasons.get(str(reason), 0) + 1

    def _handle_pilot(
        self,
        assessment,
        device_name: str,
        *,
        correlation: float = 0.0,
        marker_detected: bool = False,
    ) -> None:
        confirming_gain_adjustment = self._gain_recheck_required
        gain_recheck_required = confirming_gain_adjustment
        repeated_raise_prompts = 1
        for previous in reversed(self._pilot_history):
            if previous.get("action") != "raise_speaker":
                break
            repeated_raise_prompts += 1
        ready = bool(assessment.ready)
        action = str(assessment.action)
        message = assessment.message
        if self._settings.guided_measurement and action == "adjust_input" and self._pilot_snr_is_acceptable(assessment) and assessment.headroom_db >= self._settings.minimum_headroom_db:
            ready = True
            action = "keep_input_gain"
            message = "現在の入力GainでS/Nとクリップ余裕を確認しました。音量を固定して取得します。"
        if self._settings.guided_measurement and not ready:
            self.guide_recheck("ESSの音量・品質を確認できません。再生を停止し、ノイズの音量確認へ戻ってください。")
            return
        low_snr_continue = False
        if (
            not ready
            and action in {"raise_speaker", "level_limit_reached"}
            and self._settings.quality_gate_mode != "strict"
            and self._pilot_can_continue_with_low_snr(assessment)
            and (
                action == "level_limit_reached"
                or repeated_raise_prompts >= self._settings.low_snr_retry_limit
            )
        ):
            ready = True
            action = "low_snr_continue"
            low_snr_continue = True
            message = (
                "S/Nは推奨値未満ですが、最低品質とクリップ余裕を満たすため正式測定へ進みます。"
                "結果には低S/N警告を残します。"
            )
        record = {
            "pilot_index": len(self._pilot_history) + 1,
            "timestamp": utc_now(),
            "action": action,
            "message": message,
            "peak_dbfs": assessment.peak_dbfs,
            "headroom_db": assessment.headroom_db,
            "snr_median_db": assessment.snr_median_db,
            "snr_p10_db": assessment.snr_p10_db,
            "passing_band_fraction": assessment.passing_band_fraction,
            "active_band_start_hz": float(getattr(assessment, "active_band_start_hz", 0.0)),
            "active_band_end_hz": float(getattr(assessment, "active_band_end_hz", 0.0)),
            "active_band_count": int(getattr(assessment, "active_band_count", 0)),
            "active_band_span_octaves": float(getattr(assessment, "active_band_span_octaves", 0.0)),
            "level_mask_valid": bool(getattr(assessment, "level_mask_valid", False)),
            "playback_spl_db": getattr(assessment, "playback_spl_db", None),
            "snr_target_db": float(getattr(assessment, "snr_target_db", 0.0)),
            "snr_minimum_db": float(getattr(assessment, "snr_minimum_db", 0.0)),
            "snr_passing_fraction_required": float(getattr(assessment, "snr_passing_fraction_required", 0.0)),
            "low_snr_continue": low_snr_continue,
            "speaker_adjustment_db": assessment.speaker_adjustment_db,
            "microphone_gain_adjustment_db": assessment.microphone_gain_adjustment_db,
            "confirms_gain_adjustment": confirming_gain_adjustment,
            "ess_correlation": float(correlation),
            "synchronization": "marker + ESS verified" if marker_detected else "ESS correlation",
        }
        gain_changed_this_pilot = False
        if assessment.action in {"adjust_input", "reduce_input", "prepare_input"} and self._settings.automatic_usb_gain:
            before = read_input_gain_snapshot(device_name, self._settings.channels)
            changed = set_input_gain_adjustment(
                device_name,
                self._settings.channels,
                assessment.microphone_gain_adjustment_db,
            )
            if changed.available and before.available:
                actual_delta = float(np.mean(changed.channel_gain_db) - np.mean(before.channel_gain_db))
                if abs(actual_delta) >= 0.05:
                    gain_changed_this_pilot = True
                    if self._ambient_noise is not None:
                        self._ambient_noise = self._ambient_noise.shifted_by_gain(actual_delta)
                    self._settings = self._settings_with_input_gain_snapshot(changed)
                    self._device_info.update(
                        input_gain_db_channels=list(changed.channel_gain_db),
                        input_gain_min_db_channels=list(changed.channel_min_gain_db),
                        input_gain_max_db_channels=list(changed.channel_max_gain_db),
                        input_gain_device_name=changed.device_name,
                        input_gain_writable=changed.writable,
                    )
                    message = f"マイクGainを{actual_delta:+.1f} dB自動調整しました。次のPilot ESSを再測定してGainを確認します。"
                    record["applied_microphone_gain_db"] = actual_delta
                else:
                    message = "マイクGainは制御範囲端です。ヘッドルーム6 dB以上なら現在値で測定します。"
                    ready = self._pilot_snr_is_acceptable(assessment) and assessment.headroom_db >= self._settings.minimum_headroom_db
            else:
                reason = changed.error or before.error or "自動制御できません"
                message = f"マイクGainを自動制御できません（{reason}）。"
                ready = self._pilot_snr_is_acceptable(assessment) and assessment.headroom_db >= self._settings.minimum_headroom_db
                if ready:
                    message += " 最低ヘッドルームを満たすため現在値で続行します。"
        if gain_changed_this_pilot:
            # A clipped Pilot cannot reveal the true overload margin. Never
            # approve measurement from the same Pilot that changed Gain.
            gain_recheck_required = True
            ready = False
        elif confirming_gain_adjustment:
            if ready:
                gain_recheck_required = False
                message = (
                    f"Gain再確認Pilot合格（Peak {assessment.peak_dbfs:.1f} dBFS / "
                    f"ヘッドルーム {assessment.headroom_db:.1f} dB）。次のESSから正式測定します。"
                )
            else:
                gain_recheck_required = True
                ready = False
        record["gain_confirmation_passed"] = bool(confirming_gain_adjustment and ready and not gain_changed_this_pilot)
        record["gain_recheck_required_after_pilot"] = gain_recheck_required
        record["ready_after_pilot"] = ready
        record["message"] = message
        if ready:
            sync_reference, sync_offset = ess_band_detection_window(
                self._reference,
                self._settings,
                float(getattr(assessment, "active_band_start_hz", 0.0)),
                float(getattr(assessment, "active_band_end_hz", 0.0)),
            )
            self._sync_reference = sync_reference
            self._sync_reference_offset = sync_offset
        with self._lock:
            self._pilot_history.append(record)
            if self._settings.guided_measurement and ready:
                self._device_info["guide_stage"] = "measuring"
            self._preflight_ready = ready
            self._gain_recheck_required = gain_recheck_required
            self._device_info["preflight_ready"] = ready
            self._device_info["gain_recheck_required"] = gain_recheck_required
            self._device_info["preflight_message"] = message
            self._device_info["pilot_history"] = list(self._pilot_history)
            self._device_info["low_snr_continue"] = low_snr_continue
            playback_spl = getattr(assessment, "playback_spl_db", None)
            self._device_info["latest_playback_spl_db"] = playback_spl
            self._device_info["playback_level_warning"] = (
                "red" if playback_spl is not None and playback_spl >= self._settings.playback_red_warning_spl_db
                else "yellow" if playback_spl is not None and playback_spl >= self._settings.playback_warning_spl_db
                else "none"
            )
            self._device_info["synchronization_band_hz"] = [
                float(getattr(assessment, "active_band_start_hz", 0.0)),
                float(getattr(assessment, "active_band_end_hz", 0.0)),
            ] if ready else self._device_info.get("synchronization_band_hz", [])
            self._device_info["synchronization_reference_samples"] = int(self._sync_reference.size)

    def _pilot_snr_is_acceptable(self, assessment) -> bool:
        thresholds = pilot_snr_thresholds(self._settings)
        return bool(
            getattr(assessment, "level_mask_valid", False)
            and assessment.snr_median_db >= thresholds.median_db
            and assessment.snr_p10_db >= thresholds.p10_db
            and assessment.passing_band_fraction >= thresholds.passing_fraction
        )

    def _pilot_can_continue_with_low_snr(self, assessment) -> bool:
        if not getattr(assessment, "level_mask_valid", False):
            return False
        if assessment.headroom_db < self._settings.minimum_headroom_db:
            return False
        if self._settings.quality_gate_mode == "lenient":
            return bool(np.isfinite(assessment.snr_median_db) and assessment.snr_median_db >= 0.0)
        return bool(assessment.snr_median_db >= 6.0 and assessment.snr_p10_db >= 0.0)

    def _adaptive_sync_correlation_threshold(self) -> float:
        """Reject weak fallback peaks after a reliable acoustic lock exists."""

        floor = 0.35 if self._settings.reference_mode == "timing_marker" else 0.20
        multiplier = 0.65 if self._settings.reference_mode == "timing_marker" else 0.55
        baseline = max(float(self._settings.correlation_threshold), floor)
        recent = [
            float(shot.correlation)
            for shot in self._shots[-8:]
            if np.isfinite(shot.correlation) and shot.correlation > 0.0
        ]
        if not recent:
            return baseline
        return float(max(baseline, multiplier * np.median(recent)))

    def _preferred_center_polarity(self) -> str | None:
        if self._settings.sync_polarity != "auto":
            return self._settings.sync_polarity
        recent = [
            shot.center_polarity
            for shot in self._shots[-8:]
            if shot.center_polarity in {"positive", "negative"}
        ]
        if not recent:
            return None
        return max({"positive", "negative"}, key=recent.count)

    def _preferred_center_sample(self) -> float | None:
        for shot in reversed(self._shots):
            if np.isfinite(shot.center_sample) and shot.center_confidence >= 0.20:
                return float(shot.center_sample)
        return None

    def _expected_sync_correlation_threshold(self, recovery_threshold: float) -> float:
        """Use timing confidence to retain audible, lower-correlation repeats."""

        # This threshold applies only inside the +/-20 ms period-prediction
        # window. A long ESS correlation of 0.22 there is already a strong
        # match; tying it to the adaptive global threshold would reject the
        # same attenuated but correctly timed shot repeatedly.
        _ = recovery_threshold
        return float(max(self._settings.correlation_threshold, 0.22))

    def _save_reference(self) -> None:
        from scipy.io import wavfile

        wavfile.write(
            self._directory / "ess_reference.wav",
            self._settings.sample_rate,
            np.asarray(self._reference, dtype=np.float32),
        )
        if self._timing_marker is not None:
            wavfile.write(
                self._directory / "timing_marker.wav",
                self._settings.sample_rate,
                np.asarray(self._timing_marker, dtype=np.float32),
            )

    def _raw_recording_name(self) -> str:
        return "raw_recording_mono.wav" if self._settings.channels == 1 else "raw_recording_stereo.wav"

    def _fail(self, message: str, fatal: bool) -> None:
        with self._lock:
            self._error = str(message)
            self._state = MeasurementState.ERROR_FATAL if fatal else MeasurementState.ERROR_RECOVERABLE
            self._stopped_at = utc_now()
        self._stop_requested.set()
        if self._settings.autosave_enabled and self._directory.exists():
            try:
                autosave_manifest(self._directory, self.snapshot())
            except OSError:
                pass

    def _close_stream(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
