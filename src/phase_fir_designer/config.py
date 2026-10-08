from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal
import json
import math

from fir_design_common.kaiser import (
    KAISER_BETA_MAX,
    KAISER_BETA_MIN,
    clamp_kaiser_beta,
    kaiser_beta_in_range,
)
from .auto_iir_input_shaping import InputShapingSettings


def parse_iir_order(value: Any) -> int:
    """Recover only the known labels leaked by older selectbox serializers."""
    if isinstance(value, str):
        for order in range(1, 9):
            if value == f"{order} ({order * 6} dB/oct)":
                return order
    return int(value)


SUPPORTED_SAMPLE_RATES = (48_000, 96_000, 192_000)
SUPPORTED_WINDOWS = (
    "kaiser",
    "hann",
    "hamming",
    "blackman",
    "blackmanharris",
    "dolph_chebyshev",
    "tukey",
    "cosine_tapered",
    "rectangular",
)
SUPPORTED_ANALYSIS_FFT_SIZES = (2048, 4096, 8192, 16_384, 32_768, 65_536)
SUPPORTED_FIR_FIT_WEIGHTINGS = ("flat",)
LINEAR_FIR_CYCLES_MIN = 2.0
LINEAR_FIR_CYCLES_MAX = 7.0
# Compatibility names remain public, while the numerical policy is shared
# with Multiway through fir_design_common.kaiser.
LINEAR_FIR_BETA_MIN = KAISER_BETA_MIN
LINEAR_FIR_BETA_MAX = KAISER_BETA_MAX
LINEAR_FIR_EQ_MASK_SMOOTHING_OPTIONS = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0, 1.0, 2.0)
LINEAR_FIR_EQ_MASK_SMOOTHING_DEFAULT = 0.5
IIR_PEQ_Q_MAX = 5.0
IIR_SHELF_Q_MAX = 1.0
IIR_Q_STEP = 0.1
DEFAULT_ANALYSIS_FFT_SIZE_BY_SAMPLE_RATE = {
    48_000: 8_192,
    96_000: 16_384,
    192_000: 32_768,
}
DEFAULT_FIR_TAPS_BY_SAMPLE_RATE = {
    48_000: 4_097,
    96_000: 8_193,
    192_000: 16_385,
}


@dataclass(frozen=True)
class PhasePEQ:
    fc: float
    q: float
    phase_deg: float
    enabled: bool = True


@dataclass(frozen=True)
class PhaseShelf:
    mode: Literal["low", "high"]
    fc: float
    q: float
    phase_deg: float
    enabled: bool = True


@dataclass(frozen=True)
class PhaseTilt:
    f_start: float
    f_end: float
    slope_deg_per_oct: float
    reference_freq: float | None = None
    reference_phase_deg: float | None = None
    edge_smooth_percent: float = 0.0
    enabled: bool = True
    edge_smooth_oct: float = 0.0


@dataclass(frozen=True)
class PhaseAllPass:
    fc: float
    q: float
    polarity: Literal["positive", "negative"] = "positive"
    enabled: bool = True


@dataclass(frozen=True)
class GainPEQ:
    fc: float
    q: float
    gain_db: float
    enabled: bool = True


@dataclass(frozen=True)
class GainShelf:
    mode: Literal["low", "high"]
    fc: float
    q: float
    gain_db: float
    enabled: bool = True


@dataclass(frozen=True)
class GainTilt:
    f_start: float
    f_end: float
    slope_db_per_oct: float
    mode: Literal["high", "low"] = "high"
    reference_freq: float | None = None
    reference_gain_db: float | None = None
    edge_smooth_percent: float = 0.0
    enabled: bool = True
    edge_smooth_oct: float = 0.0


@dataclass(frozen=True)
class GainLinearFilter:
    mode: Literal["hp", "lp"]
    response: Literal["lr2"] = "lr2"
    fc: float = 1000.0
    cycles: float = 4.0
    beta: float = 12.0
    enabled: bool = True


@dataclass(frozen=True)
class IIRFilter:
    kind: Literal["peq", "high_shelf", "low_shelf", "allpass", "high_pass", "low_pass"] = "peq"
    fc: float = 1000.0
    q: float = 0.7071
    gain_db: float = 0.0
    family: Literal["linkwitz_riley", "bessel", "butterworth", "variable_q", "elliptic"] = "butterworth"
    order: int = 2
    enabled: bool = True
    origin: Literal["manual", "auto"] = "manual"
    allpass_preset: Literal["manual", "lr2_lo", "lr2_hi", "lr4", "lr8"] = "manual"
    ripple_db: float = 0.1
    stop_db: float = 80.0

    def __post_init__(self) -> None:
        if self.family == "variable_q" and (not math.isfinite(self.q) or self.q <= 0):
            raise ValueError("Variable-Q Q must be finite and > 0")
        if self.family == "elliptic" and not (
            math.isfinite(self.ripple_db) and math.isfinite(self.stop_db)
            and 0 < self.ripple_db < self.stop_db
        ):
            raise ValueError("Elliptic requires 0 < ripple_db < stop_db (positive attenuation)")
        # Keep serialized parameters and the coefficients generated from them
        # identical, even when callers construct a filter outside the UI.
        value = max(float(self.q), 0.01)
        if self.kind == "peq":
            value = min(value, IIR_PEQ_Q_MAX)
        elif self.kind in {"high_shelf", "low_shelf"}:
            value = min(value, IIR_SHELF_Q_MAX)
        object.__setattr__(self, "q", value)


def clamp_iir_q(kind: str, q: float) -> float:
    """Return the application-level Q range for editable IIR filters."""
    value = max(float(q), 0.01)
    if kind == "peq":
        return min(value, IIR_PEQ_Q_MAX)
    if kind in {"high_shelf", "low_shelf"}:
        return min(value, IIR_SHELF_Q_MAX)
    return value


def quantize_iir_q(kind: str, q: float) -> float:
    """Return an editable IIR Q value on the application 0.1 grid."""
    value = clamp_iir_q(kind, q)
    quantized = round(value, 1)
    return clamp_iir_q(kind, max(quantized, IIR_Q_STEP))


@dataclass(frozen=True)
class LinearFIRFilter:
    mode: Literal["hp", "lp"]
    # Persisted compatibility name: this record now represents one output
    # band-split boundary.  FIR methods are convolved into the FIR output;
    # IIR methods are exported separately as Linkwitz-Riley Biquads.
    response: Literal["kaiser", "linear_phase_lr2", "linear_phase_lr4", "iir_lr2", "iir_lr4", "through"] = "kaiser"
    fc: float = 500.0
    cycles: float = 4.5
    beta: float = 12.0
    enabled: bool = True
    base_crossover_hz: float | None = None
    overlap_oct: float = 0.0
    lr2_auto_taps: bool = False
    acoustic_target: bool = False


@dataclass(frozen=True)
class CandidateWaveletProfile:
    frequency_hz: list[float] = field(default_factory=list)
    weight: list[float] = field(default_factory=list)
    source_type: str = "original_ir"
    reconstruction_confidence: float = 1.0
    phase_available: bool = True
    smoothing_detected: bool = False
    estimated_smoothing_oct: float | None = None
    smoothing_likelihood: float = 0.0
    source_warning: str | None = None


@dataclass(frozen=True)
class AutoEQSection:
    enabled: bool = True
    gain_enabled: bool = True
    phase_enabled: bool = False
    f_min: float = 200.0
    f_max: float = 10_000.0
    smoothing_fraction: float = 3.0
    phase_strength: float = 1.0  # Legacy Project JSON compatibility; ignored by Auto Phase EQ.
    gain_strength: float = 1.0  # Legacy Project JSON compatibility; ignored by Auto Gain EQ.
    max_boost_db: float = 6.0
    max_cut_db: float = 12.0
    limit_mode: Literal["full", "peak_dip"] = "peak_dip"
    trend_smoothing_oct: float = 0.0
    post_limit_smoothing_fraction: float = 0.0
    candidate_min_delta_db: float = 0.0
    candidate_min_wavelet_weight: float = 0.0
    gain_phase_mode: Literal["lin", "min"] = "lin"


@dataclass(frozen=True)
class AutoEQConfig:
    phase_enabled: bool = False
    gain_enabled: bool = False
    f_min: float = 200.0
    f_max: float = 10_000.0
    smoothing_fraction: float = 3.0
    edge_smooth_percent: float = 10.0
    gain_edge_smooth_oct: float = 1.0 / 6.0
    phase_edge_smooth_oct: float = 1.0 / 3.0
    gain_boundary_smoothing_method: Literal["smooth_connect"] = "smooth_connect"
    match_target_level: bool = True
    target_level_offset_db: float = 0.0
    phase_strength: float = 1.0  # Legacy Project JSON compatibility; normalized to 1.0.
    gain_strength: float = 1.0  # Legacy Project JSON compatibility; normalized to 1.0.
    max_boost_db: float = 6.0
    max_cut_db: float = 12.0
    limit_mode: Literal["full", "peak_dip"] = "peak_dip"
    trend_smoothing_oct: float = 0.0
    post_limit_smoothing_fraction: float = 0.0
    candidate_min_delta_db: float = 0.0
    candidate_min_wavelet_weight: float = 0.0
    candidate_wavelet_profile: CandidateWaveletProfile | None = None
    sections: list[AutoEQSection] = field(default_factory=list)
    gain_sections: list[AutoEQSection] = field(default_factory=list)
    phase_sections: list[AutoEQSection] = field(default_factory=list)


@dataclass(frozen=True)
class LimiterConfig:
    mode: Literal["off", "warn", "limit"] = "off"
    threshold_db: float = 0.0


@dataclass(frozen=True)
class SpeakerResponse:
    frequency: list[float]
    gain_db: list[float]
    phase_deg: list[float] | None = None


@dataclass(frozen=True)
class DesignConfig:
    sample_rate: int
    taps: int
    analysis_fft_size: int = 8_192
    peq_filters: list[PhasePEQ] = field(default_factory=list)
    shelf_filters: list[PhaseShelf] = field(default_factory=list)
    tilt_filters: list[PhaseTilt] = field(default_factory=list)
    allpass_filters: list[PhaseAllPass] = field(default_factory=list)
    gain_peq_filters: list[GainPEQ] = field(default_factory=list)
    gain_shelf_filters: list[GainShelf] = field(default_factory=list)
    gain_tilt_filters: list[GainTilt] = field(default_factory=list)
    gain_linear_filters: list[GainLinearFilter] = field(default_factory=list)
    iir_filters: list[IIRFilter] = field(default_factory=list)
    linear_fir_filters: list[LinearFIRFilter] = field(default_factory=list)
    band_split_recipe: dict[str, Any] | None = None
    acoustic_correction_enabled: bool = True
    # Internal compatibility switch for published FIR recipes before v3.
    acoustic_boundary_continuation_enabled: bool = True
    fir_generation_grid: str = "multiway-v1"
    linear_fir_eq_mask_smoothing_oct: float = LINEAR_FIR_EQ_MASK_SMOOTHING_DEFAULT
    phase_tilt_smoothing_oct: float = 1.0 / 3.0
    gain_tilt_smoothing_oct: float = 1.0 / 3.0
    auto_eq: AutoEQConfig = field(default_factory=AutoEQConfig)
    window: str = "kaiser"
    kaiser_beta: float = 12.0
    chebyshev_attenuation_db: float = 100.0
    tukey_alpha: float = 0.5
    dc_gain_normalize: bool = True
    fir_auto_gain_enabled: bool = True
    fir_fit_weighting: Literal["flat"] = "flat"
    limiter: LimiterConfig = field(default_factory=LimiterConfig)
    speaker_response: SpeakerResponse | None = None
    target_response: SpeakerResponse | None = None
    auto_gain_input_shaping: InputShapingSettings = field(default_factory=InputShapingSettings)

    def normalized(self) -> "DesignConfig":
        taps = _normalize_taps(self.taps)
        return DesignConfig(
            sample_rate=self.sample_rate,
            taps=taps,
            band_split_recipe=self.band_split_recipe,
            acoustic_correction_enabled=bool(self.acoustic_correction_enabled),
            acoustic_boundary_continuation_enabled=bool(self.acoustic_boundary_continuation_enabled),
            fir_generation_grid=self.fir_generation_grid,
            analysis_fft_size=int(self.analysis_fft_size),
            peq_filters=[_clip_peq(f, self.sample_rate) for f in self.peq_filters],
            shelf_filters=[_clip_shelf(f, self.sample_rate) for f in self.shelf_filters],
            tilt_filters=[_clip_tilt(f, self.sample_rate) for f in self.tilt_filters],
            allpass_filters=[_clip_allpass(f, self.sample_rate) for f in self.allpass_filters],
            gain_peq_filters=[_clip_gain_peq(f, self.sample_rate) for f in self.gain_peq_filters],
            gain_shelf_filters=[
                _clip_gain_shelf(f, self.sample_rate) for f in self.gain_shelf_filters
            ],
            gain_tilt_filters=[
                _clip_gain_tilt(f, self.sample_rate) for f in self.gain_tilt_filters
            ],
            gain_linear_filters=[
                _clip_gain_linear(f, self.sample_rate) for f in self.gain_linear_filters
            ],
            iir_filters=[_clip_iir_filter(f, self.sample_rate) for f in self.iir_filters],
            linear_fir_filters=[
                _clip_linear_fir(f, self.sample_rate) for f in self.linear_fir_filters
            ],
            linear_fir_eq_mask_smoothing_oct=_normalize_linear_fir_eq_mask_smoothing_oct(
                self.linear_fir_eq_mask_smoothing_oct
            ),
            phase_tilt_smoothing_oct=float(self.phase_tilt_smoothing_oct),
            gain_tilt_smoothing_oct=float(self.gain_tilt_smoothing_oct),
            auto_eq=_clip_auto_eq(self.auto_eq, self.sample_rate),
            auto_gain_input_shaping=self.auto_gain_input_shaping,
            window=self.window,
            kaiser_beta=self.kaiser_beta,
            chebyshev_attenuation_db=self.chebyshev_attenuation_db,
            tukey_alpha=self.tukey_alpha,
            dc_gain_normalize=bool(self.dc_gain_normalize),
            fir_auto_gain_enabled=bool(self.fir_auto_gain_enabled),
            fir_fit_weighting=_normalize_fir_fit_weighting(self.fir_fit_weighting),
            limiter=self.limiter,
            speaker_response=self.speaker_response,
            target_response=self.target_response,
        )

    def validate(self) -> None:
        self.auto_gain_input_shaping.validate()
        if self.fir_generation_grid not in ("multiway-v1", "legacy-v1"):
            raise ValueError("unsupported FIR generation grid")
        if self.band_split_recipe is not None:
            from crossover_engine.recipe import validate
            validate(self.band_split_recipe)
            if self.band_split_recipe["sample_rate_hz"] != self.sample_rate:
                raise ValueError("Band split recipe sample rate mismatch")
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}")
        if self.taps < 1:
            raise ValueError("taps must be >= 1")
        if self.analysis_fft_size not in SUPPORTED_ANALYSIS_FFT_SIZES:
            raise ValueError(f"analysis_fft_size must be one of {SUPPORTED_ANALYSIS_FFT_SIZES}")
        if self.window not in SUPPORTED_WINDOWS:
            raise ValueError(f"window must be one of {SUPPORTED_WINDOWS}")
        if self.fir_fit_weighting not in SUPPORTED_FIR_FIT_WEIGHTINGS:
            raise ValueError(f"fir_fit_weighting must be one of {SUPPORTED_FIR_FIT_WEIGHTINGS}")
        if self.limiter.mode not in ("off", "warn", "limit"):
            raise ValueError("limiter.mode must be off, warn, or limit")
        if self.auto_eq.f_min < 0 or self.auto_eq.f_max < 0:
            raise ValueError("auto_eq frequencies must be >= 0")
        if self.auto_eq.f_max != 0 and self.auto_eq.f_max <= self.auto_eq.f_min:
            raise ValueError("auto_eq f_max must be greater than f_min or 0 for Nyquist")
        if self.auto_eq.smoothing_fraction < 0:
            raise ValueError("auto_eq smoothing_fraction must be >= 0")
        if self.auto_eq.max_boost_db < 0 or self.auto_eq.max_cut_db < 0:
            raise ValueError("auto_eq max boost/cut must be >= 0")
        auto_sections = self.auto_eq.sections or [
            AutoEQSection(
                enabled=self.auto_eq.gain_enabled or self.auto_eq.phase_enabled,
                gain_enabled=self.auto_eq.gain_enabled,
                phase_enabled=self.auto_eq.phase_enabled,
                f_min=self.auto_eq.f_min,
                f_max=self.auto_eq.f_max,
                smoothing_fraction=self.auto_eq.smoothing_fraction,
                phase_strength=1.0,
                gain_strength=1.0,
                max_boost_db=self.auto_eq.max_boost_db,
                max_cut_db=self.auto_eq.max_cut_db,
                limit_mode=self.auto_eq.limit_mode,
                trend_smoothing_oct=self.auto_eq.trend_smoothing_oct,
                post_limit_smoothing_fraction=self.auto_eq.post_limit_smoothing_fraction,
                candidate_min_delta_db=self.auto_eq.candidate_min_delta_db,
                candidate_min_wavelet_weight=self.auto_eq.candidate_min_wavelet_weight,
            )
        ]
        for section in [*auto_sections, *self.auto_eq.gain_sections, *self.auto_eq.phase_sections]:
            if section.gain_phase_mode not in ("lin", "min"):
                raise ValueError("auto_eq gain_phase_mode must be lin or min")
        for section in auto_sections:
            if not section.enabled:
                continue
            if section.f_min < 0 or section.f_max < 0:
                raise ValueError("auto_eq frequencies must be >= 0")
            if section.f_max != 0 and section.f_max <= section.f_min:
                raise ValueError("auto_eq f_max must be greater than f_min or 0 for Nyquist")
            if section.smoothing_fraction < 0:
                raise ValueError("auto_eq smoothing_fraction must be >= 0")
            if section.max_boost_db < 0 or section.max_cut_db < 0:
                raise ValueError("auto_eq max boost/cut must be >= 0")
            if section.limit_mode not in ("full", "peak_dip"):
                raise ValueError("auto_eq limit_mode must be full or peak_dip")
            if section.trend_smoothing_oct < 0:
                raise ValueError("auto_eq trend_smoothing_oct must be >= 0")
            if section.post_limit_smoothing_fraction < 0:
                raise ValueError("auto_eq post_limit_smoothing_fraction must be >= 0")
            if section.candidate_min_delta_db < 0:
                raise ValueError("auto_eq candidate_min_delta_db must be >= 0")
            if not 0 <= section.candidate_min_wavelet_weight <= 1:
                raise ValueError("auto_eq candidate_min_wavelet_weight must be in 0..1")
        if self.auto_eq.limit_mode not in ("full", "peak_dip"):
            raise ValueError("auto_eq limit_mode must be full or peak_dip")
        if self.auto_eq.trend_smoothing_oct < 0:
            raise ValueError("auto_eq trend_smoothing_oct must be >= 0")
        if self.auto_eq.post_limit_smoothing_fraction < 0:
            raise ValueError("auto_eq post_limit_smoothing_fraction must be >= 0")
        if self.auto_eq.candidate_min_delta_db < 0:
            raise ValueError("auto_eq candidate_min_delta_db must be >= 0")
        if not 0 <= self.auto_eq.candidate_min_wavelet_weight <= 1:
            raise ValueError("auto_eq candidate_min_wavelet_weight must be in 0..1")
        if self.auto_eq.candidate_wavelet_profile is not None:
            profile = self.auto_eq.candidate_wavelet_profile
            if len(profile.frequency_hz) != len(profile.weight):
                raise ValueError("auto_eq candidate_wavelet_profile frequency and weight length must match")
        if not 0 <= self.auto_eq.edge_smooth_percent <= 50:
            raise ValueError("auto_eq edge_smooth_percent must be in 0..50")
        if not 0 <= self.auto_eq.gain_edge_smooth_oct <= 1:
            raise ValueError("auto_eq gain_edge_smooth_oct must be in 0..1")
        if not 0 <= self.auto_eq.phase_edge_smooth_oct <= 1:
            raise ValueError("auto_eq phase_edge_smooth_oct must be in 0..1")
        if self.auto_eq.gain_boundary_smoothing_method != "smooth_connect":
            raise ValueError("auto_eq gain_boundary_smoothing_method is unsupported")
        if not 0 <= self.phase_tilt_smoothing_oct <= 1:
            raise ValueError("phase_tilt_smoothing_oct must be in 0..1")
        if not 0 <= self.gain_tilt_smoothing_oct <= 3:
            raise ValueError("gain_tilt_smoothing_oct must be in 0..3")
        if self.linear_fir_eq_mask_smoothing_oct not in LINEAR_FIR_EQ_MASK_SMOOTHING_OPTIONS:
            raise ValueError("linear FIR EQ mask smoothing must be one of the supported octave options")
        for item in [
            *self.peq_filters,
            *self.shelf_filters,
            *self.allpass_filters,
            *self.gain_peq_filters,
            *self.gain_shelf_filters,
        ]:
            if item.fc < 0:
                raise ValueError("filter fc must be >= 0")
            if item.q <= 0:
                raise ValueError("filter Q must be > 0")
        for item in self.allpass_filters:
            if item.polarity not in ("positive", "negative"):
                raise ValueError("allpass polarity must be positive or negative")
        for item in [*self.shelf_filters, *self.gain_shelf_filters]:
            if item.mode not in ("low", "high"):
                raise ValueError("shelf mode must be low or high")
        for item in [*self.tilt_filters, *self.gain_tilt_filters]:
            if item.f_start < 0 or item.f_end < 0:
                raise ValueError("tilt frequencies must be >= 0")
            if item.f_end != 0 and item.f_end <= item.f_start:
                raise ValueError("tilt f_end must be greater than f_start")
            if not 0 <= item.edge_smooth_percent <= 50:
                raise ValueError("edge_smooth_percent must be in 0..50")
            if item.reference_freq is not None and item.reference_freq <= 0:
                raise ValueError("reference_freq must be > 0")
        for item in self.tilt_filters:
            if not 0 <= item.edge_smooth_oct <= 1:
                raise ValueError("phase tilt edge_smooth_oct must be in 0..1")
        for item in self.gain_tilt_filters:
            if not 0 <= item.edge_smooth_oct <= 1:
                raise ValueError("gain tilt edge_smooth_oct must be in 0..1")
        for item in self.gain_linear_filters:
            if item.mode not in ("hp", "lp"):
                raise ValueError("gain linear mode must be hp or lp")
            if item.response != "lr2":
                raise ValueError("gain linear response must be lr2")
            if item.fc < 0:
                raise ValueError("gain linear fc must be >= 0")
            if item.cycles <= 0:
                raise ValueError("gain linear cycles must be > 0")
            if item.beta < 0:
                raise ValueError("gain linear beta must be >= 0")
        for item in self.iir_filters:
            if item.kind not in ("peq", "high_shelf", "low_shelf", "allpass", "high_pass", "low_pass"):
                raise ValueError("unsupported IIR filter type")
            if item.fc <= 0 or item.fc >= self.sample_rate / 2:
                raise ValueError("IIR filter fc must be between 0 and Nyquist")
            if item.q <= 0:
                raise ValueError("IIR filter Q must be > 0")
            if item.family not in ("linkwitz_riley", "bessel", "butterworth", "variable_q", "elliptic"):
                raise ValueError("unsupported IIR crossover family")
            if item.order not in (range(1, 9) if item.family == "elliptic" else (1, 2, 3, 4, 6, 8)):
                raise ValueError("IIR crossover order must be 1, 2, 3, 4, 6, or 8")
            if item.family == "variable_q" and item.order != 2:
                raise ValueError("Variable-Q requires order 2")
            if item.family == "linkwitz_riley" and item.kind in ("high_pass", "low_pass") and item.order not in (2, 4, 6, 8):
                raise ValueError("Linkwitz-Riley order must be 2, 4, 6, or 8")
            if item.origin not in ("manual", "auto"):
                raise ValueError("IIR filter origin must be manual or auto")
            if item.allpass_preset not in ("manual", "lr2_lo", "lr2_hi", "lr4", "lr8"):
                raise ValueError("unsupported IIR All Pass preset")
            if item.kind == "allpass" and item.allpass_preset == "manual" and item.order not in (1, 2):
                raise ValueError("manual IIR All Pass order must be 1 or 2")
        for item in self.linear_fir_filters:
            if item.mode not in ("hp", "lp"):
                raise ValueError("output band-split mode must be hp or lp")
            if item.acoustic_target and item.response in {"kaiser", "through"}:
                raise ValueError("音響ターゲットにはLR2／LR4を選択してください。")
            if item.response not in ("kaiser", "linear_phase_lr2", "linear_phase_lr4", "iir_lr2", "iir_lr4", "through"):
                raise ValueError("unsupported output band-split method")
            if item.fc < 0:
                raise ValueError("linear FIR fc must be >= 0")
            if not LINEAR_FIR_CYCLES_MIN <= item.cycles <= LINEAR_FIR_CYCLES_MAX:
                raise ValueError("linear FIR cycles must be in 2..7")
            if not kaiser_beta_in_range(item.beta):
                raise ValueError("linear FIR beta must be in 7..16")

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "DesignConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DesignConfig":
        sample_rate = int(data["sample_rate"])
        auto_eq_data = _auto_eq_data_with_edge_defaults(data.get("auto_eq", {}))
        if auto_eq_data.get("candidate_wavelet_profile") is not None:
            auto_eq_data["candidate_wavelet_profile"] = CandidateWaveletProfile(
                **auto_eq_data["candidate_wavelet_profile"]
            )
        for section_key in ("sections", "gain_sections", "phase_sections"):
            if section_key in auto_eq_data:
                auto_eq_data[section_key] = [AutoEQSection(**x) for x in auto_eq_data.get(section_key, [])]
        return cls(
            sample_rate=sample_rate,
            taps=int(data["taps"]),
            band_split_recipe=data.get("band_split_recipe"),
            acoustic_correction_enabled=bool(data.get("acoustic_correction_enabled", True)),
            acoustic_boundary_continuation_enabled=bool(data.get("acoustic_boundary_continuation_enabled", True)),
            fir_generation_grid=data.get("fir_generation_grid", "multiway-v1"),
            analysis_fft_size=int(
                data.get(
                    "analysis_fft_size",
                    DEFAULT_ANALYSIS_FFT_SIZE_BY_SAMPLE_RATE.get(sample_rate, 16_384),
                )
            ),
            peq_filters=[PhasePEQ(**x) for x in data.get("peq_filters", [])],
            shelf_filters=[PhaseShelf(**x) for x in data.get("shelf_filters", [])],
            tilt_filters=[PhaseTilt(**x) for x in data.get("tilt_filters", [])],
            allpass_filters=[PhaseAllPass(**x) for x in data.get("allpass_filters", [])],
            gain_peq_filters=[GainPEQ(**x) for x in data.get("gain_peq_filters", [])],
            gain_shelf_filters=[GainShelf(**x) for x in data.get("gain_shelf_filters", [])],
            gain_tilt_filters=[GainTilt(**x) for x in data.get("gain_tilt_filters", [])],
            gain_linear_filters=[
                _clip_gain_linear(GainLinearFilter(**x), sample_rate)
                for x in data.get("gain_linear_filters", [])
            ],
            iir_filters=[
                _clip_iir_filter(IIRFilter(**x), sample_rate)
                for x in data.get("iir_filters", [])
            ],
            linear_fir_filters=[
                _clip_linear_fir(_linear_fir_from_dict(x), sample_rate)
                for x in data.get("linear_fir_filters", [])
            ],
            linear_fir_eq_mask_smoothing_oct=_normalize_linear_fir_eq_mask_smoothing_oct(
                data.get(
                    "linear_fir_eq_mask_smoothing_oct",
                    LINEAR_FIR_EQ_MASK_SMOOTHING_DEFAULT,
                )
            ),
            phase_tilt_smoothing_oct=float(data.get("phase_tilt_smoothing_oct", 1.0 / 3.0)),
            gain_tilt_smoothing_oct=float(data.get("gain_tilt_smoothing_oct", 1.0 / 3.0)),
            auto_eq=AutoEQConfig(**auto_eq_data),
            auto_gain_input_shaping=InputShapingSettings(**data.get("auto_gain_input_shaping", {})),
            window=data.get("window", "kaiser"),
            kaiser_beta=float(data.get("kaiser_beta", 12.0)),
            chebyshev_attenuation_db=float(data.get("chebyshev_attenuation_db", 100.0)),
            tukey_alpha=float(data.get("tukey_alpha", 0.5)),
            dc_gain_normalize=bool(data.get("dc_gain_normalize", True)),
            fir_auto_gain_enabled=bool(data.get("fir_auto_gain_enabled", True)),
            fir_fit_weighting=_normalize_fir_fit_weighting(data.get("fir_fit_weighting", "flat")),
            limiter=LimiterConfig(**data.get("limiter", {})),
            speaker_response=(
                SpeakerResponse(**data["speaker_response"])
                if data.get("speaker_response") is not None
                else None
            ),
            target_response=(
                SpeakerResponse(**data["target_response"])
                if data.get("target_response") is not None
                else None
            ),
        )


def _nyquist(sample_rate: int) -> float:
    return sample_rate / 2


def _normalize_taps(value: int) -> int:
    taps = int(value)
    if taps <= 0:
        return 1
    return taps


def _normalize_fir_fit_weighting(value: Any) -> Literal["flat"]:
    return "flat"


def _normalize_linear_fir_eq_mask_smoothing_oct(value: Any) -> float:
    return min(
        LINEAR_FIR_EQ_MASK_SMOOTHING_OPTIONS,
        key=lambda option: abs(float(option) - float(value)),
    )


def _clip_unit_range(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _clip_frequency(value: float, sample_rate: int) -> float:
    return min(float(value), _nyquist(sample_rate))


def _clip_peq(item: PhasePEQ, sample_rate: int) -> PhasePEQ:
    return PhasePEQ(
        fc=_clip_frequency(item.fc, sample_rate),
        q=item.q,
        phase_deg=item.phase_deg,
        enabled=item.enabled,
    )


def _clip_shelf(item: PhaseShelf, sample_rate: int) -> PhaseShelf:
    return PhaseShelf(
        mode=item.mode,
        fc=_clip_frequency(item.fc, sample_rate),
        q=item.q,
        phase_deg=item.phase_deg,
        enabled=item.enabled,
    )


def _clip_tilt(item: PhaseTilt, sample_rate: int) -> PhaseTilt:
    return PhaseTilt(
        f_start=_clip_frequency(item.f_start, sample_rate),
        f_end=_clip_frequency(item.f_end, sample_rate),
        slope_deg_per_oct=item.slope_deg_per_oct,
        reference_freq=item.reference_freq,
        reference_phase_deg=item.reference_phase_deg,
        edge_smooth_percent=item.edge_smooth_percent,
        enabled=item.enabled,
        edge_smooth_oct=item.edge_smooth_oct,
    )


def _clip_allpass(item: PhaseAllPass, sample_rate: int) -> PhaseAllPass:
    return PhaseAllPass(
        fc=_clip_frequency(item.fc, sample_rate),
        q=item.q,
        polarity=item.polarity,
        enabled=item.enabled,
    )


def _clip_gain_peq(item: GainPEQ, sample_rate: int) -> GainPEQ:
    return GainPEQ(
        fc=_clip_frequency(item.fc, sample_rate),
        q=item.q,
        gain_db=item.gain_db,
        enabled=item.enabled,
    )


def _clip_gain_shelf(item: GainShelf, sample_rate: int) -> GainShelf:
    return GainShelf(
        mode=item.mode,
        fc=_clip_frequency(item.fc, sample_rate),
        q=item.q,
        gain_db=item.gain_db,
        enabled=item.enabled,
    )


def _clip_gain_tilt(item: GainTilt, sample_rate: int) -> GainTilt:
    return GainTilt(
        f_start=_clip_frequency(item.f_start, sample_rate),
        f_end=_clip_frequency(item.f_end, sample_rate),
        slope_db_per_oct=item.slope_db_per_oct,
        mode="low" if str(item.mode) == "low" else "high",
        reference_freq=item.reference_freq,
        reference_gain_db=item.reference_gain_db,
        edge_smooth_percent=item.edge_smooth_percent,
        enabled=item.enabled,
        edge_smooth_oct=item.edge_smooth_oct,
    )


def _clip_gain_linear(item: GainLinearFilter, sample_rate: int) -> GainLinearFilter:
    mode = item.mode if item.mode in ("hp", "lp") else "lp"
    return GainLinearFilter(
        mode=mode,
        response="lr2",
        fc=_clip_frequency(item.fc, sample_rate),
        cycles=max(float(item.cycles), 0.01),
        beta=max(float(item.beta), 0.0),
        enabled=bool(item.enabled),
    )


def _clip_iir_filter(item: IIRFilter, sample_rate: int) -> IIRFilter:
    kind = item.kind if item.kind in ("peq", "high_shelf", "low_shelf", "allpass", "high_pass", "low_pass") else "peq"
    family = item.family if item.family in ("linkwitz_riley", "bessel", "butterworth", "variable_q", "elliptic") else "butterworth"
    allpass_preset = (
        item.allpass_preset
        if item.allpass_preset in ("manual", "lr2_lo", "lr2_hi", "lr4", "lr8")
        else "manual"
    )
    order = parse_iir_order(item.order)
    allowed_orders = (2, 4, 6, 8) if family == "linkwitz_riley" and kind in ("high_pass", "low_pass") else (1, 2, 3, 4, 6, 8)
    if family == "elliptic":
        allowed_orders = tuple(range(1, 9))
    elif family == "variable_q":
        allowed_orders = (2,)
    if kind == "allpass" and allpass_preset == "manual":
        allowed_orders = (1, 2)
    if order not in allowed_orders:
        order = 4 if family == "linkwitz_riley" and kind in ("high_pass", "low_pass") else 2
    return IIRFilter(
        kind=kind,
        fc=min(max(float(item.fc), 1.0), _nyquist(sample_rate) - 1.0),
        q=(
            quantize_iir_q(kind, item.q)
            if kind in ("peq", "high_shelf", "low_shelf", "allpass")
            else clamp_iir_q(kind, item.q)
        ),
        gain_db=float(item.gain_db),
        family=family,
        ripple_db=float(item.ripple_db),
        stop_db=float(item.stop_db),
        order=order,
        allpass_preset=allpass_preset,
        enabled=bool(item.enabled),
        origin="auto" if str(item.origin) == "auto" else "manual",
    )


def _linear_fir_from_dict(data: dict[str, Any]) -> LinearFIRFilter:
    payload = dict(data)
    # Old lr4 records actually generated Kaiser coefficients.  Preserve that
    # sound while removing the misleading response name.
    response = str(payload.get("response", "kaiser")).lower()
    payload["response"] = {
        "lr2": "linear_phase_lr2",
        "lr4": "kaiser",
    }.get(response, response)
    payload.setdefault("cycles", 4.5)
    payload.setdefault("beta", 12.0)
    return LinearFIRFilter(**payload)


def _clip_linear_fir(item: LinearFIRFilter, sample_rate: int) -> LinearFIRFilter:
    fc = item.fc
    if item.base_crossover_hz is not None:
        from crossover_engine.filters import kaiser_overlap_edges_hz
        if not 0 < float(item.base_crossover_hz) < sample_rate / 2:
            raise ValueError("Band split base crossover must be in 0..Nyquist")
        edges = kaiser_overlap_edges_hz(item.base_crossover_hz, item.overlap_oct)
        fc = edges[1 if item.mode == "hp" else 0]
        if item.response in ("iir_lr2", "iir_lr4") and not 0 < fc < sample_rate / 2:
            raise ValueError("Band split IIR cutoff must be in 0..Nyquist after overlap")
    mode = item.mode if item.mode in ("hp", "lp") else "lp"
    response = (
        item.response
        if item.response in ("kaiser", "linear_phase_lr2", "linear_phase_lr4", "iir_lr2", "iir_lr4", "through")
        else "kaiser"
    )
    return LinearFIRFilter(
        mode=mode,
        response=response,
        fc=_clip_frequency(fc, sample_rate),
        base_crossover_hz=item.base_crossover_hz,
        lr2_auto_taps=bool(item.lr2_auto_taps),
        acoustic_target=bool(item.acoustic_target),
        overlap_oct=item.overlap_oct,
        cycles=min(max(float(item.cycles), LINEAR_FIR_CYCLES_MIN), LINEAR_FIR_CYCLES_MAX),
        beta=clamp_kaiser_beta(item.beta),
        enabled=bool(item.enabled),
    )


def _clip_auto_eq(item: AutoEQConfig, sample_rate: int) -> AutoEQConfig:
    return AutoEQConfig(
        phase_enabled=item.phase_enabled,
        gain_enabled=item.gain_enabled,
        f_min=_clip_frequency(item.f_min, sample_rate),
        f_max=_clip_frequency(item.f_max, sample_rate),
        smoothing_fraction=item.smoothing_fraction,
        edge_smooth_percent=item.edge_smooth_percent,
        gain_edge_smooth_oct=item.gain_edge_smooth_oct,
        phase_edge_smooth_oct=item.phase_edge_smooth_oct,
        gain_boundary_smoothing_method=item.gain_boundary_smoothing_method,
        match_target_level=item.match_target_level,
        target_level_offset_db=item.target_level_offset_db,
        # Kept in the schema for old Project JSON compatibility. Auto Phase EQ
        # is fixed at 100%, so normalized settings always store 1.0.
        phase_strength=1.0,
        # Kept in the schema for old Project JSON compatibility. Auto Gain EQ
        # is fixed at 100%, so normalized settings always store 1.0.
        gain_strength=1.0,
        max_boost_db=item.max_boost_db,
        max_cut_db=item.max_cut_db,
        limit_mode=item.limit_mode,
        trend_smoothing_oct=item.trend_smoothing_oct,
        post_limit_smoothing_fraction=item.post_limit_smoothing_fraction,
        candidate_min_delta_db=item.candidate_min_delta_db,
        # Legacy fields remain readable, but Wavelet candidate gating is no
        # longer part of FIR generation.
        candidate_min_wavelet_weight=0.0,
        candidate_wavelet_profile=None,
        sections=[_clip_auto_eq_section(section, sample_rate) for section in item.sections],
        gain_sections=[_clip_auto_eq_section(section, sample_rate) for section in item.gain_sections],
        phase_sections=[_clip_auto_eq_section(section, sample_rate) for section in item.phase_sections],
    )


def _clip_auto_eq_section(item: AutoEQSection, sample_rate: int) -> AutoEQSection:
    f_min = _clip_frequency(item.f_min, sample_rate)
    f_max = 0.0 if item.f_max == 0 else _clip_frequency(item.f_max, sample_rate)
    enabled = bool(item.enabled)
    if enabled and f_max != 0.0 and f_max <= f_min:
        nyquist = _nyquist(sample_rate)
        if f_min < nyquist:
            f_max = min(nyquist, f_min + 1.0)
        else:
            enabled = False
    return AutoEQSection(
        enabled=enabled,
        gain_enabled=item.gain_enabled,
        phase_enabled=item.phase_enabled,
        f_min=f_min,
        f_max=f_max,
        smoothing_fraction=item.smoothing_fraction,
        phase_strength=1.0,
        gain_strength=1.0,
        max_boost_db=item.max_boost_db,
        max_cut_db=item.max_cut_db,
        limit_mode=item.limit_mode,
        trend_smoothing_oct=item.trend_smoothing_oct,
        post_limit_smoothing_fraction=item.post_limit_smoothing_fraction,
        candidate_min_delta_db=item.candidate_min_delta_db,
        candidate_min_wavelet_weight=0.0,
        gain_phase_mode=item.gain_phase_mode,
    )


def _clip_candidate_wavelet_profile(profile: CandidateWaveletProfile | None) -> CandidateWaveletProfile | None:
    if profile is None:
        return None
    frequency = [float(value) for value in profile.frequency_hz]
    weight = [float(min(max(value, 0.0), 1.0)) for value in profile.weight]
    if len(frequency) != len(weight) or len(frequency) < 2:
        return None
    estimated_smoothing_oct = (
        None
        if profile.estimated_smoothing_oct is None
        else float(max(profile.estimated_smoothing_oct, 0.0))
    )
    return CandidateWaveletProfile(
        frequency_hz=frequency,
        weight=weight,
        source_type=str(profile.source_type or "original_ir"),
        reconstruction_confidence=float(min(max(profile.reconstruction_confidence, 0.0), 1.0)),
        phase_available=bool(profile.phase_available),
        smoothing_detected=bool(profile.smoothing_detected),
        estimated_smoothing_oct=estimated_smoothing_oct,
        smoothing_likelihood=float(min(max(profile.smoothing_likelihood, 0.0), 1.0)),
        source_warning=None if profile.source_warning is None else str(profile.source_warning),
    )


def _auto_eq_data_with_edge_defaults(data: dict[str, Any]) -> dict[str, Any]:
    auto_eq_data = dict(data)
    # The persisted key remains for old Project JSON compatibility. All old
    # comparison modes are migrated to the only supported batch method.
    auto_eq_data["gain_boundary_smoothing_method"] = "smooth_connect"
    # Strength was removed from both Auto EQ modes. Accept old Project JSON,
    # but normalize every legacy value before dataclass construction.
    auto_eq_data["gain_strength"] = 1.0
    auto_eq_data["phase_strength"] = 1.0
    for section_key in ("sections", "gain_sections", "phase_sections"):
        normalized_sections = []
        for section in auto_eq_data.get(section_key, []):
            normalized_section = dict(section)
            normalized_section["gain_strength"] = 1.0
            normalized_section["phase_strength"] = 1.0
            normalized_sections.append(normalized_section)
        if section_key in auto_eq_data:
            auto_eq_data[section_key] = normalized_sections
    legacy_percent = float(auto_eq_data.get("edge_smooth_percent", 10.0))
    if "gain_edge_smooth_oct" not in auto_eq_data:
        auto_eq_data["gain_edge_smooth_oct"] = _legacy_edge_percent_to_octave(
            legacy_percent,
            divisor=60.0,
        )
    if "phase_edge_smooth_oct" not in auto_eq_data:
        auto_eq_data["phase_edge_smooth_oct"] = _legacy_edge_percent_to_octave(
            legacy_percent,
            divisor=30.0,
        )
    return auto_eq_data


def _legacy_edge_percent_to_octave(percent: float, divisor: float) -> float:
    if percent <= 0:
        return 0.0
    return min(max(percent / divisor, 0.0), 0.5)
