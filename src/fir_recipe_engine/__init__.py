"""Application-neutral, versioned FIR recipe generation.

PhaseEQ and Composite Engine both call this module.  A recipe contains design
inputs and algorithm identity, never an authoritative coefficient payload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

import numpy as np

from phase_fir_designer import DesignConfig, design_phase_fir
from phase_fir_designer.auto_iir_input_shaping import InputShapingSettings
from phase_fir_designer.auto_eq import auto_gain_eq_sections, auto_phase_eq_sections
from phase_fir_designer.export_pipeline import output_fir_for_export_report


FIR_RECIPE_FORMAT = "phaseeq-fir-recipe"
FIR_RECIPE_FORMAT_VERSION = 1
FIR_RECIPE_ALGORITHM_ID = "phaseeq-manual-auto-fir"
FIR_RECIPE_ALGORITHM_VERSION = "weighted-complex-ls-v4"


@dataclass(frozen=True)
class FIRRecipePostprocess:
    remove_nyquist_enabled: bool = False
    remove_nyquist_strength: float = 0.0
    polarity_invert: bool = False
    auto_gain_enabled: bool = False


@dataclass(frozen=True)
class FIRRecipe:
    config: dict[str, Any]
    bypass: bool
    postprocess: FIRRecipePostprocess = FIRRecipePostprocess()
    format: str = FIR_RECIPE_FORMAT
    format_version: int = FIR_RECIPE_FORMAT_VERSION
    algorithm_id: str = FIR_RECIPE_ALGORITHM_ID
    algorithm_version: str = FIR_RECIPE_ALGORITHM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "format_version": self.format_version,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "bypass": self.bypass,
            "config": self.config,
            "postprocess": asdict(self.postprocess),
            "content_hash": self.content_hash,
        }

    @property
    def content_hash(self) -> str:
        payload = {
            "format": self.format,
            "format_version": self.format_version,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": self.algorithm_version,
            "bypass": self.bypass,
            "config": self.config,
            "postprocess": asdict(self.postprocess),
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FIRRecipe":
        if str(payload.get("format", "")) != FIR_RECIPE_FORMAT:
            raise ValueError("unsupported FIR recipe format")
        if int(payload.get("format_version", 0)) != FIR_RECIPE_FORMAT_VERSION:
            raise ValueError("unsupported FIR recipe format version")
        if str(payload.get("algorithm_id", "")) != FIR_RECIPE_ALGORITHM_ID:
            raise ValueError("unsupported FIR recipe algorithm")
        if str(payload.get("algorithm_version", "")) not in ("weighted-complex-ls-v1", "weighted-complex-ls-v2", "weighted-complex-ls-v3", FIR_RECIPE_ALGORITHM_VERSION):
            raise ValueError("unsupported FIR recipe algorithm version")
        config = payload.get("config")
        postprocess = payload.get("postprocess", {})
        if not isinstance(config, dict) or not isinstance(postprocess, dict):
            raise ValueError("invalid FIR recipe payload")
        recipe = cls(
            config=dict(config), bypass=bool(payload.get("bypass", False)),
            postprocess=FIRRecipePostprocess(**postprocess),
            algorithm_version=str(payload["algorithm_version"]),
        )
        expected_hash = str(payload.get("content_hash", "")).strip()
        if expected_hash and expected_hash != recipe.content_hash:
            raise ValueError("FIR recipe content hash mismatch")
        return recipe


@dataclass(frozen=True)
class FIRRecipeResult:
    coefficients: np.ndarray
    recipe_hash: str
    coefficient_hash: str
    bypass: bool
    target_dc_abs: float = 1.0


def fir_eq_is_active(config: DesignConfig) -> bool:
    from phase_fir_designer.acoustic_target import automatic_correction_active
    if automatic_correction_active(config):
        return True
    manual = (
        *config.peq_filters, *config.shelf_filters, *config.tilt_filters,
        *config.allpass_filters, *config.gain_peq_filters,
        *config.gain_shelf_filters, *config.gain_tilt_filters,
        *config.gain_linear_filters,
    )
    return bool(
        any(bool(getattr(item, "enabled", True)) for item in manual)
        or auto_gain_eq_sections(config.auto_eq)
        or auto_phase_eq_sections(config.auto_eq)
    )


def recipe_from_config(
    config: DesignConfig,
    *,
    postprocess: FIRRecipePostprocess = FIRRecipePostprocess(),
) -> FIRRecipe:
    """Create a canonical recipe, excluding Studio-owned band splitting.

    Wavelet candidate gating is permanently disabled.  IIR definitions remain
    in the design input because Auto FIR may be calculated after IIR, but the
    IIR coefficients are not convolved into the generated FIR.
    """
    normalized = config.normalized()
    auto_eq = replace(
        normalized.auto_eq,
        candidate_min_wavelet_weight=0.0,
        candidate_wavelet_profile=None,
        sections=[replace(item, candidate_min_wavelet_weight=0.0) for item in normalized.auto_eq.sections],
        gain_sections=[replace(item, candidate_min_wavelet_weight=0.0) for item in normalized.auto_eq.gain_sections],
        phase_sections=[replace(item, candidate_min_wavelet_weight=0.0) for item in normalized.auto_eq.phase_sections],
    )
    recipe_config = replace(
        normalized,
        fir_generation_grid="multiway-v1",
        auto_eq=auto_eq,
        dc_gain_normalize=False,
        fir_auto_gain_enabled=False,
        acoustic_boundary_continuation_enabled=True,
    )
    return FIRRecipe(
        config=asdict(recipe_config),
        bypass=not fir_eq_is_active(recipe_config),
        postprocess=postprocess,
    )


def generate_fir_recipe(recipe: FIRRecipe | dict[str, Any]) -> FIRRecipeResult:
    normalized = FIRRecipe.from_dict(recipe) if isinstance(recipe, dict) else recipe
    return _generate_fir_recipe(normalized)


def verified_fir_recipe_result(recipe: dict[str, Any], expected_hash: str) -> FIRRecipeResult:
    """Recover unversioned v2 endpoint changes only with an exact saved hash."""
    normalized = FIRRecipe.from_dict(recipe)
    generated = _generate_fir_recipe(normalized)
    if not expected_hash or generated.coefficient_hash == expected_hash:
        return generated
    if normalized.algorithm_version in {"weighted-complex-ls-v1", "weighted-complex-ls-v2"}:
        legacy = _generate_fir_recipe(normalized, legacy_acoustic_boundary=True)
        if legacy.coefficient_hash == expected_hash:
            return legacy
    raise ValueError("FIR recipe coefficient hash mismatch。PhaseEQでこのChannelを開き、マルチウェイへ再送信してください。")


def _generate_fir_recipe(normalized: FIRRecipe, *, legacy_acoustic_boundary: bool = False) -> FIRRecipeResult:
    config = DesignConfig.from_dict(normalized.config).normalized()
    # Shaping belongs to the versioned design input, never the stored speaker
    # response. Older recipes retain the algorithm used for their saved hash.
    if normalized.algorithm_version != FIR_RECIPE_ALGORITHM_VERSION:
        config = replace(config, auto_gain_input_shaping=InputShapingSettings())
    if legacy_acoustic_boundary:
        config = replace(config, acoustic_boundary_continuation_enabled=False)
    config = replace(config, fir_generation_grid=(
        "legacy-v1" if normalized.algorithm_version == "weighted-complex-ls-v1" else "multiway-v1"))
    if normalized.bypass:
        # The transport still requires the assigned tap count.  Composite uses
        # the explicit bypass flag and never inserts these coefficients into
        # the signal path.  For odd taps this is the centered delta; the same
        # versioned flat design keeps legacy even-tap transport deterministic.
        coefficients = design_phase_fir(
            replace(config, speaker_response=None, target_response=None),
            compute_group_delay=False, apply_linear_fir_overlay=False,
        ).fir
    else:
        design = design_phase_fir(
            config, compute_group_delay=False, apply_linear_fir_overlay=False,
        )
        coefficients = design.fir
        post = normalized.postprocess
        coefficients = output_fir_for_export_report(
            coefficients,
            output_fir_polarity_invert=post.polarity_invert,
            output_fir_remove_nyquist_enabled=post.remove_nyquist_enabled,
            output_fir_remove_nyquist_strength=post.remove_nyquist_strength,
            fir_auto_gain_enabled=post.auto_gain_enabled,
        ).fir
    values = np.ascontiguousarray(coefficients, dtype="<f8")
    return FIRRecipeResult(
        coefficients=values,
        recipe_hash=normalized.content_hash,
        coefficient_hash=hashlib.sha256(values.tobytes()).hexdigest(),
        bypass=normalized.bypass,
        target_dc_abs=(1.0 if normalized.bypass else float(10.0 ** (design.target_gain_db[0] / 20.0))),
    )


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "FIRRecipe", "FIRRecipePostprocess", "FIRRecipeResult",
    "FIR_RECIPE_ALGORITHM_ID", "FIR_RECIPE_ALGORITHM_VERSION",
    "FIR_RECIPE_FORMAT", "FIR_RECIPE_FORMAT_VERSION",
    "fir_eq_is_active", "generate_fir_recipe", "recipe_from_config",
    "verified_fir_recipe_result",
]
