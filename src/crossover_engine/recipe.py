"""Portable boundary recipes with explicit topology and derived coefficients."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .bank import MODE_BANDS, _mode_boundary_values, generate_split_filter_bank
from .filters import kaiser_overlap_edges_hz, odd_number
from .lr2_taps import ALGORITHM_VERSION, automatic_lr2_taps
from .lr4_taps import automatic_lr4_taps

LR4_TARGET_ALGORITHM_VERSION = "multiway-v3-lr4-acoustic-target"
ALIGNMENT_ALGORITHM_VERSION = "multiway-v4-phase-alignment"
POLARITY_ALGORITHM_VERSION = "multiway-v5-crossover-polarity"

METHODS = ("Kaiser FIR", "Linear-phase LR2 FIR", "Linear-phase LR4 FIR", "LR2", "LR4", "Through")


def recipe_hash(recipe):
    return hashlib.sha256(json.dumps(
        recipe, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def from_studio(mode, sample_rate, conf, way, *, fir_enabled=True, phase_alignment=None):
    cycles = float(conf.get("cycles", 3.5))
    beta = float(conf.get("beta", 12.0))
    frequencies = conf.get("cross_freqs", [])
    if mode == "2Way":
        values = ((cycles,), (beta,), (conf.get("kaiser_overlap_oct", 0.0),))
    elif mode == "3Way":
        values = (
            tuple(conf.get("cycles_" + s, cycles) for s in ("low", "high")),
            tuple(conf.get("beta_" + s, beta) for s in ("low", "high")),
            tuple(conf.get("kaiser_overlap_" + s + "_oct", 0.0) for s in ("low_mid", "mid_high")),
        )
    elif mode == "3Way+SUB":
        suffixes = ("sub_low", "low_mid", "mid_high")
        values = (
            tuple(conf.get("cycles_" + s, cycles) for s in suffixes),
            tuple(conf.get("beta_" + s, beta) for s in suffixes),
            tuple(conf.get("kaiser_overlap_" + s + "_oct", 0.0) for s in suffixes),
        )
    else:
        values = _mode_boundary_values(mode, conf)
    methods = conf.get("crossover_methods", ["Kaiser FIR"] * len(frequencies))
    recipe = dict(format="phaseeq-band-split", format_version=1,
                  algorithm_version=POLARITY_ALGORITHM_VERSION, sample_rate_hz=int(sample_rate),
                  layout=mode, way=way, ordered_ways=list(MODE_BANDS[mode]),
                  fir_enabled=bool(fir_enabled),
                  lr2_auto_polarity=bool(conf.get("iir_lr2_auto_polarity", True)),
                  boundaries=[dict(id=f"boundary-{i}", base_crossover_hz=float(fc),
                                   overlap_oct=float(values[2][i]), cycles=float(values[0][i]),
                                   beta=float(values[1][i]), method=methods[i],
                                   acoustic_target=bool((conf.get("boundary_acoustic_targets", []) + [False]*len(frequencies))[i]))
                              for i, fc in enumerate(frequencies)])
    if phase_alignment is not None:
        recipe["phase_alignment"] = dict(phase_alignment)
    validate(recipe)
    return recipe


def validate(recipe):
    if not isinstance(recipe, dict):
        raise ValueError("Band split recipe must be an object")
    if ((recipe.get("format"), recipe.get("format_version")) != ("phaseeq-band-split", 1)
            or recipe.get("algorithm_version") not in {"multiway-v1", ALGORITHM_VERSION, LR4_TARGET_ALGORITHM_VERSION, ALIGNMENT_ALGORITHM_VERSION, POLARITY_ALGORITHM_VERSION}):
        raise ValueError("Unsupported band split recipe version")
    ways = MODE_BANDS.get(recipe.get("layout"))
    if not ways or recipe.get("ordered_ways") != ways or recipe.get("way") not in ways:
        raise ValueError("Band split topology mismatch")
    fs = recipe.get("sample_rate_hz")
    boundaries = recipe.get("boundaries")
    if (not isinstance(fs, int) or isinstance(fs, bool) or fs <= 0
            or not isinstance(boundaries, list) or len(boundaries) != len(ways)-1):
        raise ValueError("Invalid band split rate or boundary count")
    if not isinstance(recipe.get("lr2_auto_polarity"), bool):
        raise ValueError("Band split polarity selection must be boolean")
    if not isinstance(recipe.get("fir_enabled"), bool):
        raise ValueError("Band split FIR enable selection must be boolean")
    if "phase_alignment" in recipe:
        from .alignment import validate_alignment
        if recipe["algorithm_version"] not in {ALIGNMENT_ALGORITHM_VERSION, POLARITY_ALGORITHM_VERSION}:
            raise ValueError("Phase alignment requires a v4 or v5 recipe")
        validate_alignment(recipe["phase_alignment"], fs)
    previous = 0.0
    identifiers = set()
    for b in boundaries:
        if not isinstance(b, dict) or not isinstance(b.get("id"), str) or not b["id"] or b["id"] in identifiers:
            raise ValueError("Invalid or duplicate band split boundary ID")
        identifiers.add(b["id"])
        try:
            fc, overlap, cycles, beta = (float(b[k]) for k in (
                "base_crossover_hz", "overlap_oct", "cycles", "beta"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing or invalid band split boundary input") from exc
        if not all(np.isfinite(v) for v in (fc, overlap, cycles, beta)):
            raise ValueError("Non-finite band split input")
        if not previous < fc < fs/2 or not -1 <= overlap <= 1 or cycles <= 0 or beta < 0:
            raise ValueError("Invalid band split boundary")
        if b.get("method") not in METHODS:
            raise ValueError("Unknown band split method")
        if (b.get("acoustic_target", False) or b["method"] == "Linear-phase LR4 FIR") and recipe["algorithm_version"] not in {LR4_TARGET_ALGORITHM_VERSION, ALIGNMENT_ALGORITHM_VERSION, POLARITY_ALGORITHM_VERSION}:
            raise ValueError("LR4／音響ターゲットには新しい帯域分割Recipeが必要です。")
        if not isinstance(b.get("acoustic_target", False), bool):
            raise ValueError("音響ターゲットの指定はON/OFFです。")
        if b.get("acoustic_target", False) and b["method"] in {"Kaiser FIR", "Through"}:
            raise ValueError("音響ターゲットにはLR2／LR4を選択してください。")
        previous = fc


def studio_config(recipe):
    validate(recipe)
    b = recipe["boundaries"]
    conf = dict(lr2_auto_taps=recipe["algorithm_version"] != "multiway-v1", cross_freqs=[v["base_crossover_hz"] for v in b],
                crossover_methods=[v["method"] for v in b],
                boundary_cycles=[v["cycles"] for v in b],
                boundary_betas=[v["beta"] for v in b],
                boundary_overlaps_oct=[v["overlap_oct"] for v in b])
    conf["boundary_acoustic_targets"] = [v.get("acoustic_target", False) for v in b]
    if b:
        conf.update(cycles=b[0]["cycles"], beta=b[0]["beta"], kaiser_overlap_oct=b[0]["overlap_oct"])
    suffixes = ("low", "high") if recipe["layout"] == "3Way" else ("sub_low", "low_mid", "mid_high")
    overlap_suffixes = ("low_mid", "mid_high") if recipe["layout"] == "3Way" else suffixes
    for v, s, o in zip(b, suffixes, overlap_suffixes):
        conf["cycles_"+s] = v["cycles"]
        conf["beta_"+s] = v["beta"]
        conf["kaiser_overlap_"+o+"_oct"] = v["overlap_oct"]
    return conf


def generate(recipe):
    conf = studio_config(recipe)
    return generate_split_filter_bank(recipe["layout"], recipe["sample_rate_hz"],
                                      conf, conf["crossover_methods"])[1]


def iir_config(recipe):
    from .iir import realized_exclusive_iir_config
    validate(recipe)
    b = recipe["boundaries"]
    return realized_exclusive_iir_config(
        way=recipe["way"], ordered_ways=tuple(recipe["ordered_ways"]),
        crossover_frequencies_hz=tuple(v["base_crossover_hz"] for v in b),
        methods=tuple("Through" if v.get("acoustic_target", False) else v["method"] for v in b),
        overlap_oct=tuple(v["overlap_oct"] for v in b),
        lr2_auto_polarity=recipe["lr2_auto_polarity"],
    )


def boundary_details(recipe):
    validate(recipe)
    result = []
    for b in recipe["boundaries"]:
        lp, hp = kaiser_overlap_edges_hz(b["base_crossover_hz"], b["overlap_oct"])
        taps = (automatic_lr4_taps(recipe["sample_rate_hz"], b["base_crossover_hz"], b["overlap_oct"]).taps
                if b["method"] == "Linear-phase LR4 FIR" else automatic_lr2_taps(recipe["sample_rate_hz"], b["base_crossover_hz"], b["overlap_oct"]).taps
                if b["method"] == "Linear-phase LR2 FIR" and recipe["algorithm_version"] != "multiway-v1"
                else odd_number(recipe["sample_rate_hz"] / max(b["base_crossover_hz"], 1) * b["cycles"]))
        result.append({**b, "lowpass_cutoff_hz": lp, "highpass_cutoff_hz": hp,
                       "natural_taps": taps
                       if b["method"] in METHODS[:3] else None})
    return result


def adjacent_filter_parameters(recipe):
    """Editable single-channel projection; never substitutes for the full bank."""
    details = boundary_details(recipe)
    index = recipe["ordered_ways"].index(recipe["way"])
    names = dict(zip(METHODS[:-1], ("kaiser", "linear_phase_lr2", "linear_phase_lr4", "iir_lr2", "iir_lr4")))
    return [dict(mode=mode, response=names[d["method"]],
                 fc=d["highpass_cutoff_hz" if mode == "hp" else "lowpass_cutoff_hz"],
                 cycles=d["cycles"], beta=d["beta"], acoustic_target=d.get("acoustic_target", False),
                 base_crossover_hz=d["base_crossover_hz"], overlap_oct=d["overlap_oct"],
                 lr2_auto_taps=recipe["algorithm_version"] != "multiway-v1")
            for mode, j in (("hp", index-1), ("lp", index))
            if 0 <= j < len(details) for d in [details[j]] if d["method"] != "Through"]


def generate_channel(recipe):
    """Return coefficients and provenance without applying output tap fitting."""
    from .iir import iir_crossover_sos
    fir = generate(recipe)[recipe["way"]]
    config = iir_config(recipe)
    sos = iir_crossover_sos(config, recipe["sample_rate_hz"])
    from .alignment import alignment_sos, output_polarity, manual_polarity_sos
    sos = np.concatenate((sos, alignment_sos(recipe), manual_polarity_sos(recipe)))
    return dict(fir=fir, sos=sos, lr2_polarity=output_polarity(recipe),
                boundaries=boundary_details(recipe), recipe_hash=recipe_hash(recipe),
                fir_sha256=hashlib.sha256(np.asarray(fir, dtype="<f8").tobytes()).hexdigest(),
                sos_sha256=hashlib.sha256(np.asarray(sos, dtype="<f8").tobytes()).hexdigest())


def verify_iir_payload(recipe, payload):
    from .iir import iir_crossover_sos
    expected = iir_config(recipe)
    if not isinstance(payload, dict):
        raise ValueError("Band split recipe requires IIR stage declaration")
    actual = np.asarray(payload.get("sos", []), dtype=float)
    if actual.size == 0:
        actual = np.empty((0, 6))
    if not np.array_equal(actual, iir_crossover_sos(expected, recipe["sample_rate_hz"])):
        raise ValueError("Band split recipe SOS mismatch")
    for key, value in expected.to_dict().items():
        if payload.get(key) != value:
            raise ValueError(f"Band split recipe IIR definition mismatch: {key}")
    if payload.get("coefficient_sample_rate_hz") != recipe["sample_rate_hz"]:
        raise ValueError("Band split recipe IIR sample rate mismatch")
