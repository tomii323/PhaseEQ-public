"""Routing for Studio settings and the matching PhaseEQ correction return."""
from crossover_engine.iir import IIRCrossoverConfig
from crossover_engine.alignment import separate_polarity


def definition(row):
    return {
        "acoustic_target": bool(row.get("phase_alignment_acoustic_target", False)),
        "polarity": int(row.get("polarity", 1)),
        "allpass": [item.to_dict() if hasattr(item, "to_dict") else dict(item)
                    for item in row.get("auto_alignment_allpass", ())],
    }


def recipe_definition(row):
    value = definition(row)
    return value if value["acoustic_target"] or value["polarity"] != 1 or value["allpass"] else None


def _target_definition(value, recipe):
    if not isinstance(value, dict):
        return value
    value = dict(value)
    if separate_polarity(recipe):
        value.pop("polarity", None)
    return value


def _target_recipe(recipe):
    if not isinstance(recipe, dict):
        return recipe
    result = dict(recipe)
    if "phase_alignment" in result:
        result["phase_alignment"] = _target_definition(result["phase_alignment"], recipe)
    return result


def target_applied(row):
    returned = row.get("phaseeq_alignment_recipe")
    return bool(row.get("phase_alignment_acoustic_target", False)
                and row.get("phaseeq_alignment_target_applied", False)
                and _target_definition(row.get("phaseeq_alignment_definition"), returned)
                == _target_definition(definition(row), returned)
                and ("phase_alignment_recipe" not in row
                     or _target_recipe(row["phase_alignment_recipe"]) == _target_recipe(returned)))


def pending(row):
    current = row.get("phase_alignment_recipe") or {}
    returned = row.get("phaseeq_alignment_recipe") or {}
    if separate_polarity(current) and returned and not separate_polarity(returned):
        if any(b.get("acoustic_target", False) for r in (current, returned)
               for b in r.get("boundaries", ())):
            return True
    previous = row.get("phaseeq_alignment_definition") or {}
    if row.get("phase_alignment_acoustic_target", False):
        return not target_applied(row)
    return bool(previous.get("acoustic_target", False))


def target_enabled(row):
    return bool(row.get("phase_alignment_acoustic_target", False))


def output_settings(row):
    """Return physical stages; user settings remain available for persistence."""
    config = row.get("iir_crossover", IIRCrossoverConfig())
    sections = () if target_enabled(row) else tuple(row.get("auto_alignment_allpass", ()))
    return config, int(row.get("polarity", 1)), sections


def receive(row, result):
    recipe = (result.get("fir_recipe") or {}).get("config", {}).get("band_split_recipe") or {}
    row["phaseeq_alignment_definition"] = recipe.get("phase_alignment")
    row["phaseeq_alignment_recipe"] = recipe
    row["phaseeq_alignment_target_applied"] = bool(result.get("alignment_target_applied", False))
