"""Fetch and parse speaker specification pages for user-managed records."""

from __future__ import annotations

from io import BytesIO
from html.parser import HTMLParser
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from utils.speaker_db import SpeakerSpecRecord


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if text:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def fetch_external_speaker_spec_url(url: str, *, timeout_sec: float = 20.0) -> str:
    normalized = str(url).strip()
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must start with http:// or https://")
    request = Request(
        normalized,
        headers={
            "User-Agent": "PhaseEQ/1.0 speaker-spec-import",
            "Accept": "text/html,text/plain,application/pdf,application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read(5_000_000)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {normalized}") from exc
    except URLError as exc:
        raise RuntimeError(f"Fetch failed: {exc.reason}") from exc
    if _is_pdf_response(normalized, content_type, data):
        try:
            return _extract_pdf_text(data)
        except RuntimeError as exc:
            fallback_url = _pdf_fallback_url(normalized)
            if not fallback_url:
                raise
            try:
                return fetch_external_speaker_spec_url(fallback_url, timeout_sec=timeout_sec)
            except Exception as fallback_exc:
                raise RuntimeError(f"{exc} Fallback URL also failed: {fallback_exc}") from fallback_exc
    encoding = _charset_from_content_type(content_type) or "utf-8"
    return data.decode(encoding, errors="replace")


def parse_external_speaker_spec(
    url: str,
    html_or_text: str,
    *,
    brand_hint: str = "",
    model_hint: str = "",
) -> SpeakerSpecRecord:
    text = _visible_text(html_or_text)
    title = _infer_title(text, url)
    product_context = _product_context(text, title)
    values = _extract_spec_values(text)
    brand = str(brand_hint).strip() or _infer_brand(product_context, title, url)
    model = str(model_hint).strip() or _infer_model(product_context, title, brand, url)
    return SpeakerSpecRecord(
        id="",
        brand=brand,
        model=model,
        driver_type=_extract_type(product_context, title=title),
        nominal_diameter_mm=values.get("nominal_diameter_mm"),
        nominal_impedance_ohm=values.get("nominal_impedance_ohm"),
        fs_hz=values.get("fs_hz"),
        qts=values.get("qts"),
        vas_l=values.get("vas_l"),
        re_ohm=values.get("re_ohm"),
        le_mh=values.get("le_mh"),
        sd_cm2=values.get("sd_cm2"),
        xmax_mm=values.get("xmax_mm"),
        spl_db=values.get("spl_db"),
        pmax_w=values.get("pmax_w"),
        overall_diameter_mm=values.get("overall_diameter_mm"),
        baffle_cutout_mm=values.get("baffle_cutout_mm"),
        mounting_depth_mm=values.get("mounting_depth_mm"),
        source_name=_source_name_from_url(url),
        source_url=str(url).strip(),
        source_type=_source_type_from_url(url),
        redistributable=False,
        user_verified=False,
        note="Fetched from external URL. Verify before saving.",
        raw_text=text[:20_000],
    )


def parse_pasted_speaker_specs(
    raw_text: str,
    *,
    brand_hint: str = "",
    source_name: str = "",
    source_url: str = "",
) -> list[SpeakerSpecRecord]:
    """Parse one or more user-managed speaker specs from pasted text.

    Supported inputs are intentionally simple and reviewable:
    - JSON object keyed by model name.
    - A table where the first column is the parameter name and the remaining
      columns are model names.
    """
    text = _visible_text(raw_text).strip()
    if not text:
        return []
    records = _parse_pasted_json_specs(
        text,
        brand_hint=brand_hint,
        source_name=source_name,
        source_url=source_url,
    )
    if records:
        return records
    return _parse_pasted_parameter_table(
        text,
        brand_hint=brand_hint,
        source_name=source_name,
        source_url=source_url,
    )


def has_external_spec_data(record: SpeakerSpecRecord) -> bool:
    return any(
        value is not None
        for value in (
            record.nominal_impedance_ohm,
            record.nominal_diameter_mm,
            record.fs_hz,
            record.qts,
            record.vas_l,
            record.re_ohm,
            record.le_mh,
            record.sd_cm2,
            record.xmax_mm,
            record.spl_db,
            record.pmax_w,
            record.overall_diameter_mm,
            record.baffle_cutout_mm,
            record.mounting_depth_mm,
        )
    )


def _parse_pasted_json_specs(
    text: str,
    *,
    brand_hint: str,
    source_name: str,
    source_url: str,
) -> list[SpeakerSpecRecord]:
    payload_text = _first_json_object(text)
    if not payload_text:
        return []
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    speaker_items = payload.get("speakers")
    if isinstance(speaker_items, list):
        return _parse_pasted_speaker_list_json(
            speaker_items,
            text=text,
            brand_hint=brand_hint,
            source_metadata=payload.get("source", {}),
            source_name=source_name,
            source_url=source_url,
        )
    records: list[SpeakerSpecRecord] = []
    for model, values in payload.items():
        if not isinstance(values, dict):
            continue
        record = _speaker_spec_from_pasted_values(
            model=str(model),
            values=values,
            brand_hint=brand_hint,
            raw_text=text,
            source_name=source_name,
            source_url=source_url,
        )
        if has_external_spec_data(record):
            records.append(record)
    return records


def _parse_pasted_speaker_list_json(
    speaker_items: list[object],
    *,
    text: str,
    brand_hint: str,
    source_metadata: object,
    source_name: str,
    source_url: str,
) -> list[SpeakerSpecRecord]:
    records: list[SpeakerSpecRecord] = []
    source_info = source_metadata if isinstance(source_metadata, dict) else {}
    for item in speaker_items:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model", "")).strip()
        if not model:
            continue
        values = _speaker_list_item_values(item)
        manufacturer = str(item.get("manufacturer", "") or source_info.get("manufacturer", "")).strip()
        product_brand = str(item.get("brand", "") or source_info.get("brand", "")).strip()
        record_brand = _combined_brand_label(manufacturer or brand_hint, product_brand)
        item_source_url = source_url or str(item.get("datasheet", "") or item.get("manual", "")).strip()
        item_source_name = source_name or (_source_name_from_url(item_source_url) if item_source_url else "Raw pasted text")
        overview = _speaker_list_item_overview(item)
        metadata = _speaker_list_item_metadata(item)
        metadata.update(
            {
                "manufacturer": manufacturer,
                "brand": product_brand,
                "series": item.get("series", ""),
                "category": item.get("category", ""),
                "type": item.get("type", ""),
                "source": source_info,
            }
        )
        record = _speaker_spec_from_pasted_values(
            model=model,
            values=values,
            brand_hint=record_brand,
            raw_text=text,
            source_name=item_source_name,
            source_url=item_source_url,
            overview=overview,
            metadata=metadata,
        )
        if has_external_spec_data(record):
            records.append(record)
    return records


def _speaker_list_item_values(item: dict[str, object]) -> dict[str, object]:
    values: dict[str, object] = {}
    for candidate in (
        item.get("thiele_small", {}),
        _nested_dict(item, "woofer", "thiele_small"),
        _nested_dict(item, "driver", "thiele_small"),
    ):
        if isinstance(candidate, dict):
            values.update(candidate)
    for key, value in item.items():
        if key not in _SPEAKER_LIST_STRUCTURAL_KEYS and not isinstance(value, (dict, list)):
            values[key] = value
    system = item.get("system", {})
    if isinstance(system, dict):
        _copy_first_present(
            values,
            system,
            {
                "nominal_impedance_ohm": ("nominal_impedance_ohm",),
                "sensitivity_dB": ("sensitivity_dB",),
                "diameter_mm": ("driver_diameter_mm", "woofer_diameter_mm"),
            },
        )
        power = system.get("power", {})
        if isinstance(power, dict):
            _copy_first_present(
                values,
                power,
                {
                    "power_rms_W": ("rated_input_W", "rms_input_W", "power_rms_W"),
                    "power_peak_W": ("maximum_input_W", "max_input_W", "power_peak_W"),
                },
            )
        enclosure = system.get("recommended_enclosure", {})
        if isinstance(enclosure, dict) and "recommended_enclosure_L" not in values:
            volume = enclosure.get("volume_L")
            if volume not in ("", None):
                values["recommended_enclosure_L"] = volume
    return values


_SPEAKER_LIST_STRUCTURAL_KEYS = {
    "manufacturer",
    "brand",
    "series",
    "model",
    "category",
    "type",
    "thiele_small",
    "woofer",
    "tweeter",
    "driver",
    "system",
    "operating_conditions",
    "country",
    "release_year",
    "datasheet",
    "manual",
    "image",
    "status",
    "frequency_response_Hz",
    "recommended_highpass_Hz",
    "recommended_lowpass_Hz",
    "recommended_slope_dB_oct",
    "recommended_enclosure_L",
    "mass_kg",
    "package_mass_kg",
    "crossover",
}


def _nested_dict(source: dict[str, object], *keys: str) -> dict[str, object]:
    current: object = source
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _copy_first_present(
    target: dict[str, object],
    source: dict[str, object],
    aliases: dict[str, tuple[str, ...]],
) -> None:
    for target_key, source_keys in aliases.items():
        if target_key in target:
            continue
        for source_key in source_keys:
            value = source.get(source_key)
            if value not in ("", None):
                target[target_key] = value
                break


def _speaker_list_item_metadata(item: dict[str, object]) -> dict[str, object]:
    metadata = {
        "country": item.get("country", ""),
        "release_year": item.get("release_year", ""),
        "status": item.get("status", ""),
        "datasheet": item.get("datasheet", ""),
        "manual": item.get("manual", ""),
        "image": item.get("image", ""),
        "frequency_response_Hz": item.get("frequency_response_Hz", ""),
        "recommended_highpass_Hz": item.get("recommended_highpass_Hz", ""),
        "recommended_lowpass_Hz": item.get("recommended_lowpass_Hz", ""),
        "recommended_slope_dB_oct": item.get("recommended_slope_dB_oct", ""),
        "recommended_enclosure_L": item.get("recommended_enclosure_L", ""),
        "mass_kg": item.get("mass_kg", ""),
        "package_mass_kg": item.get("package_mass_kg", ""),
        "crossover": item.get("crossover", ""),
        "system": item.get("system", ""),
        "operating_conditions": item.get("operating_conditions", ""),
    }
    system = item.get("system", {})
    if isinstance(system, dict):
        if not metadata["frequency_response_Hz"]:
            metadata["frequency_response_Hz"] = system.get("frequency_response_Hz", "")
        if not metadata["mass_kg"]:
            metadata["mass_kg"] = system.get("mass_with_accessories_per_unit_kg", system.get("mass_with_accessories_kg", ""))
        if not metadata["package_mass_kg"]:
            metadata["package_mass_kg"] = system.get("total_package_mass_kg", "")
        if not metadata["crossover"]:
            metadata["crossover"] = system.get("crossover_network", "")
        enclosure = system.get("recommended_enclosure", {})
        if isinstance(enclosure, dict) and not metadata["recommended_enclosure_L"]:
            metadata["recommended_enclosure_L"] = enclosure.get("volume_L", "")
    return metadata


def _combined_brand_label(manufacturer: str, brand: str) -> str:
    clean_manufacturer = str(manufacturer).strip()
    clean_brand = str(brand).strip()
    if clean_manufacturer and clean_brand and clean_manufacturer.lower() != clean_brand.lower():
        return f"{clean_manufacturer} / {clean_brand}"
    return clean_manufacturer or clean_brand


def _speaker_list_item_overview(item: dict[str, object]) -> str:
    parts = [
        str(item.get("model", "")).strip(),
        str(item.get("category", "")).strip(),
        str(item.get("type", "")).strip(),
    ]
    if isinstance(item.get("woofer", {}), dict):
        parts.append("Woofer")
    system = item.get("system", {})
    system_diameter = ""
    if isinstance(system, dict):
        system_diameter = system.get("driver_diameter_mm", system.get("woofer_diameter_mm", ""))
    diameter = item.get("diameter_mm", item.get("nominal_diameter_mm", system_diameter))
    if diameter not in ("", None):
        parts.append(f"{diameter} mm")
    return "\n".join(part for part in parts if part)


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _parse_pasted_parameter_table(
    text: str,
    *,
    brand_hint: str,
    source_name: str,
    source_url: str,
) -> list[SpeakerSpecRecord]:
    rows = [_split_pasted_table_row(line) for line in text.splitlines()]
    rows = [row for row in rows if len(row) >= 3]
    header_index = next((index for index, row in enumerate(rows) if _is_pasted_model_header(row)), -1)
    if header_index < 0:
        return []
    header = rows[header_index]
    models = [cell.strip() for cell in header[1:] if cell.strip()]
    if not models:
        return []
    values_by_model: dict[str, dict[str, float]] = {model: {} for model in models}
    for row in rows[header_index + 1 :]:
        field = _pasted_field_key(row[0])
        if not field:
            continue
        for index, model in enumerate(models, start=1):
            if index >= len(row):
                continue
            value = _parse_pasted_number(row[index])
            if value is None:
                continue
            values_by_model[model][field] = value
    overview = _pasted_overview_by_model(text, models)
    records: list[SpeakerSpecRecord] = []
    for model in models:
        record = _speaker_spec_from_pasted_values(
            model=model,
            values=values_by_model.get(model, {}),
            brand_hint=brand_hint,
            raw_text=text,
            source_name=source_name,
            source_url=source_url,
            overview=overview.get(model, ""),
        )
        if has_external_spec_data(record):
            records.append(record)
    return records


def _split_pasted_table_row(line: str) -> list[str]:
    clean = str(line).strip()
    if not clean:
        return []
    if "\t" in clean:
        return [cell.strip() for cell in clean.split("\t") if cell.strip()]
    return [cell.strip() for cell in re.split(r"\s{2,}", clean) if cell.strip()]


def _is_pasted_model_header(row: list[str]) -> bool:
    first = row[0].strip().lower()
    if first not in {"item", "parameter", "parameters", "項目"}:
        return False
    return len(row) >= 2 and any(_looks_like_model_name(cell) for cell in row[1:])


def _looks_like_model_name(value: str) -> bool:
    text = str(value).strip()
    return bool(re.search(r"[A-Za-z]{1,}[-/][A-Za-z0-9]", text) or re.search(r"[A-Za-z]{2,}\d", text))


def _pasted_field_key(label: str) -> str:
    normalized = _normalize_label(label)
    alias_prefixes = [
        ("re_ohm", ("revc", "reohm", "re")),
        ("le_mh", ("levc", "lemh", "le")),
        ("fs_hz", ("fshz", "fs")),
        ("qts", ("qts", "qt")),
        ("vas_l", ("vasl", "vas")),
        ("sd_m2", ("sdm2",)),
        ("sd_cm2", ("sdcm2", "sd")),
        ("nominal_impedance_ohm", ("nominalimpedance", "impedance", "zohm")),
        ("xmax_mm", ("xmax",)),
        ("hvc_mm", ("hvc", "voicecoilheight", "heightofvoicecoil")),
        ("hag_mm", ("hag", "hg", "airgapheight", "heightofairgap")),
        ("spl_db", ("spldb", "sensitivity")),
        ("pmax_w", ("pmax",)),
        ("overall_diameter_mm", ("overalldiameter",)),
        ("baffle_cutout_mm", ("bafflecutout", "cutout")),
        ("mounting_depth_mm", ("mountingdepth", "depth")),
    ]
    for key, prefixes in alias_prefixes:
        if any(normalized.startswith(prefix) for prefix in prefixes):
            return key
    patterns = [
        ("re_ohm", r"\bre(?:vc)?\b|dcresistance"),
        ("le_mh", r"\ble(?:vc)?\b|voicecoilinductance"),
        ("fs_hz", r"\bfs\b|freeairresonance|resonancefrequency"),
        ("qts", r"\bqts\b|\bqt\b|totalq"),
        ("vas_l", r"\bvas\b|equivalentvolume"),
        ("sd_cm2", r"\bsd\b|effectivepistonarea"),
        ("nominal_impedance_ohm", r"\bz\b|nominalimpedance|impedance"),
        ("xmax_mm", r"\bxmax\b|linearcoiltravel"),
        ("hvc_mm", r"\bhvc\b|voicecoilheight|heightofvoicecoil"),
        ("hag_mm", r"\bhag\b|\bhg\b|airgapheight|heightofairgap"),
        ("spl_db", r"\bspl\b|sensitivity|出力音圧"),
        ("pmax_w", r"\bpmax\b|ratedpower|powerhandling|入力"),
        ("overall_diameter_mm", r"overalldiameter|外径"),
        ("baffle_cutout_mm", r"cutout|バッフル開口"),
        ("mounting_depth_mm", r"mountingdepth|depth|奥行"),
    ]
    for key, pattern in patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return key
    return ""


def _normalize_label(label: str) -> str:
    text = str(label).strip().lower()
    text = text.replace("Ω", "ohm").replace("²", "2")
    return re.sub(r"[^a-z0-9_\u3040-\u30ff\u4e00-\u9fff]+", "", text)


def _parse_pasted_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = _normalize_numeric_text(str(value))
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", text, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _normalize_numeric_text(value: str) -> str:
    text = str(value).strip()
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    text = text.translate(superscripts)
    text = re.sub(r"×\s*10([+-]?\d+)", r"e\1", text)
    text = re.sub(r"x\s*10([+-]?\d+)", r"e\1", text, flags=re.IGNORECASE)
    return text


def _speaker_spec_from_pasted_values(
    *,
    model: str,
    values: dict[str, object],
    brand_hint: str,
    raw_text: str,
    source_name: str,
    source_url: str,
    overview: str = "",
    metadata: dict[str, object] | None = None,
) -> SpeakerSpecRecord:
    mapped = _mapped_pasted_values(values)
    motor_info = _motor_info_from_dimensions(mapped)
    if "xmax_mm" not in mapped and "calculated_xmax_mm" in motor_info:
        mapped["xmax_mm"] = float(motor_info["calculated_xmax_mm"])
    if "nominal_diameter_mm" not in mapped:
        diameter = _diameter_from_model_context(model, overview)
        if diameter is not None:
            mapped["nominal_diameter_mm"] = diameter
    driver_type = _extract_type(overview, title=model)
    note_parts = ["Parsed from Raw pasted text. Verify before saving."]
    unsupported = _unsupported_pasted_keys(values)
    if unsupported:
        note_parts.append("Unsupported details kept in Raw text: " + ", ".join(unsupported))
    metadata_payload = dict(metadata or {})
    metadata_payload.update(motor_info)
    metadata_note = _speaker_metadata_note(metadata_payload)
    if metadata_note:
        note_parts.append(metadata_note)
    return SpeakerSpecRecord(
        id="",
        brand=str(brand_hint).strip(),
        model=str(model).strip(),
        driver_type=driver_type,
        nominal_diameter_mm=mapped.get("nominal_diameter_mm"),
        nominal_impedance_ohm=mapped.get("nominal_impedance_ohm"),
        fs_hz=mapped.get("fs_hz"),
        qts=mapped.get("qts"),
        vas_l=mapped.get("vas_l"),
        re_ohm=mapped.get("re_ohm"),
        le_mh=mapped.get("le_mh"),
        sd_cm2=mapped.get("sd_cm2"),
        xmax_mm=mapped.get("xmax_mm"),
        spl_db=mapped.get("spl_db"),
        pmax_w=mapped.get("pmax_w"),
        overall_diameter_mm=mapped.get("overall_diameter_mm"),
        baffle_cutout_mm=mapped.get("baffle_cutout_mm"),
        mounting_depth_mm=mapped.get("mounting_depth_mm"),
        source_name=str(source_name).strip() or "Raw pasted text",
        source_url=str(source_url).strip(),
        source_type="pasted_text",
        redistributable=True,
        user_verified=False,
        note=" ".join(note_parts),
        raw_text=str(raw_text)[:20_000],
    )


def _mapped_pasted_values(values: dict[str, object]) -> dict[str, float]:
    mapped: dict[str, float] = {}
    for key, value in values.items():
        target = _pasted_value_key(str(key))
        if not target:
            continue
        numeric = _parse_pasted_number(value)
        if numeric is None:
            continue
        mapped[target] = numeric
    if "sd_m2" in mapped:
        mapped["sd_cm2"] = mapped.pop("sd_m2") * 10_000.0
    return mapped


def _motor_info_from_dimensions(values: dict[str, float]) -> dict[str, float | str]:
    hvc = values.get("hvc_mm")
    hag = values.get("hag_mm")
    if hvc is None or hag is None:
        return {}
    diff = float(hvc) - float(hag)
    if abs(diff) < 1e-9:
        topology = "equal"
    elif diff > 0:
        topology = "overhung"
    else:
        topology = "underhung"
    return {
        "hvc_mm": float(hvc),
        "hag_mm": float(hag),
        "calculated_xmax_mm": abs(diff) / 2.0,
        "motor_topology": topology,
    }


def _pasted_value_key(key: str) -> str:
    normalized = _normalize_label(key)
    aliases = {
        "reohm": "re_ohm",
        "re_ohm": "re_ohm",
        "revc": "re_ohm",
        "re": "re_ohm",
        "lemh": "le_mh",
        "le_mh": "le_mh",
        "levc": "le_mh",
        "le": "le_mh",
        "fshz": "fs_hz",
        "fs_hz": "fs_hz",
        "fs": "fs_hz",
        "qts": "qts",
        "vasl": "vas_l",
        "vas_l": "vas_l",
        "vas": "vas_l",
        "sdcm2": "sd_cm2",
        "sd_cm2": "sd_cm2",
        "sdm2": "sd_m2",
        "sd_m2": "sd_m2",
        "sd": "sd_cm2",
        "xmaxmm": "xmax_mm",
        "xmax_mm": "xmax_mm",
        "hvcmm": "hvc_mm",
        "hvc_mm": "hvc_mm",
        "voicecoilheightmm": "hvc_mm",
        "voice_coil_height_mm": "hvc_mm",
        "heightofvoicecoilmm": "hvc_mm",
        "hagmm": "hag_mm",
        "hag_mm": "hag_mm",
        "hgmm": "hag_mm",
        "hg_mm": "hag_mm",
        "airgapheightmm": "hag_mm",
        "air_gap_height_mm": "hag_mm",
        "heightofairgapmm": "hag_mm",
        "spldb": "spl_db",
        "spl_db": "spl_db",
        "sensitivitydb": "spl_db",
        "sensitivity_db": "spl_db",
        "pmaxw": "pmax_w",
        "pmax_w": "pmax_w",
        "powerrmsw": "pmax_w",
        "power_rms_w": "pmax_w",
        "powerpeakw": "pmax_w",
        "power_peak_w": "pmax_w",
        "nominalimpedanceohm": "nominal_impedance_ohm",
        "nominal_impedance_ohm": "nominal_impedance_ohm",
        "nominaldiametermm": "nominal_diameter_mm",
        "nominal_diameter_mm": "nominal_diameter_mm",
        "diameter": "nominal_diameter_mm",
        "diametermm": "nominal_diameter_mm",
        "diameter_mm": "nominal_diameter_mm",
        "overalldiametermm": "overall_diameter_mm",
        "bafflecutoutmm": "baffle_cutout_mm",
        "mountingdepthmm": "mounting_depth_mm",
    }
    return aliases.get(normalized, _pasted_field_key(key))


def _unsupported_pasted_keys(values: dict[str, object]) -> list[str]:
    unsupported: list[str] = []
    for key in values:
        normalized = _normalize_label(str(key))
        if normalized in {
            "qms",
            "qes",
            "zmaxohm",
            "zmax",
            "rmsnsm",
            "rms_ns_m",
            "mmsg",
            "mms_g",
            "cmsmn",
            "cms_m_n",
            "bltm",
            "bl_tm",
            "displacementl",
            "displacement_l",
        }:
            unsupported.append(str(key))
    return unsupported[:12]


def _speaker_metadata_note(metadata: dict[str, object]) -> str:
    visible_keys = [
        "manufacturer",
        "brand",
        "series",
        "category",
        "type",
        "country",
        "release_year",
        "status",
        "datasheet",
        "manual",
        "image",
        "hvc_mm",
        "hag_mm",
        "calculated_xmax_mm",
        "motor_topology",
        "frequency_response_Hz",
        "recommended_highpass_Hz",
        "recommended_lowpass_Hz",
        "recommended_slope_dB_oct",
        "recommended_enclosure_L",
        "mass_kg",
        "package_mass_kg",
        "crossover",
    ]
    parts = [f"{key}: {_metadata_value_text(metadata[key])}" for key in visible_keys if _metadata_value_text(metadata.get(key, "")).strip()]
    return "Metadata: " + "; ".join(parts) if parts else ""


def _metadata_value_text(value: object) -> str:
    if value in ("", None):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value).strip()


def _pasted_overview_by_model(text: str, models: list[str]) -> dict[str, str]:
    overview: dict[str, str] = {model: "" for model in models}
    lines = [line.rstrip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        clean = line.strip().lstrip("*-・").strip()
        model = next((candidate for candidate in models if clean.startswith(candidate)), "")
        if not model:
            continue
        block = [clean]
        for next_line in lines[index + 1 : index + 8]:
            stripped = next_line.strip()
            if any(stripped.lstrip("*-・").strip().startswith(candidate) for candidate in models if candidate != model):
                break
            if stripped:
                block.append(stripped)
        overview[model] = "\n".join(block)
    return overview


def _diameter_from_model_context(model: str, context: str) -> float | None:
    source = "\n".join([str(model), str(context)])
    cm_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", source, re.IGNORECASE)
    if cm_match:
        return float(cm_match.group(1)) * 10.0
    inch_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:inch|inches|in|″|”|\")", source, re.IGNORECASE)
    if inch_match:
        return float(inch_match.group(1)) * 25.4
    pioneer_match = re.search(r"\bTS-[A-Z](\d{2})\d[A-Z]*\b", source, re.IGNORECASE)
    if pioneer_match:
        return float(pioneer_match.group(1)) * 10.0
    return None


def _visible_text(html_or_text: str) -> str:
    source = str(html_or_text)
    if "<" not in source and ">" not in source:
        return source
    parser = VisibleTextParser()
    parser.feed(source)
    parsed = parser.text()
    return parsed if parsed else source


def _is_pdf_response(url: str, content_type: str, data: bytes) -> bool:
    return (
        "application/pdf" in str(content_type).lower()
        or urlparse(str(url)).path.lower().endswith(".pdf")
        or data.startswith(b"%PDF")
    )


def _pdf_fallback_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.lower()
    if host == "scan-speak.dk" and path.startswith("/datasheet/pdf/") and path.endswith(".pdf"):
        model_slug = path.rsplit("/", 1)[-1].removesuffix(".pdf")
        if model_slug:
            return f"https://www.scan-speak.dk/product/{model_slug}/"
    return ""


def _extract_pdf_text(data: bytes) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("PDF import requires pdfplumber. Install requirements.txt before fetching PDF specs.") from exc

    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            parts = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise RuntimeError(f"PDF text extraction failed: {exc}") from exc
    text = "\n".join(part for part in parts if part.strip())
    if not text.strip():
        raise RuntimeError("PDF text extraction returned no readable text.")
    if not _pdf_text_looks_readable(text):
        raise RuntimeError("PDF text extraction was not readable enough for automatic import.")
    return text


def _pdf_text_looks_readable(text: str) -> bool:
    slash_tokens = len(re.findall(r"/(?:i?\d+|\d+)", text))
    cid_tokens = len(re.findall(r"\(cid:\d+\)", text))
    readable_source = re.sub(r"\(cid:\d+\)", " ", text)
    readable_tokens = len(re.findall(r"[A-Za-z]{2,}|[\u3040-\u30ff\u4e00-\u9fff]{2,}", readable_source))
    if slash_tokens > 20 and slash_tokens > readable_tokens * 3:
        return False
    if cid_tokens > 20 and cid_tokens > readable_tokens * 2:
        return False
    return True


def _charset_from_content_type(content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", str(content_type), re.IGNORECASE)
    return match.group(1).strip("\"'") if match else ""


def _source_name_from_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    return parsed.netloc.removeprefix("www.") or "External URL"


def _source_type_from_url(url: str) -> str:
    path = urlparse(str(url).strip()).path.lower()
    return "manufacturer_pdf" if path.endswith(".pdf") else "manufacturer_url"


def _infer_brand(text: str, title: str, url: str) -> str:
    brand_candidates = [
        "Scan-Speak",
        "SB Acoustics",
        "SBAcoustics",
        "Dayton Audio",
        "Peerless",
        "Wavecor",
        "Fostex",
        "Tang Band",
        "Markaudio",
        "SEAS",
        "Morel",
        "Accuton",
        "SPK AUDIO",
    ]
    source = "\n".join([title, text, url])
    for candidate in brand_candidates:
        pattern = re.escape(candidate).replace(r"\ ", r"[\s-]*")
        if re.search(pattern, source, re.IGNORECASE):
            return "SB Acoustics" if candidate == "SBAcoustics" else candidate
    match = re.search(r"Brand(?:（ブランド）)?\s*[-:：]?\s*([A-Za-z][A-Za-z0-9 .&+-]{1,40})", text)
    if match:
        return match.group(1).strip()
    return ""


def _infer_model(text: str, title: str, brand: str, url: str) -> str:
    source = "\n".join([title, text, url])
    scan_speak = _first_match(
        source,
        [
            r"\b(?:D|R|H|P)?\d{2}[A-Z]{0,2}/[0-9A-Z-]{4,}\b",
            r"\b\d{2}[A-Z]{1,3}/[0-9A-Z-]{4,}\b",
        ],
    )
    if scan_speak and re.search(r"Scan[\s-]*Speak", "\n".join([brand, source]), re.IGNORECASE):
        return scan_speak
    sb_model = _first_match(source, [r"\b(?:SATORI\s+)?[A-Z]{2,4}\d{2}[A-Z]*-[0-9]+(?:\s*/\s*[A-Za-z0-9]+)?\b"])
    if sb_model and re.search(r"SB\s*Acoustics|SBAcoustics", "\n".join([brand, source]), re.IGNORECASE):
        return re.sub(r"\s+", " ", sb_model).strip()
    cleaned_title = _clean_model_title(title, brand)
    return cleaned_title


def _first_match(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return ""


def _clean_model_title(title: str, brand: str) -> str:
    cleaned = re.split(r"\s*[|｜]\s*", title)[0]
    cleaned = re.split(r"\s+[–-]\s+", cleaned)[0]
    cleaned = re.sub(r"（ペア）|\(pair\)|ペア", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip() or title.strip()


def _infer_title(text: str, url: str) -> str:
    for line in (line.strip() for line in text.splitlines()):
        if line and len(line) <= 120:
            return line
    path_parts = [part for part in urlparse(str(url)).path.split("/") if part]
    return path_parts[-1].replace("-", " ").replace("_", " ") if path_parts else "Fetched speaker spec"


def _product_context(text: str, title: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    title_key = str(title).split(" - ")[0].split(" – ")[0].split("｜")[0].split("|")[0].strip()
    candidate_indexes: list[int] = []
    if title_key:
        candidate_indexes = [index for index, line in enumerate(lines) if title_key in line]
    if not candidate_indexes:
        return "\n".join(lines[:120])
    section_markers = ("Category", "Categories", "Description", "商品説明", "Additional information", "Specs", "仕様")
    best_index = candidate_indexes[0]
    for index in candidate_indexes:
        window = "\n".join(lines[index : min(len(lines), index + 25)])
        if any(marker in window for marker in section_markers):
            best_index = index
    return "\n".join(lines[best_index : min(len(lines), best_index + 140)])


def _extract_type(text: str, *, title: str = "") -> str:
    type_patterns = [
        ("Subwoofer|サブウーファー", "Subwoofer"),
        ("Midwoofer|Midwoofers|\\bMW\\d{2}|ミッド\\s*[・･-]?\\s*ウーファー", "Woofer"),
        ("Woofer|Woofers|ウーファー", "Woofer"),
        ("Fullrange|Full range|フルレンジ", "Fullrange"),
        ("Midrange|ミッドレンジ", "Midrange"),
        ("Compression", "Compression"),
        ("Tweeter|ツィーター|トゥイーター|ツイーター", "Tweeter"),
    ]
    for source in (title, text):
        for pattern, value in type_patterns:
            if re.search(pattern, source, re.IGNORECASE):
                return value
    return ""


def _extract_spec_values(text: str) -> dict[str, float]:
    compact = re.sub(r"\s+", " ", text)
    number = r"([-+]?[0-9]+(?:\.[0-9]+)?)"
    sep = r"\s*[：:=]?\s*"
    patterns = {
        "fs_hz": [
            rf"\bF\s*s\b{sep}{number}\s*Hz",
            rf"Resonance frequency.*?\b{number}\s*Hz",
            rf"Free air resonance,\s*F\s*s\s+{number}\s*Hz",
            rf"最低共振周波数{sep}{number}\s*Hz",
        ],
        "qts": [
            rf"\bQ\s*ts\b{sep}{number}",
            rf"\bQ\s*t\b{sep}{number}",
            rf"Total Q-factor,\s*Q\s*ts\s+{number}",
        ],
        "vas_l": [
            rf"\bV\s*as\b{sep}{number}\s*(?:L|l|ltr\.?|liters?|litres?)",
            rf"Equivalent volume,\s*V\s*as\s+{number}\s*(?:L|l|ltr\.?|liters?|litres?)",
        ],
        "re_ohm": [
            rf"\bR\s*e\b{sep}{number}\s*(?:ohm|Ω)",
            rf"DC resistance,\s*R\s*e\s+{number}\s*(?:ohm|Ω)",
        ],
        "le_mh": [
            rf"\bL\s*e\b{sep}{number}\s*mH",
            rf"Voice coil inductance,\s*L\s*e\s+{number}\s*mH",
        ],
        "sd_cm2": [
            rf"\bS\s*d\b{sep}{number}\s*cm\s*(?:2|²)",
            rf"Effective piston area,\s*S\s*d\s+{number}\s*cm\s*(?:2|²)",
        ],
        "xmax_mm": [rf"\bX\s*max\b{sep}{number}\s*mm"],
        "spl_db": [
            rf"\bSPL\b{sep}{number}\s*dB",
            rf"Sensitivity.*?\b{number}\s*dB",
            rf"出力音圧(?:レベル)?{sep}{number}\s*dB",
        ],
        "pmax_w": [
            rf"\bP\s*max\b{sep}{number}\s*W",
            rf"(?:Power|RMS).*?\b{number}\s*W",
            rf"入力{sep}(?:[0-9]+(?:\.[0-9]+)?\s*/\s*)?{number}\s*W",
        ],
        "nominal_impedance_ohm": [
            rf"Nominal impedance.*?\b{number}\s*(?:ohm|Ω)",
            rf"\bZ\b{sep}{number}\s*(?:ohm|Ω)",
            rf"\bImpedance\b{sep}{number}\s*(?:ohm|Ω)",
            rf"インピーダンス{sep}{number}\s*(?:ohm|Ω)",
            rf"(?:ウーファー|ツィーター|トゥイーター|ツイーター|スピーカー|フルレンジ).*?\b{number}\s*Ω",
            rf"\b{number}\s*ohm\b",
        ],
        "nominal_diameter_mm": [
            rf"Nominal diameter{sep}{number}\s*mm",
            rf"\b{number}\s*mm\s*(?:Dome|Tweeter|Woofer|Midwoofer|Fullrange|speaker|driver|スピーカー)",
            rf"\bSize{sep}{number}\s*cm\b",
            rf"\b{number}\s*(?:inch|inches|in|″|”|\")(?=\s|$)",
            rf"\b{number}\s*cm\s*(?:フルレンジ|ウーファー|スピーカー|speaker|woofer|fullrange)",
        ],
        "overall_diameter_mm": [rf"Overall diameter{sep}{number}\s*mm"],
        "baffle_cutout_mm": [
            rf"(?:Baffle cutout|Cutout diameter){sep}(?:φ\s*)?{number}\s*mm",
            rf"バッフル開口径{sep}(?:φ\s*)?{number}\s*mm",
        ],
        "mounting_depth_mm": [rf"(?:Mounting depth|Depth){sep}{number}\s*mm"],
    }
    values: dict[str, float] = {}
    for key, candidates in patterns.items():
        for pattern in candidates:
            match = re.search(pattern, compact, re.IGNORECASE)
            if not match:
                continue
            try:
                values[key] = float(match.group(1))
            except ValueError:
                pass
            break
    values = _extract_columnar_pdf_values(values, text)
    values = _normalize_diameter_units(values, compact)
    if "xmax_mm" not in values:
        pp_match = re.search(rf"Linear coil travel \(p-p\)\s+{number}\s*mm", compact, re.IGNORECASE)
        if pp_match:
            try:
                values["xmax_mm"] = float(pp_match.group(1)) / 2.0
            except ValueError:
                pass
    return values


def _normalize_diameter_units(values: dict[str, float], text: str) -> dict[str, float]:
    if "nominal_diameter_mm" not in values:
        return values
    diameter = values["nominal_diameter_mm"]
    diameter_pattern = _number_value_pattern(diameter)
    inch_match = re.search(rf"\b{diameter_pattern}\s*(?:inch|inches|in|″|”|\")(?=\s|$)", text, re.IGNORECASE)
    cm_match = re.search(
        rf"(?:\bSize\s*[：:=]?\s*{diameter_pattern}\s*cm\b|"
        rf"\b{diameter_pattern}\s*cm\s*(?:フルレンジ|ウーファー|スピーカー|speaker|woofer|fullrange))",
        text,
        re.IGNORECASE,
    )
    if inch_match:
        values["nominal_diameter_mm"] = diameter * 25.4
    elif cm_match:
        values["nominal_diameter_mm"] = diameter * 10.0
    return values


def _extract_columnar_pdf_values(values: dict[str, float], text: str) -> dict[str, float]:
    if not re.search(r"Voice coil inductance,\s*Le", text, re.IGNORECASE):
        return values
    if not re.search(r"Effective piston area,\s*Sd", text, re.IGNORECASE):
        return values
    value_lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_ohm_index = next((index for index, line in enumerate(value_lines) if re.fullmatch(r"[0-9.]+\s*(?:Ω|ohm)", line, re.IGNORECASE)), -1)
    if first_ohm_index < 0:
        return values
    ordered_values = value_lines[first_ohm_index : first_ohm_index + 12]
    field_patterns = [
        ("nominal_impedance_ohm", r"^([0-9.]+)\s*(?:Ω|ohm)$", 1.0),
        ("re_ohm", r"^([0-9.]+)\s*(?:Ω|ohm)$", 1.0),
        ("le_mh", r"^([0-9.]+)\s*mH$", 1.0),
        ("sd_cm2", r"^([0-9.]+)\s*cm\s*2$", 1.0),
        (None, r"^([0-9.]+)\s*mm$", 1.0),
        (None, r"^([0-9.]+)\s*mm$", 1.0),
        (None, r"^([0-9.]+)\s*mm$", 1.0),
        ("xmax_mm", r"^([0-9.]+)\s*mm$", 0.5),
    ]
    for index, (key, pattern, multiplier) in enumerate(field_patterns):
        if index >= len(ordered_values):
            break
        match = re.match(pattern, ordered_values[index], re.IGNORECASE)
        if not match or key is None or key in values:
            continue
        try:
            values[key] = float(match.group(1)) * multiplier
        except ValueError:
            pass
    return values


def _number_value_pattern(value: float) -> str:
    if float(value).is_integer():
        return rf"{int(value)}(?:\.0+)?"
    return re.escape(f"{float(value):g}")
