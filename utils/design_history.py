"""Compact, versioned design history. Never stores generated DSP coefficients.

The caller supplies a settled pipeline generation, not a widget callback. SQLite
transactions make promotion, retention and shared-asset ownership indivisible.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import zlib

SCHEMA = 1
# Each future format bump must supply a tested, one-step migration. Originals
# remain immutable; conversions operate on a decoded copy before app validation.
MIGRATIONS = {}


def migrate_payload(payload, version):
    if version > SCHEMA:
        raise ValueError("この履歴の形式には対応していません。対応するアプリで開いてください。")
    result = deepcopy(payload)
    while version < SCHEMA:
        converter = MIGRATIONS.get(version)
        if converter is None:
            raise ValueError("この履歴の形式の変換処理がありません。元の履歴は保持しています。")
        result = converter(result)
        if not isinstance(result, dict):
            raise ValueError("履歴形式の変換結果が不正です。")
        version += 1
    return result

class HistoryPending(ValueError):
    """A pipeline is not settled yet; this is not a user-facing failure."""


DEFAULTS = {"enabled": True, "interval_minutes": 30, "limit": 20}
# Source curves are immutable shared assets; speaker input is deliberately latest-only.
ASSET_KEYS = {"target_response_raw", "target_response", "mic_cal_response_raw", "raw_response", "materialized_response"}
INPUT_KEYS = {"speaker_response_raw", "speaker_impulse_raw", "speaker_response", "speaker_source_name", "speaker_timing_provenance", "speaker_calibration_state", "current_speaker_measurement", "current_speaker_measurement_id", "speaker_package_id"}
TRANSIENT = {"snapshot_name", "active_page", "selected_display", "selected_group", "updated_at", "created_at", "latest_assignment_id"}


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode_blob(blob):
    try:
        return json.loads(zlib.decompress(blob))
    except (zlib.error, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("履歴データを読み込めません。元のデータは保持しています。") from exc


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


@contextmanager
def connection(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS history_settings (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS design_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL, owner TEXT NOT NULL,
          created REAL NOT NULL, kind TEXT NOT NULL, protected INTEGER NOT NULL DEFAULT 0,
          note TEXT NOT NULL DEFAULT '', summary TEXT NOT NULL, signature TEXT NOT NULL,
          schema_version INTEGER NOT NULL, app_version TEXT NOT NULL, payload BLOB NOT NULL,
          restored_from INTEGER);
        CREATE INDEX IF NOT EXISTS history_owner ON design_history(scope,owner,id);
        CREATE TABLE IF NOT EXISTS history_assets (hash TEXT PRIMARY KEY, payload BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS history_asset_refs (
          history_id INTEGER NOT NULL, hash TEXT NOT NULL, PRIMARY KEY(history_id,hash));
        CREATE TABLE IF NOT EXISTS history_latest_inputs (
          scope TEXT NOT NULL, owner TEXT NOT NULL, payload BLOB NOT NULL,
          PRIMARY KEY(scope,owner));
        CREATE TABLE IF NOT EXISTS history_candidates (
          scope TEXT NOT NULL, owner TEXT NOT NULL, generation TEXT NOT NULL,
          changed REAL NOT NULL, checked REAL NOT NULL DEFAULT 0,
          PRIMARY KEY(scope,owner));
        """)
        with db:
            yield db
    finally:
        db.close()


def settings(path):
    with connection(path) as db:
        row = db.execute("SELECT payload FROM history_settings WHERE id=1").fetchone()
        return {**DEFAULTS, **(json.loads(row[0]) if row else {})}


def configure(path, *, enabled, interval_minutes, limit):
    if interval_minutes not in (5, 10, 15, 30, 60) or not 5 <= limit <= 100:
        raise ValueError("Invalid history settings")
    value = dict(enabled=bool(enabled), interval_minutes=int(interval_minutes), limit=int(limit))
    with connection(path) as db:
        db.execute("INSERT OR REPLACE INTO history_settings VALUES(1,?)", (encode(value).decode(),))


def _prepare(value, assets, inputs, path=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            location = f"{path}/{key}"
            if key in TRANSIENT:
                continue
            if key in INPUT_KEYS:
                inputs[location] = item
                result[key] = {"$latest_input": location, "hash": digest(item)}
            elif key in ASSET_KEYS and item is not None:
                fingerprint = digest(item)
                assets[fingerprint] = item
                result[key] = {"$asset": fingerprint}
            else:
                result[key] = _prepare(item, assets, inputs, location)
        return result
    if isinstance(value, list):
        return [_prepare(v, assets, inputs, f"{path}/{i}") for i, v in enumerate(value)]
    return value


VIEW_KEYS = {"active_page", "navigation_page", "auto_update", "graph_max_points", "graph_mode",
             "gain_y_min_db", "gain_y_max_db", "phase_gain_mask", "result_detail_view",
             "card_open_state", "auto_iir_last_diagnostics", "target_task", "input_edit_mode",
             "fir_eq_task", "iir_eq_view", "phase_edit_group", "gain_edit_group"}


def _view_key(key):
    return key in VIEW_KEYS or key.startswith(("plot_", "chart_", "display_", "wavelet_", "dsp_result_"))


def compact_settings(payload):
    """Explicitly remove derived responses/UI navigation from a settings payload."""
    result = deepcopy(payload)
    processing = result.get("processing", {})
    crossover = deepcopy(processing.get("band_split_iir_crossover"))
    if isinstance(crossover, dict):
        crossover.pop("sos", None)
    result["processing"] = {"band_split_iir_crossover": crossover} if crossover else {}
    result.pop("revision", None)
    config = result.get("config", {})
    # target_response is the processed curve; raw target + editor settings reproduce it.
    config.pop("speaker_response", None)
    config.pop("target_response", None)
    ui = result.get("ui", {})
    for key in list(ui):
        if _view_key(key):
            ui.pop(key)
    profile = result.get("ui_profile", {})
    state = profile.get("state", {})
    for key in list(state):
        if _view_key(key):
            state.pop(key)
    return result


def _summary(old, new):
    if old is None:
        return "初回保存"
    keys = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
    labels = {"config": "EQ・設計設定", "ui": "入力処理設定", "ui_profile": "編集設定", "io": "入力・Target",
              "target_edit": "Target EQ", "level_db": "Targetレベル", "settings": "帯域分割・出力設定",
              "channels": "チャンネル設定", "phaseeq": "各WayのEQ設定", "speaker_inputs": "測定入力", "targets": "Target",
              "name": "名称", "target_response_raw": "Target入力カーブ", "phase_enabled": "入力位相",
              "level_mode": "Targetレベル調整方式", "lf_extension_enabled": "低域延長", "hf_extension_enabled": "高域延長"}
    return "、".join(labels.get(k, k) for k in keys[:4]) or "設定を復元"


def save(path, scope, owner, payload, *, kind="auto", app_version="", restored_from=None, now=None, expected_generation=None):
    if kind not in {"auto", "manual", "restore"}:
        raise ValueError("Unknown history kind")
    now = time.time() if now is None else now
    assets, inputs = {}, {}
    if scope == "target":
        payload = {**payload, "id": ""}
    prepared = _prepare(payload, assets, inputs)
    signature = digest(prepared)
    with connection(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if expected_generation is not None:
            candidate = db.execute("SELECT generation FROM history_candidates WHERE scope=? AND owner=?", (scope, owner)).fetchone()
            if candidate is None or candidate[0] != expected_generation:
                return None
            db.execute("UPDATE history_candidates SET checked=? WHERE scope=? AND owner=?", (now, scope, owner))
        db.execute("INSERT OR REPLACE INTO history_latest_inputs VALUES(?,?,?)",
                   (scope, owner, zlib.compress(encode(inputs))))
        previous = db.execute("SELECT * FROM design_history WHERE scope=? AND owner=? ORDER BY id DESC LIMIT 1", (scope, owner)).fetchone()
        if previous and previous["signature"] == signature and kind != "restore":
            if kind == "manual":
                db.execute("UPDATE design_history SET kind='manual' WHERE id=?", (previous["id"],))
            return previous["id"]
        old = decode_blob(previous["payload"]) if previous else None
        cursor = db.execute("""INSERT INTO design_history
          (scope,owner,created,kind,summary,signature,schema_version,app_version,payload,restored_from)
          VALUES(?,?,?,?,?,?,?,?,?,?)""", (scope, owner, now, kind, _summary(old, prepared), signature,
                                       SCHEMA, app_version, zlib.compress(encode(prepared)), restored_from))
        identifier = cursor.lastrowid
        for fingerprint, item in assets.items():
            db.execute("INSERT OR IGNORE INTO history_assets VALUES(?,?)", (fingerprint, zlib.compress(encode(item))))
            db.execute("INSERT INTO history_asset_refs VALUES(?,?)", (identifier, fingerprint))
        pref = db.execute("SELECT payload FROM history_settings WHERE id=1").fetchone()
        limit = json.loads(pref[0])["limit"] if pref else DEFAULTS["limit"]
        stale = db.execute("SELECT id FROM design_history WHERE scope=? AND owner=? AND kind='auto' AND protected=0 ORDER BY id DESC LIMIT -1 OFFSET ?", (scope, owner, limit)).fetchall()
        for row in stale:
            db.execute("DELETE FROM history_asset_refs WHERE history_id=?", (row[0],))
            db.execute("DELETE FROM design_history WHERE id=?", (row[0],))
        db.execute("DELETE FROM history_assets WHERE hash NOT IN (SELECT hash FROM history_asset_refs)")
        return identifier


def entries(path, scope, owner):
    with connection(path) as db:
        return [dict(row) for row in db.execute("SELECT id,created,kind,protected,note,summary,app_version,schema_version FROM design_history WHERE scope=? AND owner=? ORDER BY id DESC", (scope, owner))]


def annotate(path, identifier, *, note, protected):
    with connection(path) as db:
        db.execute("UPDATE design_history SET note=?,protected=? WHERE id=?", (str(note), int(protected), identifier))


def load(path, identifier, *, current_inputs=None):
    """Validate before returning; never mutate the application during conversion."""
    with connection(path) as db:
        row = db.execute("SELECT * FROM design_history WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("履歴が見つかりません。")
        payload = decode_blob(row["payload"])
        if digest(payload) != row["signature"]:
            raise ValueError("履歴の整合性を確認できません。")
        payload = migrate_payload(payload, row["schema_version"])
        if current_inputs is None:
            latest = db.execute("SELECT payload FROM history_latest_inputs WHERE scope=? AND owner=?", (row["scope"], row["owner"])).fetchone()
            current_inputs = decode_blob(latest[0]) if latest else {}
        warnings = []
        def resolve(value):
            if isinstance(value, dict):
                if "$asset" in value:
                    asset = db.execute("SELECT payload FROM history_assets WHERE hash=?", (value["$asset"],)).fetchone()
                    if asset is None:
                        raise ValueError("履歴に必要なTarget・校正データがありません。")
                    result = decode_blob(asset[0])
                    if digest(result) != value["$asset"]:
                        raise ValueError("参照データの整合性を確認できません。")
                    return result
                if "$latest_input" in value:
                    result = current_inputs.get(value["$latest_input"])
                    if digest(result) != value["hash"]:
                        warnings.append(value["$latest_input"])
                    return result
                return {k: resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value
        return resolve(payload), warnings, dict(row)


def consider(path, scope, owner, generation, factory, *, settled=True, busy=False, app_version="", now=None):
    """Called after pipeline apply; defer expensive serialization until due/stable."""
    if busy or not settled:
        return
    now = time.time() if now is None else now
    preference = settings(path)
    if not preference["enabled"]:
        return
    generation = str(generation)
    with connection(path) as db:
        row = db.execute("SELECT * FROM history_candidates WHERE scope=? AND owner=?", (scope, owner)).fetchone()
        if row is None:
            db.execute("INSERT INTO history_candidates VALUES(?,?,?,?,?)", (scope, owner, generation, now, now))
            return
        if row["generation"] != generation:
            db.execute("UPDATE history_candidates SET generation=?,changed=? WHERE scope=? AND owner=?", (generation, now, scope, owner))
            return
        if row["changed"] <= row["checked"] or now-row["changed"] < 30 or now-row["checked"] < preference["interval_minutes"]*60:
            return
    payload = factory()
    # A second process may have delivered a newer generation while serializing.
    with connection(path) as db:
        current = db.execute("SELECT generation FROM history_candidates WHERE scope=? AND owner=?", (scope, owner)).fetchone()
        if current is None or current[0] != generation:
            return
    save(path, scope, owner, payload, app_version=app_version, now=now, expected_generation=generation)


def input_values(payload):
    assets, inputs = {}, {}
    _prepare(payload, assets, inputs)
    return inputs


def adopt_draft(path, scope, draft_owner, owner):
    if draft_owner == owner:
        return
    with connection(path) as db:
        db.execute("UPDATE design_history SET owner=? WHERE scope=? AND owner=?", (owner, scope, draft_owner))
        for table in ("history_latest_inputs", "history_candidates"):
            db.execute(f"DELETE FROM {table} WHERE scope=? AND owner=?", (scope, draft_owner))


def display_date(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).astimezone().strftime("%Y/%m/%d %H:%M")


def validate_settings_restore(payload):
    """Reject unknown DSP fields rather than silently defaulting after an upgrade.

    Settings-format migrations must run before this boundary; the existing loader
    remains authoritative for normalization and numeric validation.
    """
    from dataclasses import fields, is_dataclass
    from utils.settings_io import config_from_payload
    config = config_from_payload(payload)
    def check(source, restored, location):
        if isinstance(source, dict) and is_dataclass(restored):
            supported = {field.name for field in fields(restored)}
            unknown = set(source) - supported
            if unknown:
                raise ValueError(f"復元できない設定があります: {location} / {', '.join(sorted(unknown))}")
            for key, value in source.items():
                check(value, getattr(restored, key), f"{location}.{key}")
        elif isinstance(source, list) and isinstance(restored, (list, tuple)):
            if len(source) != len(restored):
                raise ValueError(f"復元時に項目数が変わるため適用しません: {location}")
            for index, (value, target) in enumerate(zip(source, restored)):
                check(value, target, f"{location}[{index}]")
    missing = {field.name for field in fields(config)} - set(payload['config']) - {"speaker_response", "target_response"}
    if missing:
        raise ValueError("設定形式の移行が必要です。新しい既定値では補完しません: " + ", ".join(sorted(missing)))
    check(payload['config'], config, 'config')
    config.validate()
    return config


def unedited_channel_settings(current):
    """An older project with no PhaseEQ edits means an empty EQ, not today's EQ."""
    from dataclasses import asdict
    from phase_fir_designer import DesignConfig
    result = deepcopy(current)
    config = current['config']
    result['config'] = asdict(DesignConfig(sample_rate=int(config['sample_rate']), taps=max(3, int(config['taps']))))
    result['config'].pop('speaker_response', None)
    result['config'].pop('target_response', None)
    result['ui_profile'] = {}
    result['ui'] = {key: value for key, value in current.get('ui', {}).items()
                    if key.startswith(('speaker_', 'current_', 'input_')) or key == 'apply_mic_cal'}
    return result
