"""Register the immutable ideal input separately from selecting an input."""
from dataclasses import asdict
import json
from pathlib import Path

from phase_fir_designer.ideal_input import (
    IDEAL_INPUT_ID, IDEAL_INPUT_NAME, ideal_input_definition, ideal_speaker_response,
)
from utils.speaker_db import SpeakerMeasurementRecord, ensure_measurement_db, _connect, _now_sql, get_measurement
from utils.settings_io import speaker_response_payload


def is_builtin_ideal(record_or_id) -> bool:
    return getattr(record_or_id, "id", record_or_id) == IDEAL_INPUT_ID


def ensure_ideal_measurement(path: Path) -> SpeakerMeasurementRecord:
    """Insert once under a stable ID; concurrent startup does not replace rows."""
    ensure_measurement_db(path)
    response = speaker_response_payload(ideal_speaker_response(96000, 16384))
    response["input_definition"] = ideal_input_definition()
    record = SpeakerMeasurementRecord(
        id=IDEAL_INPUT_ID, name=IDEAL_INPUT_NAME, source_name="PhaseEQ generated ideal input",
        response_payload=response, note="生成した基準応答。実測の音圧・位相・時間情報を含まない。",
    )
    values = asdict(record)
    values["response_json"] = json.dumps(values.pop("response_payload"), ensure_ascii=False, sort_keys=True)
    with _connect(path) as db:
        now = _now_sql(db)
        values.update(created_at=now, updated_at=now, archived=0)
        names = list(values)
        db.execute(f"INSERT OR IGNORE INTO speaker_measurements ({','.join(names)}) "
                   f"VALUES ({','.join('?' for _ in names)})", tuple(values.values()))
    saved = get_measurement(path, IDEAL_INPUT_ID)
    if saved is None or saved.response_payload != response or saved.archived:
        raise ValueError("組込み理想特性の登録を確認できません。既存データを保持しています。")
    return saved
