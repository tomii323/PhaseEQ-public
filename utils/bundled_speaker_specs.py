"""Install reviewed distribution specifications without overwriting user records."""
from dataclasses import asdict
import json
from pathlib import Path

from utils.speaker_db import SpeakerSpecRecord, _connect, _now_sql, ensure_speaker_spec_db

BUNDLED_SPEAKER_SPECS = Path(__file__).resolve().parents[1] / 'resources' / 'speaker_specs.json'


def install_bundled_speaker_specs(path: Path, *, bundle_path: Path = BUNDLED_SPEAKER_SPECS) -> int:
    payload = json.loads(bundle_path.read_text(encoding='utf-8'))
    if payload.get('kind') != 'speaker_specs' or payload.get('schema_version') != 1:
        raise ValueError('Invalid bundled speaker specifications')
    records = [SpeakerSpecRecord(**item) for item in payload['records']]
    if any(not r.id or not r.brand or not r.model or not r.redistributable or r.archived for r in records):
        raise ValueError('Bundled specifications must be identified, active and redistributable')
    ensure_speaker_spec_db(path)
    count = 0
    with _connect(path) as conn:
        for record in records:
            marker = 'bundled_spec:' + record.id
            if conn.execute('SELECT 1 FROM app_meta WHERE key = ?', (marker,)).fetchone():
                continue
            # Also respect archived copies and user records created with other IDs.
            existing = conn.execute(
                'SELECT 1 FROM speaker_specs WHERE id = ? OR (brand = ? COLLATE NOCASE AND model = ? COLLATE NOCASE)',
                (record.id, record.brand, record.model),
            ).fetchone()
            if existing is None:
                now = _now_sql(conn)
                values = asdict(record) | {'created_at': now, 'updated_at': now}
                conn.execute(
                    f"INSERT INTO speaker_specs ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})",
                    tuple(values.values()),
                )
                count += 1
            conn.execute('INSERT INTO app_meta(key, value) VALUES(?, ?)', (marker, '1'))
    return count
