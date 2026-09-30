"""Recoverable standard audio preparation; files on disk, compact references in DB."""
from __future__ import annotations

import io
import json
import os
import shutil
from pathlib import Path
import threading
import uuid

import soundfile as sf

from phase_fir_designer.measurement.standard_audio import TRACKS, MEDIA, SET_VERSION, file_hash, render_track
from utils.ess_reference_db import ensure_ess_reference_db, _connect
from utils.runtime_instances import _lock_file, _unlock_file


class StandardAudioAssets:
    def __init__(self, root: Path, db: Path):
        self.root, self.db = Path(root), Path(db)
        self._lock = threading.RLock()
        self._worker = None
        self._status = {}

    def status(self, track, media):
        with self._lock:
            return dict(self._status.get(track.asset_id(media), {"state": "preparing", "message": "音源を準備しています"}))

    def start(self):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run_guarded, daemon=True, name="standard-audio-prepare")
            self._worker.start()

    def _run_guarded(self):
        try:
            self._run()
        except Exception as exc:
            with self._lock:
                for track in TRACKS:
                    for media in MEDIA:
                        self._status[track.asset_id(media)] = {"state": "failed", "message": f"保存先 {self.root}: {exc}"}

    def _run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".prepare.lock"
        with lock_path.open("a+b") as lock:
            if lock_path.stat().st_size == 0:
                lock.write(b"0")
                lock.flush()
            _lock_file(lock, blocking=True)
            try:
                for track in TRACKS:
                    for media in MEDIA:
                        try:
                            status = self.prepare(track, media)
                        except Exception as exc:
                            status = {"state": "failed", "message": str(exc)}
                        with self._lock:
                            self._status[track.asset_id(media)] = status
            finally:
                _unlock_file(lock)

    def prepare(self, track, media):
        folder = self.root / SET_VERSION / media
        folder.mkdir(parents=True, exist_ok=True)
        wav = folder / track.filename(media)
        metadata = wav.with_suffix(".json")
        manifest = None
        if wav.exists() and metadata.exists():
            try:
                candidate = json.loads(metadata.read_text())
            except (ValueError, UnicodeError):
                candidate = {}
            if candidate.get("asset_id") == track.asset_id(media) and candidate.get("file_sha256") == file_hash(wav):
                manifest = candidate
        if manifest is None:
            fs, bits = MEDIA[media]
            duration = 120 if track.kind == "noise" else 6+track.period_s*track.repeats
            required = int(duration*fs*2*(bits//8)) + 16*1024*1024
            if shutil.disk_usage(folder).free < required:
                raise OSError(f"音源の保存に約{required/1024**2:.0f} MiB必要です。保存先: {folder}")
            # Preserve interrupted/conflicting assets for recovery, never overwrite them.
            token = uuid.uuid4().hex
            for existing in (wav, metadata):
                if existing.exists():
                    existing.rename(existing.with_name(existing.name+f".preserved-{token}"))
            temporary = wav.with_name(f".{token}.wav")
            manifest = render_track(track, media, temporary)
            temp_meta = metadata.with_name(f".{token}.json")
            temp_meta.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            os.replace(temporary, wav)
            os.replace(temp_meta, metadata)
        if track.kind != "noise":
            ensure_ess_reference_db(self.db)
            with sf.SoundFile(wav) as source:
                source.seek(manifest["reference_start"])
                pcm = source.read(manifest["reference_count"], dtype="float32", always_2d=True)[:, manifest["reference_channel"]]
            audio = io.BytesIO()
            sf.write(audio, pcm, manifest["sample_rate"], format="WAV", subtype=f"PCM_{manifest['bit_depth']}")
            encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
            with _connect(self.db) as conn:
                conn.execute("INSERT OR IGNORE INTO ess_references(id,name,audio_data,manifest_data,origin,created_at,updated_at) VALUES(?,?,?,?,?,datetime('now'),datetime('now'))",
                    (track.asset_id(media), track.title, audio.getvalue(), encoded, "Standard PCM reference"))
                row = conn.execute("SELECT audio_data,manifest_data FROM ess_references WHERE id=?", (track.asset_id(media),)).fetchone()
                if bytes(row["audio_data"]) != audio.getvalue() or bytes(row["manifest_data"]) != encoded:
                    raise ValueError("同じIDのReferenceが異なります。既存データを保全しました。")
        return {"state": "ready", "path": wav, "manifest": manifest, "message": "利用できます"}
