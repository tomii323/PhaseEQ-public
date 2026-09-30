from __future__ import annotations

import shutil
import hashlib
from pathlib import Path


def application_storage_paths(app_root: Path, *, data_dir: Path | None = None) -> dict[str, Path]:
    """Return the centralized writable-storage layout for PhaseEQ."""
    data_dir = Path(data_dir) if data_dir is not None else app_root / "data"
    return {
        "data": data_dir,
        "settings": data_dir / "settings.json",
        "snapshots": data_dir / "snapshots",
        "databases": data_dir / "databases",
        "design_history_db": data_dir / "databases" / "design_history.sqlite3",
        "design_library_db": data_dir / "databases" / "phaseeq_design_library.sqlite3",
        "dsp_system_db": data_dir / "databases" / "dsp_systems.sqlite3",
        "multiway_system_db": data_dir / "databases" / "multiway_systems.sqlite3",
        "fir_artifact_db": data_dir / "databases" / "fir_artifacts.sqlite3",
        "speaker_specs_user_db": data_dir / "databases" / "speaker_specs_user.sqlite3",
        "speaker_measurements_db": data_dir / "databases" / "speaker_measurements.sqlite3",
        "microphone_db": data_dir / "databases" / "microphones.sqlite3",
        "ess_reference_db": data_dir / "databases" / "ess_references.sqlite3",
        "response_asset_db": data_dir / "databases" / "response_assets.sqlite3",
        "legacy_loudspeaker_db": data_dir / "databases" / "speaker_specs_loudspeaker_database.sqlite3",
        "external_cache": data_dir / "cache" / "external_specs" / "v1",
        "legacy_external_cache": data_dir / "cache" / "loudspeaker_database",
        # Compatibility alias for callers that only need the active cache root.
        "cache": data_dir / "cache" / "external_specs" / "v1",
        "logs": data_dir / "logs",
        "temp": data_dir / "tmp",
        "runtime_instances": data_dir / "tmp" / "runtime_instances",
        "composite_exchange": data_dir / "tmp" / "composite_exchange",
        "user_presets": data_dir / "presets",
        "legacy_settings": data_dir / "legacy" / "settings",
        "bundled_presets": app_root / "resources" / "presets",
    }


def file_revision(path: Path) -> str:
    """Return a stable revision for optimistic settings-file writes."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text_if_revision(path: Path, text: str, *, expected_revision: str) -> tuple[bool, str]:
    """Atomically write text unless another process/session changed the file."""
    current_revision = file_revision(path)
    if current_revision != expected_revision:
        return False, current_revision
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True, file_revision(path)


def _move_if_needed(source: Path, destination: Path) -> bool:
    if not source.exists() or destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return True


def migrate_legacy_storage(app_root: Path) -> list[tuple[Path, Path]]:
    """Move legacy root-level runtime data into ``data/`` without overwriting.

    SQLite sidecar files are migrated with their database. Existing files in the
    new location always win, so a partially completed migration is safe to retry.
    """
    paths = application_storage_paths(app_root)
    moved: list[tuple[Path, Path]] = []
    migrations = {
        app_root / "settings.json": paths["settings"],
        app_root / "config_export": paths["snapshots"],
        app_root / "speaker_specs_user.sqlite3": paths["speaker_specs_user_db"],
        app_root / "speaker_measurements.sqlite3": paths["speaker_measurements_db"],
        app_root / "speaker_specs_loudspeaker_database.sqlite3": paths["legacy_loudspeaker_db"],
        app_root / "cache_loudspeakerdatabase": paths["external_cache"],
        paths["legacy_external_cache"]: paths["external_cache"],
        app_root / "logs": paths["logs"],
        app_root / "tmp": paths["temp"],
        app_root / "tmp_app_check": paths["temp"] / "app_check",
        app_root / "settings": paths["legacy_settings"],
    }
    for source, destination in migrations.items():
        if _move_if_needed(source, destination):
            moved.append((source, destination))
        if source.suffix == ".sqlite3":
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar_source = Path(f"{source}{suffix}")
                sidecar_destination = Path(f"{destination}{suffix}")
                if _move_if_needed(sidecar_source, sidecar_destination):
                    moved.append((sidecar_source, sidecar_destination))

    for key in ("data", "snapshots", "databases", "user_presets", "external_cache"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return moved
