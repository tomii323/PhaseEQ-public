from __future__ import annotations

import atexit
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time
from typing import BinaryIO, Callable
import uuid


RUNTIME_INSTANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeInstance:
    instance_id: str
    pid: int
    port: int
    app_name: str
    app_root: str
    data_dir: str
    started_at_utc: str
    manifest_path: Path = field(compare=False, repr=False)
    lock_path: Path = field(compare=False, repr=False)


@dataclass(frozen=True)
class StopRuntimeInstanceResult:
    status: str
    message: str


@dataclass
class _Registration:
    instance: RuntimeInstance
    lock_file: BinaryIO


_REGISTRATIONS: dict[tuple[str, int], _Registration] = {}


def _resolved(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve())


def _lock_file(handle: BinaryIO, *, blocking: bool) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(handle.fileno(), mode, 1)
        return

    import fcntl

    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    fcntl.flock(handle.fileno(), operation)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_manifest(instance: RuntimeInstance) -> None:
    payload = {
        "schema_version": RUNTIME_INSTANCE_SCHEMA_VERSION,
        **{
            key: value
            for key, value in asdict(instance).items()
            if key not in {"manifest_path", "lock_path"}
        },
        "lock_file": instance.lock_path.name,
    }
    temporary = instance.manifest_path.with_name(
        f".{instance.manifest_path.name}.{os.getpid()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(instance.manifest_path)
    finally:
        temporary.unlink(missing_ok=True)


def _release_registration(key: tuple[str, int]) -> None:
    registration = _REGISTRATIONS.pop(key, None)
    if registration is None:
        return
    try:
        _unlock_file(registration.lock_file)
    except (OSError, ValueError):
        pass
    try:
        registration.lock_file.close()
    except OSError:
        pass
    registration.instance.manifest_path.unlink(missing_ok=True)
    registration.instance.lock_path.unlink(missing_ok=True)


def register_runtime_instance(
    runtime_dir: Path,
    *,
    port: int,
    app_name: str,
    app_root: Path,
    data_dir: Path,
) -> RuntimeInstance:
    """Register this server process in the shared configuration directory.

    The held file lock is the liveness proof. A stale manifest whose PID has
    been reused cannot be treated as active after the original process exits.
    """

    normalized_runtime_dir = Path(runtime_dir).expanduser().resolve()
    normalized_runtime_dir.mkdir(parents=True, exist_ok=True)
    key = (str(normalized_runtime_dir), int(port))
    existing = _REGISTRATIONS.get(key)
    if existing is not None:
        return existing.instance

    instance_id = str(uuid.uuid4())
    lock_path = normalized_runtime_dir / f"{instance_id}.lock"
    manifest_path = normalized_runtime_dir / f"{instance_id}.json"
    lock_file = lock_path.open("w+b")
    lock_file.write(b"1")
    lock_file.flush()
    try:
        _lock_file(lock_file, blocking=False)
        instance = RuntimeInstance(
            instance_id=instance_id,
            pid=os.getpid(),
            port=int(port),
            app_name=str(app_name),
            app_root=_resolved(app_root),
            data_dir=_resolved(data_dir),
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            manifest_path=manifest_path,
            lock_path=lock_path,
        )
        _write_manifest(instance)
    except Exception:
        try:
            _unlock_file(lock_file)
        except (OSError, ValueError):
            pass
        lock_file.close()
        manifest_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)
        raise

    _REGISTRATIONS[key] = _Registration(instance=instance, lock_file=lock_file)
    atexit.register(_release_registration, key)
    return instance


def _load_instance(manifest_path: Path) -> RuntimeInstance | None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != RUNTIME_INSTANCE_SCHEMA_VERSION:
            return None
        instance_id = str(payload["instance_id"])
        lock_name = str(payload["lock_file"])
        if Path(lock_name).name != lock_name:
            return None
        return RuntimeInstance(
            instance_id=instance_id,
            pid=int(payload["pid"]),
            port=int(payload["port"]),
            app_name=str(payload["app_name"]),
            app_root=_resolved(str(payload["app_root"])),
            data_dir=_resolved(str(payload["data_dir"])),
            started_at_utc=str(payload["started_at_utc"]),
            manifest_path=manifest_path,
            lock_path=manifest_path.parent / lock_name,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _lock_is_held(lock_path: Path) -> bool:
    try:
        handle = lock_path.open("r+b")
    except OSError:
        return False
    try:
        try:
            _lock_file(handle, blocking=False)
        except (BlockingIOError, OSError):
            return True
        _unlock_file(handle)
        return False
    finally:
        handle.close()


def _pid_exists(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _port_is_listening(port: int) -> bool:
    if not 1 <= int(port) <= 65_535:
        return False
    for host in ("127.0.0.1", "::1"):
        try:
            with socket.create_connection((host, int(port)), timeout=0.2):
                return True
        except OSError:
            continue
    return False


def _instance_matches_scope(
    instance: RuntimeInstance,
    *,
    app_name: str,
    data_dir: Path,
) -> bool:
    return (
        instance.app_name == str(app_name)
        and instance.data_dir == _resolved(data_dir)
    )


def runtime_instance_is_active(instance: RuntimeInstance) -> bool:
    return (
        _lock_is_held(instance.lock_path)
        and _pid_exists(instance.pid)
        and _port_is_listening(instance.port)
    )


def _discard_stale_instance(instance: RuntimeInstance) -> None:
    if runtime_instance_is_active(instance):
        return
    instance.manifest_path.unlink(missing_ok=True)
    instance.lock_path.unlink(missing_ok=True)


def list_other_runtime_instances(
    runtime_dir: Path,
    *,
    current: RuntimeInstance,
    app_name: str,
    data_dir: Path,
) -> list[RuntimeInstance]:
    """List live sibling servers that use this exact app and data directory."""

    instances: list[RuntimeInstance] = []
    normalized_runtime_dir = Path(runtime_dir).expanduser().resolve()
    if not normalized_runtime_dir.exists():
        return instances
    for manifest_path in normalized_runtime_dir.glob("*.json"):
        instance = _load_instance(manifest_path)
        if instance is None:
            manifest_path.unlink(missing_ok=True)
            continue
        if instance.instance_id == current.instance_id or instance.pid in {current.pid, os.getpid()}:
            continue
        if not _instance_matches_scope(
            instance,
            app_name=app_name,
            data_dir=data_dir,
        ):
            continue
        if runtime_instance_is_active(instance):
            instances.append(instance)
        else:
            _discard_stale_instance(instance)
    # Hot reloads can leave multiple locked registrations for one server.
    # Count processes/endpoints, not registration records; retain the records
    # while alive so their original cleanup/termination validation still works.
    unique = {}
    for instance in sorted(instances, key=lambda item: (item.started_at_utc, item.instance_id)):
        unique[(instance.pid, instance.port)] = instance
    return sorted(unique.values(), key=lambda item: (item.port, item.started_at_utc, item.instance_id))


def stop_runtime_instance(
    instance: RuntimeInstance,
    *,
    app_name: str,
    data_dir: Path,
    timeout_s: float = 2.0,
    terminate: Callable[[int, int], None] = os.kill,
) -> StopRuntimeInstanceResult:
    """Safely request termination after revalidating the selected sibling."""

    if instance.pid == os.getpid():
        return StopRuntimeInstanceResult("unavailable", "The current app cannot stop itself here.")
    if not _instance_matches_scope(
        instance,
        app_name=app_name,
        data_dir=data_dir,
    ):
        return StopRuntimeInstanceResult("unavailable", "The app configuration no longer matches.")

    refreshed = _load_instance(instance.manifest_path)
    if refreshed is None or refreshed != instance or not runtime_instance_is_active(refreshed):
        _discard_stale_instance(instance)
        return StopRuntimeInstanceResult("unavailable", "The other app is no longer running.")

    try:
        terminate(refreshed.pid, signal.SIGTERM)
    except ProcessLookupError:
        _discard_stale_instance(refreshed)
        return StopRuntimeInstanceResult("stopped", f"Port {refreshed.port} is already stopped.")
    except (OSError, PermissionError) as exc:
        return StopRuntimeInstanceResult(
            "unavailable",
            f"Could not stop port {refreshed.port}: {exc}",
        )

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while time.monotonic() < deadline:
        if not runtime_instance_is_active(refreshed):
            _discard_stale_instance(refreshed)
            return StopRuntimeInstanceResult("stopped", f"Stopped PhaseEQ on port {refreshed.port}.")
        time.sleep(0.05)
    return StopRuntimeInstanceResult(
        "requested",
        f"Stop requested for PhaseEQ on port {refreshed.port}.",
    )
