from __future__ import annotations

import importlib
import os
import platform
import sysconfig
import threading
from types import ModuleType


_IMPORT_LOCK = threading.RLock()


def _needs_windows_arm_x64_compatibility() -> bool:
    """Return whether an x64 Python process is emulated on Windows ARM."""

    return (
        os.name == "nt"
        and sysconfig.get_platform().casefold() == "win-amd64"
        and platform.machine().casefold() in {"arm64", "aarch64"}
    )


def import_sounddevice() -> ModuleType:
    """Import sounddevice with the PortAudio DLL matching the Python ABI.

    sounddevice 0.5.5 selects its bundled Windows DLL from
    ``platform.machine()``.  Windows ARM reports the host as ARM64 even when
    CPython is an emulated x64 process, which makes sounddevice choose an ARM64
    DLL that cannot be loaded by that process.  Temporarily report AMD64 only
    during the import; all other application platform detection remains real.
    """

    if not _needs_windows_arm_x64_compatibility():
        return importlib.import_module("sounddevice")

    with _IMPORT_LOCK:
        original_machine = platform.machine
        platform.machine = lambda: "AMD64"
        try:
            return importlib.import_module("sounddevice")
        finally:
            platform.machine = original_machine
