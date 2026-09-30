from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.metadata
import os
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_FILE = ROOT / "requirements.txt"
CONSTRAINTS_FILE = ROOT / "constraints" / "test-environment.txt"
NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
PIN_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)==([^\s;]+)\s*$")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phaseeq-matplotlib-cache"))
OPTIONAL_RUNTIME_IMPORTS = frozenset({"sounddevice"})


def import_optional_sounddevice() -> object:
    """Import sounddevice without importing PhaseEQ's dependency-heavy package.

    This checker must run in a newly created virtual environment before numpy
    and the other runtime requirements exist.  Importing the application helper
    through ``phase_fir_designer`` would execute that package's ``__init__`` and
    fail before the checker can report that installation is required.
    """

    if not (
        os.name == "nt"
        and sysconfig.get_platform().casefold() == "win-amd64"
        and platform.machine().casefold() in {"arm64", "aarch64"}
    ):
        return importlib.import_module("sounddevice")

    original_machine = platform.machine
    platform.machine = lambda: "AMD64"
    try:
        return importlib.import_module("sounddevice")
    finally:
        platform.machine = original_machine


def normalize_name(value: str) -> str:
    return value.casefold().replace("_", "-").replace(".", "-")


def version_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts = re.findall(r"\d+|[A-Za-z]+", value)
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts)


def load_runtime_pins(
    requirements_path: Path = REQUIREMENTS_FILE,
    constraints_path: Path = CONSTRAINTS_FILE,
) -> dict[str, str]:
    direct: list[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = NAME_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Unsupported runtime requirement: {line}")
        direct.append(normalize_name(match.group(1)))
    pins: dict[str, str] = {}
    for raw_line in constraints_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(line)
        if match is not None:
            pins[normalize_name(match.group(1))] = match.group(2)
    missing = sorted(name for name in direct if name not in pins)
    if missing:
        raise ValueError("Runtime dependencies lack supported pins: " + ", ".join(missing))
    return {name: pins[name] for name in direct}


@dataclass(frozen=True)
class RuntimeAssessment:
    action: str
    missing: tuple[str, ...]
    older_or_different: tuple[str, ...]
    too_new: tuple[str, ...]
    broken_imports: tuple[str, ...]
    optional_import_errors: tuple[str, ...]
    pip_ok: bool


def assess_versions(
    pins: dict[str, str],
    installed: dict[str, str | None],
    *,
    broken_imports: tuple[str, ...] = (),
    optional_import_errors: tuple[str, ...] = (),
    pip_ok: bool = True,
) -> RuntimeAssessment:
    missing: list[str] = []
    different: list[str] = []
    too_new: list[str] = []
    for name, expected in pins.items():
        actual = installed.get(name)
        if actual is None:
            missing.append(name)
        elif actual != expected:
            if version_key(actual) > version_key(expected):
                too_new.append(name)
            else:
                different.append(name)
    if broken_imports or len(too_new) > 1:
        action = "recreate"
    elif too_new:
        action = "downgrade"
    elif missing or different:
        action = "install"
    elif not pip_ok:
        action = "recreate"
    else:
        action = "ready"
    return RuntimeAssessment(
        action,
        tuple(sorted(missing)),
        tuple(sorted(different)),
        tuple(sorted(too_new)),
        tuple(sorted(broken_imports)),
        tuple(sorted(optional_import_errors)),
        bool(pip_ok),
    )


def inspect_runtime() -> RuntimeAssessment:
    pins = load_runtime_pins()
    installed: dict[str, str | None] = {}
    broken: list[str] = []
    optional_errors: list[str] = []
    for name in pins:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
            continue
        module_name = name.replace("-", "_")
        if name == "pdfplumber":
            module_name = "pdfplumber"
        try:
            if name == "sounddevice":
                import_optional_sounddevice()
            else:
                importlib.import_module(module_name)
        except Exception as exc:
            if name in OPTIONAL_RUNTIME_IMPORTS:
                optional_errors.append(f"{name}: {type(exc).__name__}: {exc}")
            else:
                broken.append(name)
    pip_ok = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    return assess_versions(
        pins,
        installed,
        broken_imports=tuple(broken),
        optional_import_errors=tuple(optional_errors),
        pip_ok=pip_ok,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the validated PhaseEQ runtime.")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--action", action="store_true", help="Print ready/install/downgrade/recreate.")
    output.add_argument("--too-new", action="store_true", help="Print packages above the supported versions.")
    output.add_argument(
        "--optional-warnings",
        action="store_true",
        help="Print optional runtime import warnings without failing.",
    )
    args = parser.parse_args()
    assessment = inspect_runtime()
    if args.action:
        print(assessment.action)
        return 0
    if args.too_new:
        print(" ".join(assessment.too_new))
        return 0
    if args.optional_warnings:
        for error in assessment.optional_import_errors:
            print(f"WARNING: Optional audio runtime is unavailable: {error}")
            print("         PhaseEQ will start, but acoustic measurement is unavailable.")
        return 0
    if assessment.action == "ready":
        print("Runtime environment: OK")
        return 0
    print(f"Runtime environment action: {assessment.action}")
    for label, values in (
        ("missing", assessment.missing),
        ("different", assessment.older_or_different),
        ("too new", assessment.too_new),
        ("broken imports", assessment.broken_imports),
    ):
        if values:
            print(f"  {label}: {', '.join(values)}")
    if not assessment.pip_ok:
        print("  pip check: failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
