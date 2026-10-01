from __future__ import annotations

from pathlib import Path
import os
import runpy
import sys
import tempfile


APP_ROOT = Path(__file__).resolve().parent
from runtime.app_update import register_running_app, writable_installation
if writable_installation(APP_ROOT):
    register_running_app(APP_ROOT)
STUDIO_ROOT = APP_ROOT / "src" / "composite_engine" / "multiway_studio"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phaseeq-matplotlib-cache"))

if str(APP_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(APP_ROOT / "src"))
runpy.run_path(str(STUDIO_ROOT / "APPY_MultiwayFIRStudio.py"), run_name="__main__")
