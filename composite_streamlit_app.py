from __future__ import annotations

from pathlib import Path
import os
import runpy
import sys
import tempfile


APP_ROOT = Path(__file__).resolve().parent
STUDIO_ROOT = APP_ROOT / "src" / "composite_engine" / "multiway_studio"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phaseeq-matplotlib-cache"))

if str(APP_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(APP_ROOT / "src"))
runpy.run_path(str(STUDIO_ROOT / "APPY_MultiwayFIRStudio.py"), run_name="__main__")
