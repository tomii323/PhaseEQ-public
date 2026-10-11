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
import streamlit as st
from utils.ui_visibility import start_visibility_diagnostics
_ui_visibility_snapshot = start_visibility_diagnostics(APP_ROOT, "multiway")
if _ui_visibility_snapshot is not None:
    st.session_state["_ui_visibility_snapshot"] = _ui_visibility_snapshot
else:
    st.session_state.pop("_ui_visibility_snapshot", None)
runpy.run_path(str(STUDIO_ROOT / "APPY_MultiwayFIRStudio.py"), run_name="__main__")
