"""Application UI language; deliberately independent of DSP design settings."""
from __future__ import annotations

from contextvars import ContextVar
import json
import os
from pathlib import Path
import tempfile


_language: ContextVar[str | None] = ContextVar("phaseeq_ui_language", default=None)
LANGUAGES = ("ja", "en")


def language_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    data = Path(os.environ.get("PHASEEQ_DATA_DIR") or root / "data")
    return data / "ui_preferences.json"


def read_language(path: Path | None = None) -> str:
    path = path or language_path()
    if not path.exists():
        return "ja"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("UI preferences must be a JSON object")
    language = payload.get("language", "ja")
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported UI language: {language}")
    return language


def save_language(language: str, path: Path | None = None) -> None:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported UI language: {language}")
    save_ui_preference("language", language, path)


def save_ui_preference(key: str, value: object, path: Path | None = None) -> None:
    """Atomically update one shared presentation preference, preserving others."""
    path = path or language_path()
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(payload, dict):
        raise ValueError("UI preferences must be a JSON object")
    payload[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ui-preferences-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def current_language() -> str | None:
    # Timed Streamlit fragments run in a fresh execution context. Session state
    # carries presentation preferences across those runs; ContextVar also keeps
    # non-Streamlit renderers and tests isolated from each other.
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is not None:
        import streamlit as st
        return st.session_state.get("_application_ui_language", _language.get())
    return _language.get()


def set_language_context(language: str | None):
    if language is not None and language not in LANGUAGES:
        raise ValueError(f"Unsupported UI language: {language}")
    return _language.set(language)


def initialize_language() -> None:
    import streamlit as st
    try:
        language = read_language()
    except (OSError, ValueError) as exc:
        language = "ja"
        st.warning(f"表示言語設定を読み込めません / Cannot read UI language preferences: {exc}")
    set_language_context(language)
    st.session_state["_application_ui_language"] = language


def render_language_setting(*, key: str) -> None:
    import streamlit as st
    language = current_language() or "ja"
    st.session_state[key] = language

    def changed():
        try:
            save_language(st.session_state[key])
        except (OSError, ValueError) as exc:
            st.session_state["_ui_language_save_error"] = str(exc)

    st.selectbox("表示言語 / Language", LANGUAGES, key=key,
                 format_func=lambda value: "日本語" if value == "ja" else "English",
                 on_change=changed)
    if error := st.session_state.pop("_ui_language_save_error", None):
        st.error(f"表示言語を保存できません / Cannot save UI language: {error}")
    st.caption("PhaseEQとマルチウェイで共通です。もう一方は次の画面更新で反映します。"
               if language == "ja" else
               "Shared by PhaseEQ and Multiway. The other app uses this language on its next rerun.")
