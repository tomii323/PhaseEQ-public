# ui_index.py
# -*- coding: utf-8 -*-
"""
UI Index Helper (for Streamlit) — v3.1.0 (breaking)

変更点（重要）
- 互換維持用の補助関数を**削除**:
  - get_index / get_index_from / resolve_index_with_prefix（完全撤廃）
- すべてのウィジェットは **“値を保存・値を返す”** 仕様に統一
- キー衝突の自動回避（widget key ≠ session_state 保存キー）は継続
- number_input の min/max 逸脱を防ぐ安全ラッパを同梱

主なAPI（値ベース）
- selectbox_indexed(label, values, *, settings_key, labels=None, default=None, value_coerce=None, ...)
- radio_indexed(label, values, *, settings_key, labels=None, default=None, value_coerce=None, ...)
- multiselect_indexed(label, values, *, settings_key, labels=None, default_values=None, value_coerce=None, ...)
- number_input_stateful_safe(label, *, settings_key, min_value, max_value, default, step=1, ...)

備考
- label_to_name は PortAudio風ラベルから“名前”部分だけ抜き出す補助（必要時のみ明示的に利用）
"""

from __future__ import annotations
from typing import Any, Callable, List, Optional, Sequence, Iterable, Hashable, Dict
import unicodedata

# ============================================================
# errorx.safe_widget_key への依存（未導入でも動作するようにフォールバック）
# ============================================================
try:
    from errorx import safe_widget_key as _safe_widget_key  # type: ignore
except Exception:
    def _safe_widget_key(prefix: str, settings_key: str, key: Optional[str]) -> str:
        """ウィジェットkeyと保存keyの衝突を回避（key未指定 or 同名なら内部keyに置換）"""
        if not key or key == settings_key:
            return f"{prefix}{settings_key}"
        return key

# ============================================================
# 基本ユーティリティ
# ============================================================
def _norm_casefold(s: str | None) -> str:
    return unicodedata.normalize("NFKC", (s or "")).strip().casefold()

def label_to_name(label: str) -> str:
    """
    PortAudio風のラベル "[PA#03] ★ Realtek XYZ | in:2 out:2" -> "Realtek XYZ"
    ※ 自動では使わず、必要時に呼び出し側から明示的に使用してください。
    """
    s = str(label)
    i = s.find("]")
    if i != -1:
        s = s[i + 1 :]
    s = s.strip()
    if s.startswith("★"):
        s = s.lstrip("★").strip()
    j = s.find(" | ")
    if j != -1:
        s = s[:j]
    return unicodedata.normalize("NFKC", s)

def _coerce_value_if_needed(x: Any, value_coerce: Optional[Callable[[Any], Any]]) -> Any:
    return value_coerce(x) if value_coerce else x

def _labels_map_from(values: Sequence[Any], labels: Optional[Sequence[str]]) -> Dict[Any, str]:
    return {v: (labels[i] if labels and i < len(labels) else str(v)) for i, v in enumerate(values)}

# ============================================================
# 値ベースのウィジェット（統一仕様）
# ============================================================
def selectbox_indexed(
    label: str,
    values: Sequence[Any],
    *,
    settings_key: str,
    labels: Optional[Sequence[str]] = None,
    default: Optional[Any] = None,
    value_coerce: Optional[Callable[[Any], Any]] = None,  # 変換が必要なときのみ明示指定（例: "96000 Hz" -> 96000）
    key: Optional[str] = None,
    help: Optional[str] = None,
    disabled: bool = False,
    **widget_kwargs,
) -> Any:
    from utils.ui_index import selectbox_indexed as shared
    return shared(label, values, settings_key=settings_key, labels=labels, default=default, value_coerce=value_coerce, help=help, disabled=disabled, key=key, **widget_kwargs)

def radio_indexed(
    label: str,
    values: Sequence[Any],
    *,
    settings_key: str,
    labels: Optional[Sequence[str]] = None,
    default: Optional[Any] = None,
    value_coerce: Optional[Callable[[Any], Any]] = None,
    key: Optional[str] = None,
    help: Optional[str] = None,
    disabled: bool = False,
    **widget_kwargs,
) -> Any:
    from utils.ui_index import radio_indexed as shared
    return shared(label, values, settings_key=settings_key, labels=labels, default=default, value_coerce=value_coerce, help=help, disabled=disabled, key=key, **widget_kwargs)

def multiselect_indexed(
    label: str,
    values: Sequence[Any],
    *,
    settings_key: str,
    labels: Optional[Sequence[str]] = None,
    default_values: Optional[Iterable[Any]] = None,
    value_coerce: Optional[Callable[[Any], Any]] = None,
    key: Optional[str] = None,
    help: Optional[str] = None,
    disabled: bool = False,
    **widget_kwargs,
) -> List[Any]:
    from utils.ui_index import multiselect_indexed as shared
    return shared(label, values, settings_key=settings_key, labels=labels, default_values=default_values, value_coerce=value_coerce, help=help, disabled=disabled, key=key, **widget_kwargs)

# ============================================================
# 入力値安全化（number_input の min/max 自動クランプ）
# ============================================================
def number_input_stateful_safe(
    label: str,
    *,
    settings_key: str,
    min_value: int | float,
    max_value: int | float,
    default: int | float,
    step: int | float = 1,
    key: Optional[str] = None,
    help: Optional[str] = None,
    disabled: bool = False,
    **widget_kwargs,
):
    from utils.ui_index import number_input_stateful_safe as shared
    return shared(label, settings_key=settings_key, min_value=min_value, max_value=max_value, default=default, step=step, help=help, disabled=disabled, key=_safe_widget_key("__num_", settings_key, key), **widget_kwargs)
