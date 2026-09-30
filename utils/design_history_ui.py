"""Quiet history controls shared by PhaseEQ and Multiway."""
from __future__ import annotations

import streamlit as st
from utils import design_history as history


def text(ja, en):
    # Use the existing application language, including standalone Studio sessions.
    from utils.ui_localization import current_language
    return en if current_language() == "en" else ja


def render_settings(path):
    with st.expander(text("履歴", "History"), expanded=False):
        preferences = history.settings(path)
        with st.form("design_history_preferences"):
            enabled = st.checkbox(text("自動履歴", "Automatic history"), value=preferences["enabled"])
            interval = st.selectbox(text("保存間隔（分）", "Interval (minutes)"), [5, 10, 15, 30, 60], index=[5, 10, 15, 30, 60].index(preferences["interval_minutes"]))
            limit = st.number_input(text("対象ごとの自動履歴上限", "Automatic entries per design"), min_value=5, max_value=100, value=preferences["limit"])
            st.caption(text("手動保存・保護した履歴は自動整理しません。", "Manual and protected entries are retained."))
            if st.form_submit_button(text("設定を保存", "Save settings")):
                history.configure(path, enabled=enabled, interval_minutes=interval, limit=limit)


def _toggle(key):
    st.session_state[key] = not st.session_state.get(key, False)


def _request(key):
    st.session_state[key] = True


def render_history(path, scope, owner, factory, restore, *, app_version="", target=False, ready=True, clone=None):
    key = f"design_history_{scope}_{owner}"
    st.button(text("履歴", "History"), key=key+"_open", icon=":material/history:",
              on_click=_toggle, args=(key+"_visible",))
    if not st.session_state.get(key+"_visible"):
        return
    with st.container(border=True):
        rows = history.entries(path, scope, owner)
        if not rows:
            st.caption(text("保存履歴はありません。", "No saved history."))
        identifiers = [row["id"] for row in rows]
        lookup = {row["id"]: row for row in rows}
        selected = st.selectbox(text("保存日時・変更内容", "Saved time / changes"), identifiers,
                                format_func=lambda value: f'{history.display_date(lookup[value]["created"])} · {lookup[value]["summary"]}', key=key+"_selection") if rows else None
        st.button(text("現在の設定を保存", "Save current settings"), key=key+"_save", disabled=not ready,
                  on_click=_request, args=(key+"_save_requested",))
        if ready and st.session_state.pop(key+"_save_requested", False):
            try:
                history.save(path, scope, owner, factory(), kind="manual", app_version=app_version)
            except history.HistoryPending:
                st.caption(text("設定の同期完了後に保存できます。", "Available after settings finish synchronizing."))
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.rerun()
        if selected is None or not ready:
            return
        try:
            current = factory()
            payload, mismatches, metadata = history.load(path, selected, current_inputs=history.input_values(current))
        except history.HistoryPending:
            return
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return
        allowed = True
        if mismatches:
            st.warning(text("保存時とスピーカー入力が異なります。再生成の結果は当時と異なります。", "Speaker input differs. Regenerated filters will differ."))
            allowed = st.checkbox(text("現在の入力を使用して設定を戻す", "Restore settings using current input"), key=key+f"_input_{selected}")
        if metadata["app_version"] != app_version:
            st.caption(text("保存時とアプリの版が異なります。設定を検証して復元します。", "Saved by a different app version; settings will be validated."))
        with st.expander(text("詳細", "Details")):
            kind = {"auto": text("自動", "Automatic"), "manual": text("手動", "Manual"), "restore": text("復元", "Restored")}[metadata["kind"]]
            st.caption(f'{kind} · {metadata["app_version"]} · #{selected}')
            counts = {kind: sum(row["kind"] == kind and not row["protected"] for row in rows) for kind in ("auto", "manual", "restore")}
            protected_count = sum(bool(row["protected"]) for row in rows)
            st.caption(text(f"自動 {counts['auto']}件・手動/復元 {counts['manual'] + counts['restore']}件・保護 {protected_count}件", f"Automatic {counts['auto']} · Manual/restored {counts['manual'] + counts['restore']} · Protected {protected_count}"))
            protected = st.checkbox(text("保護する", "Protect"), value=bool(metadata["protected"]), key=key+f"_protect_{selected}")
            note = st.text_input(text("メモ", "Note"), value=metadata["note"], key=key+f"_note_{selected}")
            if clone is not None:
                copy_name = st.text_input(text("別名", "Copy name"), key=key+f"_copy_name_{selected}")
                if st.button(text("別名の設計として開く", "Open as a separate design"),
                             key=key+"_clone", disabled=not (ready and allowed and copy_name.strip())):
                    clone(payload, copy_name.strip())
            if st.button(text("詳細を保存", "Save details"), key=key+"_annotate"):
                history.annotate(path, selected, note=note, protected=protected)
        if target:
            st.caption(text("編集内容に戻します。登録済みTargetへの反映には「更新」が必要です。", "Restores the editor. Update the Target to publish the changes."))
        # Explicit acknowledgement protects edits without a modal on normal use.
        confirmed = st.checkbox(text("現在の編集を置き換える", "Replace current edits"), key=key+f"_confirm_{selected}")
        st.button(text("この設定に戻す", "Restore these settings"), disabled=not (ready and allowed and confirmed),
                  key=key+"_restore", on_click=_request, args=(key+"_restore_requested",))
        if ready and allowed and confirmed and st.session_state.pop(key+"_restore_requested", False):
            try:
                restore(payload, selected)
            except (ValueError, OSError) as exc:
                st.error(str(exc))
