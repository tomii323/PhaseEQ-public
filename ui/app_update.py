"""Update presentation; network and installation transactions live in runtime."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from runtime.app_update import cancel_pending, latest_release, register_running_app, stage_release, update_directory, writable_installation
from utils.ui_language import current_language

TEXT = {
    'title': ('PhaseEQ本体の更新', 'Update PhaseEQ'),
    'source': ('更新元: GitHubの公開Release。設定・測定DB・ユーザープリセット・出力データは保持します。',
               'Source: public GitHub Releases. Settings, measurement databases, user presets and outputs are preserved.'),
    'development': ('本体更新は、書き込み可能な公式公開ZIPの展開先で利用できます。Git管理中の開発環境とファイル一覧のない配布物は対象外です。',
                    'App updates require a writable official public ZIP installation. Git checkouts and packages without a public file inventory are excluded.'),
    'check': ('最新版を確認', 'Check for updates'),
    'available': ('最新版 v{version}があります。設定の「PhaseEQ本体の更新」で更新できます。',
                  'Version {version} is available. Open Update PhaseEQ in Settings to prepare it.'),
    'prepare': ('更新をダウンロードして予約', 'Download and schedule update'),
    'ready': ('v{version}の更新を予約しました。設定を保存し、PhaseEQとマルチウェイを両方終了して、起動スクリプトから起動し直してください。',
              'Version {version} is ready. Save your work, close PhaseEQ and Multiway, then restart using the launcher.'),
    'cancel': ('更新予約を取り消す', 'Cancel scheduled update'),
    'current': ('現在のバージョンは v{version}です。新しい正式Releaseはありません。',
                'Current version: {version}. No newer stable release is available.'),
    'failed': ('最新版を確認できませんでした。現在のバージョンで作業を続けられます。',
               'Could not check for updates. You can continue using the current version.'),
    'download_failed': ('更新を準備できませんでした。本体は変更していません。',
                        'Could not prepare the update. The application was not changed.'),
    'release': ('更新内容をGitHubで確認', 'View release notes on GitHub'),
    'details': ('詳細', 'Details'),
    'fetching': ('更新ファイルを取得・検証しています…', 'Downloading and verifying update…'),
}


def text(key, **values):
    return TEXT[key][current_language() == 'en'].format(**values)


def check_for_updates(root, current):
    try:
        st.session_state['_app_update_release'] = latest_release(current)
        st.session_state.pop('_app_update_error', None)
    except Exception as exc:
        st.session_state['_app_update_release'] = None
        st.session_state['_app_update_error'] = str(exc)
    st.session_state['_app_update_checked'] = True


def initialize_updates(root, current):
    if not writable_installation(root):
        return
    register_running_app(root)
    if not st.session_state.get('_app_update_checked') and not (update_directory(root) / 'pending.json').exists():
        check_for_updates(root, current)


def render_update_notice():
    release = st.session_state.get('_app_update_release')
    if release:
        st.info(text('available', version=release['version']))


def render_update_settings(root: Path, current: str):
    with st.container(border=True):
        st.subheader(text('title'))
        st.caption(text('source'))
        if not writable_installation(root):
            st.info(text('development'))
            return
        pending = update_directory(root) / 'pending.json'
        if pending.exists():
            try:
                queued = json.loads(pending.read_text(encoding='utf-8'))
                st.success(text('ready', version=queued['version']))
                if st.button(text('cancel'), key='app_update_cancel'):
                    cancel_pending(root)
                    st.rerun()
            except (OSError, ValueError, KeyError):
                st.error(text('download_failed'))
            return
        if st.button(text('check'), key='app_update_check'):
            check_for_updates(root, current)
        if st.session_state.get('_app_update_error'):
            st.info(text('failed'))
            with st.expander(text('details')):
                st.code(st.session_state['_app_update_error'])
        release = st.session_state.get('_app_update_release')
        if not release:
            if not st.session_state.get('_app_update_error'):
                st.caption(text('current', version=current))
            return
        st.info(text('available', version=release['version']))
        st.link_button(text('release'), release['release_url'])
        if st.button(text('prepare'), key='app_update_prepare'):
            try:
                with st.spinner(text('fetching')):
                    stage_release(root, release)
            except Exception as exc:
                st.error(text('download_failed'))
                with st.expander(text('details')):
                    st.code(str(exc))
            else:
                st.rerun()
