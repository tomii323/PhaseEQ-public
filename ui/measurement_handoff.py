"""Explicit measurement-to-Multiway handoff from the measurement library."""
from pathlib import Path
import hashlib
import sqlite3
import streamlit as st
import pandas as pd
from utils.list_menu_ui import single_row_table_selector, database_candidates
from utils.speaker_db import list_measurement_summaries
from utils.db_viewmodels import measurement_table

from utils.ui_localization import display_text
from utils.composite_exchange import read_multiway_workspace
from utils.multiway_measurements import set_measurement_input, read_measurement_inputs, workspace_key
from utils.speaker_db import SpeakerMeasurementRecord, get_measurement, save_measurement, record_measurement_multiway_project
from utils.settings_io import speaker_response_payload
from utils.response_import import load_uploaded_speaker_response


def _send_and_record_project(root, db_path, workspace, channel, record):
    item = set_measurement_input(root, workspace, channel, record)
    try:
        record_measurement_multiway_project(db_path, record.id, item.get('project_name', ''))
    except (ValueError, OSError, sqlite3.Error) as exc:
        raise ValueError(display_text('チャンネルへのセットは保存済みですが、DBへのプロジェクト名の記録に失敗しました。再試行してください。') + str(exc)) from exc


def render_measurement_assignment(root, db_path):
    st.subheader(display_text('マルチウェイに割り当て'))
    query = st.text_input(display_text('測定データを検索'), key='measurement_assignment_query')
    records = [record for record in database_candidates(
        list_measurement_summaries, db_path, list_id='measurement_assignment',
        query=query, include_archived=False, descending=True,
    ) if record.source_type == 'speaker_input_raw']
    selected = None
    if records:
        identifier = single_row_table_selector(
            measurement_table(records), [record.id for record in records],
            selected_state_key='input_speaker_measurement_selected',
            table_key='measurement_assignment_table', height=240,
        )
        selected = get_measurement(db_path, identifier)
    else:
        st.info(display_text('該当する測定データがありません。'))
    render_measurement_handoff(root, db_path, selected)


def render_measurement_handoff(root, db_path, selected_record):
    with st.container():
        if st.session_state.get('_measurement_handoff_notice'):
            st.success(display_text(st.session_state['_measurement_handoff_notice']))
        st.caption(display_text('送信先を手動で選びます。選択したチャンネルの測定応答だけを置き換えます。'))
        if st.button(display_text('送信先・受信状態を更新'), key='measurement_dest_refresh'):
            st.rerun()
        try:
            workspace = read_multiway_workspace(root)
            if workspace is None or not workspace.get('channels'):
                st.info(display_text('先にマルチウェイを開き、チャンネル構成を設定してください。'))
                return
            roster = workspace['channels']
            st.caption(display_text('送信先プロジェクト：') + str(workspace.get('system_name') or display_text('名称未設定')))
            st.caption(display_text('セットするとプロジェクト名を測定データのメモへ追記します。一覧のプロジェクト列で確認でき、検索欄で絞り込めます。'))
            options = {row['channel_id']: f"{row['group']} / {row['name']} ({row['way']})" for row in roster}
            scope = workspace_key(workspace)
            inputs = read_measurement_inputs(root, workspace)
            st.caption(display_text('送信先のチャンネル行を選択してください。'))
            frame = pd.DataFrame([{
                display_text('チャンネル'): row['name'],
                display_text('グループ'): row['group'],
                display_text('帯域'): row['way'],
                display_text('セット中の測定データ'): inputs.get(row['channel_id'], {}).get('name', '—'),
            } for row in roster])
            channel = single_row_table_selector(
                frame, list(options), selected_state_key='measurement_destination_' + scope,
                table_key='measurement_destination_table_' + scope, height=180, required=False,
            )
            if channel:
                st.caption(display_text('選択中の送信先：') + options[channel])
            existing, upload_tab = st.tabs([display_text('選択中の測定データを使う'), display_text('ファイルを登録してセット')])
            with existing:
                if selected_record is not None:
                    st.write(f"{selected_record.name} → {options[channel]}" if channel else selected_record.name)
                    send_label = f"{options[channel]} · {display_text('マルチウェイに割り当て')}" if channel else display_text('この測定データをセット')
                    if st.button(send_label, disabled=not channel, key='measurement_send_selected'):
                        _send_and_record_project(root, db_path, workspace, channel, selected_record)
                        st.session_state['_measurement_handoff_open'] = True
                        st.session_state['_measurement_handoff_notice'] = '受け渡しを保存しました。マルチウェイと、このチャンネルを編集中のPhaseEQで順次読み込みます。'
                        st.rerun()
                else:
                    st.info(display_text('一覧で測定データを選択してください。'))
            with upload_tab:
                upload = st.file_uploader(display_text('スピーカー測定ファイル'), type=['frd', 'txt', 'csv'], key='measurement_send_upload')
                if upload is not None:
                    digest = hashlib.sha256(upload.getvalue()).hexdigest()
                    name = st.text_input(display_text('登録する測定名'), value=Path(upload.name).stem, key='measurement_upload_name_' + digest)
                    calibrated = st.checkbox(display_text('この測定データには、すでにマイク校正が適用されている'), key='measurement_upload_cal_' + digest)
                    if st.button(display_text('DBに登録してチャンネルにセット'), disabled=not channel or not name.strip(), key='measurement_upload_send'):
                        # Keep the saved ID for retries if filesystem handoff fails.
                        saved_key = '_measurement_upload_saved_' + hashlib.sha256((digest + name + str(calibrated)).encode()).hexdigest()
                        saved = get_measurement(db_path, st.session_state[saved_key]) if saved_key in st.session_state else None
                        if saved is None:
                            response = load_uploaded_speaker_response(upload).speaker_response
                            payload = speaker_response_payload(response)
                            payload['external_mic_calibration_applied'] = calibrated
                            saved = save_measurement(db_path, record=SpeakerMeasurementRecord(
                                id='', name=name.strip(), source_name=upload.name, source_type='speaker_input_raw', response_payload=payload))
                            st.session_state[saved_key] = saved.id
                        try:
                            _send_and_record_project(root, db_path, workspace, channel, saved)
                        except (ValueError, OSError) as exc:
                            st.error(display_text('DBへの登録は完了しています。受け渡し処理を再試行してください。') + str(exc))
                        else:
                            st.session_state['_measurement_handoff_open'] = True
                            st.session_state['_measurement_handoff_notice'] = 'DBへの登録と受け渡しを保存しました。マルチウェイと、このチャンネルを編集中のPhaseEQで順次読み込みます。'
                            st.rerun()
            inputs = read_measurement_inputs(root, workspace)
            if inputs:
                st.caption(display_text('セットした測定データ'))
                for channel_id, item in inputs.items():
                    received = (Path(root) / 'measurement_inputs' / 'received' / (item['request_id'] + '.json')).exists()
                    st.write(f"{options.get(channel_id, channel_id)} → {item['name']} · " + display_text('受信済み' if received else '受信待ち'))
        except (ValueError, OSError, sqlite3.Error) as exc:
            st.error(str(exc))
