"""Metadata-only editing and the independent microphone library."""
import sqlite3
import streamlit as st
import pandas as pd

from utils.ui_localization import display_text as tr
from utils.list_menu_ui import single_row_table_selector, database_candidates
from utils.speaker_db import (list_measurement_summaries, get_measurement, list_speaker_specs,
    get_speaker_spec, measurement_multiway_projects, update_measurement_details)


def render_measurement_details(db_path, specs_path, *, navigate=None):
    st.subheader(tr('測定条件・メモ'))
    if st.session_state.pop('measurement_notes_saved', False):
        st.success(tr('測定条件・メモを保存しました。'))
    if navigate is not None and st.button(tr('測定一覧・使用へ戻る'), key='measurement_notes_back'):
        navigate('Project', input_speaker_measurement_selected=str(
            st.session_state.get('input_speaker_measurement_selected', '')))
        st.rerun()
    st.caption(tr('一覧で編集する測定データを1件チェックしてください。変更は「変更を保存」で確定してから、別の測定を選択してください。'))
    query = st.text_input(tr('測定データを検索'), key='measurement_notes_query')
    records = database_candidates(list_measurement_summaries, db_path,
                                  list_id='measurement_notes', query=query)
    if not records:
        st.info(tr('該当する測定データがありません。'))
        return
    table = pd.DataFrame([{
        tr('測定名'): record.name,
        tr('測定日'): record.measurement_date,
        tr('測定チャンネル'): record.channel,
        tr('利用先プロジェクト'): ' / '.join(measurement_multiway_projects(record)),
        tr('入力元'): record.source_name,
        tr('メモ'): '\n'.join(line for line in record.note.splitlines()
                              if not line.startswith('Multiway project: ')),
    } for record in records])
    selected_id = single_row_table_selector(
        table, [record.id for record in records],
        selected_state_key='input_speaker_measurement_selected',
        table_key='measurement_notes_table', required=False,
        height=min(320, 36 + 35 * len(records)))
    selected = next((record for record in records if record.id == selected_id), None)
    if selected is None:
        st.info(tr('一覧で編集する測定データを選択してください。'))
        return
    st.markdown('**' + tr('編集中の測定：') + selected.name + '**')
    projects = measurement_multiway_projects(selected)
    st.caption(tr('利用先プロジェクト：') + (' / '.join(projects) if projects else tr('未登録')))
    if selected.speaker_spec_id and navigate is not None:
        if st.button(tr('関連付けた仕様へ'), key='measurement_view_linked_spec'):
            navigate('Speaker Specs', _speaker_db_load_user_spec_id=selected.speaker_spec_id,
                     _speaker_db_select_user_spec_id=selected.speaker_spec_id, _speaker_specs_edit_pending=True)
            st.rerun()
    st.caption(tr('メモと測定条件だけを保存します。測定カーブとマイク校正状態は変更しません。'))
    prefix = 'measurement_details_' + selected.id + '_' + selected.updated_at
    with st.form(prefix):
        note = st.text_area(tr('メモ'), value='\n'.join(line for line in selected.note.splitlines() if not line.startswith('Multiway project: ')), height=120)
        values = {}
        labels = {'name':'測定名', 'brand':'メーカー', 'model':'型番', 'measurement_date':'測定日',
                  'distance':'測定距離', 'angle':'測定角度', 'microphone':'使用マイク', 'location':'測定場所',
                  'system_name':'測定したシステム名', 'channel':'測定チャンネル', 'position_name':'測定位置',
                  'system_state':'測定時のシステム状態', 'correction_note':'測定時の補正・処理の記録'}
        for field in ('name', 'measurement_date', 'distance', 'angle'):
            values[field] = st.text_input(tr(labels[field]), value=getattr(selected, field))
        with st.expander(tr('その他の測定条件'), expanded=False):
            for field in ('brand','model','microphone','location','system_name','channel','position_name','system_state','correction_note'):
                values[field] = st.text_input(tr(labels[field]), value=getattr(selected, field))
            if selected.source_type == 'speaker_input_raw':
                scopes = {'system_channel': 'システムの単一チャンネル', 'combined_room': '左右合成・室内応答', 'driver': 'スピーカーユニット単体'}
                uses = {'correction_input': '補正設計に使用', 'verification_only': '確認用', 'archive': '記録用'}
                for field, label, options in [('measurement_scope', '測定対象', scopes), ('intended_use', '用途', uses)]:
                    current = getattr(selected, field)
                    if current not in options:
                        options[current] = current
                    values[field] = st.selectbox(tr(label), list(options), index=list(options).index(current), format_func=lambda value, names=options: tr(names[value]))
        with st.expander(tr('スピーカー仕様を関連付ける（任意）'), expanded=False):
            st.caption(tr('仕様の登録は、独立したスピーカー仕様データベースで行います。'))
            specs = list_speaker_specs(specs_path, limit=1000)
            if selected.speaker_spec_id and not any(r.id == selected.speaker_spec_id for r in specs):
                linked = get_speaker_spec(specs_path, selected.speaker_spec_id)
                if linked:
                    specs.append(linked)
            names = {r.id: f'{r.brand} / {r.model}' for r in specs}
            if selected.speaker_spec_id and selected.speaker_spec_id not in names:
                names[selected.speaker_spec_id] = selected.speaker_spec_id
            options = ['', *names]
            values['speaker_spec_id'] = st.selectbox(tr('関連付ける仕様'), options,
                index=options.index(selected.speaker_spec_id) if selected.speaker_spec_id in options else 0,
                format_func=lambda key: names.get(key, tr('関連付けなし')))
        if st.form_submit_button(tr('変更を保存'), type='primary'):
            try:
                update_measurement_details(db_path, selected.id, details=values, note=note)
            except (ValueError, OSError, sqlite3.Error) as exc:
                st.error(str(exc))
            else:
                st.session_state['measurement_notes_saved'] = True
                st.rerun()


def render_microphone_profiles(mic_path, measurement_path):
    from utils.microphone_db import list_microphone_profiles, list_microphone_units
    from ui.guided_measurement_view import _register_microphone
    st.subheader(tr('マイクプロファイル'))
    profiles = {p.id: p for p in list_microphone_profiles(mic_path)}
    st.caption(tr('対応機種と登録済みのマイク個体を確認し、個体番号と校正ファイルを登録します。'))
    st.dataframe(pd.DataFrame([{tr('メーカー'): p.manufacturer, tr('型番'): p.model, tr('接続'): p.connection_type} for p in profiles.values()]), hide_index=True, width='stretch')
    units = list_microphone_units(mic_path)
    st.dataframe(pd.DataFrame([{tr('型番'): profiles[u.profile_id].model if u.profile_id in profiles else u.profile_id,
        tr('個体番号'): u.serial_number, tr('校正データ'): u.calibration_measurement_id,
        tr('90度校正データ'): u.calibration_90_measurement_id} for u in units]), hide_index=True, width='stretch')
    if profiles:
        _register_microphone(mic_path, measurement_path, profiles, expanded=not units)
