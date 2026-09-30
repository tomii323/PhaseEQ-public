"""On-demand memory inspection without retaining response data in reports."""
from utils.ui_localization import display_text
import json
import os
import pandas as pd
import streamlit as st
from utils.memory_diagnostics import capture_memory, diagnostic_roots, inspect_memory, process_memory


def _table(rows):
    return pd.DataFrame([{**row, 'MiB': round(row['bytes'] / 2**20, 3)} for row in rows])


def render_memory_diagnostics(namespace=None):
    if os.environ.get('PHASEEQ_TEST_MEMORY_DIAGNOSTICS') != '1':
        return
    st.markdown(display_text('#### メモリ使用量の調査'))
    st.caption(display_text('ボタンを押した時だけ計測します。設定値・測定サンプルは診断結果に含めません。'))
    if st.button(display_text('メモリ使用量を計測'), key='_memory_diagnostic_capture'):
        report = capture_memory(st.session_state, namespace)
        st.session_state['_memory_diagnostic_report'] = report
        history = list(st.session_state.get('_memory_diagnostic_history', []))
        history.append({**report['process'], 'reachable_bytes': report['total_bytes']})
        st.session_state['_memory_diagnostic_history'] = history[-30:]
        st.session_state.pop('_memory_diagnostic_detail', None)
    report = st.session_state.get('_memory_diagnostic_report')
    if not report:
        return
    process = report['process']
    st.caption(f"{display_text('計測日時')}: {process['timestamp']} / PID {process['pid']}")
    st.write(f"RSS: {process.get('rss_bytes', 0) / 2**20:.1f} MiB / {display_text('調査対象の保持量')}: {report['total_bytes'] / 2**20:.1f} MiB")
    if process.get('rss_error'):
        st.warning(display_text('RSSを取得できません: ') + process['rss_error'])
    if report['truncated']:
        st.warning(display_text('調査上限に達しました。保持量は下限値です。対象を選んで詳しく調査してください。'))
    st.caption(display_text('共有参照はキー順で最初の項目に計上します。配列のviewは元バッファと重複計上しません。数値リストのスカラーは概算です。pandas共有領域・外部ライブラリ・別セッション・ブラウザーは完全には計測できません。'))
    for row in report['rows'][:5]:
        st.caption(f"{row['key']}: {row['bytes'] / 2**20:.3f} MiB")
    st.dataframe(_table(report['rows'][:30]), hide_index=True, width='stretch')
    options = [row['key'] for row in report['rows']]
    selected = st.selectbox(display_text('詳しく調べる保持項目'), options, key='_memory_diagnostic_selected') if options else None
    if selected and st.button(display_text('選択項目の内訳を調査'), key='_memory_diagnostic_inspect'):
        roots = diagnostic_roots(st.session_state, namespace)
        if selected in roots:
            detail = inspect_memory({selected: roots[selected]}, max_nodes=1_000_000, max_seconds=8)
            st.session_state['_memory_diagnostic_detail'] = {'key': selected, **detail}
        else:
            st.warning(display_text('対象が更新・解放されました。再計測してください。'))
    detail = st.session_state.get('_memory_diagnostic_detail')
    if detail and detail['key'] == selected:
        st.write(f"{display_text('この項目から参照される保持量')}: {detail['total_bytes'] / 2**20:.3f} MiB")
        st.caption(display_text('内訳は親子を含むため合計しません。bytesは共有除外後、logical_bytesは配列の論理サイズです。'))
        if detail['truncated']:
            st.warning(display_text('内訳も上限に達しました。表示値は下限値です。'))
        for item in detail['details'][:8]:
            st.caption(f"{item['path']}: {item['bytes'] / 2**20:.3f} MiB / {item['type']} / {display_text('要素数')} {item.get('length', '—')} / {item['shape']} {item['dtype']}")
        st.dataframe(_table(detail['details']), hide_index=True, width='stretch')
    st.markdown(display_text('**Streamlit共有キャッシュ・アップロード・配信ファイル**'))
    if report['streamlit'].get('error'):
        st.warning(display_text('共有キャッシュ統計を取得できません: ') + report['streamlit']['error'])
    for item in report['streamlit']['rows'][:5]:
        st.caption(f"{item['category']} / {item['key']}: {item['bytes'] / 2**20:.3f} MiB")
    st.dataframe(_table(report['streamlit']['rows']), hide_index=True, width='stretch')
    st.caption(display_text('共有キャッシュと復元後のデータは別に保持されます。上表の保持量とRSSは同じ尺度ではありません。'))
    history = st.session_state.get('_memory_diagnostic_history', [])
    st.markdown(display_text('**計測履歴（直近30回）**'))
    st.dataframe(pd.DataFrame(history), hide_index=True, width='stretch')
    st.download_button(display_text('メモリ診断JSONを保存'), json.dumps({'report': report, 'detail': detail, 'history': history}, ensure_ascii=False, indent=2),
                       file_name='PhaseEQ-memory.json', mime='application/json', on_click='ignore')
