"""Library actions backed by the saved integrated measurement original."""
from utils.ui_localization import ui_message, display_text, localized_formatter
from pathlib import Path

import streamlit as st

from phase_fir_designer.measurement.original import calibrated_export_zip
from utils.ui_work_cache import deferred_call
from utils.measurement_originals import load_measurement_original, recalibrate_measurement
from utils.speaker_db import list_measurement_summaries


def _export(path: Path, record_id: str, kind: str) -> bytes:
    return calibrated_export_zip(load_measurement_original(path, record_id), kind)


def render_original_actions(path: Path, record) -> None:
    session = (record.response_payload or {}).get('measurement_session', {})
    if not isinstance(session, dict) or not session.get('original'):
        return
    with st.expander(ui_message('ui.63efae34388297')):
        st.caption(ui_message('ui.cccc77be276819'))
        kind = st.selectbox(ui_message('ui.01267a2a66fb1c'), ['merged', 'ungated', 'gated'],
                            format_func=localized_formatter(lambda x: {'merged': '標準（Merge）', 'ungated': 'Gateなし', 'gated': 'Gateあり'}[x]),
                            key=f'original_export_kind_{record.id}')
        st.download_button(ui_message('ui.e8d51ba8561cc3'), data=deferred_call(_export, path, record.id, kind),
                           file_name=f'measurement_{record.id}_calibrated.zip', mime='application/zip',
                           key=f'original_export_{record.id}', on_click='ignore')
        calibrations = [item for item in list_measurement_summaries(path) if item.source_type == 'mic_calibration_raw']
        if calibrations:
            choices = {item.id: item.name for item in calibrations}
            current = session.get('mic_calibration_id')
            ids = list(choices)
            chosen = st.selectbox(ui_message('ui.d3a30737f6844a'), ids, format_func=choices.get,
                                  index=ids.index(current) if current in ids else 0,
                                  key=f'original_calibration_{record.id}')
            st.caption(ui_message('ui.83b36de894ef69'))
            if st.button(ui_message('ui.42b40c82c77d8f'), key=f'original_recalibrate_{record.id}'):
                try:
                    recalibrate_measurement(path, record.id, chosen)
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
                else:
                    st.success(ui_message('ui.8936eac38f4e95'))
                    st.rerun()
