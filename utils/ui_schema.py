# -*- coding: utf-8 -*-
"""APPY UI State Registry schema for PhaseEQ.

This module keeps UI structure, profile keys, and dependency definitions in one
place so the app can migrate page-by-page without changing DSP behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .ui_index import UISection


@dataclass(frozen=True)
class UIPageResultStage:
    """Right-pane result stage assigned to a navigation page."""

    stage: str
    description: str
    future_view: str = ""


UI_PAGE_OPTIONS = [
    "Project",
    "Multiway Assignment",
    "Library",
    "Measurement Notes",
    "Mic Library",
    "Mic Profiles",
    "Speaker Specs",
    "Input",
    "Target",
    "IIR EQ",
    "FIR EQ",
    "Linear FIR",
    "Export",
    "DSP System",
    "Measure",
    "Common Settings",
    "Backup & Restore",
]
UI_PAGE_GROUP_BREAK_AFTER = {"Export", "DSP System"}
UI_PAGE_TITLES = {
    "Multiway Assignment": ("MEASUREMENT", "Multiway Assignment", "測定データと送信先チャンネルを選び、マルチウェイに割り当てます。"),
    "Project": ("MEASUREMENTS", "Measurement List & Use", "測定データを検索・確認し、補正設計に使用します。"),
    "DSP System": ("DSP SYSTEM", "System", "Driver ProfileをOutput Channelへ割り当て、DSP出力までを管理します。"),
    "Library": ("REGISTER", "Register Measurement", "測定ファイルや現在の入力を登録します。"),
    "Measure": ("OPTIONAL TOOL", "Continuous ESS", "外部ESSを連続録音し、ショット単位で解析・保持します。通常フローでは必須ではありません。"),
    "Input": ("RESPONSE PROCESSING", "Response Processing", "測定応答の選択、正規化、位相基準、レベル調整を行います。"),
    "Target": ("TARGET", "Target Response", "目標応答の読み込み、編集、出力を行います。"),
    "IIR EQ": ("SOURCE CORRECTION", "IIR EQ", "Speaker/InputへIIRを適用し、Gain EQへ渡す特性を調整します。"),
    "FIR EQ": (
        "FIR CORRECTION",
        "FIR EQ",
        "Gain／Phase／Auto Gain／Auto Phaseを一覧から選択して編集します。",
    ),
    "Linear FIR": ("BAND SPLIT", "Output Band Split", "境界ごとにFIRまたはLinkwitz–Riley IIRの帯域分割方式を編集します。"),
    "Export": ("OUTPUT", "Export", "最終FIRと出力形式を設定します。"),
    "Common Settings": (
        "APPLICATION",
        "Preferences",
        "PhaseEQ全体で共有する解析条件を管理します。",
    ),
    "Backup & Restore": (
        "APPLICATION",
        "Backup & Restore",
        "設定、作業セッション、測定DB、Driver Databaseをバックアップ・復元します。",
    ),
}
UI_PAGE_TITLES.update({
    "Measurement Notes": ("MEASUREMENT", "測定条件・メモ", "登録済み測定の条件・メモを編集します。"),
    "Mic Library": ("CALIBRATION", "校正データ一覧・登録", "マイク校正データを管理します。"),
    "Mic Profiles": ("MICROPHONE", "マイクプロファイル", "マイク個体と校正ファイルを管理します。"),
    "Speaker Specs": ("OPTIONAL", "スピーカー仕様データベース（任意）", "測定データの登録には不要な、スピーカーユニット仕様の管理機能です。"),
})
UI_PAGE_RESULT_STAGES = {
    "DSP System": UIPageResultStage(stage="DSP System", description="各Wayと音響合成結果を表示します。"),
    "Project": UIPageResultStage(
        stage="Speaker/Input",
        description="選択中Speaker PackageのInput Profileを表示します。",
        future_view="Input Profile改訂履歴と測定出典を確認できます。",
    ),
    "Library": UIPageResultStage(
        stage="Speaker/Input",
        description="Speaker/Input処理後の状態を表示します。Libraryから復元したraw responseもここへ合流します。",
        future_view="メーカーURL取得、メーカー情報貼り付け、測定履歴の再利用を拡張できます。",
    ),
    "Measure": UIPageResultStage(
        stage="Speaker/Input",
        description="連続ESS測定の最新ショットと複素平均を表示します。",
    ),
    "Input": UIPageResultStage(
        stage="Speaker/Input",
        description="Speaker/Input処理後の状態を表示します。",
    ),
    "Target": UIPageResultStage(
        stage="Target Response",
        description="Speaker/InputとTarget Responseまでの状態を表示します。",
    ),
    "IIR EQ": UIPageResultStage(
        stage="IIR EQ",
        description="Speaker/InputへIIRを適用した状態を表示します。IIR係数はFIRへ畳み込みません。",
    ),
    "FIR EQ": UIPageResultStage(
        stage="Auto Phase EQ",
        description="行選択に関係なく、ONの手動Gain・Phase・Auto EQをすべて表示します。",
    ),
    "Linear FIR": UIPageResultStage(
        stage="Linear FIR",
        description="Linear FIRを畳み込み合成した状態を表示します。",
    ),
    "Export": UIPageResultStage(
        stage="Export FIR",
        description="Exportタブで実際に出力するFIRフィルターを表示します。",
    ),
    "Common Settings": UIPageResultStage(
        stage="Speaker/Input",
        description="PhaseEQ共通の解析条件と実効値を表示します。",
    ),
    "Backup & Restore": UIPageResultStage(
        stage="Speaker/Input",
        description="バックアップと復元の管理中です。現在の設計応答は変更しません。",
    ),
}

TARGET_FILTER_REGISTRY_ADAPTERS = {
    "target_gain_peq_items": {
        "prefix": "target_gain_peq",
        "count_key": "target_gain_peq_count",
        "item_type": "target_gain_peq",
        "fields": {
            "enabled": ("enabled", True),
            "fc": ("fc", 1000.0),
            "q": ("q", 1.0),
            "gain_db": ("gain_db", 0.0),
        },
    },
    "target_gain_shelf_items": {
        "prefix": "target_gain_shelf",
        "count_key": "target_gain_shelf_count",
        "item_type": "target_gain_shelf",
        "fields": {
            "enabled": ("enabled", True),
            "mode": ("mode", "low"),
            "fc": ("fc", 1000.0),
            "q": ("q", 1.0),
            "gain_db": ("gain_db", 0.0),
        },
    },
    "target_gain_linear_items": {
        "prefix": "target_gain_linear",
        "count_key": "target_gain_linear_count",
        "item_type": "target_gain_linear",
        "fields": {
            "enabled": ("enabled", True),
            "mode": ("mode", "lp"),
            "response": ("response", "lr2"),
            "fc": ("fc", 1000.0),
            "cycles": ("cycles", 4.0),
            "beta": ("beta", 12.0),
        },
    },
    "target_gain_tilt_items": {
        "prefix": "target_gain_tilt",
        "count_key": "target_gain_tilt_count",
        "item_type": "target_gain_tilt",
        "fields": {
            "enabled": ("enabled", True),
            "mode": ("mode", "high"),
            "f_start": ("f_start", 100.0),
            "f_end": ("f_end", 20_000.0),
            "slope_db_per_oct": ("slope_db_per_oct", 0.0),
        },
    },
    "target_gain_lo_tilt_items": {
        "prefix": "target_gain_lo_tilt",
        "count_key": "target_gain_lo_tilt_count",
        "item_type": "target_gain_lo_tilt",
        "fields": {
            "enabled": ("enabled", True),
            "mode": ("mode", "low"),
            "f_start": ("f_start", 1.0),
            "f_end": ("f_end", 1000.0),
            "slope_db_per_oct": ("slope_db_per_oct", 0.0),
        },
    },
    "target_phase_peq_items": {
        "prefix": "target_phase_peq",
        "count_key": "target_phase_peq_count",
        "item_type": "target_phase_peq",
        "fields": {
            "enabled": ("enabled", True),
            "fc": ("fc", 1000.0),
            "q": ("q", 1.0),
            "phase_deg": ("phase_deg", 0.0),
        },
    },
    "target_phase_shelf_items": {
        "prefix": "target_phase_shelf",
        "count_key": "target_phase_shelf_count",
        "item_type": "target_phase_shelf",
        "fields": {
            "enabled": ("enabled", True),
            "mode": ("mode", "low"),
            "fc": ("fc", 1000.0),
            "q": ("q", 1.0),
            "phase_deg": ("phase_deg", 0.0),
        },
    },
    "target_phase_tilt_items": {
        "prefix": "target_phase_tilt",
        "count_key": "target_phase_tilt_count",
        "item_type": "target_phase_tilt",
        "fields": {
            "enabled": ("enabled", True),
            "f_start": ("f_start", 100.0),
            "f_end": ("f_end", 20_000.0),
            "slope_deg_per_oct": ("slope_deg_per_oct", 0.0),
        },
    },
    "target_phase_allpass_items": {
        "prefix": "target_phase_allpass",
        "count_key": "target_phase_allpass_count",
        "item_type": "target_phase_allpass",
        "fields": {
            "enabled": ("enabled", True),
            "polarity": ("polarity", "positive"),
            "fc": ("fc", 1000.0),
            "q": ("q", 1.0),
        },
    },
    "target_iir_filter_items": {
        "prefix": "target_iir_filter",
        "count_key": "target_iir_filter_count",
        "item_type": "target_iir_filter",
        "fields": {
            "enabled": ("enabled", True),
            "kind": ("kind", "high_pass"),
            "fc": ("fc", 1000.0),
            "q": ("q", 0.7),
            "gain_db": ("gain_db", 0.0),
            "family": ("family", "linkwitz_riley"),
            "filter_order": ("filter_order", 4),
            "ripple_db": ("ripple_db", 0.1), "stop_db": ("stop_db", 80.0),
            "allpass_preset": ("allpass_preset", "lr4"),
            "origin": ("origin", "manual"),
        },
    },
}

for _measurement_page in ("Multiway Assignment", "Measurement Notes", "Mic Library", "Mic Profiles", "Speaker Specs"):
    UI_PAGE_RESULT_STAGES[_measurement_page] = UIPageResultStage(stage="Speaker/Input", description=UI_PAGE_TITLES[_measurement_page][2])

UI_SECTION_GROUPS = (
    (
        "workspace",
        "Workspace",
        ("Project", "Multiway Assignment", "Library", "Measurement Notes", "Mic Library", "Mic Profiles", "Speaker Specs"),
        "プロジェクト管理の入口です。",
    ),
    (
        "design_flow",
        "Design Flow",
        ("Input", "Target", "IIR EQ", "FIR EQ", "Linear FIR", "Export"),
        "LibraryからExportまでの通常編集フローです。",
    ),
    (
        "system_build",
        "System Build",
        ("DSP System",),
        "完成したSpeaker PackageをOutputへ接続し、Wayを割り当ててシステムを構成します。",
    ),
    (
        "tools",
        "Tools",
        ("Measure", "Backup & Restore"),
        "必要な場合だけ使う補助ツールです。",
    ),
)


def _build_section_tree() -> tuple[UISection, ...]:
    return tuple(
        UISection(
            id=group_id,
            label=group_label,
            status="unset",
            disabled=True,
            description=group_description,
            children=tuple(
                UISection(
                    id=page,
                    label=page,
                    status="unset",
                    description=UI_PAGE_TITLES[page][2],
                )
                for page in pages
            ),
        )
        for group_id, group_label, pages, group_description in UI_SECTION_GROUPS
    )


APP_SECTIONS = _build_section_tree()


UI_CARD_GROUPS = {
    "Project": (
        ("project_database", "Design Library"),
        ("project_identity", "Project identity"),
        ("project_fir_analysis", "FIR and analysis settings"),
        ("project_reset", "Reset"),
        ("settings_management_help", "Settings note"),
        ("settings_registry_diagnostics", "APPY UI State Registry diagnostics"),
    ),
    "Input": (
        ("input_source_block", "Basic: Source"),
        ("input_phase_timing_block", "Basic: Phase / Timing"),
        ("input_response_processing_block", "Basic: Response Processing"),
        ("input_diagnostics_block", "Info / Diagnostics"),
    ),
    "Measure": (),
    "Target": (
        ("target_source_level", "Target source and level"),
        ("target_gain_eq", "Target Gain EQ"),
        ("target_gain_tilt", "Target Gain Hi Tilt"),
        ("target_phase_eq", "Phase EQ"),
        ("target_phase_tilt", "Target Phase Hi Tilt"),
    ),
    "Linear FIR": (
        ("linear_fir_filters", "Linear FIR"),
    ),
    "Export": (),
    "Common Settings": (),
    "Live Result": (
        ("live_analysis_gain_phase", "Analysis Gain / Phase"),
        ("live_detail", "Detail"),
        ("live_phase_note", "Auto Phase / 位相表示について"),
        ("live_input_details", "入力処理の詳細"),
        ("live_export_fir_note", "Export FIR表示について"),
        ("wavelet_settings_status", "Wavelet settings / status"),
    ),
}

UI_CARD_IDS = frozenset(card_id for cards in UI_CARD_GROUPS.values() for card_id, _label in cards)


def ui_card_sections() -> tuple[UISection, ...]:
    """Return the page -> card hierarchy managed by the UI card state."""
    return tuple(
        UISection(
            id=page,
            label=page,
            children=tuple(UISection(id=card_id, label=label) for card_id, label in cards),
        )
        for page, cards in UI_CARD_GROUPS.items()
    )


def known_ui_card(card_id: str) -> bool:
    """Return True when card_id is part of the declared APPY UI card hierarchy."""
    return card_id in UI_CARD_IDS

PROFILE_SCALAR_KEYS = (
    "active_page",
    "project_edit_mode",
    "speaker_db_source",
    "input_edit_mode",
    "input_processing_type",
    "input_level_reference_mode",
    "input_level_reference_low",
    "input_level_reference_high",
    "export_edit_mode",
    "target_edit_mode",
    "target_gain_filter_group",
    "target_phase_filter_group",
    "gain_edit_group",
    "phase_edit_group",
    "fir_eq_view",
    "fir_eq_task",
    "linear_fir_profile",
    "current_project_id",
    "project_name",
    "driver_band_enabled",
    "driver_band_name",
    "driver_band_hpf_hz",
    "driver_band_lpf_hz",
    "project_db_search",
    "project_db_sort",
    "project_db_sort_desc",
    "project_db_show_archived",
    "project_db_name",
    "project_db_speaker_type",
    "project_db_side",
    "project_db_position",
    "project_db_tags",
    "project_db_note",
    "sample_rate",
    "taps",
    "analysis_fft_size",
    "analysis_fft_saved_size",
    "display_fft_multiplier",
    "window",
    "kaiser_beta",
    "chebyshev_attenuation_db",
    "tukey_alpha",
    "result_detail_view",
    "graph_mode",
    "gain_y_min_db",
    "gain_y_max_db",
    "phase_gain_mask",
    "speaker_url",
    "target_url",
    "apply_mic_cal",
    "speaker_polarity_invert",
    "speaker_phase_center_enabled",
    "speaker_phase_center_mode",
    "speaker_phase_center_manual_offset_ms",
    "speaker_phase_mode",
    "speaker_band_extension_enabled",
    "speaker_hf_extension_start_hz",
    "speaker_hf_gain_mode",
    "speaker_hf_phase_mode",
    "speaker_hf_custom_slope_db_per_oct",
    "speaker_band_extension_advanced",
    "speaker_lf_extension_mode",
    "speaker_extension_transition_oct",
    "speaker_extension_strength",
    "speaker_hf_phase_fit_start_hz",
    "speaker_hf_min_slope_db_per_oct",
    "speaker_hf_max_slope_db_per_oct",
    "speaker_smoothing_fraction",
    "speaker_smoothing_mode",
    "speaker_gain_shift_mode",
    "speaker_gain_shift_db",
    "target_gain_shift_mode",
    "target_gain_shift_db",
    "target_source_content_mode",
    "target_lf_extension_enabled",
    "target_editor_name",
    "target_editor_description",
    "target_editor_category",
    "target_editor_saved_hash",
    "_target_preset_editing_id",
    "_target_preset_editing_name",
    "target_hf_extension_enabled",
    "auto_iir_mode",
    "auto_iir_fit_boost_db",
    "auto_iir_fit_cut_db",
    "auto_iir_fit_max_q",
    "auto_iir_f_min",
    "auto_iir_f_max",
    "auto_iir_peq",
    "auto_iir_low_shelf",
    "auto_iir_high_shelf",
    "auto_iir_prioritize_low_frequency",
    "auto_iir_max_filters",
    "auto_iir_headroom_db",
    "output_fir_polarity_invert",
    "output_fir_cosine_taper_enabled",
    "output_fir_remove_nyquist_enabled",
    "output_fir_remove_nyquist_strength",
    "dc_gain_normalize",
    "standalone_fir_enabled",
    "fir_output_formats",
    "target_output_formats",
)

PROFILE_LIST_GROUPS = {
    "gain_peq": ("enabled", "fc", "q", "gain"),
    "gain_shelf": ("enabled", "mode", "fc", "q", "gain"),
    "gain_linear": ("enabled", "mode", "response", "fc", "cycles", "beta"),
    "linear_fir": ("enabled", "mode", "response", "fc", "cycles", "beta"),
    "gain_tilt": ("enabled", "mode", "start", "end", "slope"),
    "gain_lo_tilt": ("enabled", "mode", "start", "end", "slope"),
    "phase_peq": ("enabled", "fc", "q", "phase"),
    "phase_shelf": ("enabled", "mode", "fc", "q", "phase"),
    "phase_tilt": ("enabled", "start", "end", "slope"),
    "phase_allpass": ("enabled", "polarity", "fc", "q"),
    "auto_gain_section": (
        "enabled",
        "gain_phase_mode",
        "start",
        "end",
        "smoothing",
        "candidate_min_delta",
        "candidate_min_wavelet_weight",
    ),
    "auto_phase_section": (
        "enabled",
        "start",
        "end",
        "smoothing",
        "candidate_min_delta",
    ),
}

PROFILE_LIST_COUNT_KEYS = {
    "auto_gain_section": "auto_gain_section_count",
    "auto_phase_section": "auto_phase_section_count",
}

PROFILE_GROUP_KEYS = (
    "target_gain_tilt_smoothing_oct",
    "target_phase_tilt_smoothing_oct",
    "target_gain_lo_tilt_count",
    "gain_peq_count",
    "gain_shelf_count",
    "gain_linear_count",
    "gain_tilt_count",
    "gain_lo_tilt_count",
    "gain_tilt_smoothing_oct",
    "phase_peq_count",
    "phase_shelf_count",
    "phase_tilt_count",
    "phase_tilt_smoothing_oct",
    "phase_allpass_count",
    "auto_gain_section_count",
    "auto_phase_section_count",
)

PROFILE_ENTITY_REF_KEYS = (
    "current_project_id",
    "current_system_id",
    "current_slot_id",
    "current_profile_id",
    "current_speaker_measurement_id",
    "current_mic_calibration_id",
    "current_near_field_measurement_id",
    "current_port_measurement_id",
    "current_target_response_id",
)

PROFILE_REGISTRY_LIST_KEYS = (
    "target_gain_peq_items",
    "target_gain_shelf_items",
    "target_gain_linear_items",
    "target_gain_tilt_items",
    "target_gain_lo_tilt_items",
    "target_phase_peq_items",
    "target_phase_shelf_items",
    "target_phase_tilt_items",
    "target_phase_allpass_items",
    "target_iir_filter_items",
    "auto_gain_section_items",
    "auto_phase_section_items",
    "gain_peq_items",
    "gain_shelf_items",
    "gain_linear_items",
    "linear_fir_items",
    "gain_tilt_items",
    "gain_lo_tilt_items",
    "phase_peq_items",
    "phase_shelf_items",
    "phase_tilt_items",
    "phase_allpass_items",
)

DEPENDENCY_DEFINITIONS = {
    "current_speaker_measurement_id": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "current_mic_calibration_id": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "current_near_field_measurement_id": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "current_port_measurement_id": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "speaker_url": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "target_url": ("Target", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "driver_band_enabled": ("Project", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "driver_band_name": ("Project", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "driver_band_hpf_hz": ("Project", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "driver_band_lpf_hz": ("Project", "FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "speaker_polarity_invert": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized"),
    "speaker_phase_center_enabled": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized"),
    "speaker_phase_center_mode": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized"),
    "speaker_phase_center_manual_offset_ms": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized"),
    "speaker_gain_shift_db": ("Input", "FIR EQ", "Correction Filter", "Speaker + Realized"),
    "target_gain_shift_db": ("Target", "FIR EQ", "Correction Filter", "Speaker + Realized"),
    "gain_peq_count": ("FIR EQ", "Correction Filter", "Speaker + Realized"),
    "phase_peq_count": ("FIR EQ", "Correction Filter", "Speaker + Realized"),
    "auto_gain_section_count": ("FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "auto_phase_section_count": ("FIR EQ", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "linear_fir_count": ("Linear FIR", "Correction Filter", "Speaker + Realized", "Export FIR"),
    "output_fir_polarity_invert": ("Export FIR",),
    "output_fir_remove_nyquist_enabled": ("Export FIR",),
    "output_fir_cosine_taper_enabled": ("Export FIR",),
    "dc_gain_normalize": ("Correction Filter", "Export FIR"),
}


def profile_state_keys(
    session_state: dict,
    *,
    include_widget_keys: bool = False,
) -> list[str]:
    """Return known profile keys that exist in session_state."""
    keys: set[str] = set(PROFILE_SCALAR_KEYS)
    keys.update(PROFILE_GROUP_KEYS)
    keys.update(PROFILE_ENTITY_REF_KEYS)
    keys.update(PROFILE_REGISTRY_LIST_KEYS)
    for prefix, fields in PROFILE_LIST_GROUPS.items():
        count_key = PROFILE_LIST_COUNT_KEYS.get(prefix, f"{prefix}_count")
        count = int(session_state.get(count_key, 0) or 0)
        for idx in range(count):
            for field in fields:
                keys.add(f"{prefix}_{idx}_{field}")
    if include_widget_keys:
        keys.update(key for key in session_state.keys() if str(key).startswith("__"))
    return sorted(str(key) for key in keys if key in session_state)


def register_default_dependencies(register_dependency_func, *, scope: str | None = None) -> None:
    """Register standard PhaseEQ dependencies into UI State Registry."""
    for source, targets in DEPENDENCY_DEFINITIONS.items():
        register_dependency_func(source, targets, scope=scope, reason="PhaseEQ standard flow")


def known_page(page: str, pages: Iterable[str] = UI_PAGE_OPTIONS) -> bool:
    return page in pages


def result_stage_for_page(page: str) -> UIPageResultStage:
    """Return the right-pane result stage assigned to a navigation page."""
    return UI_PAGE_RESULT_STAGES.get(page, UI_PAGE_RESULT_STAGES["FIR EQ"])
