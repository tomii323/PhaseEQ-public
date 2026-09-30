from __future__ import annotations

import json
from string import Formatter
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from utils.ui_language import current_language


UI_WORDING_RULES: tuple[str, ...] = (
    "Visible UI text follows the shared Japanese/English application preference.",
    "Use short nouns for page and section names.",
    "Use verb + object for action buttons.",
    "Use 'Select ...' for choosing actions and '... source' for already-selected data.",
    "Use 'Library' instead of user-facing DB wording.",
    "Use 'Speaker specs' instead of ambiguous 'User specs'.",
    "Use 'Create' instead of 'Generate' for user-facing creation actions.",
    "Use 'FIR Filter' for the final FIR artifact.",
)


# User-facing names follow common DSP, measurement, and loudspeaker-design
# terminology while saved state continues to use the legacy internal values.
USER_UI_LABELS: dict[str, str] = {
    "Speaker input": "スピーカー測定",
    "Mic Calibration": "マイク校正データ",
    "Impedance": "インピーダンス（記録用）",
    "Use in Input": "Response Processingで使う",
    "Use saved in Input": "登録したデータを使う",
    "Apply selected": "選択したデータを適用",

    "Speaker Package": "Driver Profiles",
    "Project": "Driver Profiles",
    "Packages": "Driver Profiles",
    "Library": "Measurements",
    "Input Profile": "Response Processing",
    "Input source": "Measurement",
    "Timing": "Time & Phase",
    "Input process": "Response Processing",
    "Speaker specs": "ドライバーデータベース（任意）",
    "Linear FIR": "Output Band Split",
    "Common Settings": "Preferences",
    "Setup": "System Setup",
    "Routing": "Signal Flow",
    "Output": "Output Channels",
    "Way": "Band Setup",
    "Alignment": "Crossover & Alignment",
    "Files / Export": "Export",
    "Target source": "Target Source",
    "Target edit": "Target Shape",
    "Identity": "Export Label",
    "Settings": "Backup & Restore",
    "Apply selected": "Apply Selection",
    "Load selected": "Load Profile",
    "Clear current": "Clear Assignment",
    "Save and complete": "Save Design",
    "Save as new": "Save as New",
    "Update": "Recalculate",
    "Update now": "Recalculate",
}


def user_ui_label(value: object) -> str:
    """Return the adopted UI Concept v1.1 label without changing state values."""

    text = str(value)
    adopted = USER_UI_LABELS.get(text, text)
    return display_text(adopted) if current_language() else adopted


@lru_cache(maxsize=1)
def _message_catalog():
    path = Path(__file__).resolve().parents[1] / "resources" / "ui_messages.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def ui_message(message_id: str, **values: object) -> str:
    """Format only application-owned prose; supplied values are never translated."""
    catalog = _message_catalog()
    if message_id not in catalog:
        # A running app can receive new messages before its imported modules reload.
        _message_catalog.cache_clear()
        catalog = _message_catalog()
    record = catalog[message_id]
    language = current_language()
    template = (record.get(language) or record["source"]) if language else record["source"]
    return template.format(**values) if values else template


def display_text(text: object) -> str:
    """Translate a registered label without changing the corresponding stored value."""
    return _display_text_for_language(text, current_language())


@lru_cache(maxsize=1)
def _catalog_by_text():
    records = _message_catalog()
    lookup = {row["source"]: row for row in records.values()}
    for row in records.values():
        for language in ("ja", "en"):
            lookup.setdefault(row[language], row)
    return lookup


def _display_text_for_language(text: object, language: str | None) -> str:
    text = str(text)
    if language and text in _catalog_by_text():
        return _catalog_by_text()[text].get(language) or text
    if language == "ja":
        return JAPANESE_UI_LABELS.get(text, text)
    if language == "en":
        return ENGLISH_UI_LABELS.get(text, text)
    return text


def localized_formatter(formatter):
    """Translate an application enum's label while preserving its actual value."""
    # Widget serialization can call this outside the rendering thread/context.
    language = current_language()
    return lambda value: _display_text_for_language(formatter(value), language)


@lru_cache(maxsize=1)
def _notice_templates():
    patterns = []
    for key, row in _message_catalog().items():
        source = row["source"]
        if "{p0}" not in source:
            continue
        parts = list(Formatter().parse(source))
        # Exclude generic wrappers such as "{p0}" and "**{p0}**": callers may
        # be showing names or file content, which must remain untouched.
        literal_size = sum(len(literal.strip()) for literal, _, _, _ in parts)
        if literal_size < 12:
            continue
        seen = set()
        expression = ""
        for literal, field, _, _ in parts:
            expression += re.escape(literal)
            if field:
                expression += f"(?P={field})" if field in seen else f"(?P<{field}>.*?)"
                seen.add(field)
        patterns.append((literal_size, re.compile(expression, re.DOTALL), key))
    return sorted(patterns, key=lambda item: item[0], reverse=True)


def display_notice(message: object) -> str:
    """Render stored application notices without rewriting their stored payload."""
    source = str(message)
    translated = display_text(source)
    if translated != source or not current_language():
        return translated
    for _, pattern, key in _notice_templates():
        match = pattern.fullmatch(source)
        if match:
            return ui_message(key, **match.groupdict())
    return source


JAPANESE_UI_LABELS: dict[str, str] = {
    "Display FFT": "表示解析のFFT倍率",
    # Application pages
    "Project": "スピーカーパッケージ管理",
    "Speaker Package": "スピーカーパッケージ",
    "Driver Profiles": "スピーカー測定データ",
    "DSP System": "DSPシステム",
    "PhaseEQ": "音の補正・フィルター設計",
    "Library": "ライブラリ",
    "Measure": "マイク測定",
    "Input": "入力",
    "Input Profile": "入力プロファイル",
    "Response Processing": "応答処理",
    "Preferences": "環境設定",
    "Backup & Restore": "バックアップと復元",
    # Retain the unique legacy reverse-lookup alias; the message catalog displays 設定.
    "Application": "アプリケーション",
    "Open Backup & Restore": "バックアップと復元を開く",
    "Import Driver Database JSON": "Driver Database JSONを読み込む",
    "Target": "目標特性",
    "IIR EQ": "IIRイコライザー",
    "IIR": "IIR",
    "Gain": "ゲイン",
    "Gain range": "ゲイン表示幅",
    "Phase": "位相",
    "All Pass": "オールパス",
    "Linear HP / LP": "リニアフェーズHP・LP",
    "IIR HP / LP": "IIR HP・LP",
    "Gain EQ": "ゲインEQ",
    "Phase EQ": "位相EQ",
    "Auto Gain EQ": "自動ゲインEQ",
    "Auto Phase EQ": "自動位相EQ",
    "Auto Gain": "自動ゲイン",
    "Auto Phase": "自動位相",
    "Linear FIR": "出力帯域分割",
    "Output Band Split": "出力帯域分割ページ",
    "Linear Phase FIR": "リニアフェーズFIR",
    "Export": "書き出し",
    # Measure categories
    "Mic input": "マイク入力",
    "ESS": "ESS信号",
    "ESS setup": "ESS設定",
    "Calibration": "マイク校正・音圧校正",
    "Processing": "ゲート・マージ処理",
    "Gate / merge": "ゲート/合成",
    "Capture": "測定開始・収録設定",
    "Restore": "測定セッション読込",
    "Session": "セッション",
    "ESS Generator": "ESS生成画面",
    "Create ESS": "ESS生成",
    "ESS Library": "ESSライブラリ",
    # Project, database, input, and export categories
    "Project DB": "プロジェクトDB",
    "Design Library": "デザインライブラリ",
    "Packages": "パッケージ一覧",
    "Projects": "プロジェクト一覧",
    "Identity": "識別情報",
    "Export Label": "書き出し名",
    "Analysis Policy": "解析ポリシー",
    "Project info": "プロジェクト情報",
    "FIR / Analysis": "FIR・解析条件",
    "Design": "設計",
    "Settings": "設定",
    "Common Settings": "共通設定",
    "Open Common Settings": "共通設定を開く",
    "Measurements": "測定データ",
    "Add measurement": "測定データを追加",
    "Saved measurements": "登録済み測定データ",
    "Advanced registration": "詳細登録",
    "Measurement type": "測定データの種類",
    "Speaker specs": "スピーカー仕様",
    "Driver Database": "ドライバーデータベース",
    "Source": "入力ソース",
    "Input source": "入力元",
    "Select measurement": "測定データを選択",
    "Select from Library": "ライブラリから選択",
    "Speaker input": "スピーカー入力",
    "Mic Calibration": "マイク校正",
    "Mic Cal": "マイク校正（略称）",
    "Impedance": "インピーダンス測定",
    "Phase / Timing": "位相・タイミング",
    "Timing": "タイミング",
    "Response Processing": "応答処理",
    "Input process": "入力処理",
    "レベル合わせ": "レベル合わせ",
    "INPUT": "出力チャンネル入力",
    "Presets": "プリセット",
    "Target presets": "ターゲットプリセット",
    "Target source": "ターゲット元",
    "Target source content": "ターゲット元の内容",
    "Gain + Phase": "ゲイン＋位相",
    "Complex Gain": "複素音圧",
    "Gain only": "ゲインのみ",
    "Target edit": "ターゲット編集",
    "Use in PhaseEQ": "PhaseEQで使用",
    "Register current": "現在のTargetを登録",
    "Save revision": "新しいリビジョンを保存",
    "Archive selected shared Target": "選択中の共通Targetをアーカイブ",
    "Preview revision": "リビジョンをプレビュー",
    "Restore as new revision": "新しいリビジョンとして復帰",
    "Files": "ファイル",
    "System": "システム",
    "Setup": "構成",
    "System Setup": "システム構成",
    "Routing": "ルーティング",
    "Signal Flow": "信号経路",
    "Output Channels": "出力チャンネル",
    "Band Setup": "帯域設定",
    "Crossover & Alignment": "クロスオーバーとアライメント",
    "Way": "ウェイ",
    "Alignment": "アライメント",
    "FIR Gain": "FIRゲイン",
    "FIR Phase": "FIR位相",
    "FIR EQ": "FIRイコライザー",
    "FIR EQ view": "FIR EQ表示",
    "IIR EQ view": "IIR EQ表示",
    "Open FIR EQ": "FIRイコライザーを開く",
    "Open Linear FIR": "リニアFIRを開く",
    "Gain · PEQ": "ゲイン・PEQ",
    "Gain · High Shelf": "ゲイン・ハイシェルフ",
    "Gain · Low Shelf": "ゲイン・ローシェルフ",
    "Gain · High Tilt": "ゲイン・ハイチルト",
    "Gain · Low Tilt": "ゲイン・ローチルト",
    "Phase · PEQ": "位相・PEQ",
    "Phase · High Shelf": "位相・ハイシェルフ",
    "Phase · Low Shelf": "位相・ローシェルフ",
    "Phase · All Pass": "位相・オールパス",
    "Phase · Tilt": "位相・チルト",
    "Auto Gain · Section": "自動ゲイン・区間",
    "Auto Phase · Section": "自動位相・区間",
    "Files / Export": "ファイル・書き出し",
    "Continuous ESS": "連続ESS測定",
    "Test input": "テスト入力",
    "All inputs": "全入力",
    "EQ / FIR": "EQ・FIR",
    "Crossover": "クロスオーバー",
    "Crossover implementation": "帯域分割の実装場所",
    "All crossovers": "全出力クロスオーバー",
    "DSP": "DSPで帯域分割",
    "External passive": "外部パッシブネットワーク",
    "External active": "外部アクティブネットワーク",
    "Output": "出力",
    "Way / Target": "ウェイ・ターゲット",
    "Level / Delay": "レベル・ディレイ",
    "Kaiser FIR": "カイザーFIR",
    "IIR": "IIRフィルター",
    "Linear phase LR2": "LR2目標リニアフェーズFIRクロス",
    "Diagnostics": "診断",
    "Check": "確認",
    "Extension": "帯域拡張",
    "Smoothing": "スムージング",
    "Gain shift": "ゲインシフト",
    "Final FIR": "最終FIR",
    "Band Split Filters": "帯域分割フィルター",
    "FIR Filter": "FIRフィルター",
    "FIR Filter list": "FIRフィルター一覧",
    "FIR Filter output": "FIRフィルター出力",
    "Output formats": "出力形式設定",
    "Formats": "出力形式",
    "Downloads / Backup": "ダウンロード・バックアップ",
    "Downloads": "ダウンロード",
    # Filter and result categories
    "PEQ / Shelf": "PEQ・シェルフ",
    "PEQ / Shelf / All-pass": "PEQ・シェルフ・オールパス",
    "Tilt": "チルト",
    "Linear": "リニア",
    "Linear Phase LR2": "LR2目標リニアフェーズFIR",
    "Error": "誤差",
    "Response Advisor": "応答アドバイザー",
    "Wavelet": "ウェーブレット",
    "Gain / Phase": "ゲイン・位相",
    "App diagnostics": "アプリ診断",
    "Group Delay": "群遅延",
    "Step": "ステップ応答",
    "Impulse": "インパルス応答",
    "Input details": "入力詳細",
    "FIR Filter cofs": "FIR係数・窓関数",
    "Cosine Tapered window": "コサインテーパー窓",
    "Input info": "入力情報",
    "Preview": "プレビュー",
    "Selection preview": "選択内容のプレビュー",
    "Measurement Preview": "測定データのプレビュー",
    "Predicted Response": "予測応答",
    "System Response": "システム応答",
    "Summed Response": "合成応答",
    "Final Output": "最終出力",
    "Signal Path": "最終信号経路",
    "Time Response": "時間応答",
    "After Input IIR": "入力IIR後",
    "After Routing": "ルーティング後",
    "After Output IIR": "出力IIR後",
    "After Crossover / Alignment": "クロスオーバー・アライメント後",
    "After FIR": "FIR後",
    "Load Profile": "プロファイルを読込",
    "Save Design": "設計を保存",
    "Archive": "アーカイブ",
    "Recalculate": "再計算",
    "Stop Measuring": "測定を終了",
    "Abort Measurement": "測定を中止して破棄",
    "Auto (40 dB)": "自動（40 dB）",
    "20 dB": "20 dB幅",
    "40 dB": "40 dB幅",
    "60 dB": "60 dB幅",
    "80 dB": "80 dB幅",
    "120 dB": "120 dB幅",
    "Not applied": "未適用",
    "Response": "周波数特性",
    "FIR": "FIR特性",
    "Advisor": "アドバイス",
    # Measurement segmented controls
    "Wavelet Hybrid": "ウェーブレット・ハイブリッド",
    "Positive peak": "正側ピーク",
    "Negative peak": "負側ピーク",
    "Absolute peak": "絶対値ピーク",
    "Energy centroid": "エネルギー重心",
    "Manual": "手動",
    "ESS Library file": "ESSライブラリの実波形",
    "PhaseEQ nominal ESS": "PhaseEQ公称ESS",
    "USB microphone": "USBマイク",
    "Analog microphone + audio interface": "アナログマイク＋オーディオインターフェース",
    "IR peak — external ESS": "IRピーク・外部ESS",
    "Timing marker — external ESS": "タイミングマーカー・外部ESS",
    "Tweeter reference — external ESS": "ツイーター基準・外部ESS",
    "Reference Tweeter": "基準ツイーター",
    "Target Speaker": "測定対象スピーカー",
    "Auto ±": "自動 ±",
    "Positive +": "正極性 +",
    "Negative −": "負極性 −",
    "Hold edge": "端点を保持",
    "No correction": "補正なし",
    "Manual / extended file": "手動・拡張ファイル",
    "Auto": "自動",
    "Force": "強制",
    "Off": "オフ",
    "Latest": "最新のみ",
    "All retained": "保持分すべて",
    "Strict": "厳格",
    "Standard": "標準",
    "Final": "最終結果",
    "Lenient": "緩やか",
    "Raw": "生データ",
    "Denoised": "ノイズ低減済み",
    "Quality": "品質",
    "Standard Merged": "標準マージ応答",
    "Final merged": "最終合成応答",
    "Denoised Merged": "ノイズ低減マージ応答",
    "Averaged Merged": "平均マージ応答",
    "Selected Merged": "選択ショットのマージ応答",
    "Selected Ungated": "選択ショットのゲート前応答",
    "Selected Gated": "選択ショットのゲート後応答",
    "Selected full": "選択ショットのFull応答",
    "Selected gated": "選択ショットのGated応答",
    "system_channel": "システム・チャンネル別",
    "combined_room": "L+R確認用",
    "driver": "ドライバー単体・近接",
    "mic_calibration": "マイク校正データ",
    "impedance": "インピーダンス",
    "correction_input": "補正入力",
    "verification_only": "確認専用",
    "calibration": "校正",
    "archive": "保管のみ",
    "Estimated dB SPL": "推定音圧レベル",
    "dB SPL": "音圧レベル",
    "dBFS RMS": "デジタル入力レベル",
    "Frequency": "周波数指定",
    "Nyquist": "ナイキスト周波数",
    "-80 dB": "-80 dB未満を非表示",
    "-60 dB": "-60 dB未満を非表示",
    "Source - Target": "入力と目標の差分",
    # ESS generator presets
    "General measurement": "一般測定・推奨",
    "Quick check": "クイック確認",
    "Compatibility": "互換性重視",
    "Precision": "高精度",
    "Tweeter extended": "ツイーター拡張",
    "Noisy environment": "騒音環境",
    "Custom": "カスタム",
    "Both": "左右両方",
    "Left only": "左のみ",
    "Right only": "右のみ",
    "Machine-detectable timing marker": "機械検出用タイミングマーカー",
    # Action buttons
    "Abort": "中止",
    "Add": "追加",
    "Run Auto IIR": "IIR自動調整を実行",
    "Clear Auto IIR": "自動IIRを削除",
    "Reset Auto IIR settings": "設定を初期値に戻す",
    "Add current": "現在設定を追加",
    "Apply": "適用",
    "Apply to Input": "入力に適用",
    "Apply target to DSP System": "TargetをDSPシステムへ適用",
    "Apply Gate / Center": "ゲート・センターを適用",
    "Apply Gate / Merge": "ゲート・マージを適用",
    "Auto range": "自動範囲",
    "Apply first": "最初の候補を適用",
    "Apply generator preset": "生成プリセットを適用",
    "Apply measurements": "測定データを適用",
    "Apply spec": "仕様を適用",
    "Capture calibration level (1 s)": "校正レベルを1秒取得",
    "Estimate microphone sensitivity (1 s)": "マイク感度を1秒で推定",
    "Capture SPL meter comparison (1 s)": "音圧計比較を1秒取得",
    "Accept sound pressure calibration": "音圧校正を採用",
    "Cancel": "キャンセル",
    "Clear reference": "参照を解除",
    "Delete": "削除",
    "Delete selected user preset": "選択中ユーザープリセットを削除",
    "Demote to User": "ユーザー項目へ移動",
    "Duplicate": "複製",
    "Open Library": "ライブラリを開く",
    "Open Target editor": "Target編集を開く",
    "Go to Input": "Inputへ進む",
    "Back to live": "ライブ結果へ戻る",
    "Open Measure": "マイク測定を開く",
    "Fetch URL": "URLから取得",
    "Create ESS files": "ESSファイルを生成",
    "Create level-check WAV": "音量確認WAVを生成",
    "Create marker version": "タイミングマーカー付き版を生成",
    "Find measurements": "測定データを探す",
    "Find speaker specs": "スピーカー仕様を探す",
    "Import measurements JSON": "測定JSONを読み込む",
    "Import specs JSON": "仕様JSONを読み込む",
    "Load into form": "フォームへ読み込む",
    "Load selected": "選択項目を読み込む",
    "Load session": "セッションを読み込む",
    "Lower 10 dB": "10 dB下げる",
    "Move down": "下へ移動",
    "Move up": "上へ移動",
    "New form": "新規フォーム",
    "Open speaker specs": "スピーカー仕様を開く",
    "Parse raw text": "貼り付けテキストを解析",
    "Prepare snapshot": "スナップショットを準備",
    "Promote to Built-in": "組み込み項目へ移動",
    "Prepare ZIP": "ZIPを準備",
    "Download ZIP": "ZIPをダウンロード",
    "Rebuild Final": "最終結果を再構築",
    "Recalculate selected Wavelet": "選択ショットのウェーブレットを再計算",
    "Refresh input devices": "入力機器を再検索",
    "Save to Library": "ライブラリへ保存",
    "Restore defaults": "初期設定を復元",
    "Reset defaults": "初期設定に戻す",
    "Restore from ZIP": "ZIPから環境を復元",
    "Apply loaded settings": "読み込んだ設定を適用",
    "Restore saved settings": "保存済み設定を復元",
    "Save current settings": "現在設定を保存",
    "Register unit": "マイク個体を登録",
    "Raise 10 dB": "10 dB上げる",
    "Reset selection": "選択を解除",
    "Restore Target settings": "目標特性設定を復元",
    "Restore to Input": "入力へ復元",
    "Use in Input": "入力に使用",
    "Use saved in Input": "保存したデータを入力に使用",
    "Run Post denoise": "後処理ノイズ低減を実行",
    "Post process": "後処理",
    "Run diagnostics": "診断を実行",
    "Save impedance": "インピーダンスを保存",
    "Save mic": "マイクデータを保存",
    "Save mic file": "マイクファイルを保存",
    "Save speaker": "スピーカーデータを保存",
    "Save speaker file": "スピーカーファイル一式を保存",
    "Save as new": "別レコード保存",
    "Save and complete": "保存して完了",
    "Save form": "フォームを保存",
    "Save new": "新規保存",
    "Sort by frequency": "周波数順に並べ替え",
    "Start": "開始",
    "Start noise check": "暗騒音測定を開始",
    "Start download watch": "ダウンロード監視を開始",
    "Stop": "停止",
    "Stop watch": "監視を停止",
    "Sync app name": "アプリ名と同期",
    "Update": "更新",
    "Update selected": "選択項目を更新",
    "Edit DSP setup": "DSP構成を編集",
    "Use metadata": "メタデータを使用",
    "Validate and register selected files": "選択ファイルを検証・登録",
    # Headings and compact status labels
    "Measurement prep / results": "測定準備・結果",
    "Prep flow": "測定準備フロー",
    "View settings": "表示設定",
    "Result select": "結果選択",
    "Current input": "現在の入力",
    "Library status": "ライブラリ状態",
    "Measurement actions": "測定操作",
    "Session ZIP": "セッションZIP",
    "Cancel countdown": "開始カウントダウンをキャンセル",
    "Clear and start": "結果をクリアして開始",
    "Apply selected": "選択項目を適用",
    "Apply Target": "ターゲットを適用",
    "Save Group Target": "グループターゲットを保存",
    "Clear current": "現在の設定を解除",
    "Continue to Input": "入力設定へ進む",
    "Continue to Target": "ターゲット設定へ進む",
    "Clear and load session": "結果をクリアしてセッション読込",
    "Cancel load": "読込をキャンセル",
    "FIR Filter formats": "FIRフィルター形式",
    "Measurement settings": "測定設定",
    "Measurement results": "測定結果",
    "Measurement control": "測定コントロール",
    "Done shots": "完了shot",
    "Accepted shots": "採用shot",
    "Start delay": "開始までの遅延時間",
    "Save": "保存",
    "Clear and start": "結果をクリアして開始",
    "App diagnostics status": "アプリ診断ステータス",
    "Final clock correction": "最終結果のClock補正",
}
JAPANESE_UI_LABELS.update({"Multiway Assignment": "マルチウェイに割り当て", 'Measurement List & Use': '測定一覧・使用', 'Register Measurement': '測定データを登録', 'Measurement Notes': '測定条件・メモ', 'Calibration Library': '校正データ一覧・登録', 'Microphone Profiles': 'マイクプロファイル', 'Speaker Specifications': 'スピーカー仕様データベース（任意）'})
JAPANESE_UI_LABELS.update({"Mic Library": "マイク校正データ管理", "Mic Profiles": "マイク個体管理", "Speaker Specs": "スピーカー仕様管理"})
ENGLISH_UI_LABELS: dict[str, str] = {
    japanese: english for english, japanese in JAPANESE_UI_LABELS.items()
}
if len(ENGLISH_UI_LABELS) != len(JAPANESE_UI_LABELS):
    raise RuntimeError("Japanese UI labels must be unique for bidirectional lookup")


PHASEEQ_TRANSLATED_MENU_GROUPS: dict[str, tuple[str, ...]] = {
    "Page selector": (
        "Project", "Multiway Assignment", "Library", "Measurement Notes", "Mic Library", "Mic Profiles", "Speaker Specs", "Input", "Target", "IIR EQ", "FIR EQ", "Linear FIR",
        "Export", "DSP System", "Measure", "Common Settings", "Backup & Restore",
    ),
    "Measure categories": ("Mic input", "ESS setup", "Calibration", "Gate / merge", "Capture", "Session"),
    "ESS source": ("Create ESS", "ESS Library"),
    "ESS generator presets": (
        "General measurement", "Quick check", "Compatibility", "Precision",
        "Tweeter extended", "Noisy environment", "Custom",
    ),
    "ESS output channels": ("Both", "Left only", "Right only"),
    "Converted ESS output channels": ("Both", "Left only", "Right only"),
    "Project edit mode": ("Design Library", "Identity", "FIR / Analysis", "Settings"),
    "Library mode": ("Measurements", "Speaker specs"),
    "Input edit mode": ("Input source", "Timing", "Input process"),
    "Input processing type": ("Extension", "Smoothing", "Gain shift"),
    "Smoothing mode": ("Gain + Phase", "Complex Gain"),
    "Phase centering peak mode": (
        "Wavelet Hybrid", "Positive peak", "Negative peak",
        "Absolute peak", "Energy centroid", "Manual", "Off",
    ),
    "Linear FIR mode": ("FIR Filter", "Presets"),
    "Target task": ("Target presets", "Target source", "Target edit", "Files"),
    "Target source content": ("Gain + Phase", "Gain only"),
    "Target edit mode": ("Gain", "Phase"),
    "IIR EQ view": ("Gain", "Phase"),
    "FIR EQ view": ("Gain", "Phase"),
    "Gain filter group": ("PEQ / Shelf", "Tilt"),
    "Phase filter group": ("PEQ / Shelf / All-pass", "Tilt"),
    "Export edit mode": ("FIR Filter", "Formats", "Downloads"),
    "Live Result detail": ("Error", "Advisor", "Wavelet", "Group Delay", "Step", "Impulse", "Input info"),
    "Wavelet mode": ("Source", "Target", "Source - Target"),
    "ESS reference": ("ESS Library file", "PhaseEQ nominal ESS"),
    "Calibration extrapolation": ("Hold edge", "No correction", "Manual / extended file"),
    "Clock correction": ("Auto", "Force", "Off"),
    "Reprocess scope": ("Latest", "All retained"),
    "Quality gate": ("Strict", "Standard", "Lenient"),
    "Measurement result layer": ("Raw", "Final", "Denoised"),
    "Measurement result view": ("Gain / Phase", "Impulse", "Wavelet", "App diagnostics"),
    "Measurement detail": ("Impulse", "Wavelet", "Quality"),
    "Measurement DB response": (
        "Final merged", "Denoised Merged", "Averaged Merged",
        "Selected Merged", "Selected full", "Selected gated",
    ),
    "Measurement target": ("system_channel", "combined_room", "driver", "mic_calibration", "impedance"),
    "Library measurement type": ("Speaker input", "Mic Calibration", "Impedance"),
    "Measurement use": ("correction_input", "verification_only", "calibration", "archive"),
    "Measurement gain unit": ("Estimated dB SPL", "dB SPL", "dBFS RMS"),
    "ESS end mode": ("Frequency", "Nyquist"),
    "Phase mask": ("Off", "-80 dB", "-60 dB"),
    "Workspace": ("Speaker Package", "DSP System", "Measure", "Application"),
    "DSP System pages": (
        "System", "Target", "IIR EQ", "FIR EQ", "Linear FIR", "Files / Export",
    ),
    "FIR EQ item type": (
        "Gain · PEQ", "Gain · High Shelf", "Gain · Low Shelf",
        "Gain · High Tilt", "Gain · Low Tilt", "Phase · PEQ",
        "Phase · High Shelf", "Phase · Low Shelf", "Phase · All Pass",
        "Phase · Tilt", "Auto Gain · Section", "Auto Phase · Section",
    ),
    "DSP System task": ("Setup", "Routing", "Output", "Way", "Alignment", "Files"),
    "DSP Output stage": ("INPUT", "Target", "IIR", "FIR"),
    "DSP Way stage": ("Way / Target", "Crossover", "Level / Delay"),
    "DSP Way edit": ("Source", "EQ / FIR", "All crossovers", "Output"),
    "DSP routing preview": ("Test input", "All inputs"),
    "DSP crossover engine": ("Linear phase LR2", "Kaiser FIR", "IIR", "Off"),
    "DSP crossover implementation": (
        "DSP", "External passive", "External active",
    ),
}


def japanese_ui_label(label: str) -> str | None:
    """Return the centrally registered Japanese name for an English UI label."""
    return JAPANESE_UI_LABELS.get(str(label))


def english_ui_label(label: str) -> str | None:
    """Return the English name paired with a centrally registered Japanese label."""
    return ENGLISH_UI_LABELS.get(str(label))


def localized_ui_label(label: str, language: str) -> str | None:
    """Translate a registered UI label in either direction."""
    normalized = str(language).strip().lower()
    if normalized in {"ja", "jp", "japanese", "日本語"}:
        return japanese_ui_label(label)
    if normalized in {"en", "english", "英語"}:
        return english_ui_label(label)
    raise ValueError(f"Unsupported UI language: {language}")


def missing_japanese_ui_labels(labels: Sequence[str]) -> tuple[str, ...]:
    """List labels that would otherwise silently lose their Japanese help."""
    return tuple(str(label) for label in labels if japanese_ui_label(str(label)) is None)


def _hover_translation_css(
    widget_key: str,
    labels: Sequence[str],
    *,
    item_selector: str,
) -> str:
    if current_language():
        return ""
    missing = missing_japanese_ui_labels(labels)
    if missing:
        raise ValueError(f"Japanese UI labels are not registered: {', '.join(missing)}")
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "-", str(widget_key))
    scope = f".st-key-{safe_key}"
    rules = [
        f"{scope} {item_selector} {{ position: relative; overflow: visible; }}",
        (
            f"{scope} {item_selector}::after {{"
            "position: absolute; top: 50%; left: 50%; z-index: 1000; "
            "transform: translate(-50%, -50%); box-sizing: border-box; "
            "display: flex; align-items: center; justify-content: center; "
            "width: max-content; min-width: calc(100% + 4px); max-width: min(18rem, 80vw); "
            "min-height: calc(100% + 4px); padding: 0.35rem 0.65rem; "
            "border: 1px solid var(--rf-page-accent, var(--primary-color, #806943)); "
            "border-radius: 0.4rem; "
            "background: var(--rf-panel, var(--secondary-background-color, #fffdf9)); "
            "color: var(--text-color, #22252a); "
            "box-shadow: 0 4px 12px rgba(0, 0, 0, 0.28); "
            "font-size: 0.86rem; font-weight: 700; line-height: 1.25; "
            "text-align: center; white-space: normal; pointer-events: none; opacity: 0; "
            "transition: opacity 100ms ease; }"
        ),
        (
            f"{scope} {item_selector}:is(:hover, :focus-visible)::after {{ "
            "opacity: 1; transition-delay: var(--phaseeq-help-delay, 2.5s); }"
        ),
        (
            f"{scope} {item_selector}:is(:hover, :focus-visible) > * {{ "
            "opacity: 0; transition: opacity 0s var(--phaseeq-help-delay, 2.5s); }"
        ),
    ]
    for index, label in enumerate(labels, start=1):
        japanese = japanese_ui_label(str(label))
        rules.append(
            f"{scope} {item_selector}:nth-of-type({index})::after "
            f'{{ content: {json.dumps(japanese, ensure_ascii=False)}; }}'
        )
    return "\n".join(rules)


def tab_hover_translation_css(
    widget_key: str,
    labels: Sequence[str],
    *,
    bordered: bool = False,
) -> str:
    """Build key-scoped Japanese tooltips for a Streamlit tab group."""
    css = _hover_translation_css(widget_key, labels, item_selector='[role="tab"]')
    if not bordered:
        return css
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "-", str(widget_key))
    scope = f".st-key-{safe_key}"
    frame_rules = (
        f'{scope} [data-baseweb="tab-list"] {{ gap: 0.35rem; }}\n'
        f'{scope} [role="tab"] {{ border: 1px solid var(--rf-card-border, rgba(49, 51, 63, 0.24)) !important; '
        "border-radius: 0.4rem !important; padding-inline: 0.85rem !important; "
        "background: var(--rf-panel, var(--secondary-background-color, transparent)); }}\n"
        f'{scope} [role="tab"][aria-selected="true"] {{ border-color: var(--rf-page-accent, var(--primary-color)) !important; '
        "box-shadow: inset 0 -2px 0 var(--rf-page-accent, var(--primary-color)); }}"
    )
    return f"{css}\n{frame_rules}"


def segmented_hover_translation_css(widget_key: str, labels: Sequence[str]) -> str:
    """Build key-scoped Japanese tooltips for a Streamlit segmented control."""
    return _hover_translation_css(
        widget_key,
        labels,
        item_selector='button[data-variant="segmented_control"]',
    )


def required_segmented_control(label: str, options: Sequence[object], **kwargs: object):
    """Keep required single selections valid and synchronized with the frontend."""
    import streamlit as st

    values = list(options)
    if current_language():
        label = display_text(label)
        kwargs["format_func"] = localized_formatter(kwargs.get("format_func", str))
    if kwargs.get("selection_mode", "single") == "single":
        kwargs.setdefault("required", True)
    widget_key = kwargs.get("key")
    if kwargs.get("selection_mode", "single") == "single" and kwargs.get("required") and values:
        fallback = kwargs.get("default")
        if fallback not in values:
            fallback = values[0]
        if widget_key is not None:
            selected = st.session_state.get(widget_key, fallback)
            # Restore one valid value through Session State only, so the
            # frontend receives the selection even after an empty/legacy seed.
            st.session_state[widget_key] = selected if selected in values else fallback
            kwargs.pop("default", None)
        else:
            kwargs["default"] = fallback
    return st.segmented_control(label, values, **kwargs)


def localized_segmented_control(
    label: str,
    options: Sequence[object],
    **kwargs: object,
):
    """Render a segmented control and automatically add Japanese label help."""
    import streamlit as st

    values = list(options)
    translation_labels = kwargs.pop("translation_labels", None)
    if kwargs.get("selection_mode", "single") == "single":
        # PhaseEQ uses segmented controls as compact radio groups. A current
        # mode, page, route, or display choice must always remain selected;
        # required=True also makes clicking the active segment idempotent.
        kwargs.setdefault("required", True)
    kwargs.setdefault("format_func", user_ui_label)
    if current_language():
        kwargs["format_func"] = localized_formatter(kwargs["format_func"])
        label = display_text(label)
    widget_key = kwargs.get("key")
    labels_for_translation = values if translation_labels is None else list(translation_labels)
    if len(labels_for_translation) != len(values):
        raise ValueError("translation_labels must match the segmented control options")
    if (
        widget_key is not None
        and labels_for_translation
        and all(isinstance(value, str) for value in labels_for_translation)
    ):
        string_values = [str(value) for value in labels_for_translation]
        st.markdown(
            f"<style>{segmented_hover_translation_css(str(widget_key), string_values)}</style>",
            unsafe_allow_html=True,
        )
    return required_segmented_control(label, values, **kwargs)


def localized_button_help(label: str, help_text: str | None = None) -> str | None:
    """Add a stable native Japanese tooltip to an English action button."""
    if current_language():
        return display_text(help_text) if help_text else None
    raw_label = str(label)
    normalized = re.sub(r"\s*\(\d+\)\s*$", "", raw_label).strip()
    japanese = japanese_ui_label(raw_label) or japanese_ui_label(normalized)
    if japanese is None:
        return help_text
    heading = japanese
    if help_text:
        return f"{heading}\n\n{help_text}"
    return heading


def localized_button(container: object, label: str, **kwargs: object):
    """Render a Streamlit button with centrally managed Japanese hover help."""
    help_text = kwargs.pop("help", None)
    kwargs["help"] = localized_button_help(label, str(help_text) if help_text is not None else None)
    return container.button(user_ui_label(label), **kwargs)
