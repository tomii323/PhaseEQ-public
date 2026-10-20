"""Present input kinds without changing alignment policy or stored assets."""
from utils.ui_localization import display_text


def speaker_source_label(source):
    if source is None:
        return display_text("入力未設定（Unity）")
    if source.generated_ideal:
        return source.label + " · " + display_text("理想特性")
    return source.label


def alignment_input_notice(readiness):
    if readiness.mode == "speaker":
        return ""
    details = []
    if readiness.ideal_channels:
        details.append(display_text("理想特性：") + " / ".join(readiness.ideal_channels))
    if readiness.missing_channels:
        details.append(display_text("入力未設定（Unity）：") + " / ".join(readiness.missing_channels))
    if not readiness.ready:
        return " ".join([
            display_text("Speaker応答と理想特性／Unityが混在しているため、自動時間整合を実行できません。"),
            *details,
            display_text("全Channelに実測応答を適用するか、全Channelを理想特性またはUnityにしてフィルターだけを解析してください。"),
        ])
    return " ".join([
        *details,
        display_text("Speakerの音響時間差を含めず、Crossover／Filterだけを位相・時間整合します。"),
    ])


def returned_speaker_status(row):
    asset = row.get("phaseeq_speaker_response")
    if isinstance(asset, dict):
        return display_text("理想特性を反映") if asset.get("generated_ideal", False) else display_text("反映済み")
    return display_text("なし") if row.get("phaseeq_assignment_id") else display_text("未返却")
