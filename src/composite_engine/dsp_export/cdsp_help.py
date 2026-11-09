"""Shared dedicated help for the UI and C-DSP delivery README."""
from .cdsp_dirac import CAUTION


def cdsp_guide(language='ja'):
    if language == 'en':
        return '''C-DSP 8x12 DL Dirac/APF export

Purpose: use equivalent APFs for the Dirac Live measurement stage, then switch
to the existing crossover coefficients for playback with the same phase reference.
Pre-Dirac and Post-Dirac are alternative measurement/playback states of the OUTPUT
Xover blocks. They do not mean input blocks before/after the internal Dirac processor.
48 kHz only; DSP FIR output OFF; automatic structural phase compensation ON;
all boundaries LR4 or NTM4, acoustic targets OFF.
Up to four bands (3Way+SUB or 4Way). Import target: Outputs → Xover → Advanced Biquad.
Pre: equivalent APFs + automatic phase APFs. Post: normal XO + automatic phase APFs.
Channel IIR PEQ files are included in PEQ/CHxx_PEQ.txt (maximum 10 sections).
Each channel has Dirac_Targets/CHxx.targetcurve with Group Target magnitude only.
Normal crossover magnitude and Group Target phase are excluded.
Before measurement, the output Xover contains equivalent APFs + automatic phase APFs.
Measure this response including APFs and use Dirac phase correction with the aim of
linearizing phase. Check the corrected phase with measurements.
Group Target selection is separate from the crossover acoustic-target switches.
To use Group Target as the IIR Auto PEQ target, keep acoustic-target reflection
OFF at every crossover boundary. Also keep the phase/time alignment reflection OFF.
Automatic structural phase compensation stays ON; its APFs are loaded into Xover.
Bass Management and Baffle EQ are excluded.

1. Save the current device configuration. Mute outputs before loading coefficients.
   Match manifest.json output_index (CH01-CH12), group and way to physical outputs.
2. Pre-Dirac: open Outputs, select the target channel's Xover, and open Advanced Biquad.
   Load the matching Pre_Dirac/CHxx_APF.txt into all 8 slots (paste and click Process).
   Repeat for every channel and check the settings.
   Apply Gain, Polarity and Delay from manifest.json in both states.
3. Before microphone measurement, open Outputs → PEQ and IMPORT the matching
   PEQ/CHxx_PEQ.txt. All 10 slots are replaced, including unity in unused slots.
   Keep this PEQ active for measurement and playback; do not add it to Xover.
   Independently provide driver protection and choose safe measurement levels
   before starting Dirac Live measurement. Launch it with Start Dirac Live Software
   in the C-DSP plugin. Keep routing and channel settings fixed.
4. After measurement, load the matching Dirac_Targets/CHxx.targetcurve in Dirac Live's
   filter-design view. Use manifest.json to map Group to the actual Dirac channel
   or group; multiple output Ways can share a Dirac channel. Check correction limits.
   Only magnitude is transferred; target phase is not imported into Dirac Live.
   Complete filter creation and transfer. Mute outputs.
   Follow the same Outputs → Xover → Advanced Biquad steps and replace the Pre
   coefficients with Post_Dirac/CHxx_XO.txt (paste and click Process).
   Replace the same 8 slots; do not add or cascade Pre and Post coefficients.
   Keep PEQ, Gain, Polarity, Delay and routing unchanged.
5. Save the playback configuration, enable the intended Dirac filter and verify
   imported slots, channel mapping and output level before playback.

If export is disabled: set 48 kHz, FIR OFF, automatic structural phase compensation
ON and LR4/NTM4, then recalculate. Use at most four bands, 8 Xover sections and 10 PEQ sections per channel,
outputs 1-12, Delay 0-20 ms and Gain -72 to +12 dB. Reduce the relevant configuration
if a limit is exceeded; coefficients are never truncated. Every unused slot is unity.

CAUTION: Pre-Dirac APFs do not attenuate out-of-band signals. You are responsible
for protecting the drivers during measurement. No protection filters or automatic
level limiting are added by this export.
Hardware import and acoustic operation require verification on the target device.
Reference: https://www.minidsp.com/images/documents/C-DSP%208x12%20DL%20User%20Manual.pdf
'''
    return '''C-DSP 8x12 DL Dirac/APF出力

目的：Dirac Liveの測定時に等価APFを使い、再生時に既存のクロスオーバー係数へ切り替えます。
Pre/Postは同じ位相基準を使う、出力Xoverの測定用／再生用の切替状態です。
機器内部のDirac処理より前／後の入力ブロックへ配置する意味ではありません。
48 kHz専用、DSP FIR出力OFF、位相構造補償の自動適用ON、全境界LR4またはNTM4、音響ターゲットOFFで出力します。
最大4帯域（3Way+SUBまたは4Way）です。読み込み先はOutputs → Xover → Advanced Biquadに固定します。
Preは位相等価APF＋自動位相保証APF、Postは通常XO＋自動位相保証APFです。
各チャンネルのIIR PEQをPEQ/CHxx_PEQ.txtへ出力します（最大10段）。
各チャンネルのDirac_Targets/CHxx.targetcurveにはGroup Targetの振幅だけを出力します。
通常XOの振幅とGroup Targetの位相は含めません。
測定前の出力Xoverには位相等価APF＋自動位相保証APFを設定します。
このAPFを含む応答を測定し、Diracの位相補正で直線位相化を目指します。
補正後の位相は実測で確認してください。
Group Targetの選択は、帯域分割の「音響ターゲット」ON/OFFとは別の設定です。
Group TargetをIIR Auto PEQの目標に使うため、帯域分割の「音響ターゲットに反映」は全境界でOFFにします。
位相・時間整合側の同名設定もOFFにします。位相構造補償の自動適用はONのまま、APFをXoverへ設定します。
Bass ManagementとBaffle EQは出力対象に含めません。

1. 現在の機器設定を保存し、係数を読み込む前に出力をミュートしてください。
   manifest.jsonのoutput_index（CH01〜CH12）、Group、Wayを実際の出力端子と照合します。
2. Pre-Dirac適用時は、C-DSPプラグインで次の順に操作します。
   ① Outputsを開きます。
   ② 対象チャンネルのXoverを選択します。
   ③ Advanced Biquadを開きます。
   ④ 対応するPre_Dirac/CHxx_APF.txtを全8段へ読み込みます（貼り付け後にProcess）。
   ⑤ 全チャンネルについて設定を確認します。
   Gain、極性、Delayはmanifest.jsonの値を設定し、Pre/Postで共通に使用します。
3. マイク測定前にOutputs → PEQを開き、対応するPEQ/CHxx_PEQ.txtをIMPORTします。
   未使用段の通過係数を含めて全10段を置き換えます。
   このPEQは測定時と再生時で保持し、Xoverへ追加しないでください。
   ユニット保護と安全な測定音量をユーザー自身で設定してください。
   C-DSPのStart Dirac Live SoftwareでDirac Liveを起動します。
   接続・チャンネル設定を保持してDirac Liveの測定を行います。
4. 測定後、Dirac Liveのフィルター設計画面で対応するDirac_Targets/CHxx.targetcurveを読み込みます。
   manifest.jsonのGroupと実機のDiracチャンネル／グループを照合し、補正範囲を確認してください。
   複数Wayが同じDiracチャンネルを共有する場合があります。目標の位相はDirac Liveへ転送しません。
   フィルター作成・転送を完了したら、出力をミュートします。
   同じOutputs → Xover → Advanced Biquadの操作で、Post_Dirac/CHxx_XO.txtを読み込みます。
   重要：Pre/Postを追加・重ね掛けせず、同じ8段のXover Advanced Biquad領域で置換します。
   PEQ・Gain・極性・Delay・ルーティングを保持してください。
5. 再生用設定を保存し、使用するDiracフィルターを有効にします。
   再生前に、各段の読込結果、出力割当、音量を確認してください。

出力できない場合は48 kHz、FIR OFF、自動適用ON、LR4／NTM4を設定して再計算します。
最大4帯域、各チャンネルのXoverは8段、PEQは10段、出力はCH01〜CH12、Delayは0〜20 ms、Gainは-72〜+12 dBです。
上限を超えた項目は構成を見直してください。段数の切り捨ては行いません。空き段は通過係数で上書きします。

''' + CAUTION + '''
この出力は保護フィルターや自動レベル制限を追加しません。
実機への読込み・音響動作は、使用する機器で確認してください。
参照：https://www.minidsp.com/images/documents/C-DSP%208x12%20DL%20User%20Manual.pdf
'''
