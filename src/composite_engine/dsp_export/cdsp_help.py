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
PEQ design is optional. REW and a ten-filter design are not prerequisites.
Without PEQ, no per-channel microphone measurements need to be registered in PhaseEQ.
With PEQ, measure each channel using REW or equivalent software, register the
measurements for the corresponding channels in PhaseEQ, then configure PEQ.
Dirac Live microphone measurement is a separate step required in either case.
For listening preferences or track-specific fine tuning, adjust Group Target in
PhaseEQ and export the target curve file. Load the same curve for every channel
in Dirac Live, including each group when channels are grouped, then create and
transfer the correction filter again.
For a listening comparison, first measure each channel with REW or equivalent
software, register the results and use PEQ to flatten its usable frequency range.
Then calibrate Dirac Live. This preparation is recommended for comparison, not
a prerequisite for using the APF workflow without PEQ.
Keep the playback Post crossover, structural APs, PEQ and output Delay unchanged
when comparing Dirac Live ON/OFF. The AP configuration aligns the channels to
a common phase reference; individual AP sections depend on each Way's role.
The device also retains Dirac's calculated Gain and Delay when filtering is OFF
(C-DSP manual, section 4.7). Check relative arrival times and listening levels.
Listen for differences between Dirac correction ON and IIR crossover playback
with Dirac OFF. Dirac Live uses mixed-phase correction (manual, section 1.2);
the audible difference depends on the system, measurements and target curve.
Use only the IIR PEQ needed for your system (0-10 sections). With no PEQ,
the exported PEQ file contains unity in all 10 slots; load it to clear old filters.
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

Sources and scope
- Practical reference: Sonnie Parker, miniDSP C-DSP 8x12 DL Processor – Review
  and Setup/Tuning Guide (AV NIRVANA, June 25, 2026), sections All Pass Filters
  Method and Pre-/Post-Dirac C-DSP Configuration:
  https://www.avnirvana.com/threads/minidsp-c-dsp-8x12-dl-processor-%E2%80%93-review-and-setup-tuning-guide.16912/
  The author describes measuring with APFs representing crossover phase, then
  switching to crossovers. This is a practical method, not a miniDSP standard procedure.
  The article enters APFs in PEQ; PhaseEQ puts Pre/Post coefficients in Xover
  and keeps the channel IIR PEQ in PEQ. Follow the destinations below.
  The article's Bass Management, rear fill and example frequencies are outside this export.
- Device reference: miniDSP, C-DSP 8x12 DL User Manual, Revision 2.7
  (August 5, 2021), sections 4.4 (signal flow), 7.1 (output controls),
  7.2.2 (coefficient loading) and 8.1 (specifications):
  https://www.minidsp.com/images/documents/C-DSP%208x12%20DL%20User%20Manual.pdf
- Calibration reference: miniDSP Dirac Live User Manual, Revision 1.5
  (November 29, 2021), sections 4.12 (measurement), 4.13 (groups),
  4.14 (target curves and correction range) and Filter Export:
  https://www.minidsp.com/images/documents/miniDSP%20Dirac%20Live%20User%20Manual.pdf

What PhaseEQ supports
1. Preparation → reuse the configured LR4/NTM4 crossover and structural APFs
   to generate the measurement Pre APFs and playback Post crossover files.
2. Delivery → package channel IIR PEQ, Group Target magnitude curves and
   manifest.json (output mapping, Gain, Polarity, Delay) in one ZIP.
3. Compatibility → check export conditions and section/device limits;
   block incompatible exports and fill unused slots with unity coefficients.
What you do in C-DSP / Dirac Live
Load the files, set routing and common output controls, protect drivers and set
measurement levels, take microphone measurements, map Dirac groups, load targets,
choose correction limits, create/transfer filters and replace Pre with Post.
PhaseEQ provides files for these steps; device connection, automatic loading,
Dirac Live control and post-correction acoustic verification are performed externally.
The NTM4 and structural-compensation coefficients follow PhaseEQ's existing design;
the practical article does not establish their behavior or guarantee linear phase.

Loading and measurement steps
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
'''
    return '''C-DSP 8x12 DL Dirac/APF出力

目的：Dirac Liveの測定時に等価APFを使い、再生時に既存のクロスオーバー係数へ切り替えます。
Pre/Postは同じ位相基準を使う、出力Xoverの測定用／再生用の切替状態です。
機器内部のDirac処理より前／後の入力ブロックへ配置する意味ではありません。
48 kHz専用、DSP FIR出力OFF、位相構造補償の自動適用ON、全境界LR4またはNTM4、音響ターゲットOFFで出力します。
最大4帯域（3Way+SUBまたは4Way）です。読み込み先はOutputs → Xover → Advanced Biquadに固定します。
Preは位相等価APF＋自動位相保証APF、Postは通常XO＋自動位相保証APFです。
各チャンネルのIIR PEQをPEQ/CHxx_PEQ.txtへ出力します（最大10段）。
PEQの設計は任意です。REWなどの使用や10点すべてのPEQ設計は必須ではありません。
PEQを使わない場合、PhaseEQへのチャンネルごとのマイク測定データ入力は不要です。
PEQを使う場合は、REWなどでチャンネルごとのマイク測定を行い、
測定結果をPhaseEQの対応するチャンネルへ登録した上で、PEQを設定してください。
Dirac Live側のマイク測定は別の工程で、どちらの場合も実施します。
聴こえ方や曲目に合わせた微調整は、PhaseEQのGroup Targetを微調整し、ターゲットカーブをファイル出力します。
Dirac Liveの全チャンネルへ同じターゲットカーブを読み込んでください。
グループ分けしている場合は各グループへ同じカーブを読み込み、補正フィルターを再作成・転送します。
聴き比べを重視する場合は、REWなどで各チャンネルを測定し、測定結果を登録して、
使用帯域内の周波数特性をPEQで平坦に整えた後にDirac Liveを設定することをお勧めします。
これは比較のための推奨手順です。PEQなしでAPF方式を使う場合の必須条件ではありません。
Dirac LiveのON/OFF比較では、再生用Postの帯域分割・構造補償AP・PEQ・出力Delayを保持します。
AP構成は全チャンネルの位相基準をそろえるためのもので、各AP段の配置はWayの役割によって異なります。
機器はDirac補正OFF時にもDiracが算出したGain・Delayを保持します（機器マニュアル4.7）。
ディレー設定を切り替えずに比較し、相対的な到達時間と試聴音量を確認してください。
Dirac補正ONと、OFF時のIIR帯域分割による再生を聴き比べられます。
Dirac Liveは混合位相補正です（同1.2）。音の違いは機器構成・測定・ターゲット設定によって異なります。
必要なIIR PEQだけを使います（0〜10段）。PEQなしの場合は全10段が通過係数のファイルを出力します。
以前のPEQを残さないため、このファイルを読み込んでください。
各チャンネルのDirac_Targets/CHxx.targetcurveにはGroup Targetの振幅だけを出力します。
通常XOの振幅とGroup Targetの位相は含めません。
測定前の出力Xoverには位相等価APF＋自動位相保証APFを設定します。
このAPFを含む応答を測定し、Diracの位相補正で直線位相化を目指します。
補正後の位相は実測で確認してください。
Group Targetの選択は、帯域分割の「音響ターゲット」ON/OFFとは別の設定です。
Group TargetをIIR Auto PEQの目標に使うため、帯域分割の「音響ターゲットに反映」は全境界でOFFにします。
位相・時間整合側の同名設定もOFFにします。位相構造補償の自動適用はONのまま、APFをXoverへ設定します。
Bass ManagementとBaffle EQは出力対象に含めません。

参照原典と適用範囲
- 実践手順：Sonnie Parker「miniDSP C-DSP 8x12 DL Processor – Review and
  Setup/Tuning Guide」（AV NIRVANA、2026年6月25日）。参照箇所はAll Pass
  Filters Method、Pre-/Post-Dirac C-DSP Configurationです。
  https://www.avnirvana.com/threads/minidsp-c-dsp-8x12-dl-processor-%E2%80%93-review-and-setup-tuning-guide.16912/
  クロスオーバーの位相を表すAPFで測定し、測定後にクロスオーバーへ切り替える実践例です。
  メーカー標準手順としてではなく、APF方式の参考にします。
  記事はAPFをPEQへ手入力します。PhaseEQはPre/PostをXover用に出力し、IIR PEQはPEQへ分けます。
  読み込み先は下記手順に従ってください。記事のBass Management・リアフィル・周波数例は本出力の対象外です。
- 機器仕様：miniDSP「C-DSP 8x12 DL User Manual」Revision 2.7（2021年8月5日）。
  4.4（信号フロー）、7.1（出力調整）、7.2.2（係数読込み）、8.1（機器仕様）を参照します。
  https://www.minidsp.com/images/documents/C-DSP%208x12%20DL%20User%20Manual.pdf
- 測定操作：miniDSP「Dirac Live User Manual」Revision 1.5（2021年11月29日）。
  4.12（測定）、4.13（グループ）、4.14（ターゲットと補正範囲）、Filter Exportを参照します。
  https://www.minidsp.com/images/documents/miniDSP%20Dirac%20Live%20User%20Manual.pdf

PhaseEQがサポートする範囲
1. 準備 → 設定したLR4／NTM4のクロスオーバーと構造補償APFを再利用し、
   測定用Pre APFと再生用Postクロスオーバーの係数ファイルを生成します。
2. 受渡し → IIR PEQ、Group Targetの目標振幅、出力対応・Gain・極性・Delayを記した
   manifest.jsonを、チャンネル別係数と一緒にZIPへまとめます。
3. 適合確認 → 出力条件と段数・機器上限を確認します。条件不適合時は出力を停止し、空き段は通過係数で埋めます。
機器・Dirac Live側で行う操作
ファイル読込み、ルーティング・共通出力値の設定、ユニット保護・測定音量の設定、マイク測定、
Diracグループの対応付け、ターゲット読込み・補正範囲の選択、フィルター作成・転送、PreからPostへの置換を行います。
PhaseEQはこれらの操作に使うファイルを用意します。実機接続・自動書込み・Dirac Liveの操作・補正後の実測確認は外部で行います。
NTM4と構造補償の係数はPhaseEQの既存設計に基づきます。実践記事によるこれらの方式の検証や直線位相の保証はありません。

読込み・測定の手順
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
'''
