"""Structured guide content shared by the UI, delivery README and user guide."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GuideSection:
    title: str
    body: str


_SECTIONS = {
    'ja': (
        GuideSection('概要', """目的：Dirac Liveの測定時に等価APFを使い、再生時に既存のクロスオーバー係数へ切り替えます。
Pre/Postは同じ位相基準を使う、出力Xoverの測定用／再生用の切替状態です。
機器内部のDirac処理より前／後の入力ブロックへ配置する意味ではありません。

Preは位相等価APF＋自動位相保証APF、Postは通常XO＋自動位相保証APFです。


### PhaseEQが用意するもの
1. 準備 → 設定したLR4／NTM4のクロスオーバーと構造補償APFを再利用し、
   測定用Pre APFと再生用Postクロスオーバーの係数ファイルを生成します。
2. 受渡し → IIR PEQ、Group Targetの目標振幅、出力対応・Gain・極性・Delayを記した
   manifest.jsonを、チャンネル別係数と一緒にZIPへまとめます。
3. 適合確認 → 出力条件と段数・機器上限を確認します。条件不適合時は出力を停止し、空き段は通過係数で埋めます。


### 機器・Dirac Liveで行うこと
ファイル読込み、ルーティング・共通出力値の設定、ユニット保護・測定音量の設定、マイク測定、
Diracグループの対応付け、ターゲット読込み・補正範囲の選択、フィルター作成・転送、PreからPostへの置換を行います。
PhaseEQはこれらの操作に使うファイルを用意します。実機接続・自動書込み・Dirac Liveの操作・補正後の実測確認は外部で行います。
NTM4と構造補償の係数はPhaseEQの既存設計に基づきます。実践記事によるこれらの方式の検証や直線位相の保証はありません。"""),
        GuideSection('準備', """### PEQは必要な場合だけ使用

| PEQの使用 | PhaseEQで行う準備 |
|---|---|
| 使わない | チャンネルごとのマイク測定データ登録は不要 |
| 使う（0〜10段） | REWなどで各チャンネルを測定 → 対応チャンネルへ測定結果を登録 → PEQを設定 |

10点すべてのPEQ設計は必須ではありません。**Dirac Live側のマイク測定は、どちらの場合も実施します。**

PEQなしでも、`PEQ/CHxx_PEQ.txt`を読み込んでください。全10段の通過係数で、以前のPEQを消去します。


### Group Targetの設定

- Group TargetをIIR Auto PEQの目標に使います。
- 帯域分割の「音響ターゲットに反映」は**全境界でOFF**にします。位相・時間整合側の同名設定もOFFにします。
- 位相構造補償の自動適用は**ON**にします。

`Dirac_Targets/CHxx.targetcurve`へ出力するのはGroup Targetの振幅だけです。通常XOの振幅とGroup Targetの位相は含めません。


### 位相補正の考え方

測定前の出力Xoverに、位相等価APF＋自動位相保証APFを設定します。この応答を測定し、Diracの位相補正で直線位相化を目指します。補正後の位相は実測で確認してください。"""),
        GuideSection('読込み・測定', """### ファイルと読込み先

| ファイル | 読込み先・操作 |
|---|---|
| `Pre_Dirac/CHxx_APF.txt` | Outputs → Xover → Advanced Biquad。全8段へ貼付け → Process |
| `PEQ/CHxx_PEQ.txt` | Outputs → PEQ → IMPORT。通過係数を含む全10段を置換 |
| `Dirac_Targets/CHxx.targetcurve` | Dirac Liveのフィルター設計画面 |
| `Post_Dirac/CHxx_XO.txt` | 測定・転送後、Preと同じXover全8段へ貼付け → Process |


### 1. 保存・出力の照合

機器設定を保存し、出力をミュートします。`manifest.json`のoutput_index（CH01〜CH12）・Group・Wayを出力端子と照合します。Gain・極性・Delayは記載値を設定し、Pre/Postで共通に使います。


### 2. 測定用PreとPEQを読込み

全チャンネルのXoverへPre、PEQへIIR PEQを読み込みます。PEQは測定時・再生時とも保持します。Xoverへ追加しないでください。


### 3. ユニット保護・マイク測定

**Pre APFは帯域外信号を減衰させません。** ユニット保護と安全な測定音量をユーザー自身で設定してください。

C-DSPの`Start Dirac Live Software`から起動し、ルーティング・チャンネル設定を保持して測定します。


### 4. ターゲット・フィルター転送

`manifest.json`のGroupを実機のDiracチャンネル／グループと照合し、対応するターゲットを読み込みます。複数Wayが同じDiracチャンネルを共有する場合があります。補正範囲を確認し、フィルターを作成・転送します。目標の位相は転送しません。


### 5. 再生用Postへ置換

出力をミュートし、**Preと同じXover全8段をPostで置換**します。Pre/Postを追加・重ね掛けしないでください。PEQ・Gain・極性・Delay・ルーティングは保持します。


### 6. 保存・再生前の確認

再生用設定を保存し、使用するDiracフィルターを有効にします。各段の読込結果・出力割当・音量を確認してから再生します。"""),
        GuideSection('再生・比較', """聴こえ方や曲目に合わせた微調整は、PhaseEQのGroup Targetを微調整し、ターゲットカーブをファイル出力します。
Dirac Liveの全チャンネルへ同じターゲットカーブを読み込んでください。
グループ分けしている場合は各グループへ同じカーブを読み込み、補正フィルターを再作成・転送します。

### 聴き比べの準備

聴き比べを重視する場合は、REWなどで各チャンネルを測定し、測定結果を登録して、
使用帯域内の周波数特性をPEQで平坦に整えた後にDirac Liveを設定することをお勧めします。
これは比較のための推奨手順です。PEQなしでAPF方式を使う場合の必須条件ではありません。

### Dirac Live ON/OFFの比較

Dirac LiveのON/OFF比較では、再生用Postの帯域分割・構造補償AP・PEQ・出力Delayを保持します。
AP構成は全チャンネルの位相基準をそろえるためのもので、各AP段の配置はWayの役割によって異なります。
機器はDirac補正OFF時にもDiracが算出したGain・Delayを保持します（機器マニュアル4.7）。
ディレー設定を切り替えずに比較し、相対的な到達時間と試聴音量を確認してください。
Dirac補正ONと、OFF時のIIR帯域分割による再生を聴き比べられます。
Dirac Liveは混合位相補正です（同1.2）。音の違いは機器構成・測定・ターゲット設定によって異なります。"""),
        GuideSection('条件・注意', """### 出力条件

| 項目 | 条件 |
|---|---|
| サンプルレート／DSP FIR | 48 kHz専用／OFF |
| 位相構造補償の自動適用 | ON |
| 全境界の方式 | LR4またはNTM4 |
| 音響ターゲットへの反映 | 全境界・位相時間整合ともOFF |
| 帯域数 | 最大4帯域（3Way+SUB／4Way） |
| チャンネルごとの段数 | Xover 8段／PEQ 10段 |
| 出力／Delay／Gain | CH01〜CH12／0〜20 ms／-72〜+12 dB |

出力できない場合は、表示された原因に合わせて設定を見直し、再計算します。段数は切り捨てず、条件不適合時は出力を停止します。空き段は通過係数で上書きします。


### 測定時の注意

**CAUTION：Pre-Dirac APFは帯域外信号を減衰させません。測定時のユニット保護はユーザー自身で行ってください。**

この出力は保護フィルターや自動レベル制限を追加しません。実機への読込み・音響動作は、使用する機器で確認してください。

Bass ManagementとBaffle EQは出力対象に含めません。"""),
        GuideSection('原典', """- 実践手順：Sonnie Parker「miniDSP C-DSP 8x12 DL Processor – Review and
  Setup/Tuning Guide」（AV NIRVANA、2026年6月25日）。参照箇所はAll Pass
  Filters Method、Pre-/Post-Dirac C-DSP Configurationです。
  https://www.avnirvana.com/threads/minidsp-c-dsp-8x12-dl-processor-%E2%80%93-review-and-setup-tuning-guide.16912/
  クロスオーバーの位相を表すAPFで測定し、測定後にクロスオーバーへ切り替える実践例です。
  メーカー標準手順としてではなく、APF方式の参考にします。
  記事はAPFをPEQへ手入力します。PhaseEQはPre/PostをXover用に出力し、IIR PEQはPEQへ分けます。
  読み込み先は「読込み・測定」の手順に従ってください。記事のBass Management・リアフィル・周波数例は本出力の対象外です。

- 機器仕様：miniDSP「C-DSP 8x12 DL User Manual」Revision 2.7（2021年8月5日）。
  4.4（信号フロー）、7.1（出力調整）、7.2.2（係数読込み）、8.1（機器仕様）を参照します。
  https://www.minidsp.com/images/documents/C-DSP%208x12%20DL%20User%20Manual.pdf

- 測定操作：miniDSP「Dirac Live User Manual」Revision 1.5（2021年11月29日）。
  4.12（測定）、4.13（グループ）、4.14（ターゲットと補正範囲）、Filter Exportを参照します。
  https://www.minidsp.com/images/documents/miniDSP%20Dirac%20Live%20User%20Manual.pdf"""),
    ),
    'en': (
        GuideSection('Overview', """Purpose: use equivalent APFs for the Dirac Live measurement stage, then switch to the existing crossover coefficients for playback with the same phase reference. Pre-Dirac and Post-Dirac are alternative measurement/playback states of the OUTPUT Xover blocks. They do not mean input blocks before/after the internal Dirac processor.

Pre: equivalent APFs + automatic phase APFs. Post: normal XO + automatic phase APFs.


### What PhaseEQ provides
1. Preparation → reuse the configured LR4/NTM4 crossover and structural APFs to generate the measurement Pre APFs and playback Post crossover files.
2. Delivery → package channel IIR PEQ, Group Target magnitude curves and manifest.json (output mapping, Gain, Polarity, Delay) in one ZIP.
3. Compatibility → check export conditions and section/device limits; block incompatible exports and fill unused slots with unity coefficients.


### What you do in C-DSP / Dirac Live

Load the files, set routing and common output controls, protect drivers and set measurement levels, take microphone measurements, map Dirac groups, load targets, choose correction limits, create/transfer filters and replace Pre with Post. PhaseEQ provides files for these steps; device connection, automatic loading, Dirac Live control and post-correction acoustic verification are performed externally. The NTM4 and structural-compensation coefficients follow PhaseEQ's existing design; the practical article does not establish their behavior or guarantee linear phase."""),
        GuideSection('Preparation', """### PEQ is optional

| PEQ use | Preparation in PhaseEQ |
|---|---|
| None | No per-channel microphone measurement registration |
| Used (0–10 sections) | Measure each channel with REW or equivalent → register the results for the matching channels → configure PEQ |

A ten-filter design is not required. **Dirac Live microphone measurement is required in either case.**

Even without PEQ, load `PEQ/CHxx_PEQ.txt`. Unity coefficients replace all 10 slots and clear old filters.


### Group Target settings

- Use Group Target as the IIR Auto PEQ target.
- Keep acoustic-target reflection **OFF at every crossover boundary**, and also OFF in phase/time alignment.
- Keep automatic structural phase compensation **ON**.

`Dirac_Targets/CHxx.targetcurve` contains Group Target magnitude only. Normal crossover magnitude and Group Target phase are excluded.


### Phase correction

Load equivalent APFs + automatic phase APFs into the output Xover before measurement. Measure this response and use Dirac phase correction with the aim of linearizing phase. Check the corrected phase with measurements."""),
        GuideSection('Loading & measurement', """### Files and destinations

| File | Destination / operation |
|---|---|
| `Pre_Dirac/CHxx_APF.txt` | Outputs → Xover → Advanced Biquad. Paste all 8 slots → Process |
| `PEQ/CHxx_PEQ.txt` | Outputs → PEQ → IMPORT. Replace all 10 slots including unity |
| `Dirac_Targets/CHxx.targetcurve` | Dirac Live filter-design view |
| `Post_Dirac/CHxx_XO.txt` | After measurement and transfer, paste into the same 8 Xover slots → Process |


### 1. Save and map outputs

Save the device configuration and mute outputs. Match `manifest.json` output_index (CH01–CH12), Group and Way to physical outputs. Apply its Gain, Polarity and Delay values in both Pre and Post states.


### 2. Load measurement Pre and PEQ

Load Pre into every channel's Xover and IIR PEQ into PEQ. Keep PEQ active for measurement and playback; do not add it to Xover.


### 3. Protect drivers and measure

**Pre APFs do not attenuate out-of-band signals.** Provide driver protection and choose safe measurement levels yourself.

Launch `Start Dirac Live Software` from C-DSP. Keep routing and channel settings fixed during microphone measurement.


### 4. Load targets and transfer filters

Match Group in `manifest.json` to the actual Dirac channel or group and load the corresponding target. Multiple output Ways can share one Dirac channel. Check correction limits, create the filter and transfer it. Target phase is not transferred.


### 5. Replace Pre with playback Post

Mute outputs. **Replace the same 8 Xover slots with Post.** Do not add or cascade Pre and Post. Keep PEQ, Gain, Polarity, Delay and routing unchanged.


### 6. Save and check playback

Save the playback configuration and enable the intended Dirac filter. Check imported slots, channel mapping and output level before playback."""),
        GuideSection('Playback & comparison', """For listening preferences or track-specific fine tuning, adjust Group Target in PhaseEQ and export the target curve file. Load the same curve for every channel in Dirac Live, including each group when channels are grouped, then create and transfer the correction filter again.

### Preparing a listening comparison

For a listening comparison, first measure each channel with REW or equivalent software, register the results and use PEQ to flatten its usable frequency range. Then calibrate Dirac Live. This preparation is recommended for comparison, not a prerequisite for using the APF workflow without PEQ.

### Dirac Live ON/OFF comparison

Keep the playback Post crossover, structural APs, PEQ and output Delay unchanged when comparing Dirac Live ON/OFF. The AP configuration aligns the channels to a common phase reference; individual AP sections depend on each Way's role. The device also retains Dirac's calculated Gain and Delay when filtering is OFF (C-DSP manual, section 4.7). Check relative arrival times and listening levels. Listen for differences between Dirac correction ON and IIR crossover playback with Dirac OFF. Dirac Live uses mixed-phase correction (manual, section 1.2); the audible difference depends on the system, measurements and target curve."""),
        GuideSection('Conditions & caution', """### Export conditions

| Item | Requirement |
|---|---|
| Sample rate / DSP FIR | 48 kHz only / OFF |
| Automatic structural phase compensation | ON |
| Every boundary | LR4 or NTM4 |
| Acoustic-target reflection | OFF at every boundary and in phase/time alignment |
| Bands | Up to four (3Way+SUB / 4Way) |
| Sections per channel | Xover 8 / PEQ 10 |
| Output / Delay / Gain | CH01–CH12 / 0–20 ms / -72 to +12 dB |

If export is disabled, fix the indicated settings and recalculate. Incompatible exports are blocked; coefficients are never truncated. Unused slots are overwritten with unity.


### Measurement caution

**CAUTION: Pre-Dirac APFs do not attenuate out-of-band signals. You are responsible for protecting the drivers during measurement.**

This export adds no protection filters or automatic level limiting. Verify hardware import and acoustic operation on the target device.

Bass Management and Baffle EQ are excluded."""),
        GuideSection('Sources', """- Practical reference: Sonnie Parker, miniDSP C-DSP 8x12 DL Processor – Review   and Setup/Tuning Guide (AV NIRVANA, June 25, 2026), sections All Pass Filters   Method and Pre-/Post-Dirac C-DSP Configuration:   https://www.avnirvana.com/threads/minidsp-c-dsp-8x12-dl-processor-%E2%80%93-review-and-setup-tuning-guide.16912/   The author describes measuring with APFs representing crossover phase, then   switching to crossovers. This is a practical method, not a miniDSP standard procedure.   The article enters APFs in PEQ; PhaseEQ puts Pre/Post coefficients in Xover   and keeps the channel IIR PEQ in PEQ. Follow the destinations in Loading & measurement.   The article's Bass Management, rear fill and example frequencies are outside this export.

- Device reference: miniDSP, C-DSP 8x12 DL User Manual, Revision 2.7   (August 5, 2021), sections 4.4 (signal flow), 7.1 (output controls),   7.2.2 (coefficient loading) and 8.1 (specifications):   https://www.minidsp.com/images/documents/C-DSP%208x12%20DL%20User%20Manual.pdf

- Calibration reference: miniDSP Dirac Live User Manual, Revision 1.5   (November 29, 2021), sections 4.12 (measurement), 4.13 (groups),   4.14 (target curves and correction range) and Filter Export:   https://www.minidsp.com/images/documents/miniDSP%20Dirac%20Live%20User%20Manual.pdf"""),
    ),
}


def cdsp_guide_sections(language='ja'):
    """Return ordered content; rendering and device operations remain separate."""
    return _SECTIONS['en' if language == 'en' else 'ja']


def cdsp_guide(language='ja'):
    """Keep the existing README API and filename; Markdown stays readable as text."""
    title = 'C-DSP 8x12 DL Dirac/APF export' if language == 'en' else 'C-DSP 8x12 DL Dirac/APF出力'
    return title + '\n\n' + '\n\n'.join(
        f'## {section.title}\n\n{section.body}' for section in cdsp_guide_sections(language)) + '\n'
