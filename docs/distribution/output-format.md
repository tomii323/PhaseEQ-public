# PhaseEQ 出力ファイル仕様

- 文書更新日: 2026-09-25
- 対象公開版: PhaseEQ v1.20.8
- 文書状態: v1.20.1リリース版

PhaseEQの「書き出し」、マルチウェイの「保存・出力」、PhaseEQの「設定 → バックアップと復元」で保存できるファイルを説明します。操作順は[操作ガイド](user-guide.md)、過去の変更は[更新履歴](history.md)を参照してください。

各ダウンロード操作時に確定済み設定から必要な出力を準備します。別の準備ボタンは不要です。表示FFT倍率やグラフの間引き点数は出力FIR係数を変更しません。

作業セッションには帯域分割生成定義と生成条件を保持し、帯域分割IIR／Combined IIR chainのJSONおよび極性情報も収録します。新形式FIR 生成定義はalgorithm v4です。自動ゲイン計算専用の入力整形設定を保持し、整形済みの測定配列は保存しません。旧v1〜v3は入力整形OFFで再生成します。他PCで再生成する場合は受取側も対応版へ更新してください。旧形式v1の再生成は旧グリッドを維持しますが、現在の作業を新条件で再計算・再出力した係数は従来版と変わる場合があります。

## 出力の分類

| 出力 | 用途 | 復元用か |
|---|---|---|
| 目標応答 | 目標応答の受け渡し | いいえ |
| IIR バイクアッド | 外部DSPへのIIR実装 | いいえ |
| Generic IIR Parameters | バイクアッドを直接入力できないIIR-only DSPへの設計値転記 | いいえ |
| FIR coefficients / WAV | 畳み込みエンジンへのFIR実装 | いいえ |
| システム プリセット JSON | DSPシステム設定の保存・再利用 | はい |
| DSP Configuration Package ZIP | Device別の実装ファイル一式 | いいえ |
| Manifest JSON | Package内容、署名、安全判定の監査 | いいえ |
| Current settings JSON / Local snapshot | 現在設定と編集状態の復元 | はい |
| 作業セッションZIP | 作業状態と読み込み済み応答の移動・再開 | はい |
| 測定データ JSON | 測定DBの移行 | はい |
| ドライバーデータベース JSON | ドライバー諸元DBの移行 | はい |
| 測定パッケージ | 未校正統合IR、別校正データ、測定条件・時間情報の保管 | 保存済み原本からの再計算用 |

## 目標応答

目標特性の周波数、ゲイン、位相をCSV／テキスト形式で出力します。周波数はHz、ゲインはdB、位相はdegreeです。出力対象はEditorで適用済みの目標特性であり、未適用のプレビューは含めません。

## IIR バイクアッド

IIRイコライザー、出力帯域分割、両者を直列にしたCombined IIR chainを選択して出力します。出力帯域分割には直接出力のLR2／LR4・1次／2次オールパス・極性を含めます。音響ターゲットへ移した成分は直接出力から除き、LR2自動極性は帯域分割の設定、オールパスは位相・時間整合の設定に従います。手動極性は独立して直接適用します。完全な出力係数に含まれる極性を外部DSPで二重に適用しないでください。

| 形式 | 係数規約 |
|---|---|
| Generic / internal | `b0, b1, b2, a1, a2` |
| CamillaDSP | `DiffEq` |
| SigmaDSP / SigmaStudio | `B0, B1, B2, A1, A2` |
| miniDSP | 内部`a1/a2`の符号を反転した機器規約 |

量子化後の極半径が`0.9999`以上になる構成は安全判定を通過せず、正式なDSP Package出力を許可しません。表示用の理論応答ではなく、実際に出力する量子化係数で判定します。

## Generic IIR Parameters

`汎用IIRパラメーター`は、バイクアッド係数を直接入力できず、IIRフィルターの設計値を手入力するDSP向けのMarkdown文書です。サンプルレートと、各フィルターの`Apply order / Type / Frequency [Hz] / Gain [dB] / Q / Order / Enabled / Family / Origin / All-pass preset`を記録します。無効なフィルターも`Enabled: No`として残し、編集内容と並び順を失いません。

PhaseEQ 書き出しではIIRイコライザー、出力帯域分割ページ、両者を直列にしたCombined IIR chainごとに個別ダウンロードできます。作業セッションZIPでは`outputs/<Project>_<Band>_IIR_Generic-Parameters_<SampleRate>_<Timestamp>.md`として収録します。帯域がない場合はファイル名から省略します。

転記時は適用 orderの上から順に設定します。DSPごとにShelf Q、クロス family、オールパスの定義や設定可能範囲が異なるため、この文書は無条件な機種互換を保証しません。転記後は対象DSPの実現応答を確認します。

追加のオールパス・極性段はGeneric IIR Parameters内でもバイクアッド係数表として記録します。通常のパラメーター表だけを転記しても全段は再現できません。

## FIR

FIRはテキスト係数またはWAVとして出力します。サンプルレート、タップ数、極性、正規化、対象出力は、適用済みのDSPシステム設計条件に従います。

- `出力FIRの極性反転`を有効にすると、個別FIRとPackage内FIRの両方へ反映します。
- システムまたはDeviceがFIR非対応、または対象出力がFIR bypassの場合、そのDevice Packageから該当FIR段を省略します。
- FIR Peak、係数長、サンプルレートの不整合は書き出し Gateで検査します。

書き出しグラフはFIR出力、IIR出力、FIR＋IIR出力を分けて表示します。スピーカー＋実現応答と時間波形には両方を一度ずつ適用します。FIR OFF時のFIR出力はバイパス表示で、FIRファイルを生成する意味ではありません。合成グラフによってFIR係数へIIRを畳み込むことはありません。

## システム プリセット JSON

`Download System preset`は、DSPシステム、Device、入力、Route、出力、クロス、アライメント、適用済み目標特性／IIR／FIRを再利用するためのJSONです。未適用のEditor Working Copyは最終出力条件として扱いません。

## DSP Configuration Package ZIP

`Download DSP Package ZIP`は、接続Deviceごとの設定と係数をまとめた実装用アーカイブです。現在のPackage manifestは`schema_version: 8`です。

Packageには構成に応じて次を含みます。

- Device／入力／Route／出力 チャンネル設定
- 目標特性、IIR、FIR、クロス、ゲイン、遅延、極性
- CamillaDSP、miniDSP、SigmaDSP／SigmaStudio向け設定または係数
- 設計 Signatureと生成条件
- Headroom、Peak、IIR安定性などの安全判定
- Manifest

Packageは、現在の設計 Signatureに対応するシステム応答を更新し、`設計を保存`で正式保存した後だけ出力できます。設定変更後に古い計算結果を流用することはできません。

## マルチウェイ Studio DSP 書き出し v3

PhaseEQ同梱マルチウェイ Studioの`DSP 書き出し v3`は、上記PhaseEQ DSPシステム Packageとは独立した後段DSP用アーカイブです。ファイル名はシステム名、モード、DSP Adapter、サンプルレート、生成日時を含みます。Adapter選択はStudioのスピーカー、PhaseEQ、クロス、位相整合、グラフ状態を変更しません。

共通構成は次のとおりです。

- チャンネル別の最終FIR、`phaseeq_iir.json`、`crossover_iir.json`、ゲイン／極性／遅延
- PhaseEQ IIR、IIRバッフル補正、IIR帯域分割、帯域分割補償オールパスを別Stage／別ファイルで保持
- `manifest.json`（`format: phaseeq-multiway-dsp-export`, `format_version: 3`）、SHA-256、ワークスペース、チャンネル設定表、検証Report
- 処理順は`Final FIR → PhaseEQ IIR → Baffle IIR → IIR crossover → crossover All-pass → Gain → Polarity → Delay`
- `Delay [samples]`は画面の「タップ長とディレイ目安」と同じく、`FIR長差Delay + 手動Delay + 位相・時間整合Delay`を出力します。FIR係数自体が持つ固有センター遅延`(N-1)/2`は二重加算しません。出力タップ統一時のFIR長差遅延は0です。
- 外部アプリで測った距離を時間基準にした場合、チャンネル別`channel_config.json`の`timing_provenance.external_distance_alignment`へ距離[m]、公称音速343.42 m/s、入力元を監査情報として記録します。距離から求めた実適用値は`delay.phase_alignment_samples`へ一度だけ含め、書き出し時に再加算しません。
- FIR OFFのチャンネルは`tap_count=null`、`final_fir=null`として扱い、FIR WAV／CSV／BIN、FIR 生成定義、FIR reportを出力しません。IIR、ゲイン、極性、遅延はFIRと独立して出力します。
- `documentation/system_summary.md`、`documentation/system_specification.md`、`documentation/system_specification.json`を同梱します。3ファイルは同じCanonical DSP Packageから生成し、保存済みマルチウェイ システムがある場合はシステム ID、管理No、システム 改訂番号、content hashを記録します。チャンネル、FIR固有センター遅延、DSP追加遅延、FIR／IIR、DSP profile、signature、検証結果も同じ型番から出力します。単独編集出力は保存済みシステムを偽装しません。
- PhaseEQ 作業セッションには`documentation/session_specification.md`と`documentation/session_specification.json`を同梱します。割り当てにシステム provenanceがある場合はそのシステム／目標特性参照を記録し、それ以外は単独編集 sessionとして明示します。

| Adapter | 主な出力 |
|---|---|
| miniDSP | チャンネル別float32 little-endian `final_fir.bin`、PhaseEQ／クロス別Advanced バイクアッド TXT、チャンネル設定 |
| CamillaDSP | float32 WAV係数、`camilladsp.yml`、PhaseEQ／クロス別Filter YAML |
| SigmaStudio | FIR float／5.23、General 2nd 次数係数 float／5.23、Block map、チャンネル設定 |
| Generic SOS | PhaseEQ IIR、Baffle IIR、IIR クロス、補償オールパスを別CSVにした標準SOS出力 |
| Generic Parameters | IIR-only機器向けPhaseEQ `type / Hz / dB / Q` CSVとクロス／オールパス JSON。FIR非対応、整数遅延前提 |

画面上の`作業バックアップ`（内部形式：DSP package + Resume v2）と`実機DSP用パッケージ`（内部形式：DSP 書き出し v3）は目的とVersion体系が異なります。前者は作業継続、後者は後段DSPへの投入が主目的で、相互のVersion番号を読み替えません。DSP 書き出し v3にも`configuration/workspace.json`が含まれるため、マルチウェイ Studioの区間 7「設定ファイルを読み込む／作業を再開」で読み込めます。読込前にパスとSHA-256を検証し、初期のv3で表示選択値が欠けている場合は、保存されたモノラル／ステレオのチャンネル構成からMainまたはL+Rへ安全に補完します。

## Manifest JSON

`Download Manifest JSON`はPackageと同じ生成条件を単独で確認するための監査用JSONです。Device、出力ファイル、サンプルレート、係数形式、設計 Signature、安全判定を記録します。Package本体の代替ではありません。

## 作業セッションZIP

旧称`Project ZIP`です。現行UIでは`Application > バックアップと復元`から書き出し／Importします。

含むもの:

- 現在設定と編集状態
- 読み込み済みスピーカー入力、マイク校正、目標応答の数値
- 作業再開に必要な選択状態と応答参照

含まないもの:

- 測定データ DB全体
- ドライバーデータベース全体
- DSPシステム DB全体の完全バックアップ
- 測定データ セッションの生データ音声や測定単位NPZ

別PC移行、長期保管、DB初期化前には、作業セッションZIPだけでなく測定データ JSONとドライバーデータベース JSONも保存してください。

## 測定データ セッションZIP

連続測定中の一時ファイルは作業用です。`Save Measurement`がDBへ成功すると対象セッションの一時領域は整理されます。測定パッケージ（schema 5）は統合未校正IR 1本の圧縮NPZ、calibration.json、measurement.jsonだけを含みます。生データ・各ショットIRは含めません。ライブラリの外部出力ZIPはcalibrated_ir.wav、calibrated_response.frd、calibrated_output.jsonです。FRDは校正有効帯域の校正済み応答、WAVは同条件の校正済みfloat32 IRで、ピーク正規化しません。単位・感度・Gate・時間原点と別保持の到達時刻は付属JSONを参照してください。

### 新形式の内容と既存形式との関係

| 出力/保存 | 内容 | 使用目的 |
|---|---|---|
| 測定DB | 統合未校正IRの圧縮BLOB＋別校正/recipe/metadata、校正済みFRキャッシュ | 通常のライブラリ利用 |
| 測定パッケージschema 5 | `integrated_uncalibrated_ir.npz`、`calibration.json`、`measurement.json` | 任意の移動・保管。停止・統合後に出力 |
| 校正済み出力ZIP | `calibrated_ir.wav`、`calibrated_response.frd`、`calibrated_output.json` | 外部利用。merged/ungated/gatedを選択 |
| 測定データ JSON | 原本NPZのBase64＋校正・metadata・測定レコード | 測定DB全体のバックアップ/別DB復元 |
| 作業セッションZIP | 現在の設計設定・入力応答・設計成果物 | 作業再開。測定原本DB全体のバックアップではない |

外部WAVのsample 0は円環IRの中心で、実測の到達時刻は別metadataです。波形の長さにはFFT paddingを含み得るため、長さから観測精度を推定しないでください。外部アプリの音圧・時間原点の解釈は実読込未検証です。旧セッションZIPは互換読込を維持し、既存ファイルを自動変換しません。

## ファイル名と時刻

ダウンロード名には設計／システム名と生成時刻を含めます。時刻はファイル同士の対応確認に使い、内容の同一性は設計 SignatureとManifestで判定します。ファイル名が同じでも署名が異なる場合は同一設計として扱いません。

## 書き出し前の確認

1. 入力、目標特性、使用するFIR／IIRを確認します。
2. PhaseEQの「書き出し」でFIR・IIRそれぞれの応答と合成応答を確認します。
3. マルチウェイへ返す場合は「マルチウェイへ送信」を押し、受信後の各チャンネルと合成応答を確認します。
4. 作業を再開するための設定や作業セッションを保存します。
5. 実機で使う係数やDSP用パッケージを保存し、サンプルレート、タップ数、チャンネルの対応を確認します。

作業保存と実機用係数の出力は別の操作です。保存物の用途は本書冒頭の表で確認してください。

実画面で手順を確認する場合は、[画像付き操作マニュアル（PDF）](manual/PhaseEQ_v1.20.8_画像付き操作マニュアル_日本語.pdf)の第9章「書き出しとマルチウェイ連携」と第12章「マルチウェイ FIR Studio」を参照してください。
