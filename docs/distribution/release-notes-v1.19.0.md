# PhaseEQ v1.19.0 更新案内

PhaseEQ v1.19.0では、Multiway StudioのSystem管理、Target連携、Mono／Stereo／SUB構成、グラフ表示、位相・時間整合、保存・復元・Exportを一つの流れとして整理しました。

## 主な更新

### FullrangeとFullrange+SUB

帯域分割に`Fullrange`と`Fullrange+SUB`を追加しました。`Fullrange`はMainを分割せず、そのまま通します。`Fullrange+SUB`はSUBのLow-passとFullrangeのHigh-passを同じ境界から生成し、両者の合成が平坦になる相補応答とします。

Mono／Stereo、共有SUB／左右独立SUB、Channel設定、PhaseEQ Assignment、System保存、Resume、DSP Exportを、既存の2Way～4Wayと同じ操作体系で扱えます。

### L／R／L+R表示

Stereo時の表示を`L`、`R`、`L+R`へ統一しました。SUB単独表示はありません。

- `L`：Leftに属する全Wayと`L System Sum`
- `R`：Rightに属する全Wayと`R System Sum`
- `L+R`：左右の全Way、`L System Sum`、`R System Sum`を同じグラフに表示

`L+R`は左右を電気的に加算した新しいDSP出力ではなく、比較しやすく並べる表示専用ビューです。WaveletではLeftとRightを別々のパネルに表示します。

共有SUBはChannel線を1本だけ表示し、左右それぞれのSystem Sumへ1回ずつ含めます。左右独立SUBはChannel IDで区別するため、同じ表示名でも混同しません。

### Targetの扱い

Group TargetとChannel Targetは参照線として表示し、Multiway／CompositeのChannel応答やSystem Sumへ直接加算・乗算しません。Targetの表示切替だけではDSPを再計算しません。

Composite EngineからGroup Targetを指定してPhaseEQを開くと、そのTargetを通常のPhaseEQ Target編集へ引き継ぎます。Target Source、Level、Gain／Phase編集、Presetは同じTarget定義として保存します。`Gain only`を選んだ場合はTarget SourceのPhase列だけを使用せず、Target Shapeで追加したPhase編集は維持します。

### Multiway SystemとRevision

Multiway System Libraryが、System、Channel、Target、Assignmentの関連を管理します。管理番号、System名、Revision、編集中のGroup／Way／Channel／Targetを画面上で確認できます。

通常は最新Target Revisionを参照できます。過去の設定へ戻す場合は、保存済みRevisionを上書きせず、新しいRevisionとして復帰します。複数Systemを切り替えても、別SystemのAssignmentやTargetを誤って流用しないよう識別情報を検証します。

### PhaseEQ Assignment

Composite Assignment中は、帯域分割とそのFIR／IIR方式をMultiway設定へ固定します。PhaseEQ側では確認できますが変更できません。PhaseEQのFIR EQは設定情報からComposite Engine内で再生成し、係数だけを無条件に引き継ぎません。

Speaker/Inputが設定されていない場合は、古いSpeaker/Input、Auto IIR Preview、Envelopeを表示しません。IIRだけのChannelでは、Unity FIRをFIR適用済みとして扱いません。

### 位相・時間整合とDelay

Mono／Stereoで実行ボタンを分けず、現在Systemの全有効Channelを一度に整合します。StereoはLeft／Rightを同時に処理し、共有SUBは1つの物理Channelとして扱います。Displayed channelの選択は解析対象へ影響しません。

FIRとIIRを混在する場合も、FIRの理論中心、IIR境界、手動Delay、自動整合Delayを分離して扱います。DSP Export v3の`Delay [samples]`には、機器へ設定する合計Delayを出力します。

### 起動処理

macOS／Linux／Windowsの連動ランチャーは、PhaseEQとComposite Engineの応答を0.3秒間隔で確認し、利用可能なLoopback URLが確認できてからBrowserを開きます。起動直後に`localhost`で表示できない場合も、自動再試行して待機します。

## 更新後の確認

1. Multiway Studioで`Fullrange+SUB`を選びます。
2. `Channel layout`をMonoまたはStereoへ切り替えます。
3. Stereoでは結果表示が`L`、`R`、`L+R`だけであることを確認します。
4. Group Targetを表示しても、SUBやSystem SumのLevelが変わらないことを確認します。
5. 必要に応じて`位相・時間整合を適用`を1回実行します。
6. Systemを保存し、ResumeまたはDSP Export v3を作成します。

## 互換性

既存の2Way、3Way、4Wayと各+SUBを継続利用できます。旧AssignmentとResumeは対応Versionの範囲で読み込みますが、Target非適用を確認できない古い返却FIRは安全のためDSP経路へ接続しません。元データは削除せず、再編集・再保存できる状態を維持します。
