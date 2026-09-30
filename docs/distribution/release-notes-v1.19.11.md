# PhaseEQ v1.19.11 更新案内

本書はこの版を公開した時点の変更記録です。現在の操作・設定条件は[操作ガイド](user-guide.md)、後続版の変更は[更新履歴](history.md)を参照してください。

PhaseEQ v1.19.11では、設定編集・復元の安定性を改善し、FIR出力設定と応答グラフを拡充しました。Target presetsに「Harman H4 Target Curve」を追加しています。

## 設定の編集・復元

- 数値・文字・選択・ON/OFFの編集内容を変更イベント時に確定し、画面の切り替えや復元で古い初期値へ戻る経路と二重初期化警告を修正しました。
- Target level、Manual shift、Phase centering、Phase maskなどの保存・再表示を改善しました。Compositeの受信監視も安定化しました。
- EQ編集リストの追加・複製は種類ごとに暫定128件までです。既存の超過設定は切り捨てません。DB候補は暫定8,192件ずつ表示します。

## FIR出力とグラフ

- FIRタップ数の指定とApplyをExportへ移しました。Assignment接続中は受信したタップ数を使用します。
- PhaseEQ ExportとCompositeの最終DSP用FIRにCosine Tapered窓を追加しました。α=0.10固定、初期値OFFで、最終切り詰め後に一度だけ適用します。係数グラフで窓形状も確認できます。
- Compositeの最終DSP用FIR設定・特性を集約し、Nyquist成分除去と強度を指定できるようにしました。グラフ・バックアップ・実機出力は同じ最終係数を使用します。
- CompositeのFIR係数入力は、全段を全長で線形畳み込みしてから一度だけ中央切り出しします。従来方式から再生成すると係数が変わる場合があります。
- PhaseEQのInteractive応答グラフをPlotlyへ変更しました。ブラウザー内で拡大・移動できます。FIR EQは選択行に関係なく全有効EQを表示し、IIR／FIRの合計領域表示も復旧しました。
- Compositeの解析結果とLight画像を再利用し、解析用Group応答ZIPはダウンロード時に生成します。

## 入力応答とTarget

- Speakerの帯域外補完を共通化し、高域を測定最大周波数から滑らかに接続します。LF／HF extensionの新規既定値はONです。保存済みの明示的なOFFは保持します。
- 入力の時間原点・位相の由来・校正状態を一貫して引き継ぎます。マイク校正はP0で一度だけ適用します。
- Target presetsはApply前にGain／Phaseをプレビューできます。「Harman H4 Target Curve」も選択できます。PhaseEQのGain only Targetは欠損位相を0°で補完し、位相無効指定は優先します。

## 更新時の確認

1. 編集内容を保存し、必要に応じてWorking Session ZIPでバックアップしてから、PhaseEQとComposite Engineを再起動してください。
2. 補完ONの既存設定も新方式で再計算するため、AutoEQ・生成FIRが変わる場合があります。IR／Step誤差の一律改善を保証するものではありません。再出力前に最終特性を確認してください。
3. 最終FIR設定・時間情報を別PCへ渡す場合は、受取側も本版へ更新してください。既に書き出した完成済みファイルは変更しません。

操作の詳細は[利用者ガイド](user-guide.md)、出力内容は[出力ファイル仕様](output-format.md)を参照してください。
