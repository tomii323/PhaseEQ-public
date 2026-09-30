# PhaseEQ v1.19.4 更新案内

PhaseEQ v1.19.4では、Multiway StudioのFIR出力状態を明示し、IIRだけを使用する構成で不要なFIRタップ数やFIRファイルが生成されないようにしました。また、品質チェックを変更内容に応じて分類し、リリース時の重複テストを安全に省略します。

## Multiway FIR出力

- Section 6へ`DSP FIR出力を有効にする`を追加しました。
- Fullrangeでも帯域タップ長を表示し、FIRを使用する場合に設定できます。
- FIRを初めてONにした時、分割FIRから長さを取得できない帯域は1023 tapsになります。
- FIRクロスを使用する帯域でタップ長が0の場合は、実際に生成された分割FIRの長さを使用します。
- IIRのみ、またはFullrangeでタップ長が0の場合はFIR OFFです。1 tapのUnity FIRへ置き換えません。

## Composite・PhaseEQ・DSP Export

- FIR OFFではFIR係数、FIR Recipe、FIRファイル、FIR report、FIRタップ数を出力しません。
- FIR OFFでもPhaseEQ Assignmentは利用でき、IIR EQ、Target、処理済みSpeaker応答、Working SessionをMultiwayへ戻せます。
- グラフ、タップ長一覧、Composite、Assignment、DSP Exportは同じFIR ON／OFF状態を参照します。
- User-defined miniDSPとSigmaStudioから、機種に依存する4096 taps固定制限を除外しました。実機へ投入する前に、所有するDSPの仕様を確認してください。

## 品質チェックとリリース

- テストを文書、ソース、開発／配布基盤へ分類し、通常コミットでは変更範囲に応じたSmartチェックを実行します。
- 正式リリースでは全カテゴリーを確認します。
- 全品質チェック後にGit treeが変わっていない場合だけ、release commit時の同一品質hookを省略し、全テストの二重実行を防ぎます。
