# PhaseEQ v1.19.8 更新案内

PhaseEQ v1.19.8では、測定からMultiway設計、Composite Assignment、DSP出力までの時間基準と受け渡しを強化しました。単一選択ボタンの動作と配色もPhaseEQ／Multiwayで統一しています。

## 測定と時間基準

- 連続測定で基準トラック、入出力レベル、ESS出力Channelを同じ流れで確認できるようにしました。
- ESS出力Channelは常に1つを選ぶ操作とし、選択中のボタンを再度押しても全OFFになりません。
- Speakerの位相中心とTweeter相対タイミングを、後段のMultiway位相・時間整合へ引き渡せるようにしました。
- 時間基準の由来を保存し、表示、再計算、Assignmentで二重補正しない共通経路へ整理しました。

## Multiway Studio

- System全Channelの距離／時間基準を揃え、最も遠いDriverを基準に非負Delayを適用する位相・時間整合を追加しました。
- PhaseEQ Tweeter基準、外部測定距離、Crossover位相推定の優先順位と、全Channelが揃わない場合の安全な停止条件を追加しました。
- バッフル補正を現在のChannel構成に応じてFIRまたはIIRで自動実現し、グラフ、Channel一覧、保存、DSP Exportへ同じ状態を反映します。
- Crossover、Speaker、PhaseEQ EQ、Gain、Polarity、Delay、All-passを含む実現済み応答から、Phase、Group Delay、Impulse、Stepを表示します。

## PhaseEQとMultiwayの自動受け渡し

- 送信側が新しいAssignment revisionを公開すると、起動中の受信側が2秒間隔で状態を確認して自動受領します。
- PhaseEQは未受領のChannel Assignmentを1件ずつ開き、Multiwayは現在のSystemに属する更新だけを接続します。
- 受信側が停止中でもAssignmentを保持し、次回起動時に同じ受信処理を行います。

## UIの選択動作と配色

- どれか1つを必ず選ぶ項目を連結型コントロールへ統一し、再クリックによる選択解除を防止しました。
- 機能が相関しない操作は独立ボタンのままとし、意味上のまとまりを維持しています。
- Gain関連を青、Phase関連を紫、警告／主要実行をオレンジとする意味別配色をPhaseEQとMultiwayで共有します。
- 日本語表示で一部の位相中心ラベルが未登録となり、画面起動時に`ValueError`となる問題を修正しました。

## 文書

- Composite Assignment、入力応答／IR基準、Multiway UI、Phase／時間整合の実装仕様を更新しました。
- REW Remote測定連携は将来実装の詳細仕様として追加し、現行機能と区別して記載しています。
