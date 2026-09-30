# PhaseEQ v1.19.6 更新案内

PhaseEQ v1.19.6では、利用者ガイドを現行アプリへ再照合し、初めて使う機能を探しやすく、短い単位で読める構成へ改訂しました。また、GitHub Releaseの自動公開が開始前に停止する場合がある問題を修正しました。

## 利用者ガイドの再構成

- 目的別と画面別の索引を追加し、必要な操作へ直接移動できるようにしました。
- 各章を目的、場所、完了条件、短い操作手順へ統一しました。
- 詳細説明を番号付き補足へまとめ、本文の密度を均一にしました。
- 現行アプリに存在しない旧DSP System操作など、古い説明を除去しました。

## Composite EngineとFIRの説明

- Composite EngineのSection 1～8と、最初のSystemを作る順序を追加しました。
- FIRとIIRの違い、tap数、FIR出力ON／OFF、Crossover methodの選び方を説明しました。
- All-passと通常Delayの違い、位相・時間整合での選択方法を追加しました。
- Crossover / Design Response、Channel Response、System Sum、Group Targetの役割を分けて説明しました。
- Phase、Group Delay、Waveletの読み方と、Composite AssignmentをPhaseEQへ送って戻す流れを追加しました。

## Releaseワークフローの修正

- GitHub Actionsの一時ファイルパスをRunner開始後に評価するよう修正しました。
- Release Jobが実行開始前に停止し、配布ZIPとSHA-256を公開できない場合がある問題を解消しました。

音響処理、DSP係数、保存形式の仕様変更はありません。
