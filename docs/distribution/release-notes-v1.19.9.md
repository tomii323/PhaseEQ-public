# PhaseEQ v1.19.9 更新案内

PhaseEQ v1.19.9では、PhaseEQを単独利用する場合とComposite Engineから複数Channelを編集する場合を整理し、接続状態と保存先を見失いにくくしました。成果物が変わった場合だけ自動受信するため、監視中の通知や再描画の繰り返しも抑えています。

## PhaseEQとComposite Engineの作業状態

- PhaseEQの状態を`Standalone`、`Composite Assignment`、`Target Edit`の3つに整理しました。
- StandaloneではAssignmentやComposite Engineへの接続操作を表示せず、単独作業用のFIRタップ数を指定できます。
- Composite Assignmentでは、画面ID、Assignment ID、編集中Channel、保存先を必要な場所へ簡潔に表示します。
- 複数のPhaseEQ画面を開いても、それぞれの画面とAssignmentを識別できます。同じChannelを複数画面で編集すること自体は防止せず、状態を確認して運用できる仕様です。
- Target Editでは対象Targetと戻り先を示し、関係しないメニューを隠して誤操作を減らします。

## Channel編集と自動受信

- Composite Assignment内で別Channelへ切り替えても、Channelごとの編集中内容をdraftとして保持します。
- EQの追加・削除などで編集中Channelが初期化されないよう、Channel選択とEQ一覧更新の状態を分離しました。
- 接続確認用heartbeatと、編集結果を表すoutput revisionを別々に管理します。新しい実出力が届いた場合だけMultiwayが自動受信し、同じ通知を繰り返しません。
- Assignment情報が不足、不一致、別System由来の場合は状態マトリクスで判定し、誤ったChannelへ自動適用しません。

## DSP処理とグラフ

- Multiwayの実現済みChannel応答へPhaseEQ FIRの位相を含む処理状態を渡し、PhaseEQとHighなどのChannelグラフの差を修正しました。
- Stereo／Mono、Fullrange／Multiway、FIR／IIRの各経路で同じ実現済み状態を参照します。
- Gain、Phase、警告／実行の意味別UI色をPhaseEQとMultiwayで揃え、グラフ系列も同じ意味体系で見分けやすくしました。
- 必ず1つ選ぶ項目は連結型ボタンへ統一し、選択中のボタンを再度押しても全OFFになりません。

## Exportと他PCでの再現

- Export edit modeをWorking Session ZIPとして出力し、別PCのPhaseEQで読み込んで編集状態を再現できます。
- ZIP内のmanifestとSHA-256を検証し、ファイル欠落や内容不一致を検出します。
- Biquadを直接入力できないIIR専用DSP向けに、Type、Frequency、Gain、Q、Order、Enabled、適用順を記載したGeneric IIR Parametersを追加しました。

## 文書とバージョン表示

- システムフロー図、将来構想、簡易マニュアル、利用者ガイド、出力仕様、アプリケーション仕様を現行実装へ同期しました。
- 将来構想は現行仕様と候補案を分離し、今後その方向へ進むことが未決定である項目を明示しました。
- Composite Engineでは`PhaseEQ Composite Engine v1.19.9`と`Multiwayベース版 v1.1.3`を併記し、統合版の版と由来となったベース版を区別します。

## 更新後の確認

1. PhaseEQをStandaloneで開き、Composite接続操作が表示されずFIRタップ数を設定できることを確認します。
2. Composite Engineから2つ以上のChannelをPhaseEQで開き、画面ID、Assignment ID、Channel名を確認します。
3. 各PhaseEQで編集結果を保存し、Multiwayが新しい結果を1回だけ受信することを確認します。
4. 必要に応じてWorking Session ZIPを別環境へ移し、PhaseEQで読み込んで設定を確認します。

既存のPhaseEQ Project、DSP Package、FIR WAV／CSV、Biquad CSVは引き続き利用できます。複数画面で同じChannelを同時編集する場合は、最後に保存した結果がMainへ戻るため、表示される画面IDとAssignment IDを確認してください。
