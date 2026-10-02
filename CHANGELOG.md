# 更新履歴

PhaseEQの主な変更点を、バージョンごとにまとめます。

## 未リリース

## v1.20.14 - 2026-10-02

- Stand-aloneを1チャンネルの単独編集へ変更し、左右選択・追加チャンネル・Stereo Link操作を非表示化。旧版の左右リンク付き設定は保存時の選択チャンネルの実効EQと入力条件を引き継ぐ。PhaseEQ側のMultiwayチャンネルのリンクON/OFFと設定を編集対象選択と同じ行の右側へ配置。

## v1.20.13 - 2026-10-02

- 公開ZIPにファイル一覧へ記載した公開用Release workflowを含め、本体更新時の「Release ZIP does not match its public inventory」を修正。v1.20.11／v1.20.12の更新処理でも検証できる配布構成へ変更。
- 公開前のgit archive、公開Actions、匿名取得した公開ZIPを実際の本体更新検証で確認し、一覧とZIPの不一致を公開前に検出。

## v1.20.12 - 2026-10-02

- Stereo Linkで選択した複数チャンネルのIIR EQ、FIR EQ・Auto EQ、Target設定を共有。採用元と共有対象を選び、確認後に適用。解除時は設定を各チャンネルへコピーして独立編集へ移行。
- リンク状態とチャンネル設定を保存・復元。Multiwayでは古い共有設定の返却FIR／IIRを検出して合成を停止し、再生成・送信を案内。
- Stereo Linkの操作説明を操作ガイドと画像付きPDFへ追加し、文書一覧と現行版へのリンクを同期。

## v1.20.11 - 2026-10-02

- 本体更新の成功時に今回と前回の成功更新のバックアップを即時削除。削除失敗は通知して次の更新成功時に再試行し、更新失敗・中断時は復元用バックアップを保持。

- チャンネル間および単独編集との切り替えで表示中のメニューを維持。切り替え先で利用できない場合は左側の利用可能なメニューへ移動（FIR OFFではFIR EQからIIR EQへ）。

- 公開GitHub Releaseの起動時確認と、設定画面からの本体更新予約を追加。ZIP・ファイル別SHA-256を検証し、両アプリ終了後のランチャー起動時に適用。個人データを保持し、旧ソースの退避と失敗時の復元に対応。Git管理中の開発環境は更新対象外。

## v1.20.10 - 2026-10-01

- 個人利用向け無償フリーウェアとして正式な利用条件をLICENSE.mdに集約。本体の商用・業務利用禁止と、個人利用で生成した成果物の事後の商用利用を区別し、GitHub上の閲覧・Fork・Pull Requestを許可。
- 生成物の出力ZIPへ「生成物の利用条件.txt」を同梱。DSP処理・係数・波形・設定データの形式は維持。本体の配布ZIPにはLICENSE.md全文を同梱し、公開対象・配布検査へ追加。
- README・操作ガイド・画像付きマニュアルの利用条件を確定内容へ更新。

## v1.20.9 - 2026-10-01

- 一般向け文書と画像付き操作マニュアルに概要・仕様、設計パラメーターの決め方、書き出し後の共通確認、利用条件の確認先を追加。音楽再生機能がないことを明記し、実機未検証の出力機器の説明を限定。基礎説明を集約し、既存の12章構成を維持。

## v1.20.8 - 2026-09-30

- 一般公開用リポジトリと非公開の開発リポジトリを分離。公開対象を明示一覧で検査し、開発文書・テスト・開発スクリプト・廃止資産と開発履歴を公開側へ送信しない構成へ変更。
- 起動用の環境検査・Windows連携スクリプトをruntimeへ移動し、ランチャーの参照先を更新。

## v1.20.7 - 2026-09-30

- 現行日本語UIを再撮影した画像付き操作マニュアルを配布ZIPへ同梱し、利用者向け文書から案内。
- 配布ZIPからテスト・開発補助スクリプト・開発文書を除外。開発手順はソース側の開発文書へ集約し、起動に必要な補助スクリプトと依存関係制約は保持。

## v1.20.6 - 2026-09-27

- PhaseEQ下段の編集・結果表示領域をブラウザー下端まで拡大。
- 帯域分割の音響ターゲット設定をIIR／FIRページ間で保持。「音響ターゲットに反映」へ名称を変更し、手動IIRによる調整も案内。
- 音響ターゲットと自動補正を分離。単独編集・マルチウェイとも、EQなしでは補正FIRをフラットとし、手動EQまたは明示的なAuto EQで調整。
- 目標応答の登録・編集グラフに選択Targetと保存対象の編集応答を追加。Target保存・出力には音響ターゲット用帯域分割を含めない。
- WAV Targetの元ナイキストより上は、高域延長ON時にGain・Phaseを推定。OFF時は元データを超える表示を省略。
- 応答処理・Target・表示解析・DSP System・Compositeの範囲外補完を共通Lo／Hi方式へ統一。対応形式の旧設定は読み込み時に新方式へ正規化し、元応答・EQ・明示的なON／OFFを保持。
- 旧延長計算関数と専用補助関数、旧方式比較スクリプト・テストを通常の実行対象から削除。復元用原本は配布対象外に退避。

## v1.20.5 - 2026-09-26

- マルチウェイのチャンネル別DCゲイン正規化を追加。返却補正FIRを設計ターゲットのDC値へ合わせてから帯域分割と合成し、設定をSystem・履歴・再開データへ保存。初期値OFF、旧形式で基準値がない場合は未適用を案内。
- DSP FIR出力OFF時の帯域分割方式をPhaseEQと共通化。FIR LR2をIIR LR2、FIR LR4／カイザーをIIR LR4へ切り替え、OFF中はFIR方式を選択不可に変更。登録済みEQファイルは保持し、OFF中の読込・合成・編集を無効化。
- 最終FIRタップ数とDSP追加Delayを保存・出力と同じ結果から表示。タップ数統一は割り当て長で合成・後処理した後のゼロ追加へ統一し、FIRなしと偶奇混在のDelay補償を修正。明細、EQファイル合成、最終FIRグラフを折りたたみ表示へ整理。
- クロス近傍の補正前後比較から、FIR有効帯域に実適用した共通出力ゲインだけを除去。IIRのみの帯域と手動Gain・極性・Delayは維持。
- 共通解析FFTをSpeaker Package読込、チャンネル・Workspace切替、サンプルレート変更で保持。明示的な設定復元では保存ファイルを優先し、選択欄の旧候補を破棄。

## v1.20.4 - 2026-09-26

- マルチウェイの帯域タップ数変更時に、入力欄より上の位相・時間整合とバッフル実現方式の表示領域が一時的に消え、スクロール位置が変動する問題を修正。表示領域を保持し、計算後に同じ場所へ結果を表示。
- 「出力タップ数を最長フィルターに揃える」を「設定」から「3 FIR出力・タップ長」の帯域別入力欄の下へ移動。設定キー、保存値、再計算処理、FIR OFF時の無効化は維持。

## v1.20.3 - 2026-09-25

- 英語名称を説明する箇所、英和対応表、旧称・内部名の引用を英語表記へ修正。通常の操作説明は日本語を維持し、文書作成規則にも明記。アプリの動作・保存形式・生成係数は変更なし。

## v1.20.2 - 2026-09-25

- リリース準備を事前検証と書込に分離。現行文書の版表記を明示対象のみ同期し、再実行時の履歴重複を防止。GitHub公開本文の文書リンクを同じタグの絶対URLへ変換。

- リリース履歴の記載先を更新履歴・版別更新案内へ集約。仕様書末尾の版別補足を機能別の節へ移し、同じ追記形式を防ぐ文書検査を追加。

- v1.19.11〜v1.20.1の文書を再点検。目標特性の選択・未保存編集ロック、グループターゲットの編集経路、自動レベル合わせ、表示解析FFT倍率、標準目標特性と一覧件数の説明を訂正・補記。旧版の更新案内を当時の履歴と明示し、用語辞書に沿った日本語文書整理と版別照合表を追加。プログラムの挙動は変更なし。

## v1.20.1 - 2026-09-25

- 操作ガイド・出力仕様・更新案内を現行実装へ更新。Standaloneのターゲット補正、FIR OFF、IIR合成、位相整合と極性、設定配置、自動クロップの対象・探索下限を明記。

- 書き出しにFIR／IIR／両者の合成応答を追加。IIR EQ・直接帯域分割・All Pass・極性を含め、測定入力との合成および時間解析にも各段を一度だけ適用。FIR OFF時はFIRをバイパス表示。

- 書き出しグラフのページ差分をFIR同士で比較するよう修正。別出力のLR2／LR4 IIR帯域分割が上下反転した増幅として表示される問題を解消。

- 初期化後に方式の選択表示だけが旧値に残る不一致を修正。共通の選択UIでブラウザーへ状態変更を通知し、表示と計算値を同期。

- PhaseEQの選択中Hi／BP／Lowとマルチウェイの選択中分割構成に、帯域分割設定を初期値へ戻すボタンを追加。他の構成とEQを保持し、共通の復元・再計算経路を利用。

- 出力帯域分割の標準プリセットに境界別Overlap 0 octを明記し、プリセット入力欄の再表示時に最小値へ戻らず保存値・初期値を復元するよう修正。

- PhaseEQとマルチウェイのカイザー窓β初期値・未設定時補完値を12.0へ統一。全Way／SUB境界に適用し、保存済みの明示値は維持。

- 出力帯域分割の登録済み標準カイザー窓プリセット11件（HP／LP計13フィルター）のβを8.6から12.0へ変更。

- スタンドアローンのHi／BP／LowでLR2／LR4の音響ターゲットON／OFFを操作可能に変更。IIR LR2／LR4はFIR OFFでもターゲットへ反映し、直接帯域分割出力から除外してIIR EQで調整可能。カイザー窓／ThroughのOFF固定とマルチウェイの引き継ぎ表示を維持。

- 出力帯域分割の表示名変更で発生した未定義の翻訳関数参照を修正。日本語／英語、操作可能／引き継ぎによる無効表示で、実際の選択ボタンの描画を回帰検証。

- 書き出しのFIRタップ数の隣に独立したFIR ON/OFFを追加。スタンドアローンのOFF時はFIR EQ設定を保持して無効化し、FIR LR2／LR4をIIR LR2／LR4、カイザー窓をIIR LR4へ変換。マルチウェイ接続中は引き継ぎ状態を無効表示し、帯域分割を変更しない。出力帯域分割の案内もFIR／IIR対応へ更新。

- FIR OFFチャンネルではFIRイコライザーを無効表示にし、選択中ならIIRイコライザーへ移動。内部のFIR EQ設定を保持し、IIR対応の出力帯域分割は操作可能なまま維持。

- マルチウェイの「言語設定」を「設定」へ変更し、サンプリング周波数・FIR出力条件・グラフ表示設定を集約。設計操作と手動更新を元の位置に維持し、既存の保存値と処理順を保持。

- PhaseEQ／マルチウェイ共通のヘルプ表示待ち時間を追加（初期2.5秒、0〜5秒）。設定はプロジェクトから独立して保存し、標準ヘルプと翻訳ヘルプへ適用。表示待ち中のヘルプはボタン操作を遮らない。

- IIR Errorの入力未設定時も0°を基準にPhase誤差を表示。測定に位相がない場合とは区別し、一定の位相差も除去せずTargetとの差を表示。Gain誤差は維持。

- 測定入力が未設定の音響ターゲットをFIRの直接フィルターとして生成・表示してしまう問題を修正。ターゲット表示は維持し、測定に対する補正経路と独立したTarget直接設計を分離。

- LR2自動極性補償を帯域分割ターゲットのON／OFFに連動する経路へ移動。位相・時間整合ターゲットはAll Passのみを制御し、手動Polarityは独立して直接適用。旧Recipeの応答を保持し、再送信時に新ルールへ移行。

- PhaseEQの出力帯域分割に直接IIRの段数・係数を表示。既存のLR2／LR4・All Pass出力へLR2極性を含め、手動Polarityと合わせた完全なBiquadをFIR OFFでも書き出せるよう統一。外部DSPでのLR2極性の別適用は不要。

- 位相・時間整合の音響ターゲット補正ON時は、返送待ち・設定変更直後もマルチウェイの直接All Passを生成せず、グラフ・合成・DSP出力の適用条件を統一。DelayとPhaseEQ向けRecipeは維持。

- 位相構造補償の自動適用を追加（初期OFF）。ONではWay・分割方式・周波数の変更に理論All Pass配置と極性が追従し、適用ボタンは時間整合／Delayだけを更新。音響ターゲット設定との共通経路、OFF時の従来一括適用、取り消しを維持。

- 「3. チャンネル設定」のチャンネルごとのPhaseEQ操作案内・リンクを削除し、測定割り当ての入口を「5」に集約。手動ファイル入力を「追加IIR応答」と明記し、自動返送との重複を避ける案内に変更。

- マルチウェイの独立した「PhaseEQ連携」欄を廃止し、FIR・IIR・測定の反映状況を「PhaseEQでチャンネルを編集」の一覧へ集約。現在の出力に一致しない返送データを追加入力として自動採用せず、除外名を案内。保存済みデータは保持。

- PhaseEQの編集対象切替で、一覧更新前に新しいAssignmentの選択を受け付けるよう修正。移動先の保存データを旧編集状態より優先し、新規Channelでは旧EQ・入力・Targetを引き継がない。同一System・Channelの再発行では未保存EQを維持し、保存状態の探索をSystem内に限定。

- 位相・時間整合に「音響ターゲットとして補正」を追加（初期OFF）。ONではチャンネル別の1次／2次All Passを既存の音響ターゲットへ反映。Delayはマルチウェイ側で維持。
- PhaseEQの出力帯域分割に、マルチウェイから受け取った位相整合の表示とBiquad出力を追加。OFFでは従来の応答を保持し、ONでは直接All Passを除外して二重適用を防止。設定変更後の古い補正FIRによる出力を停止。

## v1.19.21 - 2026-09-23

- 表示解析のFFT倍率を「設定 → 環境設定 → 予測応答の表示設定」へ移動。倍率と保存値を維持。
- LR2／LR4のFIR・IIR選択時に音響ターゲット補正を初期ONへ変更。保存済みON／OFFは保持し、カイザー窓・スルーは常時OFF。
- 測定データの送信先を現在の有効チャンネルに限定し、過去のExchange返却入力の混入を修正。残るチャンネルの測定割り当てを保持し、旧データは削除しない。

## v1.19.20 - 2026-09-22

- 翻訳漏れを追加点検し、Target操作、測定の割り当て案内、履歴、ZIP生成再試行、Multiway受信案内、テスト用メモリ診断を日英対応。固定文言の翻訳経路と表示名の回帰テストを追加。

- 応答処理の操作メニューに残っていた英語表示と、英語モードの編集対象・接続状態・詳細欄に残っていた日本語を修正。保存済みの測定名・プロジェクト名は変更しない。

## v1.19.19 - 2026-09-22

- Auto Gain等の数値入力で、保存・連動した値と初期値が二重指定される警告を修正。Waveletの日本語軸名・測定名に共通の日本語フォント検出を適用し、文字欠落の警告を解消。

- スピーカー測定データの上部メニューに「マルチウェイに割り当て」を追加。測定一覧の選択を引き継ぎ、専用ページでも測定データと送信先チャンネルを選択可能に。既存の登録・受け渡し処理は維持。

- PhaseEQの左右ペインの最小高さ360px制限を撤廃し、低いウィンドウでも画面の高さに追従するよう変更。左右独立のスクロールは維持。

- Multiway の PhaseEQ 送信対象を現在の有効な帯域チャンネルに限定。過去の返却データや追加音声入力に現在の帯域分割Recipeを誤適用し、IIR次数不一致で全チャンネルの送信が止まる問題を修正。返却データ・追加音声入力の合成処理と通信形式は維持。

## v1.19.18 - 2026-09-22

- FIR Auto Gain EQ の各セクションに Lin／Min. Phase の選択を追加。既存の振幅補正・境界処理を維持し、Min の位相を後段の Auto Phase に反映。旧設定は Lin として読み込み、保存・レジューム・既存の FIR Recipe 引き継ぎに対応。
- 新規 FIR Auto Gain EQ の Smoothing 初期値を 1/12 oct に変更。保存済みの値と Auto Phase の初期値は維持。
- Auto EQ 入力整形の「深いディップを浅くして計算」「帯域別oct平滑化」の保存値と再表示の不整合を修正。Config に入力整形がない旧設定は UI 側の保存値を復元し、明示された Config の ON／OFF を優先。
- 予測応答の自動更新とターゲット切り替えロックのチェック表示を、再表示時にも保存値と一致させる。v1.19.15 以降の追加・変更 UI の確認記録と回帰テストを追加。

## v1.19.17 - 2026-09-21

- Auto EQ入力整形を全FIR Auto Gainへ共通適用。IIR適用後の入力だけを整形し、元測定・Auto Phase・帯域分割・後段の制限処理は保持。共通設定をConfigにも保存し、FIR EQから調整・計算用入力を確認可能に。
- DSP System・MultiwayのFIR再生成にも共通の入力整形設定を引き継ぐ。新Recipeをv4とし、旧Recipeの生成結果・係数ハッシュ検証を維持。合成応答は元測定と実フィルターから計算する。

- Power V4のUI・専用エンジン・局所評価・専用テスト・比較スクリプトを削除し、Auto EQをターゲット追従式に統一。旧設定の生成方式を正規化し、生成済みIIRは保持。

- Auto EQ計算用入力を常時表示し、入力整形の未設定値を両ONに統一。oct幅を1/Nの分母入力へ変更し、従来の保存幅は維持。

- Gain / Phase chartsを初期表示で開き、塗りつぶし・背景帯・P0補完範囲・系列の説明は「グラフの見方」ボタン内へ集約。

- Auto EQ入力整形の各設定に日英の調整目安を追加。新規・リセット時の圧縮初期値をしきい値3 dB・9:1・ニー3 dB（深さ12 dB→約4 dB）へ変更。保存済み値は保持し、チェック切替でも設定欄の開閉状態を維持。

- ターゲット追従式Auto EQに、計算用データの深いディップ整形・帯域別oct平滑化を追加。ディップはソフトニー付き圧縮型で整形し、後段のAuto EQは従来のまま。元の入力で予測応答を表示。両OFFは従来経路・従来結果を維持する。

- メモリ使用量の調査をテスト専用化。通常UIから非表示にし、明示的な環境変数指定時だけ診断画面を表示。UIなしの計測APIと自動テストは維持。

- Streamlit 1.62系への更新時に固定依存が1.59.2のまま残り起動できない問題を修正。固定版を1.62.0へ統一し、requirements・pyprojectと固定版の適合性テストを追加。

- Streamlitを1.62系列へ更新。数値編集の0.1秒監視を編集待ちの間だけに限定し、ページ遷移時の古い待ち合わせを解除。ZIP準備失敗・終了時の監視継続とTarget ACKの反復書き込みを修正。
- PhaseEQ／Multiwayの共有更新をOSロックと固有一時ファイルで保護。新送信は公開済みSpeaker／FIR／Working Sessionを上書きせず、世代別資産とstatus確定点を使用。再送の重複防止、資産読取の上限付き再試行、旧形式の読取補完を整備。更新後は両アプリを再起動する。

- 入力処理の未変更列を結果内で共有し、段階別キャッシュ復元後にも共有を維持。呼び出し間のデータ分離と数値結果を保ち、実測入力のキャッシュ取得結果の保持量を約31%削減。

- メモリ診断にRSS履歴、セッション・アプリ・計算キャッシュの保持量、上位項目の内訳、配列の形状・型、共有キャッシュ・アップロード量、JSON出力を追加。手動計測と探索上限で診断負荷を制限。
- ターゲット・測定・スピーカーパッケージ・Target履歴の一覧を概要読み込みへ変更し、選択時に主データを読み込み。保存・複製・エクスポートでは従来の完全データを使用。スピーカーパッケージ一覧は保存内容IDを表示。

- 自動レベル合わせで平坦部が見つからない場合、高く安定した実測帯域を選ぶ代替処理を追加。入力Auto・固定Targetへの合わせ込み・Target Autoに適用し、低域ノイズや急な減衰帯域を除外。

## v1.19.16 - 2026-09-20

- 設定にターゲット選択時の未保存編集ロックを追加。既定はOFFで、OFFでは未保存編集を破棄して切り替え。

- 予測応答の自動更新・モード・表示点数を設定画面へ移動。予測応答の見出し行の右端に手動更新ボタンを残し、グラフ上部を省スペース化。

- Power v4の谷Boost上限をQ 2.0へ緩和し、検出した谷の近傍で1/6 octaveの形状に再フィット。新規ディップ・帯域誤差・中心レベル・ピークPowerの関連許容量を調整。鋭い谷の除外と合成ヘッドルーム制限は維持し、実行時の判定閾値を診断へ保存。

## v1.19.15 - 2026-09-20

- 上部のナビゲーション・編集対象選択・状態表示を圧縮。単独／接続状態とプロジェクト・入力情報を横並びにし、画面IDとメモは「詳細」に集約。下部の編集欄・グラフは画面の高さに合わせて拡大。

## v1.19.14 - 2026-09-20

- 系列切り替えで表示が変わるQ・リップル・阻止域減衰の入力を通常再描画へ統一。部分再描画の遅延更新が消えた入力欄へ届き、ブラウザーが空画面になる競合を防止。

- IIR次数の表示形式を各フィルターの系列に固定し、系列切り替え後に「4 (24 dB/oct)」が数値設定へ混入して落ちる問題を修正。既知の表示文字列が保存済みの場合も次数を復元。

- IIR EQとTarget IIRに可変Qの2次HP／LPとEllipticの1〜8次HP／LPを実装。Ellipticの通過域リップル・阻止域減衰を保存・復元・共通Target応答へ反映。廃止済みFIR Designer解析・Q/BW変換の参照テストを整理。

- 同梱TargetからAudiofrog Target Curveを除外し、Harman H4 Target Curve・Flat Referenceの2件へ整理。

- 「測定条件・メモ」に検索付き測定一覧を追加。チェックした1件の条件・メモを編集し、保存後は一覧を更新。「測定一覧・使用へ戻る」で選択を引き継ぐ。

- Target一覧の読み込み・正規化結果を最新1組だけキャッシュし、画面更新ごとの大容量WAV特性データの再解析・重複コピーを削減。ファイル更新・作成・削除を検知して再読み込みし、呼び出し側の編集はキャッシュから分離。

- Target一覧からの切り替え時、未保存変更の確認を一覧直下へ移動。切り替え待ちのTarget名と現在の編集対象を併記し、選択が反映されないように見える状態を改善。

- Targetの手動シフト欄で暗黙の初期値が下限の−120 dBになる経路を除去。Session Stateの値を使用し、空欄を確定した場合は0 dBとして扱う。

- 新規Targetの初期名だけが復元され、自動提案の状態が失われた場合も、読み込み済みファイル名を保存名へ補完。手入力名と登録済みTarget名は保持。

- 未保存編集の切り替え確認が残っていると「書き出し・読み戻し」などから登録・編集へ戻される問題を修正。編集要求時の移動を一度に限定し、その後のメニュー選択を保持。

- 再読み込みで同一サーバーの稼働記録が重複した場合、他のPhaseEQの数を過大表示する問題を修正。PID・ポートが同じ記録は1件として表示。

- Targetから画面を切り替えた際、破棄済みの位相設定ウィジェット通知でKeyErrorが発生する問題を修正。保存済み設定を保持して古い通知を無視。

- Targetの履歴ボタンを「登録・更新」欄へ集約。書き出し・読み戻しと詳細管理では非表示にし、自動履歴取得は維持。

- Targetの「書き出し・読み戻し」から重複する「05 登録・更新」を除き、「登録・編集」に表示を集約。

- Targetの登録・編集と詳細管理の一覧・選択状態を共通化。標準／ユーザーの種類、編集可否、入力元を表示。詳細管理の操作を選択対象の下に集約し、別の編集中Targetの登録・更新欄を非表示に整理。

- 新規Targetの保存名を空欄にした場合も、読み込み済みファイル名から補完。空欄のままアップロードする場合にも対応。

- 新規Targetへのファイル読み込み直後に、拡張子を除いたファイル名を保存名へ仮入力。重複回避と手入力名の保持を維持。

- Targetの「選択を解除して新規入力へ」を1回で反映するよう修正。一覧の描画前に解除要求を処理し、未保存編集の確認と未選択時の無効化を維持。

- 設計設定の履歴を追加。Standalone・Target・Multiwayに控えめな履歴入口を設け、同期完了後の自動取得（既定30分・20件）、重複抑止、手動保存・保護・復元に対応。測定入力は最新のみ、Target元データは共有し、生成済み出力係数は保持しない。旧形式データは保全。

- IIR Auto EQにターゲット追従の自動PEQモードを追加。振幅誤差を評価してFc・Gain・Qを調整し、補正上限と手動IIRを保持。従来のPower v4と切り替え可能。

- Target一覧の選択・解除時に古い入力イベントが復元値を上書きする問題を修正。未編集の新規状態を未保存変更と誤判定しないよう改善。
- Targetの位相ON/OFFを入力ファイルの位相だけに限定し、Phase EQは有効のまま保持。操作をファイル読み込み欄へ移動。

- 目標応答の一覧選択と入力元・Gain EQ・Phase EQ・レベル調整を1画面に統合。標準／ユーザーTargetの表示切り替えと、未選択時の新規登録、選択時の更新を明示し、標準Targetの上書き制限と未保存変更の確認を維持。

- WAV由来のTargetは低域・高域延長をOFF・操作不可にし、プリセット復元時も無効化。

- Target入力にモノラルWAV（IR）を追加。元のサンプルレートで振幅・位相に変換し、IRピークを時間原点として取込遅延を除去。

## v1.19.13 - 2026-09-17

- 同梱TargetをAudiofrog Target Curve・Harman H4 Target Curve・Flat Referenceの3件へ整理。Gentle House Curveを同梱対象から除外。

- 再配布可のスピーカー仕様5機種を配布用JSONとして同梱。仕様DBを開いた際に未登録分のみ追加し、既存編集・アーカイブ・削除を保持。

- 仕様DBで一覧未選択のまま既存データの編集モードが残る問題を修正。更新対象をチェックした行に同期し、選択解除・検索対象外では入力を新規下書きとして保持。

- スピーカー仕様DBの一覧選択と編集を1画面に統合。行選択で自動読込し、新規登録と既存IDの更新を明示的に分離。チャンネルセットへの移動ボタンを「測定データのチャンネル選択へ」に変更。

- スピーカー仕様データベースを全幅表示に変更。右側の応答グラフを非表示にし、仕様値を基本性能・T/Sパラメーター・外形／取付寸法に整理。

- マルチウェイChannel編集中の入力変更をチャンネルセットに限定。PhaseEQの直接適用・解除を抑止し、設定復元時も入力を保持。各Wayの直接Speaker読込欄を整理し、現在入力名・統一した変更先案内を表示。Standaloneの直接適用は維持。

- 測定データをマルチウェイへセットしても、同じChannelを編集中のPhaseEQ入力が更新されなかった問題を修正。チャンネル初回表示と更新検知時に割り当てを読み込み、別Channel・Standaloneへの適用を防止。受け渡し保存と画面への読込を区別する案内へ変更。

- スピーカー測定データを「測定一覧・使用」「測定データを登録」「測定条件・メモ」に分離。マイク校正・個体プロファイルと任意のスピーカー仕様DBを独立化。測定条件の更新では応答・校正情報・プロジェクト履歴を保持。

- マルチウェイへの測定データ受け渡し時に、プロジェクト名をDBのメモへ重複なく追記。一覧に利用プロジェクト列を追加し、プロジェクト名での検索に対応。元の測定条件・メモ・測定応答を保持。

- スピーカー測定データから既存のマルチウェイChannelを手動指定して測定応答をセットする機能を追加。FRD／TXT／CSVの直接読込ではDB登録と受け渡しを実行し、登録済みデータも再利用可能。Multiwayの自動再読込・測定名表示・受信状態確認に対応。自動振り分けや保護処理は追加しない。

- 「アプリケーション」のメニュー名を「設定」（英語：Settings）へ変更。

- 「PhaseEQ設計」のメニュー名を「音の補正・フィルター設計」（英語：Sound Correction & Filter Design）へ変更し、操作目的を明確化。

- 「ドライバープロファイル」の画面名を「スピーカー測定データ」（英語：Speaker Measurements）に変更。測定データの保管・選択場所を明確化し、保存した処理プロファイルの名称と区別。既存の設定・ページ識別子は維持。

- Targetの詳細管理で管理対象を選択すると、保存済みTargetのグラフへ切り替わるよう修正。編集内容と保存先は変更せず、詳細管理へ戻った場合も選択中の対象をプレビュー。

- 音源ボタンを「マイク測定用音源のダウンロード」に変更。「初回必須：音源の準備」を独立させ、CD作成／WAVプレーヤーの手順と毎回の測定前確認を分離。旧音源案内と内部ZIP管理の説明を画面から削除。

- ガイド測定の音源ダウンロードを全15曲一括に統一。ZIP展開、曲順表、オーディオCD／WAVプレーヤーの準備、接続と測定開始前の操作を画面と取扱説明書へ追加。

- 音源ZIPの初回ダウンロードで「File not found」になる競合を修正。遅延生成の完了前から転送データの参照を保持し、ブラウザーの取得が遅れても定期更新で消えないように変更。測定開始前の解放は維持。

- ガイド測定の音源ZIPを1回のダウンロード操作で準備・転送。定期再描画からダウンロードを分離し、準備ZIPは接続・配置へ進む時に解放し、暗騒音取得・測定開始前に不要な転送用ZIPも解放。測定中の削除は行わず、他接続が使用中のデータは保護。メモリーのみで保持するためアプリ再起動時に一時ZIPを残さず、元のWAVと保存済みZIPは保持。

- PhaseEQとマルチウェイに共通の日本語／英語表示設定を追加。既存の和英辞書を拡張し、各機能のメニュー、説明、ヘルプ、通知、グラフ表示へ適用。言語設定は設計条件から独立して保存し、選択値・数値・ユーザー名を保持。定期更新画面とグラフキャッシュにも表示言語を反映。

- 新規Targetの保存名を入力ファイル名・コピー元から仮入力し、登録済み名との重複は連番で回避。手入力した名前と既存Targetの編集名は保持。
- Targetを「選ぶ／編集する／書き出し・読み戻し／詳細管理」へ整理。編集中の名前・未保存状態・上書き／新規保存を共通表示し、標準Targetと自分のTargetを区別。新規作成、保存時点への復帰、未保存切替時の3択を追加し、独立した確定処理のなかったApply Targetを廃止。
- デスクトップの編集欄・Predicted Responseを幅の制約付き2列で配置し、グラフが編集欄の下へ回り込む表示を防止。左右独立スクロールと小画面の縦配置を維持。
- Target presetsの基本操作を選択・読み込んで編集・保存へ整理。編集中の名前、保存への導線、保存完了を明示し、未読込の別プリセットへの上書きを防止。共通Targetの履歴と配布用管理は折りたたみへ移動。
- FIR Recipeをv3へ更新し、音響補正の帯域端減衰保持を版管理。旧v1/v2返却は新旧どちらかの再生成係数が保存ハッシュと完全一致した場合だけ受信し、旧Highの検証失敗に伴うSpeaker未設定を解消。
- EQマスクはEQ補正だけに適用し、Kaiser・LRなどの帯域分割本体へ適用しない既存仕様を画面ヘルプ・取扱説明書に明記。係数と出力FIRの回帰確認を追加。
- 入力の「Auto: 平坦部を0 dBへシフト」をTargetレベル合わせと共通の高音圧・平坦帯域判定へ変更。ノイズ部と補完帯域を除外し、80 Hz未満も対応。候補なしの場合は手動調整を案内。
- 入力のTarget自動レベル合わせで、Targetの基準レベルをスピーカーの高音圧候補帯域内から求めるよう修正。低域が持ち上がったTargetでもツイーターの主帯域を選択し、ノイズ部・急な帯域分割の減衰部の除外は維持。
- MultiwayのTargetプリセット復元時の初期値とSession Stateの二重指定を解消し、選択保持と起動時の警告防止を両立。
- Response Processingの「Targetに自動で合わせる」で、Targetだけでなくスピーカー測定にも高音圧・平坦条件を適用。広いノイズ部への誤整合を防ぎ、主帯域が重ならない場合は手動指定を案内。
- Target SourceにLF／HF extensionの個別ON/OFFを追加（既定OFF）。端の振幅傾斜を延長し、設定・プリセット・MultiwayへのTarget引継ぎで共通適用。
- 音響ターゲット補正の帯域外が0 dBに戻る処理を是正。低域端からDC、高域端からNyquistまで減衰を保持し、短いFIRで減衰域へUnity成分が漏れる問題を抑制。
- MultiwayのTarget選択IDを通常設定へ保存し、再起動・Web再読み込みでも復元。一覧変更時も選択IDを保持し、未登録になった場合は先頭へ戻さず案内を表示。
- Target levelのAutoをスピーカー実測の高音圧・平坦帯域に合わせる方式へ変更。疎なTargetの両端や密なFFTのノイズ部に偏る問題を修正し、補完帯域を除外。平坦帯域がない場合は手動調整を案内。
- StandaloneをChannelから独立したDraftへ復元する仕様に統一。画面再描画・設定復元でもMultiwayの帯域分割ターゲットを適用せず、Standaloneの通常のOutput Band SplitとTarget編集を分離。
- ChannelからTarget編集を開くときはDraft退避後にStandaloneへ切り替え、元の登録済みTargetを保存データから再読込。Channel用の帯域分割を編集画面へ引き継がないよう修正。
- PhaseEQ Targetメニューの切替要求を非表示タブの復帰まで保持し、切替確認後だけ完了通知を表示。PhaseEQのタブ／ウィンドウをユーザーが選ぶ手順と、受信待ち時の操作を画面・取扱説明書に追加。
- MultiwayのTargetをPhaseEQ登録済みプリセットの選択・プレビューに限定。ファイル登録・編集・Gain／位相設定はPhaseEQへ集約し、「PhaseEQのTargetを開く」と登録一覧更新ボタンを追加。接続中は既存のPhaseEQ画面をDraft退避付きでStandalone Targetへ切り替える。
- MultiwayのTargetプリセットを画面部品とは別の設定値として保持し、再描画・一時的な非表示で先頭へ戻る問題を修正。復元Targetの候補を表示・送信で統一し、別プリセットを選んだ際は以前のTarget編集結果の適用を解除（保存済み編集は保持）。
- 「音響ターゲットとして補正」のヘルプと取扱説明書を拡充。High／Lowの具体例、ON／OFFの出力差、測定・レベル合わせ・補正FIR返却の手順、設定変更後の更新を明記。
- 音響Target合成で0 Hzを対数補間から除き、FFT由来Targetの反復警告を修正。ブラウザー接続通知の一時ファイルを書き込みごとに分離し、同一セッションの並行更新で監視が停止する競合を修正。
- FIR EQ／IIR EQ／Target EQのフィルター一覧にAll ON・All OFF・All Clearを追加。表示中のGain／Phase一覧だけを操作し、空の一覧では無効化。Auto EQの重複区間制約と受信Targetの編集ロックを維持。
- Multiwayから受信したChannel TargetをPhaseEQでは閲覧専用に変更。Targetの選択をMultiwayの左側へ移し、選択中カーブの振幅・位相を表示。専用のGroup Target編集は維持。
- Response Processingに「レベル合わせ」を追加。Targetの高く平坦な実測帯域へスピーカー入力の基準レベルを自動調整し、調整量・基準帯域・比較グラフを表示。基準帯域の手動指定と微調整にも対応し、元データ・再生音量は変更しない。

- PhaseEQ／MultiwayにリニアフェーズLR4 FIRを追加。−100 dB以上・最大誤差0.05 dB以内を実応答で検証し、余裕代5%の3次近似からタップ長を割り当て、目安を表示。
- LR2／LR4の各境界に「音響ターゲットとして補正」を追加。ONの境界だけIIR応答をPhaseEQ Targetへ合成し、リニアフェーズ方式は振幅を保持して位相を0°にする。振幅・位相を補正し、ON境界の分割FIR／IIRは出力しない。OFFで元のTargetと通常分割出力へ戻り、出力の位相・時間整合は維持。

- リニアフェーズLR2 FIRのタップ数を自動算出・割り当て。LP／HPの5 Hz〜Nyquist、目標応答−80 dB以上で最大誤差0.05 dB以内を数値評価。周期数とβ値を無効化し、生成・表示・新Assignmentを統一。旧Assignmentの周期数方式はアルゴリズム版で保持。
- LR2帯域の手動タップ長を0へ強制して入力を無効化する制限を解除。自動クロップOFF時は指定値を保持して生成へ反映。
- Composite Channel workspaceの選択解除（X）を廃止。Channel選択を維持し、旧画面由来の未選択値は現在のAssignmentへ戻す。Standaloneへの切り替えはリストから選択可能。
- 開いているPhaseEQが約2秒間隔でMultiwayの最新Assignmentを確認し、更新時だけ編集内容をDraftへ退避して同じChannelの最新条件へ自動切り替え。Standalone・Target Editでは切り替えず、同じ更新の再試行ループを防止。
- 全境界がIIR／スルー、またはFullrangeでは自動クロップをOFF・操作不可にし、保存済みONも補正。DSP FIR出力ONではWayごとのタップ長を手動指定可能にし、0＝FIR OFFを維持。
- Assignment接続状態を返却済みFIRのIDと分離し、現在のPhaseEQセッションで判定。同じChannelの新Assignmentへは現在の編集をDraft保存して引き継ぎ、最新の帯域条件に更新。前回返却データは保持したまま返却待ちとして表示。
- MultiwayのWayタップ長0では、自動クロップのON／OFFにかかわらず、実生成した帯域分割FIR長×2＋1を仮割り当てし、PhaseEQ Assignmentにも反映。明示タップ長、FIR OFF、自動クロップ下限は維持。
- PhaseEQの帯域分割合成後FIRを補正用タップ数へ切り詰めず、全畳み込み長で解析・個別Exportするよう修正。カイザー窓HPの阻止帯域減衰の悪化を解消し、実出力長を表示。Multiway返却の補正専用FIRは指定長を維持。
- PhaseEQのOutput Band Splitにもカイザー窓の目標減衰・周期数表と方式別タップ数案内を追加。LR2はMultiwayと同じ誤差条件で算出し、目安／必要長と現在の生成長を区別。
- ガイド測定の「測定する側」を初回は左のみ、以後は保存済みの選択で表示。必須の単一選択ボタンは共通処理で未選択・不正な復元値を補正し、音源プリセット・マイク位置・測定画面も統一。
- 測定データ登録の「登録のみ」は入力へ未適用であることを明記。データ種類の未選択・不正な復元値をスピーカー測定へ揃え、初期表示でも選択状態を反映。
- Driver Profilesでスピーカー測定データの行を選ぶと、右側に周波数特性をプレビュー表示。選択と入力への適用を分離し、検索結果が空になった場合はプレビューを解除。
- Response ProcessingのCurrent input案内に、下の「ライブラリを開く（Open Library）」からの登録手順と、画面下のMeasurementで測定データの行を選びApply Selectionを押す適用手順を明記。
- カイザー窓の周期数入力を2.7〜15.0・小数4桁へ拡張。キーボードと±ボタンの増減は0.1刻み。減衰スロープ表の全値を入力でき、保存・読み込みでも保持。
- カイザー窓のヘルプと利用ガイドに、β＝8／10／12／14と24〜192 dB/octの平均減衰スロープに対応する周期数表を追加。評価条件と現行設定範囲外の値を明記。
- LR2の96 kHz・重ね合わせ0・20 Hz〜20 kHzでは余裕代5%の3次補正式でタップ数を算出。生成後の誤差検証を維持し、不合格または適用範囲外は数値探索へ戻す。
- タップ数案内をカイザー窓FIR／リニアフェーズLR2 FIRの選択方式に合わせて表示。LR2ではカイザー窓の減衰案内を表示せず、使用しないβ値を無効化。
- カイザー窓β値の目標減衰案内に、現在のFs・クロス周波数・周期数に基づく境界FIRタップ数の目安を併記。
- MultiwayのFIR周期数を3.0〜7.0、カイザー窓β値を7.0〜14.0に統一。設定読み込み時も範囲内へ補正。
- Response ProcessingのCurrent inputに、適用中データの意味と測定データの登録・選択・適用手順を日本語で案内。
- 全境界がスルーの場合はMultiwayの自動位相・時間整合を無効化。整合の適用・取り消し・帯域分割変更による解除時は、Auto OFFや重い描画の自動スキップにかかわらずグラフを再計算し、インパルス表示を現在のDelayへ同期。
- PhaseEQのページ・セクション見出しに「最初に／よく調整／必要時／確認／出力／保存」の用途表示を追加。測定データ登録の入口は「最初に」と表示。
- Multiwayから有効な全Channelを一括送信。送信Channelの選択とメイン画面最上部の送信枠を撤去し、サイドバーの操作に集約。「PhaseEQで開く」の表示を固定し、再計算中もブラウザー接続通知を継続することで接続表示の揺れを抑制。
- Compositeの接続情報を「編集中のChannel」へ集約し、常時表示の説明をヘルプへ移動。PhaseEQの左右下部ペインに独立スクロールを追加。

- スピーカー測定の登録をLibraryの主導線に変更し、登録案内を日本語化。Driver Profilesの測定一覧をSpeaker/Inputとそろえ、保存プロファイル・マイク校正・マイク機種設定・任意のドライバー仕様を分離。
- 外部測定のマイク校正済みチェックと訂正操作を追加。状態を保存し、校正済み入力では校正メニューと適用処理の両方で追加校正を防止。
- Channel選択の右側へ送信枠を移動。相手側のブラウザーが接続中は起動リンクを無効化し、切断後に再有効化。

## v1.19.12 - 2026-09-15

- Multiwayのメイン画面上部に送信先選択・「PhaseEQへ送信」・作業画面リンクを集約。StandaloneのChannel候補を現在の編集構成へ限定し、返却FIR ManifestをChannel所属判定に使って未返却Channelの初回接続を拒否する不具合を修正。

- PhaseEQの画面上部へ送信先Channel付き「マルチウェイへ送信」を追加。Export下部の既存ボタンも同名に統一し、同じ送信条件と処理を使用。送信結果を上部にも表示し、Workspace保存失敗時には成功を表示しない。

- Multiwayの境界別分割方式に「スルー（外部で帯域分割済み）」を追加。外部パッシブ／アクティブネットワーク適用後の測定にLP/HPを重ねず、設定保存・Assignment・共通帯域分割生成へ反映。スルー境界の構造的All-pass補償は行わない。

- 通常pytest・品質チェック・pre-commit・リリース前検証を軽量版に統一。アプリ起動を伴う統合試験と計測で確認した重い試験は、明示的な`--full`指定時だけ実行する。品質チェックに工程別時間、並列テストに収集時間と遅い試験の表示を追加。

- 単独UI検証・ベンチマーク17本の一時作業領域を共通化。通常検証の測定音源自動生成を無効化し、画像・DB・ログ・作業キャッシュを成功・失敗時とも削除。テストサーバーの停止待ちが時間切れでも終了させてから削除する。

- 品質チェックの終了時に、実行専用の一時データ・検証用音源・pipキャッシュを共通スクリプトで自動削除し、削除完了を検証。個別テストにも同じ後処理を適用する手順をテスト仕様へ追加。

- Standalone／各Wayを同じセッションで切替可能にし、共通EQを引き継ぎながら移動先AssignmentのSample Rate・Tap数・帯域分割を優先適用。WayからStandaloneへ戻る場合はRecipeを保持し、FIR OFFの0 tapsは元のStandalone有効Tap数へ置き換える。既存Draft・設定読込を共通経路へ統合し、検証失敗時は元のContextを保持する。

- Measureにプリセット起点のガイド測定を追加。標準15曲のCD版44.1k/16bitのみを起動時に準備し、Referenceを自動登録。対応する個体感度・Gain構成で、通常応答をノイズ確認→ESS Pilot→3件取得→統合IR保存へ接続。歪み・残響・相対測距の未対応解析は実行不可と明示し、従来測定画面を維持する。

- マイク測定の恒久原本を統合未校正IR 1本と別校正データへ変更。Library保存と同一DB transactionで確定し、録音Raw・各ショットIRを新規パッケージに含めない。Measurements JSONバックアップも新原本を保持する。
- 未校正IRの統合後にGate・偏差減算・Mergeを行う処理順へ統一。Libraryで原本から校正済み応答／IRを再生成し、PhaseEQ・Multiwayへの二重校正を禁止。Libraryに周波数校正変更と校正済みIR・FRD出力を追加。

- マイク校正データを偏差に統一し、Speaker入力に残っていた加算を減算へ是正。マイク測定側の偏差減算と整合させ、入力処理キャッシュの版も更新。
- 校正済みのマイク測定応答をLibraryからSpeaker入力へ渡した際のマイク校正再適用を防止。新規保存では振幅・位相の実適用状態を記録し、旧形式で校正IDだけがある結果は適用状態不明として追加校正を抑止する。

- マイク測定の逆畳み込みで、計算済みIRを固定FFT長で切り捨てる処理を修正。録音・Reference長に合わせた内部FFTの全IRを保持し、長い残響と負時間側の成分をGate再解析・保存へ引き継ぐ。

- CompositeのAssignment一覧でrevと未取得表示が混在する場合のArrow変換警告と、Nyquist除去強度など共通スライダーの設定復元時の二重初期化警告を修正。

## v1.19.11 - 2026-09-12

- Target presetsに「Harman H4 Target Curve」を追加。

- Compositeのバッフル周波数など共通数値入力の復元時の二重初期化警告を修正。PhaseEQ受信監視fragmentをグラフ構成に依存しない固定位置へ移し、全体再描画で監視IDが変わる経路を解消。

- Project名など共通文字入力の設定復元時に、初期値とSession Stateを二重指定する警告を修正。編集確定と非表示からの復帰時の値保持は維持。

- Compositeの段応答専用評価を共通化し、途中段で使わないIR/Step/GDの生成を省略。Target付加時は既存ch/Group結果を保持し、変更のない解析結果を内容キャッシュから復元する。最終FIRは生成元signature付きartifactとしてグラフ・実機出力・作業バックアップで共有し、不一致を検出する。
- Studioの6図とComposite解析図のLight表示に、確定入力・表示条件をキーとしたPNGキャッシュを追加。図オブジェクトのイベント番号には依存しない。最終FIR処理、描画点数・テーマ・操作仕様は維持。
- Nyquist除去の共通処理をアプリ非依存モジュールへ移し、CompositeからPhaseEQへの実行依存を解消。PhaseEQのGain/Phase・Error図は同値引数の参照共有状態による不要なキャッシュ無効化を抑止。

- Compositeの解析用Group応答ファイル・ZIP生成をダウンロード時へ移動し、通常更新では生成しない。FIR係数入力の最終DSP合成は共通A方式（全長線形畳み込み後に一度だけ中央切り出し）へ変更。Nyquist除去・Cosine Tapered窓のON/OFFと適用順は維持。

- Compositeの最終DSP用FIR設定にRemove FIR Nyquist componentと除去強度を追加。PhaseEQと同じ除去処理を窓より前に適用し、最終特性・係数グラフ、バックアップ、DSP出力へ反映。System設定・JSON／ZIPの保存復元にも対応。

- Compositeに「最終DSP用FIR設定・特性」枠を追加し、窓スイッチと各チャンネルのGain／Phase・係数表示を集約。DSP出力と同じ最終係数から描画し、窓ON/OFFに追従する。解析用Group応答とは区別し、追加応答計算は内容キャッシュ・非表示時省略に対応。

- PhaseEQ ExportとComposite最終DSP出力にCosine Tapered窓（α=0.10固定、初期値OFF）を追加。最終切り詰め後に一度だけ適用し、Assignment返却では重複適用しない。FIR Filter cofsグラフでlinear値、dB re 1.0、窓形状を実タップ範囲で確認可能。Working SessionのAuto Gain指定も個別出力と一致させる。

- FIRタップ長の入力／ApplyをPreferencesからOUTPUTのExportへ移動。適用後もExportに留まり、Formats／Downloadsからも設定可能。Assignment接続時の受信タップ数固定、設定キー、生成方式、ERRORの低Gainマスクは維持する。
- PhaseEQの共通応答グラフ（Gain／Phase、IR／Step、群遅延、誤差、入力・Targetプレビュー）のInteractive描画をPlotlyへ変更。現在のPhaseEQの配色、線種、軸設定、EQ合計領域、補完範囲を維持。Light、Wavelet、DSP生成・パイプラインは変更せず、軸操作はブラウザー内だけで実行する。
- IIR EQ／FIR EQの合計領域表示を復旧。共通描画へIIR Filter／Correction Filterを強調対象として指定し、既存合計データを0基準のオレンジ／真鍮色の面で表示する。Output Band Splitの既存表示とDSP生成は変更しない。仕様書の「選択中ステージまで」という古いFIR表示説明も全有効EQの合計へ更新。
- 一覧の数値入力fragmentが全体描画時の古い値を再表示用初期値として送り続ける経路も修正。今回受理した入力値を送信し、世代検証・変更イベントの待ち合わせは維持する。
- 同種の確定漏れをドロップダウン／ラジオ／連結型選択／複数選択／ON-OFF／スライダー／文字入力にも確認し、数値入力と同じイベント確定モジュールへ統一。EQ一覧の選択・ON/OFF・スライダー・連動数値も現在IDへ確定し、削除・設定置換後の古いイベントを拒否する。計算式やApply／Save／Exportの明示操作は変更しない。
- 共通数値入力を変更イベント時に正本設定へ確定する方式へ変更。別イベントで編集欄の描画が省略されると+8 dBの編集が旧値+4 dBのまま残る経路を回帰テストで再現・修正。既存の個別コールバックと正規化を維持する。High Pass連続操作によるユーザー報告の実ブラウザー再現は未確認であり、今回再現した経路と区別する。
- PhaseEQ／Compositeの選択UIを共通アダプター化し、リスト操作をStreamlit非依存の中核へ分離。編集リストは暫定128件、DB候補は暫定8,192件／ページとし、個別ポリシーで変更可能。上限超過の既存設定を切り捨てず、追加・複製で増やさない。Measurements／Speaker specs／InputはSQLページング、Projectは取得後ページング。DB全件ExportとDSP計算方式は変更しない。
- IIRのQ入力の二重初期化警告を修正。PEQ／Shelf／2次All Pass（Target側共用）の個別初期化を除去し、共通数値入力へ一本化。併せて周波数クリップの未変更値への再代入を省略。Qの量子化・制限・DSP計算は変更しない。
- Manual shiftなどの数値を編集した直後、内部値は更新されてもブラウザー再表示用の初期値に読み込み時の旧値が残る不整合を修正。共通数値入力で今回の入力値を初期値にも反映する。4 dB読み込み後の0／−2／＋2 dBへの編集、Auto IIR実計算へのTarget入力・グラフ用Target・保存値の一致を回帰テスト化。Auto IIRの計算方式は変更しない。実ブラウザーで報告された値の戻り方の再現は未確認。
- Target levelのラジオ選択が再表示時に先頭のAutoを初期値として送る不整合を修正。共通radioで保存値のindexを毎回送信する。Auto IIR実行・Target画面との往復で、手動／Autoの選択・Manual shift・保存値・実効Target応答が維持されることを確認。Auto IIR演算自体は変更しない。
- LF / HF extensionの新規設定・未保存時の既定値をONへ統一。保存済みの明示的なOFFは保持する。数値入力のvalueとSession Stateへの二重初期化も解消し、再表示用初期値・古いイベントの保護は維持する。
- Phase centeringのPeak modeなど、連結型選択の再表示時に保存済みの初期選択がブラウザーへ送られない不整合を修正。共通segmented controlで毎回保存値をdefaultへ渡し、タスク移動・復元後も選択を維持する。中心検出の演算式・既定モードは変更しない。
- Target presetsの一覧選択で、読み込み前のGain／Phaseプレビューを表示。Applyまでは現在Target・EQを変更しない。PhaseEQのTarget処理はGain only／位相欠損を0°で補完し、位相編集の基準とする。位相無効指定は優先し、Speaker測定とComposite直接処理は変更しない。
- Gain filters等の行切り替えでON/OFFチェックボックスがOFF初期値で再表示される不整合を修正。共通チェックボックスと一覧編集の両方で保存状態を再表示用初期値へ渡す。同じ欠落があった一覧の数値入力も修正し、Gain等が下限へ変わることを防ぐ。明示的なOFFは維持し、既存の保存済みOFFを一括ONにはしない。

- Auto IIRの3診断線がDCを非表示にするためのNaNを、グラフ検査が計算異常として警告する不整合を修正。該当系列の0 HzのNaNだけを許容し、正周波数のNaN、inf、全欠損、他系列の異常は引き続き警告する。Auto IIRの計算・係数は変更しない。

- FIR EQ一覧で手動／Auto行の選択によりグラフが途中段階へ戻り、ONの後段処理が表示から外れる挙動を修正。行選択を編集専用とし、Gain／PhaseどちらでもONの全FIR EQを表示する。生成・Exportの演算方式やフィルターON/OFFは変更しない。

- メニュー翻訳の再発防止テストを実ソース照合へ拡張。43文字列メニューの登録と隔離描画を確認し、新規の未解析メニューを検査漏れとして検出する。追加の製品動作変更なし。

- 平準化を有効にすると「Complex Gain」の日本語ラベル未登録で停止する不具合を修正。平準化メニューを翻訳検査対象へ追加し、両モードの描画・再実行を回帰テスト化。DSP計算・保存値は変更しない。

- P0とPhaseEQ表示のSpeaker帯域外補完を独立`response_completion`へ共通化。Loは従来表示方式、Hiは測定最大周波数から回帰傾きを滑らかに接続する方式へ変更し、Nyquist強制ゼロを廃止。測定点と時間原点を保持する。旧補完詳細設定は保存互換用に残すが標準補完では使用しない。旧設定も補完ONなら新方式で再計算するため、AutoEQ・生成FIRは変わり得る。AutoEQのユーザー指定範囲は制限しない。Gain改善を優先した採用であり、IR・Step誤差の一律改善は保証しない。

- P0→Compositeの時間契約に原点移動・位相の由来・応答hashを追加。位相欠損の0°置換、推定位相への実測時間復元、受信失敗時の応答/時間情報の世代混在を防ぎ、外部距離使用時も既知のP0原点移動を相殺する。
- マイク校正をP0へ集約し、生入力と校正状態を保存。P0段階別Cacheを追加し、手動Gain変更時の中心検出・平準化・Auto Gain推定の再実行を省略する。
- PhaseEQのGain/PhaseグラフにP0補完と表示専用補完の背景帯を追加。背景帯の追加自体は計算と時間原点を変更しない。補完方式の変更と生成への影響は上記の別項目を参照。

- グラフのPhase maskが設定保存・UI Profileから欠落し、再起動で既定の−80 dBへ戻る不具合を修正。Off／−80 dB／−60 dBの選択を保存・復元する。DSP演算条件は変更しない。

- Phase centeringのOFF→ON等でManual offsetの入力欄が下限の−1000 msになる不具合を修正。共通数値入力が再表示用の初期値を毎回渡し、保存済みの手動値を維持する。IR演算・中心検出の方式は変更しない。

## v1.19.10 - 2026-09-08

- 独立共通化後に未使用となったPhaseEQ時間解析の旧ヘルパー6件、旧Wavelet APIの再export2か所、未使用importを除去。過去比較で使用中の4関数は検証fixtureへ隔離し、現行の位置合わせ・描画・生成処理は保持する。

- Standaloneで保存済みMultiway帯域分割Recipeがあると、Output Band Splitの選択メニューが接続中と同様に無効になる不具合を修正。接続中だけ選択を制限し、未接続ではPresetを選べる。閲覧ではRecipeを保持し、Apply時だけ単独設定へ置き換える。

- PhaseEQの表示解析を独立`response_analysis`へ集約。Speakerの補間・帯域外補完を共通化し、IR/Step/Waveletの入力と中心原点を統一。Stepは全長IRの累積和後に切り出す。表示FFT倍率1/2/4（既定1、96kHz時96001点）を保存・復元し、無変更解析をセッション内キャッシュで再利用する。位相欠損の時間応答やTarget未設定の差分を代用生成しない。FIR生成・出力係数とComposite/Multiway側の新規接続は変更しない。下記の旧Wavelet移設項目で別課題だったPhaseEQ入口差に対応した。

- Wavelet演算をPhaseEQ実装から独立`wavelet_analysis`へ移設し、Compositeと共通化。利用側を直接importへ移行し、旧Wavelet互換ファイル2件を削除。旧モジュール名の一時pickleキャッシュは対象外とし、両アプリ再起動を要する。保存NPZ、IR入口、位置合わせ、演算式は変更しない。移設前との12条件完全一致と関連185テストを確認。Gain/Phase・IR・Wavelet入口の比較結果を記録（中心crop/帯域外の差は別課題で未修正）。

- Multiwayの合成IRグラフの時間軸を各帯域IR・ステップ応答と統一し、奇数長で1サンプル早く表示される問題を修正。FIR係数・実遅延・周波数応答は変更しない。

- PhaseEQのゲイン・位相／誤差図を内容・表示条件別に再利用し、保存manifest・payload・JSONの未変更時生成を省略。実書き込みとrevision競合チェックは維持。Export／IIR／Target／DBバックアップ、測定Session ZIP・UI診断JSON、Multiway作業バックアップ・実機DSP用ZIPはダウンロード1回で必要な出力を生成・取得。確定済み入力をコピーし、生成中の編集混在とダウンロードによる全画面再実行を防止。測定ZIP取得では保存フラグを変更しない。Composite保存は操作時に係数とWorking Sessionを準備する。

- 再読み込み前の`SpeakerResponse`が残った場合に、入力キャッシュの`PicklingError`で停止する問題を修正。直列化できない入力／結果はキャッシュを使わず通常計算し、復元できないキャッシュ項目だけを破棄。編集中データは初期化しない。

- Auto IIRへ「Reset Auto IIR settings（設定を初期値に戻す）」を追加。生成条件だけを初期化し、生成済みAuto／Manual IIR、Input、Target、編集中チャンネルは保持。自動再生成は行わない。

- 入力Pipelineと表示系列を内容ベース・容量上限付きでキャッシュ。PhaseEQ／Multiwayの周波数間引きを共通化し、Light／InteractiveともUIの1024／2048／4096点を上限に使用。位相ラップ補間点の増加とマスク区間の誤接続を抑制。測定平滑化UIと保存・DSP処理条件を維持。
- Complex Gainのcoherent窓平均を累積和で高速化し、非有限値・極端な振幅差・打ち消しの数値保護と回帰テストを追加。

- 数値入力の遅延callbackが消去済みwidgetキーを参照して停止する問題を修正。キー欠落・削除済みEQ・設定入替前のイベントを無効化し、古い部分画面は現在の設定から再構築する。
- EQ数値の連続入力を部分再実行へ分離し、入力停止後200～300 msで最新値のグラフ更新を開始する。中間値ごとの全画面処理を抑え、最終値の保存をChromiumで検証する。
- IIR／FIR編集後の再表示を計測し、位相線分割の配列処理化と自動保存時の応答署名重複・再帰コピー削減を実施。係数生成と署名形式を維持し、代表条件で約29～30%短縮。再計測スクリプトと分析記録を追加。
- FIR EQのGain／Phase表示・編集段階をUI Profileへ保存。ページ切替で段階別FIR設計キャッシュを破棄しないようにし、未変更のグラフ基礎系列をセッション内で再利用する。
- FIR生成FFTをMultiway基準の偶数`max(sample_rate, taps)`へ共通化し、表示Analysis FFTから分離。96 kHz／351 tapsは96,000点で生成する。
- Multiway最終FIR投影の余分な1 sample遅延を修正し、奇数／偶数tapの整数／半sample中心を共通化。新FIR Recipeはalgorithm v2とし、既存v1 Recipeは旧グリッドで再生成する。
- Multiwayの帯域分割FIR bankを`crossover_engine`へ共通化。全Way構成・境界別基準クロス／overlap／方式／cycles／betaとLR2極性補償をAssignmentへ保持し、PhaseEQが同じ自然係数を再現する。
- PhaseEQの新規Standalone境界は基準クロスからcutoffと自然tap数を自動計算。旧設定は従来方式を保持し、明示操作で移行する。IIRのoverlapは一度だけ適用する。
- 共通Recipeを設定・再起動・Working Session ZIPへ保持し、受領SOSとの不一致を検出する。ZIPへ帯域分割IIR／Combined chainのJSONと極性説明を追加。
- 詳細仕様、操作説明、旧Mid合成説明の矛盾を更新し、共通化前の係数goldenと保存・復元回帰テストを追加。

## v1.19.9 - 2026-09-05

- PhaseEQのStandalone／Composite Assignment／Target Editを明示的な作業モードとして整理し、接続先、画面ID、Assignment ID、Channel、保存先を必要な場面だけ表示する。StandaloneではComposite連携操作を隠し、単独作業用のFIRタップ数を指定可能にする
- Multiwayから複数のPhaseEQ画面を開いた場合も、Channel別draftとAssignment参照を保持して自由に切り替えられるようにする。EQの追加・削除などで編集中Channelが初期化されないよう、選択状態と編集内容の寿命を分離する
- PhaseEQ／Multiway間の監視を接続heartbeatと成果物revisionに分離し、新しい実出力だけを自動受信する。監視読込みだけで通知や画面再描画を繰り返さず、欠落・不一致・別Systemの更新は状態マトリクスに従って扱う
- Multiwayの実現済みChannel応答へPhaseEQ FIR位相を含む全処理を渡し、Stereo／Mono、Fullrange／Multiway、FIR／IIRの各経路でグラフと出力の参照状態を一致させる
- Gain、Phase、警告／実行などの意味別UI色とグラフ系列色をPhaseEQ／Multiwayで統一し、単一選択が必須の連結型ボタンは再クリックでもOFFにならない共通動作とする
- Export edit modeを別PCのPhaseEQで再現できるWorking Session ZIPへ拡張し、manifestとSHA-256で内容を検証する。Biquadを直接扱わないIIR専用DSP向けにType、Frequency、Gain、Q、Order、Enabled、適用順を記載したGeneric IIR Parametersを追加する
- システムフロー図、将来構想、簡易マニュアル、利用者ガイド、仕様書を現行実装へ同期し、未確定の将来案を現行仕様と分離する。Composite EngineはPhaseEQ統合版とMultiwayベース版を併記し、派生版であることを明確にする

## v1.19.8 - 2026-09-02

- 測定画面を連続測定ワークフローへ拡張し、基準トラック、入出力レベル確認、ESS出力Channelの必須単一選択、Tweeter相対タイミングと時間基準情報を一貫して扱えるようにする
- Multiway StudioへSystem全Channelの位相・時間整合、距離／Tweeter基準の選択、FIR／IIR自動選択のバッフル補正を追加し、表示、保存、Assignment、DSP Exportへ同じ実現済み状態を反映する
- PhaseEQとMultiway間のAssignmentを2秒間隔で自動受領し、手動Refreshやリンク移動なしで編集中Channelへ接続する。停止中に届いたAssignmentも次回起動時に受領する
- 単一選択が必要なUIを連結型コントロールへ統一し、選択中項目の再クリックで全OFFにならないようにする。Gainを青、Phaseを紫、警告／実行をオレンジとする共通配色をPhaseEQとMultiwayへ適用する
- 日本語UIで`Wavelet Hybrid`、ピーク基準、`Energy centroid`、`Manual`が未登録として起動を停止する問題を修正する
- Composite Assignment、IR時間基準、Multiway UI、Phase／時間整合、REW Remote測定の詳細仕様と利用者ガイドを更新する

## v1.19.7 - 2026-08-29

- 新規`.venv`の依存確認がPhaseEQ本体を先にimportして`numpy`未導入で停止する問題を修正する。環境判定を標準ライブラリだけで開始し、不足パッケージは再作成を繰り返さずmacOS／Linux・Windows共通のインストール経路へ進める

## v1.19.6 - 2026-08-29

- GitHub Actionsの一時ファイルパスをRunner割当後のStep内で評価し、ReleaseワークフローがJob開始前に停止する問題を修正する
- 利用者ガイドを現行Workspace／Page構成へ再照合し、目的別・画面別の索引、短い操作単位、番号付き補足へ再構成する。旧DSP System操作など現行アプリにない説明を除去する
- Composite EngineのSection 1～8、FIR／IIRとtap数、All-pass、位相・時間整合、Crossover／Channel／System Sum／Group Targetの違い、Phase・Group Delay・Wavelet、Assignment往復を初心者向けに説明する

## v1.19.5 - 2026-08-29

- GitHub Release本文を配布用の日本語更新案内から自動生成し、前バージョンとの差分リンクを追加する。手動再実行でも既存Release本文を同じ正本へ同期する

## v1.19.4 - 2026-08-29

- Multiway StudioへDSP FIR出力ON／OFFを追加し、Fullrangeでもタップ長を設定可能にする。FIR初回ON時は1023 taps、FIRクロスの帯域タップ長0は実生成長、IIRのみ／Fullrangeの0はFIR OFFとして扱う
- FIR OFFでは1 tapのUnity FIRを生成せず、タップ数、FIR Recipe、FIR係数、FIRファイル、FIR reportを出力しない。グラフ、タップ一覧、Composite、PhaseEQ Assignment、DSP Exportで同じFIR状態を参照する
- FIR OFFのPhaseEQ AssignmentでもIIR EQ、Target、処理済みSpeaker応答、Working Sessionの往復を継続する。汎用miniDSP／SigmaStudioから機種非依存の4096 taps固定制限を除外する
- Release前に全品質チェック済みのGit tree IDを固定し、内容が変わっていない場合だけrelease commitの重複品質hookを省略する。通常コミットと他のpre-commit hookは維持し、リリース全体を約6分から約3分へ短縮する
- テストを文書、ソース、開発／配布基盤の大区分へ分類し、通常コミットではstage差分に応じたSmartチェックを行う。責務が重なる変更はカテゴリーを併合し、切り分けが難しい変更は全件確認へ戻す
- 「リリース」依頼を既定の`tomii323/PhaseEQ`へのバージョン更新、commit、tag、atomic push、GitHub Release公開までの常設承認として明文化する。送信先が異なる場合や公開範囲が不明な場合は停止する

## v1.19.3 - 2026-08-29

- Release前に配布文書契約を先行検査し、更新案内の命名不整合を全品質ゲートより前に検出する
- 非公開GitHubリポジトリでは既存の読み取り可能なghセッションへ限定フォールバックし、Release ZIPとSHA-256の両資産を確認する。認証開始・修復は行わない
- 全pytestを省略せず最大4個の隔離プロセスへ均等配分し、重いStreamlit AppTestと大規模数値テストのフル品質チェックを約10分55秒から約3分17秒へ短縮する。`--jobs 1`による直列再現も維持する

## v1.19.2 - 2026-08-28

- Fullrange+SUBのFullrangeへSUBと相補になるHigh-passを生成し、境界を含むSystem Sumを平坦にする
- Stereoで片側ChannelだけをCompositeへ含めた場合もL／R／L+R表示を維持し、別GroupやGroup Targetへ置き換えない
- Shared SUBをL／R双方のSystem Sumへ各1回だけ含め、L+RのChannel線では1系列だけ表示する。左右独立SUBは相互加算しない
- 全ChannelをComposite対象外にした場合は空のグラフとして安全に表示し、時間応答描画の空配列例外を防止する

## v1.19.1 - 2026-08-28

- GitHub Releaseワークフローへタグpushと手動再実行の二つの起動経路を追加し、初回登録時など自動起動できなかった場合も既存タグから復旧できるようにする
- 既存Releaseを再利用し、同名のZIP／SHA-256資産だけを検証済み生成物へ差し替える冪等処理を追加する。再実行時にReleaseを重複作成しない
- GitHub Actionsの配布ZIP検査をローカルReleaseと同じ共通契約へ統一し、開発文書や未分類Markdownの混入を自動的に拒否する

## v1.19.0 - 2026-08-28

- Multiway StudioへFullrange／Fullrange+SUBを追加し、Mono／Stereo、共有SUB／左右独立SUBを既存2～4Wayと同じChannel、Assignment、整合、保存、復元、Export経路へ接続する
- Composite結果表示をMonoのMain、StereoのL／R／L+Rへ統一し、SUB単独表示を廃止する。L+Rは左右の全Wayと左右別System Sumを同一グラフへ表示し、Waveletは左右別パネルで確認できるようにする
- Group Target／Channel Targetを係数やSystem Sumへ乗算しない表示専用系列として分離し、共有SUBおよび同名の左右独立SUBをChannel IDで識別する
- Multiway SystemをProject、Target、Assignment、Revisionの管理正本として拡張し、PhaseEQとCompositeでTarget定義とFIR生成Recipeを共有する。Composite Assignment中の帯域分割と暗黙FIR加工をMultiway設定へ一本化する
- FIR／IIR混在時の理論中心、実遅延、群遅延、位相・時間整合、奇数長FIR、DSP Export v3のDelay出力を共通経路で扱い、関連する保存・Resume・DB連携を補強する
- macOS／Linux／Windowsの連動起動でlocalhostの起動待機を0.3秒間隔の再試行へ統一し、利用可能なloopback URLを確認してからブラウザーを開く

## v1.18.3 - 2026-08-22

- PhaseEQとComposite EngineでBuilt-in／User Target preset DBとTarget生成処理を共有し、カテゴリ・利用範囲付き定義、内容snapshot、revision付き編集sessionを追加。CompositeのChannel／Group TargetをPhaseEQで編集し、保存完了後だけCompositeへ反映できるようにする
- CompositeのTarget選択、Assignment、Resume、DSP ExportへChannel別Target定義を接続。再起動時に保存済みPhaseEQ Targetを古いAssignmentで上書きせず、別環境でUser presetが不足してもsnapshotから復元できるようにする
- REWのTarget Curve CSVとDirac Live TargetをTarget Responseから読み込めることを明示し、コメント・列見出し付きREW CSVの回帰テストを追加する
- PhaseEQとCompositeの小数表示を一般的な四捨五入へ統一し、Waveletと時間応答グラフの表示範囲・軸・系列スタイルを共通化する。表示だけを丸め、DSP計算値、保存値、Export係数は変更しない
- macOS／LinuxとWindowsの連動ランチャーを`127.0.0.1`限定待受へ変更し、表示上の`local-only`と実際のネットワーク公開範囲を一致させる

## v1.18.2 - 2026-08-15

- Windowsランチャーの実行順と回復動作をmacOS/Linux版へ統一。Main／Compositeの事前確認、オフライン優先、最終import失敗時のオンライン修復、更新後の再検証、Check-only、2ポート起動を同じ順序で行い、PhaseEQ終了時は連動起動したComposite Engineプロセスも確実に終了する

- `sounddevice`のimport失敗をアプリ全体のRuntime破損から分離。Windows 11 ARMのx64エミュレーション等でPortAudioを読み込めない場合も、例外内容と測定機能が利用不可であることを警告し、設計、グラフ、Multiway、ファイル入出力は起動可能とする
- Windows 11 ARM上のx64 PythonでSoundDevice 0.5.5がホストCPUを基に`libportaudioarm64.dll`を誤選択する問題を回避。SoundDeviceのimport中だけPython ABIに合わせてAMD64を通知し、同梱`libportaudio64bit.dll`を読み込んで測定機能を利用可能にする

- Windows 11の新しいPython Install Managerが、未導入の`py -3.13`候補で成功終了コードを返す場合にランチャーが誤採用する問題を修正。実Pythonだけが返せる識別値を確認し、利用不可候補を静かに除外して`py -3.12`、`py -3`、`python`へ継続する。固定依存関係の最終検証に失敗するPython 3.14は採用せず、検証済みの3.12／3.13だけを使用する。Windows 11 ARMでは`pyarrow`等のWindows ARM64 wheel不足によるソースビルド失敗を避けるため、公式x64 Python 3.12をwingetで強制導入しOSのx64エミュレーションで実行する。x64判定にはホストCPUを返し得る`platform.machine()`ではなくPythonのwheel ABIを表す`sysconfig.get_platform() == win-amd64`を使用する。導入直後は標準インストール先、Python登録情報、ユーザー領域のインストール直下から再検出し、Python同梱の`Lib\venv\scripts`にある内部リダイレクターは候補から除外する

## v1.18.1 - 2026-08-14

- Multiway関連文書と画像付き操作マニュアルを現行UIへ再照合。グラフ更新を`Auto`／`Recalculate`、Gain軸下限を`-144/-40/-30/-20/-10 dB`へ訂正し、位相・時間整合を単一の適用操作として記載する。単体Multiway FIR Studio 1.1.3とPhaseEQ同梱Composite拡張も明確に区別し、v1.18.0版PDFを再生成する

## v1.18.0 - 2026-08-14

- Multiway Studioの上段`Crossover / Design Response`と関連表を、生成FIRだけの表示から実現済みChannel応答へ拡張。Speaker source、PhaseEQ FIR／IIR EQ、FIR／IIR帯域分割、Gain、Polarity、手動Delay、適用済み位相整合Delay／All-passを同じ信号経路で反映し、Way別応答と複素`Design Way Sum`を表示する
- PhaseEQの`Response Processing`最終応答をSpeaker sourceとしてAssignmentへ保存し、FIR Auto Gainを常にOFFにしたFIR EQ、IIR定義、実現SOS、Working Sessionと共にMultiwayへ返却する。Assignmentを再度開いた場合も直前の有効なSpeaker／FIR／IIR結果を保持し、保存完了時だけ同じAssignment revisionを更新する
- Multiwayの位相・時間整合を単一の`位相・時間整合を適用`へ統合。押下時の最新応答を解析して原子的に適用し、通常時は要約だけ、Channel／境界の数値は`解析結果を確認`内だけに表示する。旧解析ボタン、解析Preview、空のシミュレーショングラフを廃止する
- 位相・時間整合を最低域Way基準のLow→High逐次解法へ変更。3WayではLow/Midを先に整合し、その結果を含むMidを基準にMid/Highを解く。理論All-passの後に残差Delayを求め、因果化共通Offset適用後のDSP Delayを最寄りの1 sampleへ丸める
- 再解析時は前回の自動Delay／All-passだけを除去して同じ現在入力から絶対再解析し、補正値の累積ずれを防止する。All-pass構造補償を使用不可にしてもDelay解析を継続し、`位相整合を元に戻す`ではDelayとAll-passを共に復元、Crossover method変更時は旧自動整合結果とUndoをクリアする
- Speaker／PhaseEQ応答が整合適用やUndo後に消える問題を修正。Speaker Package、手動FRD／IR、PhaseEQ Response Processing出力、Unityの排他優先順位とDSP入力Signatureを定義し、3Way IIR LR2／LR4を含む再実行で入力AssetとChannel identityを維持する
- Speaker FRDのPhase列を明示的なUnwrapped Phaseとして取り込み、Speaker、PhaseEQ FIR／IIR、crossover、All-pass、Polarity、Delayの連続位相をChannel結果まで保持する。下段Unwrapped Phaseの等価な360°枝をクロス位置で表示上だけ接続し、Group Delayを同じ連続位相とGain maskから算出する
- MultiwayグラフUIをPhaseEQ側へ整合。右ペイン上部へAuto、Recalculate、Mode Light／Interactive、Pointsを集約し、Gain軸を-144～+40 dB、Phase maskを-80／-60 dBまで拡張する。Impulse／Stepの横方向だけの拡大、全Impulse／Step／Group Delayの-2～+10 ms表示、Impulse／Stepの1 tap解像度を追加する
- IIR方式のタップ長表示を修正。IIRは生成FIRタップを持たないため`—`とし、グラフ用IIR→IR変換だけを2 Hzまで評価できる長さで生成して全Wayを同じ中心へ揃える。クロス近傍変動量、タップ長とDelay目安、出力Gain調整結果も実現済み応答経路へ統一する
- 後段DSP用`DSP Export v3`を追加。Channel別最終FIR、PhaseEQ IIR、IIR crossover＋補償All-pass、Gain、Polarity、手動／自動Delay、Workspace、設定表、検証Report、Manifest、SHA-256を、miniDSP、CamillaDSP、SigmaStudio、Generic SOS、IIR-only Hz／dB／Q Adapterで機器別Packageへ変換する。作業復元用Resume Format v2とは独立して扱う
- 現行コードとUIを基準にREADME、システム仕様、操作ガイド、Composite Engine、グラフ、回帰テスト、出力形式、UI用語、文書マップを再監査し、旧High基準、Analyze／Preview／Apply、表示時だけunwrapという旧仕様記述を更新する

## v1.17.0 - 2026-08-11

- Auto IIRをDeterministic Power v4へ更新。上側包絡のWide+Cap Peak Cut、広い山への低Q Cut、広い谷への低Q Boost、Sharp Dip保護、弱い最終PEQの除外、候補代表化と敗者復活探索を追加し、最新診断とPreview曲線を保存・再表示する
- Auto IIRのPEQ中心周波数を、振幅結果を維持しながら高域側群遅延の突出と傾斜が改善する場合だけQ連動範囲で微調整する。Gain符号による固定方向シフトは行わず、改善しない場合は元の周波数を維持する
- TargetへIIR EQと同じ実High Pass／Low Pass／Matched All Passを追加。Gain側9種類、Phase側5種類を、それぞれ1つの追加メニュー、一覧、選択中編集欄へ統合し、重複していたPhase All Pass UIを廃止する
- IIR EQとFIR EQをGain／Phase表示へ統一し、一覧へ`Lin.`／`Min.`／`Max.`のPhase方式を表示する。Target、IIR EQ、FIR EQの名称、配色、番号見出し、共通Add／Duplicate／Delete操作を揃える
- Auto IIR完了要約を`Auto IIR details`へ永続表示し、仕様書、UIフロー、ユーザーガイド、画像付きマニュアル生成・画面取得スクリプトを現行UIへ更新する

## v1.16.0 - 2026-08-08

- PhaseEQのLinear FIR出力帯域分割を、Kaiser FIR、LR2-target Linear-phase FIR、IIR LR2、IIR LR4の境界ごとの排他選択へ拡張。Midを含む境界間でFIR／IIR、LR2／LR4、Kaiser／Linear-phase LR2の混在を許容し、FIR EQと帯域分割FIRの合成、IIR Biquad分離出力を維持する
- LR用EQ mask／Phase内側ガードをSpec v2へ更新。標準LR境界は−40 dB、緩いLR2 HPだけは−20 dBとし、Kaiser条件とLinear FIR本体へは従来どおり適用しない
- Multiway Composite Engineの位相・時間整合へ、通過していないIIR境界の理論All-pass構造補償と残差Delay解析を追加。FIR境界へAll-passを誤適用しないよう修正し、結果グラフの描画負荷によるブラックアウトを防止する
- 独立SQLite `Multiway System Library`を追加。複数System、Channel、Studio設定、Speaker Package参照、最新PhaseEQ Assignment参照をRevision付きで保存し、Open／Duplicate／Archive／Restoreを提供する
- DSP Resume ZIPをFormat v2へ更新。System出自とChannel参照を保存し、旧v1互換読込、未知版・版不一致・Checksum不一致・危険なArchive Pathの適用前拒否を追加する
- Multiway UIへMono／Stereo、左右別Speaker Package、左右独立Channel、共有／左右独立SUB、Left／Right／Sub結果切替、PhaseEQ Channel workflowを追加。2Way／3WayからのSUB追加は未検証フィルターを生成せず、既存4Wayクロスへ明示的に切り替える

## v1.15.0 - 2026-08-03

- DSP System OutputのSpeaker Package一覧で、行選択PreviewとApplied Sourceの違いが分からずLive Resultへ反映されていないように見える問題を修正。未適用時の警告と、Live Result上部のWay別Applied Source名を追加し、Apply後にOutput・Way・再生成元応答が同じPackageへ切り替わることを検証する
- `LIVE · DSP System Result`のGainグラフへAnalysis Gainと同じMin dB／Max dB操作を追加。-144～-10 dBの下限、+10～+40 dBの上限を選択でき、DSP System専用の表示範囲として設定JSON、Local snapshot、Project ZIPへ保存・復元する
- 設定JSONへ整数の`settings_file_revision`を追加し、構造移行用`schema_version`、内容署名`revision`と責務を分離。現行は無印の旧ファイルをrevision 0として継続読込し、将来最小対応revisionを引き上げた後は、古い自動保存を内容へ適用せず復元可能な廃却領域へ退避して初期設定から再開する。新しすぎる設定、手動Import、Project ZIPは自動クリアせず非対応として拒否する
- Spec v2.1の段階実装を完了。Speaker Package保存をInput Profile専用APIへ切り替え、Target／EQ／DSP条件の誤保存を防止した。DSP System schema 13でDevice別Analysis FFTを廃止し、共通Analysis FFT、Output TargetのFollow／独立コピー保持、VariantのActive／Working／Applied永続化、Applied／Live／Saved完成署名、未保存Export停止を接続した
- Speaker Package、Application Analysis Policy、DSP Device、Physical Output、Wayの責務と、Variant Working／Applied、保存完了、Export署名Gateを`PhaseEQ System Spec v2.1`へ再整理。現行実装との差分と受け入れ条件を明示し、新システムの規範仕様として既存仕様・フロー・テスト文書から参照する
- Speaker Packageの責務をInput Profileまでへ限定し、新規保存と改訂署名からSample rate／FIR taps／Analysis FFTを除外。Sample rate／tapsはDSP Device、Analysis FFTはPhaseEQ共通ポリシーとしてSetupへ集約し、必要時だけFIR長以上の高速FFT長へ自動昇格する
- DSP Outputへの適用をTarget／IIR EQ／FIR EQ／Linear FIRで分離し、別工程の設定上書きを防止。Target編集先をWay groupまたはOutputとして固定表示し、Target／IIR EQ／FIR EQ／Wayへ独立8枠Variantバンクを追加する。Linear FIRはVariant対象外とする
- Speaker PackageとDSP Systemに`保存して完了`の節目を追加。DSP Systemは最新Live ResultとDesign Signatureが一致するまで完了保存を無効化し、自動復元状態と正式保存を区別する
- Auto IIRからFIR taps由来の低域優先境界と`Prioritize LF for IIR`を廃止。Deterministic Power v2と互換エンジンの候補評価を、ピーク高さ、Power面積、Wavelet、凹凸幅、聴感重みによる全帯域評価へ統一する
- DSP SystemのOutput `Speaker Package / INPUT`を即時反映Selectboxから、最大10行の一覧選択、Raw Input／Input ProfileのGain／Phase Preview、明示的なApply／Clearによる確定操作へ変更する。Input ProfileのMeasurement一覧も約10件を縦表示し、FIR processingのSession State二重指定警告を解消する
- DSP Systemの`FIR processing`をOFFで保存しても、Load時に直前のStreamlitウィジェット値が保存値を上書きする問題を修正。SystemのLoad／New／Duplicate、JSON Restore、Delete後の再選択でSetupウィジェットを読み込んだSystemへ同期する
- DSP SystemでFIR processingをOFFにした場合も右側のIIR-only Live Resultを維持する。WayごとのFIR OFF、FIR Crossover Bypass、Common FIR Auto Gain未適用の元メッセージは削除・要約せず、閉じた`FIR OFF／Bypass details`へまとめ、FIR以外の警告は通常表示する
- DSP Systemの旧UI名称を廃止し、外側の`System`ページと重複していた内側の`System`を`Setup / 構成`へ変更。Systemページ内を`Setup / Routing / Output / Way / Alignment / Files`へ統一し、構築時に最初に決める`Output topology`をSetupの先頭へ移動した。保存済みの旧`System`選択状態は画面へ再表示せず`Setup`へ自動移行する
- FIR補正のGain、Phase、Auto Gain、Auto Phaseを`FIR EQ`の1画面へ統合。全項目を1つの表で選択編集し、新規作成を種類選択＋共通`Add`ボタン1つへ集約した。Duplicate／Deleteも共通一覧の選択行操作へ統一し、各編集欄に残っていたSortと重複操作を削除した。Linear FIRは帯域分割／生成条件を担う独立メニューとして維持し、DSP OutputからもFIR EQとLinear FIRを別々に開く
- 旧Project DB隔離後の現行構成をPlantUMLで再整理。Measurement→Speaker Package→DSP System→Live Result→Device Exportの全体図、主要機能別8図、UI画面構成・実作業・更新タイミングの3図を追加し、FIR OFF、外部Crossover、Wayへの明示適用、Design Signature不一致時のExport停止を明示した
- 従来Project 6件をSpeaker Packageと独立した1 Input／1 Output／1 Way DSP Systemへ最終移行。Target、IIR／FIR EQ、Sample Rate、Device FIR tapsを移し、Crossoverと旧Gain／Delay／Polarity／Alignment／Mute／Soloは仕様どおり移行せず初期化した。全6件でSpeaker Package接続、Target、係数有限性、Design Signature、Exportを検証し、既存の無関係なDSP System 1件は保持した
- DSP System保存形式をschema 12限定に変更し、旧schema自動変換、Way内Linear FIRからのCrossover推測、Routeの旧Polarity／Delayを削除。Crossover Plan、物理OutputのSpeaker Package／IIR／FIR、WayのGain／Delay／Polarityをそれぞれ唯一の正本とした
- Design Libraryから旧→新移行台帳の作成・更新・一覧API、`migration_source_id`によるOrigin表示を削除。受入済み6件の台帳は専用SQLiteとして廃却領域へ移し、現行DBの台帳テーブルと移行マーカーを解除した
- 旧Project DB、WAL／SHM、専用CRUD／read-only Reader、移行スクリプト、専用テスト、完了済み移行計画を`decommission_pending`へ隔離。最終1Way移行モジュール／実行スクリプト／受入テストも実行完了後に追加隔離し、完全削除は機能ベースの最終サマリーと人の承認まで保留した

## v1.14.0 - 2026-07-30

- DSP Systemの物理`Output`と音響`Way`を分離。OutputはSpeaker Package／Flat INPUT、Way Target追従または独立コピー、IIR、FIRを保持し、WayはCrossover／Gain／Delay／Polarity／Alignmentを保持する。Way × Output matrixで割当を交換しても両者の設定責務を維持し、Speaker Package・IIR・FIRをすべてスルーした構成も保存・再生成できる

- DSP Systemへ`FIR processing`一括スイッチを追加。OFFではGain／Phase／Auto／Linear／Way FIR設定を保持したまま全FIRとFIR由来遅延をBypassし、FIR編集ボタンを無効化して、miniDSP／CamillaDSP／SigmaStudio出力からFIR段を除外する。Device Max I/Oを実際の論理Input／Way／Routingへ反映する1/2入力レイアウト操作と、容量・実使用構成を分けた表示も追加した
- DSP System Targetへ帯域分割構成を読取表示し、Targetへ同じフィルターを二重追加しない構成へ統一。Main境界とSUBごとにDSP、外部パッシブ、外部アクティブを選択でき、外部帯域分割時はCrossover係数だけを生成・Exportから除外しながら、DSPのPEQ、FIR EQ、ゲイン、極性、ディレイ、Routingを継続利用できるようにした。IIR All Passへ独立Matched All PassエンジンによるLR2 Lo／Hi、LR4、LR8位相presetを接続した
- IIRフィルターのAdd／Delete／Duplicateで右側Detail viewがErrorへ戻る状態競合を修正。IIR画面の入力系列名を`Speaker/Input`、適用後を`IIR Result`へ固定し、同じ入力が別名で切り替わって見える状態を解消した
- IIR EQのAuto IIR説明を2行と主要結果の1行要約へ圧縮し、生成フィルター、詳細指標、Boostしなかった局所ディップ、探索条件を閉じた詳細タブへ移動。`01 IIR filters`の番号欄を`Auto PEQ1`／`Manual PEQ1`形式へ変更し、生成元と種類を一覧上で識別可能にした
- Target画面を選択中DSP Systemと音響グループの正規Target編集画面へ統合。既存のSource／Gain／Phase／Preset／Files編集を維持したまま、対象WayとDSP draft状態を表示し、完成した有効Targetをグループ単位でDSP Systemへ取り込めるようにした。System画面の重複Target編集は要約とTarget画面への導線へ整理し、DSP配下全ページの上部条件表示をDSP System基準へ統一した
- Workspace／DSP Systemの選択UIへ英語表示・日本語ホバーの中央翻訳を接続し、DSP System画面上部を選択中Speaker PackageではなくSystem名、入出力数、Device、Target、Design Signatureの専用条件表示へ変更。WorkspaceのSession Stateと初期値の二重指定警告を解消した
- `Apply current editors to this Way`をLive Resultキャッシュから分離し、同じ実行で完成した最新Target／IIR／FIR設定をWayへ取り込むよう修正。Live ResultのAuto updateがOFFでも古い編集値を転記しない
- 旧`project_db.sqlite3`を通常実装から切り離し、Speaker Package、Target Package、Designを分離保存する`phaseeq_design_library.sqlite3`へ移行。旧Project IDを同じDesign IDとして引き継ぐためDSP Systemの接続を維持し、現有6件はConfig、Speaker、Target、UI、メタデータ比較に全件合格した
- DSP SystemのLive Result、Target表示、Wavelet、Manifest、CamillaDSP pipelineを単一の最終信号経路へ統合。Input IIRをFIR残差設計へ反映しつつBiquadとして1回だけ出力し、Way Gain／Polarity／DelayをFIRから分離してFIR OFF時も有効にした。応答Assetのない旧Target Packageは保存済み編集値から実応答を再構築する
- 旧Project DBを作成・更新しないread-only Reader、再実行可能な全件移行、SQLiteバックアップ、旧新ID対応台帳を独立実装。Project画面とDSP System、Auto IIRベンチマークはDesign Libraryを正本とし、旧DBへの通常参照を除去した
- 旧ソース／DBのM5廃却候補調査を共通モジュール化。類似名称、静的参照、SQLite整合性、移行状況をEvidence Bundleへまとめ、技術チェックを機能ベースの削除内容サマリーへ変換するレビュー用プロンプトを生成する。M5では隔離・削除を行わない

- DSP Systemの再生成判定をProject Revision／Sessionのstaleフラグから`Design Signature schema 1`へ移行。System設定、Routing、Device、Way、Crossover、Target、Projectの実Input／IIR EQ／FIR EQ／Auto EQ、生成エンジン版から決定的署名を生成し、Project名や保存時刻だけの変更では再生成しない。生成結果と現在署名が一致しない場合は古いFIRのDSP Package出力を停止し、Manifest schema 6へSystem／Way署名を記録する
- 旧DSP System preset／DBはDesign Signature未生成としてschema 8へ読込可能とし、最初のUpdateでSystem署名とWay別FIR署名を付与する。従来のProject Revisionは移行互換と履歴情報として保持する
- Projectの`Input`、`IIR EQ`、`FIR EQ`、`Auto EQ`のUIと保存設定を変更せず、DSP SystemのWay再生成元として接続。Project処理後のInput応答を起点に、補正Target、Project FIR EQ、DSP SystemのFIR Crossover／Alignmentを独立`Way FIR Single Projection`でDevice tapsへ1回だけ投影し、段別畳み込み後のcropを廃止した
- Deviceに`Way FIR supported`、各Wayに`Way FIR` ON／OFFを追加。OFFまたは非対応Deviceでは保存済みFIR設定を消さずIIRのみでLive Resultを計算し、CamillaDSP／miniDSP／SigmaStudio packageからFIR係数とFIR段を省略する
- DSP SystemでWayのProjectを未選択にした場合、0 dB／0°のFlat inputとProject filterなしのSourceとして生成し、Crossover、位相補正、Way Level／Delay／Polarity、Routing、Live Result、Device packageを利用可能にした。保存済みProject IDが見つからない場合も警告を表示してFlat sourceへ退避する
- DSP SystemのRoutingを、実機DSP Input → 論理Input → Route → Output / Way → Sourceの一貫した接続判定へ統一。物理Input未割当をLive Result、Headroom、未接続表示、Device packageで同じく未接続として扱い、ProjectとFlat sourceを区別して表示する
- Routing画面へInput要約、物理Input binding、Outputカード、Project、Crossover、Route数値／詳細設定を統合。OutputカードはDevice・Out番号を常時表示し、InputボタンでON／OFF、数値表示からGain／Delay、Settingsから極性／Muteを編集できる
- `Main ways`を入力グループごとの帯域数として扱い、StereoのWay変更で左右が1系統へ潰れる問題を修正。SUBを0～8個の別枠出力として設定でき、Stereoの共有SUBはL/R各−6.02 dB、2 SUBは左右独立Routeを生成する
- DSP System内の再描画先を現在ページと選択セクションへ固定し、RoutingやWayの編集後にInput画面へ戻る状態競合を防止した

## v1.13.1 - 2026-07-28

- Deterministic Power v2のLo／Hi Shelf許可時に、同じフィルター予算のShelfなし全PEQ構成を必ず追加設計して最終指標で比較するよう修正。Shelf混在探索が有効な全PEQ枝を押し出して、Shelf ON時だけRMS、1/3 oct ABS、Peak Powerが悪化する問題を防止した

## v1.13.0 - 2026-07-28

- `Run Auto IIR`／`Clear Auto IIR`の再描画先をIIR EQへ固定し、処理完了後にInput画面へ戻ることがあるPage selectorの状態競合を修正した
- `Run Auto IIR`を乱数なしのDeterministic Power v2へ切り替え。Beam 8／分岐候補6、正ピーク初期Cut 85%、広いディップの初期Boost 80%、正負Gainの広域輪郭Q 0.3～0.8、局所ピーク面積を仮に15%残す予約Cut PEQ、Peak Power最大値・面積優先、新規ディップ0.35 dB、Peak Power許容2%のGuardを採用した。Manual IIRを残差と合成Headroomへ含め、実Impulse／FRD再構成の1/6 oct Wavelet、Target -40 dBマスク、Shelf許可、低域優先を接続した
- Auto IIR実行時だけ、Targetの1/3 oct Gainが帯域内ピーク比−40 dB以上となる最初／最後の外側境界へStart／Endを一時調整。Speaker/InputとManual IIRはマスク判定に使わず、保存中のStart／Endを変更しない。診断の指定／実効補正範囲へ調整前後を表示する
- Auto IIRへ検証済み28/48マルチ解像度方式を接続。ピーク検出36点/octを維持したまま反復評価を28点/octへ減らし、Fc 0.5%・Gain 0.05 dB・Q 0.1へ量子化後に48点/octで再評価する。1/12 RMS 0.01 dB、1/3 ABS 0.01 dB、知覚ピーク0.01 dB、正ピーク面積0.02 dB-oct、ヘッドルーム、本数、Shelf構成のGuardを外れた場合は従来48点/octへ自動フォールバックする。現在の96 kHz／15～2500 Hz／7 PEQ設定では13.36秒から6.66秒へ50.2%短縮し、Guard合格を確認した
- Auto IIRの探索中に固定済みIIR応答と周波数カーネルを再利用し、候補応答を誤差評価とヘッドルーム評価で共有するよう高速化。探索回数、評価点、目的関数、最終再順位付けを変えず、8フィルターのStandard比較条件で27.47秒から15.24秒へ短縮した
- Auto IIRを1/1 oct音色、1/3 oct輪郭、正ピーク高さ／面積の3層評価へ更新。正ピークだけを追加優先し、周囲の音色を残す部分Cut、Wavelet最大1.5倍、広い凹凸候補の継続、BoostのQ 1.2制限、0.25 dBデッドバンド付き過補正Guard、1本除去後の再最適化による冗長削減を追加
- Auto IIRの個別Max Boost／Max CutとAuto PEQ Boost +3 dB上限を廃止。Manualを含む合成IIRの最大正Gainを`IIR headroom dB`で制限し、初期値+12 dB、0～120 dB／0.5 dB刻みで設定・保存可能にした。Legacy、Sobol／Optuna、Q量子化後、最終出力へ同じ制約を適用する
- Auto IIRのLo／Hi Shelfスイッチを必須生成ではなく候補許可へ変更。Shelfなし、Loのみ、Hiのみ、両方の構成でPEQの残り枠まで比較し、誤差面積、RMS、P95、正ピーク、下側面積、最大ディップをShelfなしより悪化させない構成だけを採用する
- DSP Systemの帯域分割を独立`CrossoverPlan`へ接続。全Outputの共有境界とSUBをWay画面の1画面で編集し、Linear Phase LR2、Kaiser FIR、主要IIR、Offを選択可能にした。FIRはProject補正FIRへ合成し、IIRは予測応答とCamillaDSP／miniDSP／SigmaStudio向けBiquad出力へ追加する。旧Systemは保存済みLR2／Kaiser設定から自動移行する
- Auto IIRへ実Impulse由来の1/3 oct Waveletピークプロファイルを接続。Impulseがない位相付きFRDは複素IR、GainのみFRDは最小位相IRを再構成してWaveletを生成し、使用した入力種別と信頼度を診断へ表示する。再構成不能時だけ従来の1/3 oct応答近似へ退避する
- DSP SystemのKaiser帯域分割で、Device tap不足時にMain 3.0／SUB 2.4の安全下限までCyclesを先に縮小。同一Mainクロス境界はLP／HPの小さいtap予算を基準に共通Cyclesを両側へ適用し、片側だけ遷移形状が変わる非対称を防止
- Target GainのLR2 HP / LPを純粋なLinear Phase LR2目標として維持し、`LinearFIRFilter.response="lr2"`を実係数生成へ反映。Kaiser FIRは`response="kaiser"`へ分離し、別系統のLR2独立モジュールを削除
- Auto IIRの最終共同最適化を、SciPy Sobol初期探索、Optuna multivariate TPE、SciPyロバスト最小二乗の3段構成へ変更。Legacyと探索上位候補を同じQ量子化・Fc/Gain再調整・範囲外Guardへ通し、実表示する1/1・1/3 oct ABS、8次ピーク、1/12 oct誤差面積／RMS／P95で再順位付けするため、探索量を増やしてもLegacyより悪い最終解を採用しない
- Auto IIRのRun／Clearを次回再実行の先頭で1回だけ処理するコマンド方式へ変更。IIRリスト差し替え時に旧Registry widget stateと計算Cacheを消去し、連続Run、Auto置換、Manual保持、Clear後の値復活を安定化

## v1.12.0 - 2026-07-25

- `DSP System`画面を追加。System、DSP、Way、Alignment、Filesを左画面で設定し、既存ProjectをHi／Mid／Low／Subへ接続して、元のSpeaker測定をSystem側のSmoothing、Gain Shift、Target条件で再処理する
- 2Way、3Way、4Way、3Way + Sub、上段96 kHz／下段48 kHzのTemplate、複数DSP、入出力Channel、System側Linear FIR帯域分割、Way Level／Delay／Polarity／Mute／Soloに対応する
- 各WayとSystem Sumを複素応答で合成し、右画面へGain、Phase、Wavelet、Impulse、Step、Group Delayを表示。DSP SystemではError、Response Advisor、Input detailsを表示しない
- Auto Level、到達ピーク基準Auto Delay、Polarity比較、Common FIR Auto Gain、異なるSample Rate／Clock／Latency同期警告、出力Headroom表示を追加する
- DSP System専用DB、System JSON保存・復元、接続Project Revisionの記録・更新／欠落検知、利用可能なWayだけでの再生成継続を追加する
- FilesからminiDSP、CamillaDSP、SigmaDSP／SigmaStudio向けDSP Package ZIPとManifestを出力。System名、用途、年月日・時分、短い重複回避IDを外側の出力名にも統一する

## v1.11.0 - 2026-07-25

- INPUT／TARGET処理をStreamlitから独立したPipelineへ接続。Mic Calibration、位相、帯域拡張、Smoothing、Gain Shift、Target編集の中間結果とRevisionをProjectへ保存し、旧Project／旧設定JSONをschema 3へ自動移行する
- DSP System向けに、複数DSP、Way別Sample Rate、Project参照、Linear FIR帯域分割、一時Config再生成、元Speaker測定とIIR／FIRを使ったWay応答、異なるSample Rateの複素合成、共通Auto Gain付き最終係数の独立APIを追加する
- DSP別Package出力を追加。CamillaDSPはConv WAVとIIR DiffEq YAML、miniDSPはチャンネル別float32 LE BIN／貼り付けTXT、SigmaDSP／SigmaStudioはFIR係数／逆順メモリ係数／IIR係数を機器別フォルダへ生成する
- IIR EQの個別DownloadとProject ZIPへCamillaDSP DiffEq YAMLを追加し、Generic JSON／CSV、miniDSP、SigmaStudioと同じBiquadから生成する
- Target Response入力へDirac Liveの`.targetcurve`と`BREAKPOINTS`形式TXTを追加し、FilesからEffective TargetをDirac Live形式で出力できるようにする
- 外部DSP向けファイル名をProject、Hi／Mid／LowなどのBand／Way、Channel、用途、対象DSP、Sample Rate、生成日時で統一。秒を表示せず、同一分内の短いIDで重複を避け、同時生成したZIP一式を識別可能にする

## v1.10.0 - 2026-07-25

- Detail viewでWaveletからErrorなどへ切り替えて戻った際、Preset／Bandwidth／Range／Wavelet Mode／Window Previewが初期値へ戻る場合がある問題を修正。表示中の一時Widgetと保存用の正本設定を分離し、再表示時は正本から復元する
- Clear Auto IIR実行後にAuto IIR操作部が閉じる問題を修正。開閉状態を保持する共通カードへ変更し、説明、生成結果、診断表、停止理由を日本語化
- Auto IIRへ初期値ONの`Prioritize LF for IIR`を追加。FIR tapsから求めた低域境界以下を緩く優先し、OFFでも信頼帯域判定とStart／End外の悪化防止は維持する
- Auto IIRの数値データ帯域と信頼帯域を分離。測定ESS帯域／IR分解能を保存・再利用し、旧FFTデータの根拠がない20 Hz未満や端点外挿を評価から除外。Start／End内を主目的、外側を補正前比Guardとし、境界直外のピーク／ディップに引かれないよう変更
- Auto IIRを帯域幅適応の専用対数軸と上位候補Shortlistへ変更。通常は上位2候補、ほぼ同点時だけ最大4候補を非線形最適化し、Analysis FFT sizeを一時増加させずに高解像度設定時の速度低下を抑制。診断へ指定／実効／信頼帯域、独立分解能、低域優先境界、実行時間を追加し、同一条件の結果を最大16件再利用
- Analysis FFT size初期値のマニュアル表を実装値の48 kHz:8192／96 kHz:16384／192 kHz:32768へ修正

## v1.9.1 - 2026-07-25

- IIR EQのQ入力、一覧、保存正規化を0.1刻みへ統一。Auto IIRは連続値で安定した候補を探索した後、Qを0.1刻みに確定し、Q固定でFrequency／Gainを再最適化する
- Auto IIR内部の周波数応答をSOS伝達多項式の直接評価へ高速化。推定初期値から正常収束した場合はマルチスタートを省略し、未収束、評価上限、非有限値、目的関数悪化時だけ別Q初期値へ退避する
- オクターブ境界処理のPython中央値走査をSciPyのベクトル化処理へ置換。Target Gain／Phase／IIR応答は同一周波数軸の共有計算へ統合し、Gain／Phase／FIRの数値結果を維持したまま重複計算を削減する
- Live Resultの工程別設計結果をSession内LRUで再利用。Gain EQ／Phase EQ／Auto Gain／Auto Phaseの差分表示は、前工程FIRを再設計せずTarget曲線だけを計算する
- 自動保存設定をschema 2へ更新。周波数特性は`data/databases/response_assets.sqlite3`へchecksum付きの不変assetとして分離し、`data/settings.json`は参照と軽量設定だけを保持する。同一内容の再保存を省略し、手動保存する設定JSON、Snapshot、Project ZIPは従来どおり応答を内包して単体で移動できる
- WaveletのComplex Morlet畳み込みを表示時間範囲とWavelet半幅に必要なIR区間だけへ限定。ModeごとにSource／Target mapを必要時だけ生成し、Range／Graph mode／Window Previewなどの表示設定を解析Cache keyから分離する
- Group Delayは通常のAnalysis FFT周波数軸でFIR係数とsample番号付き係数の2回の実FFTから算出し、任意周波数軸だけSciPy方式へ退避する
- Measurement resultsの1秒更新を軽量な測定状態監視へ限定。新shot、Gate／Center再処理、測定状態変更時だけ結果領域を更新し、停止後は結果選択や表示操作時だけ再描画する。Gain／Phaseの平滑化・間引き結果とImpulseの整列・ピーク保持間引きをMeasurement revision単位で再利用し、表示軸やLight／Interactive切替から解析Cacheを分離する

## v1.9.0 - 2026-07-24

- Gain EQ、Phase EQ、IIR EQ、Target Gain／Phase、Gain Tiltの`Add`／`Duplicate`後に、新しく追加・複製した設定を一覧で自動選択し、その編集欄を即時表示するよう統一。PresetとAuto EQも同じ動作とし、Auto EQ一覧では選択行の表示も同期する回帰テストを追加
- 同じ`data`設定領域を使うPhaseEQが複数起動した場合、上部警告へ別インスタンスのポート番号と`Stop port ...`を表示。停止直前にアプリ／設定領域／別PID／待受ポート／プロセス保持ロックを再検証して通常終了を要求し、現在操作中の画面や古いPID記録、別設定のPhaseEQ、無関係なプロセスは停止対象にしない。macOS／Linux／Windowsランチャーから実ポートもアプリへ通知する
- IIR EQのLive Resultを`Speaker/Input before IIR`、`IIR Filter`、`Speaker/Input after IIR`、`Target Response`の共通複素応答へ統一。Wavelet SourceがTargetへ退避する誤接続、Response AdvisorがPass FIRの誤差を評価する問題、Impulse／Step／Group DelayへFIR系列が混在する問題を修正し、IIR EQではLinear FIR EQ mask境界を表示しない
- Auto IIRをShelf先行の階層型最適化へ更新。1/3 octの局所ピーク除外済みトレンドでLo／Hi Shelfを先に判定し、残り枠を1/12 octの1山1PEQへ割り当てる。上限をShelf込みの`Max filters`へ統一し、近接する重複・逆符号PEQを抑制。`Start Hz`／`End Hz`は候補Frequency制約だけに使い、採否・共同最適化・冗長削除はInput／Target共通全帯域で行って範囲外のBiquad裾野も評価する。診断へ補正／評価範囲と範囲外誤差面積を追加し、全帯域の誤差面積、P95/P99、重み付きRMS、ヘッドルームを優先。Raw最大誤差は警告指標へ分離し、保存済み6フィルター例を回帰テスト化
- IIR filters一覧へQ列を追加。PEQ／Shelf／All Passは現在Qを小数3桁で表示し、Qを直接使用しないHP／LPは`-`で区別する
- IIRのQ上限をPEQは5、Lo／Hi Shelfは1に統一。手動UI、Auto IIR、保存設定の再読込へ同じClampを適用し、All Passは従来範囲を維持する。Auto Shelfは端部中央値による固定初期値をやめ、対数周波数上の重み付き絶対誤差面積が最大限減るFrequency／Q／Gain候補を選ぶ
- TargetとGain EQの間にIIR EQを追加。PEQ、Lo／Hi Shelf、All Pass、Linkwitz-Riley／Bessel／ButterworthのHP／LP、編集可能なAuto PEQ／Shelf候補に対応。IIR適用後のSpeaker/InputをFIR残差計算へ渡すが、IIR係数自体はFIRへ畳み込まず、Generic CSV／JSON、miniDSP、SigmaStudio BiquadとFIRを別出力する。Live Result、設定保存・復元、Project ZIP、係数安定性・符号規約・FIR分離の回帰テストも追加
- 将来構想を、Correction Planner / DSP Optimizer中心から、v2.xのMultiway Projectとv3.xのIIR / FIR Hybrid・試聴一体化へ変更。Project内の複数Channel、クロスオーバー、Delay / Polarity、複素応答による合成特性、Snapshot / A-B、組み込みDSP Profile、試聴から再調整までのWorkspace方針を明文化。既存Candidate Planは現行Auto Gain / Auto Phaseの内部処理と回帰テスト用途に限定する

## v1.8.0

- Inputを含むLive ResultのWaveletでBandwidthなどを変更した際、Widget側の一時状態が`Auto`の保存値を上書きしてOFFになる場合がある問題を修正。`Auto`は独立した正本状態を持ち、Autoチェックボックス自身の操作時だけcallbackで更新する
- Phase EQとTarget PhaseのHi Tiltに行選択表を追加し、2件以上のTiltを一覧から切り替えて編集できるよう修正。Value列へdeg/octを表示する。併せてTarget Editの各filter groupから重複する開閉カードを外し、group選択後は対象一覧を必ず表示する
- TargetでSource／Edit／Filesを切り替えた際、非表示になったfile uploaderと入力欄に残る旧URLの組み合わせで、読み込み済みTarget Responseが別データへ置き換わる問題を修正。読み込み成功時のTargetを正本snapshotとして保持し、URL変更時だけ再読込し、一時的な読込失敗時も現在Targetを維持する
- Linear FIR EQ maskのLo／Hi境界線をInputとTargetのLive Resultでは非表示にし、補正処理を確認するGain EQ以降のグラフだけに表示する
- Input、Project、Target、Gain EQ、Phase EQ、Auto Gain/Auto Phase、Linear FIR、Export、Measureを「左＝選択・設定・実行、右＝結果確認」の共通構成へ再整理。TargetはSource／Edit／Files、ExportはFIR Filter／Formats／Downloadsに分け、EQは全件重複一覧を廃止して選択中グループの1表と現在状態要約に統一。InputはSpeaker input／Mic Calibrationを1表で切り替え、`Apply selected`と`Clear current`を分離。Projectは現在Project／Input／設計条件を要約し、Linear FIRは現在Filter編集とPreset管理を切替。Measureは準備フロー、Start／Stop、Gate／Center、マイクGainを左上へ移し、右側を最新測定結果の常時表示へ統一。アプリ名／Workflow選択と左右ペインのタイトルもコンパクト化
- キャッシュ管理を`CacheCoordinator`へ統合。Design Result、Wavelet、Measurement Result、Audio capability、ESS analysis、External downloadを独立DomainとしてRevision管理し、変更イベントから依存Domainだけを無効化する。Live Resultの分散削除処理、Wavelet、ESS解析、Audio形式確認を共通Cache keyへ揃え、High precisionのPython object ID依存を測定Revisionへ置換。測定開始時はUIキャッシュに関係なくAudio形式を再確認し、旧`data/cache/loudspeaker_database`は`data/cache/external_specs/v1`へ安全移行する
- Page selectorを`Library → Input → Target → Gain EQ → Phase EQ → Auto Gain → Auto Phase → Linear FIR → Export`の通常フロー優先へ整理し、`Measure`を任意のContinuous ESSツールとして末尾へ移動
- 画面名を`Speaker DB`から`Library`、`Gain`／`Phase`から`Gain EQ`／`Phase EQ`、`Auto Gain EQ`／`Auto Phase EQ`から`Auto Gain`／`Auto Phase`へ統一
- 旧画面名で保存された設定やページ遷移を新名称へ正規化して読み込む互換処理を追加
- `Measurement DB`や`Speaker DB`などの通常操作向け表示を`Library`へ寄せ、`User specs`を`Speaker specs`へ統一
- ドキュメント一覧と読む順番を示す`docs/documentation-map.md`を追加
- UI文言、英語表示、日本語hover/help、短い名称の基準をまとめた`docs/ui_wording.md`を追加
- 開発・検証用依存関係を固定版の`constraints/test-environment.txt`へ集約
- macOS／Linuxの`setup_test_env_mac_linux.sh`を、オプションなしで依存関係、Playwright同梱Chromium、pre-commit、品質チェックまで実行するフルセットアップへ変更
- Windowsの`setup_test_env_windows.bat`でPlaywright同梱Chromiumを標準導入するよう変更
- Playwright UI確認は全OSで同梱Chromiumを標準使用し、必要時だけ環境変数でシステムChrome channelへ切り替えられるようにした
- Library > Measurementsの先頭へクイック登録を追加。Speaker/Input、Mic Calibration、Impedanceを種類選択、ファイル選択、`Save to Library`の3操作で登録し、保存後は該当レコードを自動選択して`Use in Input`へ続けられるようにした
- Continuous ESSの結果UIを、測定中は最新の有効Gain／Phaseを常時保持表示し、停止後は`Measurement result selector`表でStandard／Denoised／各Shotを選択してグラフ表示、DB登録、後処理対象を切り替える方式へ変更
- 測定準備フロー上部に`ESS再生タイミング`相当の状態案内を追加。カウントダウン／暗騒音中は「まだESSを再生しない」、Pilot待ちでは「ESSを再生してください」、正式shot中は「そのまま継続」を色付きメッセージで明示
- マイク測定の右画面をグラフ優先レイアウトへ再整理。測定操作、Gate / Center、マイクGainを上部の横並び操作バーへ圧縮し、Pilot詳細、測定判定、後処理を折りたたみ／ポップオーバーへ移して縦長構造を緩和
- Library / InputなどのDB一覧表から内部識別子の列を非表示化。選択・更新・削除は内部識別子で維持しつつ、ユーザー画面には表示名、条件、Created / Updatedを表示する
- 測定中に操作するGate start／Gate end／IR center polarity（Auto ±／Positive +／Negative −）をLive Result近接の`Live adjustment`へ集約し、極性変更時は必要に応じてRawからIRセンターを再捕捉して再解析するよう修正
- Stop後の旧`Measurement result selector`切替を廃止し、表の選択状態をLibrary登録候補にも反映することで、確認対象と保存・後処理対象がずれないようにした
- LibraryのSaved measurementsとInputのMeasurement selectionで、一覧行の選択だけで右側に保存特性を表示する`Selection preview`を追加。Gain縦軸は初期状態で選択曲線のピークへ合わせた40 dB幅とし、20／40／60／80／120 dBへ手動切替できる。`Not applied`を明示し、Use／Applyを押すまで現在設定を変更しない。現在Projectの全体結果を確認するProject DB、編集とLive Resultの連動を優先するTarget presets、単独曲線では判断材料にならないLinear FIR presetsはPreview対象外とする
- Live Resultの`Auto`がOFFでも、ページ移動で表示段階が変わった場合は必ず再計算する。Targetへ移動しても前ページの結果が残る問題を修正し、ページ内の設定変更だけを`Update`まで保留する
- Project DBの一覧選択と現在Projectを分離し、行選択だけでは保存・更新対象のProject IDを切り替えず、`Load selected`実行時だけ現在Projectとして復元するよう変更

## v1.7.0

- LibraryへSystem、Channel、Position、System state、Useを追加。スピーカーシステム全体のL/R測定を補正入力として明示保存でき、L+R combined／Verification onlyの測定は履歴には残しつつInputの補正候補から除外する
- LibraryのSpeaker specs通常保存は従来の重複整理を維持しつつ、Duplicate / Save as newで同じBrand / Model / Sourceでも明示的に別レコード登録できるUIへ変更。一覧にCreatedを表示し、削除は選択レコードだけをArchive化する仕様を明確化
- macOS／Linux／Windowsランチャーをoffline-firstへ変更。既存`.venv`の必須module、Streamlit 1.50以上2.0未満、`pip check`がすべて合格すれば、pip・Homebrew・apt・winget等へ接続せずlocalhostで起動する。不足時だけオンライン導入へ進み、`--update`指定時は明示的にpipと依存関係を更新。オフライン通常起動ではLinux日本語font不足も警告だけとして起動を継続し、Streamlit利用統計送信を無効化
- macOS CoreAudio／UMIK入力が`PaMacCore (AUHAL) err='-50'`後に終了通知を返さず停止する場合を、音声callbackの3秒生存監視で検出。終了通知あり／なしの両方で、旧streamを閉じ、PortAudioを再列挙し、機器名で同じ入力を再選択して最大3回自動再接続する。再接続後は途中Sweepを破棄してESS同期区間を更新し、完了Shot・番号・セッションを保持したまま測定を継続。失敗時だけ復旧可能エラーとして停止
- 測定前PilotのS/N条件をQuality gate連動へ変更。Strictは従来の中央値25 dB／p10 18 dB／80%、Standardは15 dB／6 dB／25%、Lenientは6 dB／0 dB／最小有効帯域とし、音量上昇案内を1回最大+3 dBへ制限。Standard／Lenientで同じ案内が3回続くか校正済み75 dB SPLへ達した場合は、最低品質を満たせば低S/N警告付きで正式測定へ進む
- 再生レベルの目安を75 dB SPL、80 dB SPL以上を黄、85 dB SPL以上を赤の警告としてPilot履歴と画面へ保存・表示。警告だけで測定をReject／停止せず、外部プレーヤーも自動停止しない仕様を明示
- UMIK-1の書込み可能なCoreAudio Gainを測定開始時に+24 dBへ初期化し、通常は固定。実クリップまたはヘッドルーム6 dB未満の場合だけGainを下げ、変更後は別Pilotで必ず再確認する
- マイク測定のFull IRをGated IRで確定した同じ小数sampleセンターへ時間整列し、Impulse、Gain／Phase、Waveletの0 ms基準を統一。Full IR内にGate外の強い反射や次ショット由来のpeakがあっても、Waveletだけが別の最大peakへ移動しないよう修正
- 連続ESSの各shot収録終端を、公称周期と直近実測周期の短い方から次のTiming marker／ESS開始位置と2 ms guardを差し引いて決定。次ショット混入見込みは収録を短縮して除外し、除外量と保護後capture長をアプリ診断へ記録
- Measureの測定操作列へ常時見える`ファイル保存`メニューを追加。測定中も完了済みshotまでのセッションZIPを明示生成して保存でき、Stop後は同じ導線から確定版を保存可能化
- Gate + Fullの自動合成が成立しないshotでは、壊れた合成値ではなくFullへ退避していること、現在Gateから決まる探索下限、Gate延長による再処理案を結果画面とshot診断へ明示
- 測定中のStop／AbortをFragment内のクリック戻り値処理からコールバックへ変更。測定開始後も設定画面に残っていた古い`running=False`ではなく、エンジンの最新状態からボタンの有効／無効を決定。満杯の音声キュー待ちとStandard確定をUIスレッドから分離し、録音済みブロックを保持したままStop要求を即時受理するよう修正
- 暗騒音測定の開始確認をダイアログから測定操作枠内へ変更。開始クリック後の同期sleepを廃止し、画面を止めずにカウントダウン表示・キャンセル・期限到達後の録音開始を行うよう修正

- マイク測定のImpulse表示で、直接音ピークが循環IR末尾にある場合もピーク後応答を全長展開し、2 ms付近で表示が途切れないよう修正
- マイク測定WaveletをWavelet Modeと同じComplex Morlet計算・色域・Peak／Centroid描画へ統合し、負荷時に省略された最新shotも表示時に自動計算
- 測定結果をGain / Phase、Impulse、Wavelet、アプリ診断タブへ整理。利用者向けグラフは常に最新shotへ追従し、Start／Stop／Abortを測定準備フロー直下へ移動

- IRセンター探索をIR全長の絶対最大peakから、ESS相関開始のFFT末尾20 ms（負時間）～`min(1周期−Sweep長, 250 ms)`（正時間）に限定。2shot目以降は直前センター±最大5 msで追跡し、マイク移動時だけ周期内で再捕捉。FFT末尾の直接音を負sample化し、Gateを末尾から先頭へ循環させて直接音後の欠落を修正。shot表へ追跡／再捕捉状態と探索sample範囲を追加
- 正式shot取得後もマイクGainを変更可能化。変更時は完了済みshotを保持してREADYを解除し、次のESSを非保存Pilotとしてクリップ／S/N／ヘッドルームを再確認。各shotの実Gainとセッション基準Gainとの差を複素応答へ逆補正し、異なるマイク位置・Gainのshotを同じレベル基準で統合
- Measureへ`IR peak — external ESS`（既定）と`Timing marker — external ESS`を追加。IR peak方式は外部再生周期とRaw peak移動を診断専用とし、正／負／自動極性で直接音を小数sample中心合わせしてGainと相対Phaseを保存。Timing marker方式はmarker検証済みTiming応答とpeak中心応答を分離保存
- 測定合格を基本／高精度の二段階へ変更。1 Accepted shotで基本測定とLibrary登録を許可し、3shot、coherence、repeatability、center条件は高精度判定へ限定。Clock条件はTiming marker方式だけに適用し、Capture／Gain／Relative phase／Timing／High precisionを独立表示
- IR peak方式では20 ms周期差と2 ms Raw peak移動によるhard rejectを廃止。中心合わせ後IR類似度をWarning／平均除外判断に使い、既知WAV相関の適応下限を0.20および直近中央値×0.55とした。帯域maskは音量調整・同期・診断だけに使用
- Gate + Fullの自動crossover下限を直接音後Gateの`1/T`とし、負側Gateを除外。信頼できるFull/Gated重複帯域がない場合の固定3 kHz fallbackを廃止し、Fullを維持してMerge unavailableを表示
- CaptureのStartが無効な場合に、ESS未選択、入力機器未検出、マイク非対応Sample rate、CoreAudio入力形式、絶対レベル校正値の各理由と修正先を表示。ESS Generator／ESS Library切替を追加し、生成・読込済みESSをSQLite DBへ永続保存して再起動後もWAV再読込なしで選択可能化
- ESS Generatorを常時stereo出力とし、Lのみ／Rのみ／L+Rを選択可能化。未指定chは完全な無音とし、stereo ESS読込時はESSを検出したchだけを候補に、両chにある場合は最初に検出したch（同時ならL）を測定基準へ採用。外部プレーヤーのファイルリピート間隔は不定として新しいtiming segmentを開始し、欠測数とclock drift回帰から除外
- ESS Generator v2のTiming marker検出後、予測開始位置の前後50 msを実ESS全文との相関で再検証・微調整する同期確認を追加。不一致時はESS全長を保持したbufferで従来相関へフォールバックし、Pilot履歴へ同期方式とESS correlationを保存。正式shotは周期誤差20 ms以内とDirect-IR peakの反復移動10 ms以内も検証し、誤同期を結果から除外
- 外部プレーヤーが宣言反復数の途中でファイル先頭へ戻っても、streaming buffer内の最初の相関候補を優先し、検証済みMarkerまたは強いESS全文相関を新しいtiming segmentとして即時採用。最初の同期後は既知周期の予測位置でESS全文をmarkerより先に照合し、短いmarkerのSweep内偽相関を抑制。その他の周期異常も1shot破棄後に同期anchorをリセットし、Last rejectionを成功後に解除して測定を自動復帰
- Pilotで自動抽出した有効帯域に対応するESS時間区間を同期検出専用referenceとして使用。低域・中域だけを再生するスピーカーでも全帯域ESSとの低相関による偽同期を避け、正式なGain／Phase／IRと保存周波数には帯域maskを適用しない
- 各shotで絶対IR peakを小数sample精度で0 msへ移し、Standard統合時にRaw IRの伝搬遅延を重ねて再付加していた処理を廃止。同期anchorが切れたshotは別timing segmentとして統合せず、Direct-IR移動許容を10 msから2 msへ厳格化
- `測定OK`を同一同期区間3shot以上、Median coherence 0.900以上、Repeatability P90 1.50 dB以下、残留center 0.50 sample以下、Clock residual 1 sample以下かつdrift 5000 ppm以内で明示判定し、未合格時のLibrary登録を停止
- 予測周期の探索窓がbufferへ完全到着する前には相関探索・fallbackを開始しないよう修正。最初の正式shotは相関0.35以上、その後は直近8shotの相関中央値の65%も満たす適応閾値とし、悪条件で約0.2の偽peakへ同期が移ることを抑制
- Accepted shot表へReliability scoreとFailure analysisを追加し、ESS相関低下、S/N 20 dB未満、IR center移動0.5 ms超、IR peakレベル変動6 dB超、連続周期未確認をshotごとに表示
- MeasureのGain／Phaseグラフへ上部凡例と固定説明を追加。青実線=Selected Ungated、オレンジ点線=Selected Gated、緑破線=Selected Merged、紫実線=Standard／Live preview Mergedとして表示
- ESS Generator／Marker変換の主DB音声をFLACとし、WAV・FLAC・対応manifest JSONを常に同一レコードへ登録。manifestなしで入力したWAV／FLACは、反復Sweep・周期・marker・PCM hashの解析結果からmanifestを自動生成してDBへ追記し、既存の音声のみのレコードも選択時に補完
- MeasurementのGate初期値をStart `-2.0 ms`／End `5.0 ms`へ変更し、Gate startを数値入力化。Gate / Mergeの再処理範囲は`Latest`を初期値とした。PhaseはRaw IRを保持したままAbsolute peakを0 msへ補正し、Gain軸はデータ追従の40 dB幅と10 dB単位の上下移動へ変更。結果ヘッダー、Gain／Phase、Impulse／Waveletの操作と表示寸法をLive Resultへ統一
- Measureの縦長設定を`Input / ESS`、`Calibration`、`Processing`、`Capture`、`Restore`のカテゴリー切替へ整理。PhaseEQ全体のページ選択・カテゴリー・英語単一選択ボタンへ日本語ホバー／キーボードフォーカス表示を展開し、英和・和英の双方向対応表、対象メニュー監査リスト、未登録・重複検出を共通化
- Page selectorやMeasureカテゴリーの日本語表示が横スクロール領域／再描画で見えない問題を修正。英語アクションボタンのhelpから不要な`日本語名:`接頭辞を削除し、Measureの6カテゴリーを枠付き表示へ統一
- MeasurementのStartで開始条件を表示し、3／5／10秒から選んだ遅延時間のカウントダウン後に5秒間の暗騒音測定を開始する操作を追加
- メーカー公式校正アシスタントをminiDSP、Dayton Audio、Sonarworksへ共通化。公式ページでSubmitした後のダウンロードフォルダーを5分間監視し、UMIK 0°／90°、OmniMic OMM、UMM-6／EMM-6 TXT、XREF 20 ZIP内0°／30°／90°を検証・自動登録・個体リンク。既存ファイルを除外し、手動選択へフォールバック可能
- Dayton Audio UMM-6をMicrophone DBへ追加し、実ファイル互換のためUMM-6／EMM-6シリアルは4～6桁を受理

- Measurement開始時に5秒の暗騒音測定と非保存Pilotを追加。音量調整専用のスピーカー非依存maskでS/N中央値25 dB／p10 18 dB／合格帯域80%を先に確保し、その後に制御可能なUSBマイクGainを10 dBヘッドルーム目標／6 dB最低へ自動調整して再Pilotする。maskと調整前PilotはRaw・グラフ・Standardへ適用・混在させず診断履歴だけを保存
- Measurementの準備状況を暗騒音／スピーカー音量／マイクGain／最終確認の4段階で表示し、自動調整後のCoreAudioマイクGainを絶対dB値で手動微調整できるようにした。手動変更後はREADYを解除し、別のPilot ESSが合格するまで正式測定へ進まない

- UMIK-1／UMIK-2の公式Product Brief情報をMicrophone DBへ追加し、UMIK-2を48／96／192 kHz対応へ更新。UMIK-1は校正時20 Hz～20 kHz ±1 dB、UMIK-2は公開精度未指定として区別
- miniDSP公式個体校正のSens Factorを、0°／90°・SERNO・UMIK-2 AGain・CoreAudio入力Gainが整合した場合だけManufacturer calibrated SPLへ自動適用。測定中のGain変化ショットを除外し、代表感度はEstimated dB SPLとして分離
- Mic CalibrationのGain／Phase列をマイク実測偏差として複素応答から減算するよう修正し、UMIK／OmniMic校正の符号をメーカー校正ファイル定義へ整合
- MeasureへUMIK-1／UMIK-2のminiDSP公式校正導線と0°／90°一括登録を追加。製品シリアル、ファイル内`SERNO`、角度、`Sens Factor`、周波数応答を検証し、元TXT・取得元URL・SHA-256をLibraryへ保存してマイク個体へ自動リンク

- Measurementへマイク距離と絶対レベル校正を追加。USBマイクのdBFS/Pa感度、アナログマイクのmV/Pa＋I/F 0 dBFS電圧＋入力Gain、音響校正器、音圧計比較に対応し、校正音の1秒dBFS RMS取得を追加
- 実ESSのPeakを反映したマイク入力dBFS RMSと校正済みdB SPLをGainグラフで切替表示。セッションZIPへ`level_*_dbfs.frd`／`level_*_spl.frd`、Libraryへ単位・距離・校正根拠を保存
- Measurementの`Gate start ms`でImpulse peak基準の負値（-100 msまで）を数値入力できるよう修正。Gate / Merge再解析時はshot数とCombined Raw IRを保持し、shot NPZ／manifestへ反映したうえでStandardを自動再確定
- MeasurementのGainが既知ESSに対する相対伝達ゲインであることを明示し、自動追従する固定40 dBスパンと10 dB単位の上下移動、表示専用のOff〜1/3 oct Gain smoothing（初期値1/12 oct）を追加
- MeasurementのQuality表示で未定義の`phase_frame`を参照して停止する問題を修正し、Phaseグラフを本来のAnalysis Gain / Phaseへ戻してQualityをCoherence／Confidence／Repeatabilityだけに分離
- UMIKなど複数の校正角度を持つマイクで、初回表示時の`Calibration angle`が未初期化の`None`になり停止する問題を修正
- PhaseEQ起動後に接続したUSBマイク／オーディオインターフェースを認識できるよう、測定停止中にPortAudioを再初期化する`Refresh input devices`を追加。既知ESSモードでは再スキャン後のOmniMic／UMIKを優先選択

## v1.6.0

- Linear FIR presetへEQ mask smoothingを統合し、周波数フィルターなしの`Pass` presetを追加。preset適用時にlayoutとdomain backupを同時更新し、不要な`02 Linear FIR`見出しと個別入力カードを削除
- Linear FIRページへ移動した際、非表示中にStreamlitが`linear_fir_items`を除去するとlayoutが`None`へ変わる問題を修正。正本の独立バックアップから復元し、明示的に`None`を選んだ場合だけLinear FIRを解除
- ソフトウェア名を `PhaseEQ` へ戻し、画面タイトル、起動表示、入口ファイル、Pythonパッケージ、ESSマニフェスト、Project ZIP、現行ドキュメントを統一

- `Measure`ページへ外部ESSの連続録音、相関検出、デュアルゲインL/R合成、ショット単位解析、自動保存、Stop後のGate/Merge再解析、複素平均、Library登録を追加
- 測定停止・入力エラー後も確定済みショットを保持し、測定セッションZIPの保存と再読込に対応
- `Measure`ページへの移動で補正設計側のLive Result表示ステージや表示設定が変わらないよう、直前の設計コンテキストを退避・復元
- `Measure`ページのGate／Merge／平均／保存Widgetで、Session Stateとdefault/valueを二重指定するStreamlit警告を解消
- 1chのシステム既定入力を2ch測定へ誤採用しないよう、Stereo input deviceを実デバイス選択へ変更し、エンジン側にもチャンネル数検証と2chフォールバックを追加
- OmniMic／UMIK-1／UMIK-2向けに、ナレーション付きCDトラックWAVから反復スイープの実波形と周期を自動抽出し、ゲイン違いL/Rをステレオ収録して音圧とヘッドルームに合うチャンネルを選ぶ既知ESSモードを追加
- マイク機種の対応Sample rate・公称bit深度・チャンネル数と、所有個体の製品シリアル・Mic Calibrationリンクを分離管理するMicrophone DBを追加
- USB記述子の汎用シリアル値を校正識別へ流用せず、登録した製品シリアルと校正データ名／ファイル名等が一意に一致した場合だけ自動選択
- UMIK-1／2の`3桁-4桁`シリアルと0°／90°校正、OmniMicの7桁シリアル・0°校正・48 kHz Phase校正をMicrophone DBとMeasureへ追加
- アナログ1chのEMM-6（数字5桁）／XREF 20（英数6桁）を追加し、オーディオインターフェースと物理入力chを指定してmono収録できるよう対応
- 実校正ファイルに合わせ、OmniMic `.omm`の同一ファイル内Phase列、UMIKのハイフン有無を無視したシリアル照合、XREF 20の0°／30°／90°校正をサポート
- NumPy 2.5とsounddevice 0.5.5の組み合わせで録音callbackごとに出る既知のshape代入DeprecationWarningだけを限定抑制
- `Measure`のMic calibration／Calibration angleへ丸いヘルプアイコンを追加し、Libraryでの校正ファイル登録手順を表示
- 完了済みマイク測定結果を新規Startまたはセッション読込で表示置換する場合、保存済みでも必ず確認してからクリアするよう変更
- マイク測定結果へLive Result準拠のGain／Phase、Light／Interactive、表示点数、Impulse／Wavelet Detail表示を追加
- 複数ショットをサブサンプル整列し、S/N・相関・周波数別外れ値を考慮した位相保持の複素統合から「高精度1ショット」を生成するHigh precisionモードを追加
- 48／96／192 kHz・24-bitの反復ESSをWAV／FLAC／manifestへ出力するESS Generatorを追加し、生成ファイルのPCM hashと正確なshot位置を再利用可能化
- Stop後に反復周期からDAC/ADC clock driftを推定し、信頼できる場合だけRawを時間伸縮して再deconvolutionするStandard処理を追加
- Raw／Standard／Denoisedを分離し、1shotのCoherenceをN/A、9shot以上を最良連続5～8shotで統合する測定回数別処理を追加
- Calibration範囲外をHold edge／No correction／Manual extensionから選べるようにし、20 kHzやCalibration上限によるNyquistまでの測定・FIR補正停止を行わない仕様を明示
- 測定中のHigh precision／Running選択を自動Live previewへ統合し、Stop後はClock補正済みStandardへ自動切替する単一ワークフローへ変更
- 一次Quality gateを追加し、利用不能shotは配列・グラフ・NPZ・FRD・Standardへ残さず理由別counterだけを保持。既定Raw保存をAccepted shots onlyとし、全連続WAVは任意のForensic archiveへ分離

## v1.5.1

- Linear FIR preset一覧にHP / LP別のCycles・Betaを追加し、Preset編集欄から個別に追加・更新できるように改善
- Linear FIR preset適用直後にCycles・Beta編集欄が最小値へ戻るStreamlit再実行問題を修正
- EQ mask smoothingなど無関係なWidgetの再実行で、保存済みLinear FIRレイアウトが`None`へ戻る問題を修正
- 再起動時のSpeaker/Input測定、Target Response、Target編集設定を保持するresume回帰テストを追加
- 複数画面・複数プロセスの古いセッションが最新の自動保存設定を上書きしないよう、settingsファイルへrevision検証付きatomic writeを追加
- Streamlit AppTestの設定、DB、User presetをテスト専用データ領域へ分離し、実利用データを変更しない品質チェック環境へ改善

## v1.5.0

- 実行時設定、DB、snapshot、user preset、cache、log、tmpを `data/` へ集約し、旧ルート配置をSQLite sidecarごと安全に自動移行
- Project、Library、Input、Local snapshotsの選択を一覧表のチェック選択へ統一し、読込・複製・編集・削除を一覧直下へ配置
- Target、Gain、Phaseの全フィルター一覧をチェック選択式の唯一の編集入口とし、重複するSelected filter表示を削除
- Target ResponseとLinear FIRにBuilt-in / User分離型JSON preset管理を追加し、追加、更新、複製、削除、配布用への昇格、Userへの降格に対応
- Project、snapshot、preset読込時は保存設定を正としてWidget mirrorと非永続layout memoryを一括更新し、Linear FIRのHi / Mid / Loや周波数が古い画面状態へ戻る問題を修正
- Gain / Phase共通の境界処理を、ロバスト傾き推定、制約付きHermite、破綻時のsmootherstepに統一し、Hi / Lo外側端でも入力側の傾きを考慮
- Phase EQ mask Lo側に30 deg/oct保守案と60 deg/oct候補の動的選択を追加し、意地悪データを含むDAシミュレーションと回帰テストを整備
- Gain / Phaseの選択ボタン、主要操作、見出し、編集中カードを機能系統色で統一し、Light / Darkの選択状態とコントラストを改善

## v1.4.0

- Target、Gain、Phaseの全フィルター一覧を編集入口へ統一し、ネイティブな単一行チェック選択から対応する編集モードとカードを直接開く構成へ変更
- Project、Library、Input、Exportを一覧＋segmented controlで再編し、InputをSource、Phase / Timing、Response Processing、Diagnosticsの4ブロックへ整理
- 全メニューの一覧表示をライブWidget値から導出し、設定変更が一覧へ1回遅れて反映される問題を修正
- Auto Gain／Auto Phaseのセクション編集を一覧＋選択中編集へ統一し、Strength、Max Boost／Max Cut、旧Boundary smoothing方式選択UIを削除
- Auto Gain/Auto Phaseの重複領域を、通常編集では操作対象を優先して競合先をOFF、Project読込時は競合参加セクションを全件OFFとして警告する仕様へ統一
- Gain／Phase共通のオクターブ境界スムージングライブラリを追加し、ロバスト傾き推定、制約付きHermite、破綻時の幅拡張／傾き緩和、smootherstepフォールバックを実装
- Auto Gain／Auto Phaseの共有境界を一括処理し、旧Logistic／preserve-edges境界ライブラリを削除
- Auto Gain/Auto PhaseとHi TiltのNyquist処理を仮想延長＋Hi側境界なしへ統一し、Auto PhaseがNyquistで0°へ収束するデグレを修正
- Gain／Phaseグラフの系列、色、線種、Current page delta表示を整理し、設定のない補正線やNyquist端点の誤表示を修正
- Linear FIRの全フィルター一覧、Midレイアウトの周波数正規化、一覧と編集欄の同一rerun同期を追加
- Light／Darkテーマを温かいアイボリーと深いグラファイト基調へ更新し、表ヘッダー、背景、選択状態の視認性を改善
- 独立実行可能な境界スムージングデモ、運用ドキュメント、UI／DSP／設定互換性の回帰テストを追加

## v1.3.0

- 上部のPage selectorと36:64のEdit / Live Result構成を維持しながら、左編集領域の情報密度を再編
- InputページをSource、Phase / Timing、Response Processing、Info / Diagnosticsの順で整理し、補助設定を初期状態で閉じる構成へ変更
- TargetページにGain / Phase切り替え、Gain / Phaseページにフィルター群切り替えを追加し、表示中の編集対象を限定
- 各ページへ設定レジストリから生成するコンパクトな状態サマリーを追加し、非表示側のフィルターも計算・保存対象として維持
- ヘッダーのProject概要表示、Project ZIPとMeasurements / Speaker specs JSONのバックアップ範囲を明確化

## v1.2.12

- Export出力直前の最終FIR係数へ、固定仕様のFIR Auto Gainを追加
- FIR Auto GainはTarget FIR Peak -0.3 dB、Attenuation only、FFT oversampling 64固定とし、最大応答が-0.3 dBを超える場合のみ全係数へ同一倍率を1回だけ適用
- 個別FIR出力、Project ZIP、Export FIRプレビュー、Project ZIP manifestにFIR Auto Gain結果を反映

## v1.2.11

- macOS / Python 3.13環境で、Streamlitの表表示時にPyArrowのmimalloc allocatorでネイティブクラッシュする可能性があるため、アプリ起動時にPyArrowのmemory poolをsystem allocatorへ切り替える対策を追加

## v1.2.10

- Library > Speaker specsのRaw pasted text欄に、AI生成用のJSONプロンプトをアプリ内ヘルプとして追加
- ユーザーガイドへRaw pasted text AI生成プロンプトを追記し、仕様書から標準テンプレートとして参照するように更新

## v1.2.9

- Auto Gain / Auto Phaseの隣接セクション境界をロジスティック遷移weightで接続し、接触境界で段差が残る問題を修正
- Auto smoothingを補正カーブ生成後ではなく、Target / Speaker/Input + Manual / Manualのソース側へ適用する流れへ変更
- Multi Section weight、外側端transition、狭い境界幅の自動縮小を扱う `logistic_transition` モジュールと境界条件テストを追加
- Project共通のSample rate、FIR taps、Analysis FFT sizeを保存用settings値からProject入力widgetへ同期し、リロード後に古い `48000 Hz / 63 taps / 2048 FFT` が入力欄へ戻る問題を修正
- README、ユーザーガイド、アプリ仕様、将来構想のAuto Gain/Auto Phase処理フローとバージョン表記を更新

## v1.2.8

- Speaker/InputのInvert before center、Phase centering、Phase handling、Speaker/Input smoothing変更時に、Live Resultが即時更新されない問題を修正
- Live Result更新判定を前回UI状態との差分で一元管理し、処理に影響するUI変更ではキャッシュを破棄して再計算するように変更
- FIR tapsの読取キーを `taps_widget` に統一し、ブラウザ再読み込み後に古い `48000 Hz / 63 taps` 表示へ戻る問題を修正
- Wavelet Gain Shiftの品質判定で全NaN行がある場合の `All-NaN slice encountered` 警告を抑制
- Target Response編集のみがある場合のAuto Gain表示判定を修正し、Target未設定扱いにならないように変更

## v1.2.7

- Save current settings / Project DB保存時に、widget由来の最新設定を保存payloadへ同期し、古いUI値と新しいconfig値が混在する問題を修正
- Project DBの保存メタ情報を最新widget値から取得するように変更し、Project名、Speaker、Side、Position、Tags、Noteの保存遅れを抑制
- MeasurementsのSpeaker spec referenceでApply specを押したとき、Measurement nameが空欄または自動生成名の場合だけSpeaker spec由来の名前へ更新するように変更
- Measurement nameの自動更新条件をUIヘルプ、ユーザーガイド、仕様書へ追記

## v1.2.6

- Project設定変更後のDesign conditions、Live Result、Export確認表示の再描画同期を一元化
- 新規UI設定追加時の `_read_scalar_ui_state()` 登録ルール、1回遅れ防止、テスト観点をAI開発ポリシーと仕様書へ明記
- ProjectのFIR tapsで63未満または偶数を入力した場合に、軽い通知を表示して有効な奇数tapへ補正するように変更
- 通常Gain EQからLin. HP / Lin. LPを削除し、Target Gain EQのLR2 HP / LR2 LPへ役割を分離
- Target Gain EQのLR2 HP / LR2 LPを素のLinkwitz-Riley式へ統一し、Kaiser近似LRの探索コードとドキュメントを削除
- Target Gain EQカードの表示名を通常Gain EQと分離し、LR2 HP / LR2 LPが通常Gain EQに見えないように変更
- Target Gain EQ / Gain EQのShelfにGain入力を追加し、Gain Lo Tiltを追加
- Gain Tilt / Phase TiltをHi Tilt表記へ整理し、既存設定はHi Tiltとして互換維持

## v1.2.5

- Project共通のSample rate、FIR taps、FIR length、Analysis FFTをヘッダー直下へ表示し、変更先をProject > FIR / Analysisへ誘導
- Export前のDesign conditions確認表示を追加し、Project設定とExport条件の確認経路を整理
- Page selectorと誘導ボタンの状態同期を見直し、Project settings、Library Measurements、Linear FIRなどへのページ遷移を安定化
- Speaker/Input measurement setにNear-field woofer / portを内包する構成へ整理し、InputからMeasurements選択でセット適用できる導線を強化
- MeasurementsからMic CalibrationをInputへ適用し、未使用時はNone / clearで明示的に外せるUIへ整理
- Project DB一覧とCurrent Input measurementsで、Inputに適用したMeasurement参照を確認しやすく改善
- 表示用テーブル/ラベル、Project ZIP/Export処理、Measurement payload処理をutilsへ分離し、メインプログラムの責務を縮小
- Library、Input、Project、Exportのヘルプ文とドキュメントを、Measurements起点の基本フローへ更新

## v1.1.5

- Linear FIR EQ maskを再導入し、手動Gain / Phase EQ、Auto Gain、Auto Phaseの遮断領域補正を制限
- EQ mask境界をLinear FIR単体応答から検出し、Gain / Phase / Error / Group Delayの周波数グラフへ境界線を表示
- Auto Gain / Auto Phaseの候補表示とCurrent page deltaを、mask適用後の補正特性に統一
- Gain maskの外側フェードを符号に関係なく6 dB/octで0 dBへ戻す仕様へ変更
- Phase maskのLo側外側フェードを15 deg/octへ変更し、unwrap処理を前提に整理
- Target Gain EQにTarget Response専用のLR2 HP / LR2 LPを追加し、通常Gain EQとは分離
- Interactive表示でImpulse / Step応答のtap軸拡大ができない問題を修正
- LibraryをSpeaker specsとMeasurementsの2系統へ整理し、外部WEB/PDFからのSpeaker specs取得補助と測定履歴管理を追加
- MeasurementsにSpeaker/Input、Mic Calibration、Near-field woofer / port、Impedanceの登録導線を追加し、Near-fieldもFRD/TXT/CSV/WAV入力へ対応
- Inputページの直接Speaker/Input / Mic Calibrationファイル入力をMeasurementsへ統合し、InputではLibraryから選択する導線へ整理
- Project DBへInputで適用したSpeaker/Input / Mic Calibration / Near-field woofer / portのMeasurement参照を保存し、Project一覧へ表示
- 再配布リスクを避けるため、Loudspeaker Database由来DBの取得・管理機能を削除

## v1.1.3

- Page selectorに `Linear FIR` を追加し、Auto Phase後からExport前にLinear FIRを直接加算できるように変更
- Target ResponseのLin. HP / Lin. LPを追加
- Gain EQのLin. HP / Lin. LPはLR2固定へ整理し、LR4を使用不可としてUIと設定復元を更新
- Live Result、Error、Correction Filter、Speaker + Realized、Export FIRの表示段階をLinear FIR工程に追従
- Wavelet、Response Advisor、FIR feasibility、開発環境、Git / GitHub、AI開発ポリシー関連のドキュメントと検証を整備

## v1.1.2

- Export FIR stageのWavelet Source / Deltaを、出力FIR反映後のSpeaker + Realized基準へ変更
- Speaker/InputのLF / HF extensionを追加し、HF gain / phase補完とLF slope extrapolationに対応
- Phase Centeringの探索範囲を自動設定に変更し、元IRまたは再構成IRのピーク周辺を使う仕様へ整理
- Gain Shift Autoを最大Gainから-15 dB以内の主帯域基準へ変更し、低域ノイズに引っ張られにくく改善
- Wavelet解析のconfidence / reflection / spreadをGain ShiftやPhase Centeringの品質判断に活用
- Analysis Gain / Phase表示の軸レンジと表示前間引きを改善し、大きなWAV入力時の表示待ちを短縮
- macOS / Linux起動スクリプトのPython / venv / 日本語フォント環境チェックを強化

## v1.1.1

- 未使用の `soundfile` 依存関係を削除
- WAV入出力はSciPyの `scipy.io.wavfile` で行う仕様として整理
- 開発・検証用依存関係にPlaywright / pytest-playwrightを明記
- インストールガイドと配布手順の依存関係説明を更新

## v1.1.0

- Live ResultのDetail viewにComplex Morlet Wavelet解析を追加
- Source / Target / Source - Target表示に対応
- Wavelet Sourceを表示ステージから自動選択し、Auto update / Update now / Graph modeを通常グラフと同じ制御へ統合
- WaveletのInteractive表示をPlotly Heatmapへ変更し、拡大縮小時の応答性を改善
- Wavelet grid解像度をGraph pointsに連動し、最大1024 x 2048 gridへ変更
- Wavelet用AlignmentとしてBand-limited Cross Correlation + Fractional DelayとIR Peak Abs fallbackを追加
- Waveletの低域表示が不自然に落ちないよう、周波数ごとの表示正規化を改善
- Waveletに各周波数ごとのピークトレース線を追加

## v1.0.9

- Project DBを追加し、複数Projectの検索、ソート、追加、編集、複製、削除、素早い復元に対応
- Project DBは設定と読み込み済み応答データを保存し、出力FIR係数は保存せず復元後に再生成する仕様にしました
- Speaker/Inputでmono WAVインパルス応答の読み込みに対応
- Speaker/Input Phase Centeringを追加し、インパルス応答由来のピーク位置から位相中心を調整できるようにしました
- Delay Adjustmentメニューを削除し、Phase Centeringへ整理
- 左側Navigationの選択状態を設定保存・復元の対象に追加
- Live Resultの表示ステージとDetail viewを設定保存・復元の対象に追加
- Target Edit / Gain / Phase / Auto Gain/Auto PhaseのEQリスト操作を、APPY UI State Registryの階層化listを正とするAdd / 行削除 / ソート方式へ変更
- EQ item名を `EQ1 | 周波数 | 主要パラメーター` 形式へ変更し、同じ周波数でも識別しやすくしました
- Gain EQ / Phase EQの1周波数系フィルターを統合表示し、全件を表形式で編集できる構成へ変更しました
- Auto Gain/Auto Phaseセクションを開始周波数の低い順へ自動ソートするように変更

## v1.0.8

- Streamlitの依存バージョンを `>=1.30,<1.51` に制限
- 配布スクリプトの `--upgrade` 実行時に、互換性未確認のStreamlit新バージョンへ上がりすぎないようにしました

## v1.0.7

- 配布起動スクリプトで依存関係を `pip install --upgrade -r requirements.txt` に変更
- 既存 `.venv` でも古い依存関係が残りにくいように起動手順を更新

## v1.0.6

- 配布環境のStreamlitで `st.altair_chart(..., width="stretch")` が未対応の場合に起動できない問題を修正
- 設定読み込みプレビューの `st.dataframe(..., width="stretch")` も互換APIへ戻しました

## v1.0.5

- 配布環境のStreamlitで `st.tabs(..., key=...)` が未対応の場合に起動できない問題を修正
- Live ResultのResponse / Time Analysisタブを、Streamlit 1.30以降で動く互換APIへ戻しました

## v1.0.4

- macOS / Linux用の即実行スクリプト `run_PhaseEQ_mac_linux.sh` を追加
- Windows用の即実行スクリプト `run_PhaseEQ_windows.bat` を追加
- 起動スクリプトで仮想環境作成、依存関係インストール、空きポート選択、ブラウザ起動まで行えるように整理
- 起動スクリプトの起動ファイル名を `phaseeq.py` に統一
- 起動スクリプトのPython要件をアプリ仕様と同じPython 3.9以降に統一
- README、インストールガイド、配布ファイルビルド手順に起動スクリプトの使い方を追加

## v1.0.3

- アプリバージョンを1.0.3へ更新
- Nyquist超え周波数応答行を処理対象外にする仕様を README / 仕様書 / ガイドへ反映
- 元データは設定JSON / Project ZIP用に保持する説明を追加
- Target Response作成は Gain / Phase ではなく Target Edit と明確化
- Auto Gain が Speaker/Input + Manual Gain EQ 後の残差を補正する説明へ修正
- Phaseタブ説明をAutoタブ前へ移動し、構成を整理
- Exportタブ説明を後半へ移動し、内容を現行仕様に修正
- Window設定の説明を削除
- 用語を Speaker Response から Speaker/Input に統一

## v1.0.2

- アプリバージョンを1.0.2へ更新
- 周波数応答データのNyquist超え行を処理対象外にするガードを追加
- Auto GainがGain EQ後の結果を入力として参照するように修正
- 表示ステージでPhase/Gain Tilt total smoothingが反映されるように修正

## v1.0.1

- アプリバージョンを1.0.1へ更新
- Analysis FFT sizeの初期値を48kHz:4096 / 96kHz:8192 / 192kHz:16384へ変更
- 未使用のWindow関連設定を左側UIから削除
- Window関連値は旧設定JSON互換のため内部設定として維持
- Tilt total smoothingの不要実行を抑制
- fractional octave smoothingを高速化
- タイトル下のバージョン/作者表示の重なりを修正

## v1.0.0

- アプリバージョンを1.0.0へ更新
- Editor / Live Resultをウィンドウ風に分離
- Edit側を独立スクロール領域に変更
- Live Resultは表示ステージ切替を固定し、結果表示エリアのみスクロールする構成へ変更
- Speaker/InputにHigh Tail compensationを追加
- High Tail compensationをDelay Adjustment後、Speaker/Input smoothing前に適用
- High TailはGainのみを対象とし、Phaseには適用しない仕様に整理
- Auto GainにLimit modeを追加
  - Full correction
  - Peak/Dip only, preserve Tilt
- Peak/Dip onlyでは広域Trend/Tiltを保持し、局所ピーク/ディップ成分へMax Boost / Max Cutを適用
- Auto GainにPost limit smoothingを追加
- Auto Gain/Auto Phaseセクション設定の保存/復帰に新しいLimit設定を追加
- システム全体と各処理セクションのPlantUMLフロー図を追加
- pytest検証を拡充

## v0.1.4

- アプリバージョンを0.1.4へ更新
- Target ResponseのNyquist以上の扱いを、Nyquist手前の最後の有効値保持へ統一
- FIR生成とError評価に使うTarget Gain / Target Phaseも、Nyquist点で直前値保持へ統一
- Auto Gain / Auto Phaseのedge smoothがNyquist端を0 dB / 0°へ戻さないよう修正
- Phase Nyquist Guardを削除
- FIR設計方式をWeighted Complex LSへ一本化
- ErrorグラフにGain Error / Phase Errorの凡例を追加
- Speaker/Input、Target Response、Auto Gain/Auto Phase、ExportまわりのUIと仕様説明を更新
- pytest検証を91件へ拡充

## v0.1.3

- アプリバージョンを0.1.3へ更新
- Speaker/Inputにpolarity invertを追加
- polarity invertをphase handling、Delay Adjustment、unwrap処理より前に適用
- Time / AnalysisタブをImpulse Response / Step Response / Group Delayの切替表示へ変更
- Time / AnalysisにSpeaker/Input、Target Response、Correction Filter Realized、Speaker + Realizedを表示
- Group DelayはGroup Delay表示を選んだ時だけ計算するように変更
- Exportタブを編集側の右端へ移動し、出力形式、DC Gain Normalize、ダウンロード操作を集約
- Output FIR polarity invertを追加し、書き出すFIR係数だけ逆相出力できるように変更
- ExportタブにSpeaker/Input polarity invert、Output FIR polarity invert、DC Gain Normalizeの状態表示を追加
- README、操作ガイド、仕様書を更新

## v0.1.2

- アプリバージョンを0.1.2へ更新
- Project name、出力形式選択、Project ZIP一式出力/復元を追加
- Target ResponseタブからGain Shift後のTarget Responseを直接ダウンロード可能に変更
- DC Gain Normalizeをサイドバー設定からON/OFFできるように変更
- Speaker/Input smoothingにComplex Gainモードを追加
- Speaker/Input / Auto Gain/Auto Phase / edge smooth系の平準化選択肢に1/1 octを追加
- UIの基本設定、出力設定、Live Result設定の並びを整理
- README、操作ガイド、仕様書、出力仕様を更新
- pytest検証を72件へ拡充

## v0.1.1

- アプリバージョンを0.1.1へ更新
- Group DelayをTime / Analysisタブへ分離し、必要時のみ計算するように変更
- Responseタブのライブ更新時はGroup Delay計算を省略し、操作レスポンスを改善
- FFT処理をSciPy FFTへ寄せ、Analysis FFT sizeの実効値を高速FFT長へ最適化
- README、操作ガイド、仕様書、配布手順をv0.1.1に更新
- Core APIに遅延Group Delay計算用の `calculate_group_delay_ms()` を追加
- pytest検証を70件へ拡充

## v0.1.0

- PhaseEQとして初期実装
- Speaker/Input読み込み
- Mic Calibration読み込み
- Target Response読み込みと出力
- Speaker/Input / Target ResponseのGain Shift
- Gain PEQ / Shelf / Tilt
- Phase PEQ / Shelf / Tilt
- Auto Gain / Auto Phase
- FIR生成、解析、WAV / TXT / CSV / BIN出力
- 現在設定の自動保存、次回起動時の復元、スナップショット管理
- 日本語README、操作ガイド、仕様文書を追加
