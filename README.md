# PhaseEQ

リリースバージョン: **v1.20.8**

文書更新日: **2026-09-29**

PhaseEQはスピーカーの測定応答から、目標特性、IIR／FIR補正、帯域分割を設計するアプリです。マルチウェイ画面では各チャンネルを合成し、位相・時間整合を調整して実機DSP用の係数を出力できます。

## 起動する

- macOS／Linux: `./run_PhaseEQ_mac_linux.sh`
- Windows: `run_PhaseEQ_windows.bat`

ランチャーがPhaseEQとマルチウェイを起動します。通常の起動では、依存関係が揃っていればネット接続は不要です。初回導入や更新は[インストールガイド](docs/distribution/installation.md)を参照してください。

Windows 11 ARMでは、x64エミュレーション上のx64 Pythonを使用します。検証対象はPython 3.12／3.13です。

## 設計を進める

1. 「スピーカー測定データ」で測定を登録し、「応答処理」で補正入力を準備します。
2. 「目標特性」を選び、「IIRイコライザー」「FIRイコライザー」で補正します。
3. 必要に応じて「出力帯域分割ページ」を設定します。
4. 「書き出し」でFIR ON／OFFとタップ数、FIR・IIRの合成応答を確認し、係数を保存します。

マルチウェイでは、先に構成と帯域分割を設定してPhaseEQへ送信します。PhaseEQでチャンネルごとに補正し、「マルチウェイへ送信」で返した後、合成応答を確認します。接続中のFIR条件や帯域分割はマルチウェイ側で管理します。

FIR OFFでもIIRによる設計を続けられます。FIRイコライザーの設定は保持されますが、操作できません。FIRとIIRは別のDSP段として出力されるため、両方を使う場合は実機にも両方を設定してください。

## 日本語マニュアルとドキュメント

| 文書 | 内容 |
|---|---|
| [画像付き操作マニュアル（PDF）](docs/distribution/manual/PhaseEQ_v1.20.8_画像付き操作マニュアル_日本語.pdf) | 現行UIの実画面、操作順、図解を収録（対応版はPDF表紙に記載） |
| [操作ガイド](docs/distribution/user-guide.md) | 現在の画面名、操作順、設定条件、グラフの読み方 |
| [出力ファイル仕様](docs/distribution/output-format.md) | FIR／IIR係数、作業保存、DSP用パッケージ |
| [インストールガイド](docs/distribution/installation.md) | 導入・更新・起動 |
| [更新履歴](docs/distribution/history.md) | バージョン別の変更と旧ガイドの更新記録 |
| [v1.20.8 更新案内](docs/distribution/release-notes-v1.20.8.md) | 最新公開版の変更点 |

リリースZIPには画像付きPDFと利用者向け文書を同梱します。テストプログラム・開発補助スクリプト・開発文書は含みません。[一般公開リポジトリ](https://github.com/tomii323/PhaseEQ-public)から実行用ソースを取得できます。開発資産は非公開リポジトリで管理しています。

## 実行時データ

個人設定、DB、測定一時ファイル、生成物、キャッシュは`data/`以下へ保存され、通常はGitや
配布ソースへ含めません。主な対象は`data/settings.json`、`data/databases/`、`data/cache/`、
`data/snapshots/`、`data/presets/`、`data/tmp/`です。

## 主な依存関係

Streamlit、NumPy、pandas、SciPy、Altair、Matplotlib、Plotly、SoundFile、pdfplumberを使用します。
