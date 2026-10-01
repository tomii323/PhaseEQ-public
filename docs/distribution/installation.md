# インストールガイド

PhaseEQの環境構築と起動方法を説明します。
対応形式や操作範囲は[操作ガイド](user-guide.md)を参照してください。

Streamlitは `1.62` 以上、`1.63` 未満を使用し、現在の検証版は`1.62.0`です。
起動スクリプトは、起動時に検証済み依存版をローカル確認します。不足がなければ更新せずオフラインで起動します。1パッケージだけ検証版より新しい場合は仮想環境からそのパッケージを明示的に削除して対応版を入れ直し、複数の上限超過または壊れた依存関係がある場合は仮想環境を再作成します。修復にはネット接続が必要です。

Windows版はPython 3.12／3.13のx64を対象とします。Windows 11 ARMでは、必須依存パッケージにWindows ARM64ネイティブwheelがないため、OSのx64エミュレーションで公式x64 Pythonを使用します。ランチャーはARM64環境を検出するとx64 Python 3.12をwingetで導入し、ネイティブARM Pythonを採用しません。初回導入直後は、同じコマンドプロンプトへ反映されていないPATHを標準インストール先、Python公式インストーラーのレジストリ情報、ユーザー領域のPythonインストール直下から再検出するため、Windowsや端末を再起動せず続行できます。Python同梱の仮想環境用内部実行ファイルは探索対象にしません。検出時は`Detected Python installation:`に続けて使用候補の場所を表示します。

## 配布物とマニュアル

リリースZIPを展開すると、`docs/distribution/manual/` に[画像付き操作マニュアル（PDF）](manual/PhaseEQ_v1.20.8_画像付き操作マニュアル_日本語.pdf)があります。PDF表紙の対応版を確認してください。通常の利用にテスト環境の構築は不要です。テスト・開発文書・開発用スクリプトはZIPに含みません。実行用ソースは[一般公開リポジトリ](https://github.com/tomii323/PhaseEQ-public)から取得できます。

## 新版へ更新する

編集中の内容を設定ファイルまたは作業セッションへ保存し、PhaseEQとマルチウェイの両方を終了してから新版を展開します。両アプリは同じ配布版を使ってください。旧ブラウザー画面が残る場合は再読み込みし、画面の版表示を確認します。保存済みの測定データや作業DBを削除する必要はありません。変更点は[更新履歴](history.md)を参照してください。

## macOS

すぐに起動したい場合は、プロジェクトフォルダで次を実行します。

```sh
chmod +x run_PhaseEQ_mac_linux.sh
./run_PhaseEQ_mac_linux.sh
```

この起動スクリプトは、仮想環境と依存関係の確認、空きポートの選択、ブラウザ起動まで行います。既存`.venv`がStreamlit 1.62系を含む検証環境と一致する場合は、pip、Homebrew等のネットワーク処理を行いません。初回構築や依存修復時はオンライン導入が必要です。

通常のオフライン優先起動:

```sh
./run_PhaseEQ_mac_linux.sh
```

オンライン環境でpipと依存関係を明示更新する場合:

```sh
./run_PhaseEQ_mac_linux.sh --update
```

起動ログを保存する場合は`--log`を併用できます。オフライン起動ではStreamlit利用統計送信も無効になります。スピーカー仕様のURL読込やメーカー校正ページを開く機能はネットワークを必要としますが、測定、設計、DB、ローカルファイル入出力、ESS生成はローカルで動作します。

### 1. Pythonを用意する

Homebrewを使う場合は次を実行します。

```sh
brew install python
```

### 2. 仮想環境を作成する

ターミナルでプロジェクトフォルダへ移動し、仮想環境を作成します。
macOSでは通常 `py` は使わず、`python3` を使います。
仮想環境を有効化した後は、環境内の `python` を使います。

```sh
cd "/path/to/PhaseEQ"
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 依存関係をインストールする

```sh
python -m pip install --constraint constraints/test-environment.txt pip setuptools wheel
python -m pip install --upgrade --constraint constraints/test-environment.txt -r requirements.txt
```

### 4. 起動する

```sh
python -m streamlit run phaseeq.py
```

ブラウザで `http://localhost:8501` を開きます。

## Windows 11

すぐに起動したい場合は、`run_PhaseEQ_windows.bat` をダブルクリックします。
仮想環境の作成、依存関係の確認、空きポートの選択、ブラウザ起動まで自動で行います。既存`.venv`がStreamlit 1.62系を含む検証環境と一致する場合は、pipやwingetへ接続せずオフラインで起動します。初回構築や依存修復時はオンライン導入が必要です。
macOS/Linux版と同じく、固定バージョンが揃っていても最終import確認が失敗した場合はオンライン修復を試みます。PhaseEQ終了時は、同時起動したマルチウェイも自動終了します。

通常のオフライン優先起動:

```bat
run_PhaseEQ_windows.bat
```

オンライン環境でpipと依存関係を明示更新する場合:

```bat
run_PhaseEQ_windows.bat --update
```

起動ログを保存する場合は`--log`を併用できます。オフライン起動ではStreamlit利用統計送信も無効になります。

### 初回設定の要点

| 項目 | 要点 |
|---|---|
| Python | 検証済みの公式x64版Python 3.12または3.13を使用 |
| Windows ARM | ARM64ネイティブPythonではなくx64 Python 3.12を使用 |
| PATH | インストール時にPythonのPATH登録を有効にする |
| PowerShell | 仮想環境を有効化できない場合だけ実行ポリシーを変更する |
| 仮想環境 | プロジェクト内の`.venv`へ依存ライブラリを分離する |
| pip | `pip`単独ではなく`python -m pip`を使用する |
| 起動 | 仮想環境を有効化してから`python -m streamlit`で起動する |

Windowsの新しいPython Install Managerで`py -3.13`に対応Runtimeが登録されていない場合も、ランチャーはその候補を使用せず、`py -3.12`、`py -3`、PATH上の`python`の順に実際に起動できるx64 Python 3.12／3.13を探します。`No runtime installed that matches ...`やARM64ネイティブPythonは候補不採用として扱います。Python 3.14だけがある場合も採用せず、`winget`で検証済みのPython 3.12を導入します。

プロジェクトは、短く分かりやすいパスへ展開することを推奨します。

```text
C:\Projects\PhaseEQ
```

### 1. Pythonを用意する

通常はランチャーへPythonの検出と導入を任せてください。手動構築する場合はx64版Python 3.12または3.13をインストールします。公式インストーラーを使う場合は、
`Add python.exe to PATH` を有効にしてください。

Windows 11 ARMで手動導入する場合は、管理者権限を必要としないユーザー領域へx64版を指定します。

```bat
winget install -e --id Python.Python.3.12 --architecture x64 --scope user --force
```

ARM64版Pythonが既に入っていてもアンインストールは不要です。PhaseEQ用`.venv`だけをx64 Pythonから作成します。

SoundDevice 0.5.5はWindows x64／ARM64の両wheelを提供します。Windows ARM上のx64 Pythonでは、SoundDeviceがホストCPUを見てARM64版PortAudio DLLを誤選択するため、PhaseEQはSoundDeviceのimport時だけPython ABIに合わせてx64版`libportaudio64bit.dll`を選択します。それでも音声ドライバー等により読み込めない場合は、`Optional audio runtime is unavailable`と具体的な例外を警告してPhaseEQを起動します。

インストール後、新しいPowerShellを開いて確認します。

```powershell
python --version
py --version
python -m pip --version
```

Windowsでは`py`はPython Launcherです。x64 Windowsでは仮想環境作成に`py -3.12`または`py -3.13`を使用できます。Windows ARMでは`py`がARM64版を選ぶ場合があるため、手動構築より`run_PhaseEQ_windows.bat`を推奨します。

### 2. 仮想環境を作成する

PowerShellでプロジェクトフォルダへ移動します。

```powershell
cd "C:\Projects\PhaseEQ"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`py` が見つからない場合は、次のように `python` で作成します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

有効化できない場合は、現在のPowerShellだけ一時的に許可します。

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. 依存関係をインストールする

```powershell
python -m pip install --constraint constraints/test-environment.txt pip setuptools wheel
python -m pip install --upgrade --constraint constraints/test-environment.txt -r requirements.txt
```

### 4. 起動する

```powershell
python -m streamlit run phaseeq.py
```

自動的にブラウザが開かない場合は、次へアクセスします。

```text
http://localhost:8501
```

### 5. 2回目以降の起動

```powershell
cd "C:\Projects\PhaseEQ"
.\.venv\Scripts\Activate.ps1
python -m streamlit run phaseeq.py
```

## Linux

Ubuntu / Debian系の例です。

すぐに起動したい場合は、プロジェクトフォルダで次を実行します。

```sh
chmod +x run_PhaseEQ_mac_linux.sh
./run_PhaseEQ_mac_linux.sh
```

Linuxでは、起動スクリプトがPython環境、venv、pip、日本語フォントの有無を確認します。通常起動でランタイムが揃っていればネットワーク処理を行いません。日本語フォントだけが不足する場合は警告して起動を継続します。依存不足時または`--update`指定時は、`apt` / `dnf` / `pacman`による自動インストールを試みるため、ネットワークと`sudo`入力が必要になることがあります。

```sh
sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 --version
cd /path/to/PhaseEQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --constraint constraints/test-environment.txt pip setuptools wheel
python -m pip install --upgrade --constraint constraints/test-environment.txt -r requirements.txt
python -m streamlit run phaseeq.py
```

手動インストールの場合もPython 3.12または3.13が必要です。
ディストリビューション標準の`python3`が3.12未満または3.14以降の場合は、OSの手順に従ってPython 3.12／3.13を導入するか、起動スクリプトを使用してください。

## 終了方法

起動しているターミナルで `Ctrl + C` を押します。

## トラブル対処

### `ModuleNotFoundError` が表示される

標準ランチャーの初回起動では、`Recreating virtual environment`の後に不足依存関係を検出し、オンラインインストールへ進みます。インストール開始前に`numpy`などの`ModuleNotFoundError`で停止する場合は、修正済みの最新Releaseへ更新してください。

手動起動の場合は、プロジェクト直下で起動しているか、仮想環境が有効かを確認してください。

```sh
pwd
python -m pip list
python -m streamlit run phaseeq.py
```

### `app.py` が見つからない

現在の起動ファイルは `phaseeq.py` です。

```sh
python -m streamlit run phaseeq.py
```

### ポート8501を使用中と表示される

別ポートを指定します。

```sh
python -m streamlit run phaseeq.py --server.port 8502
```

### グラフやUIの色が想定と違う

Streamlitのライト / ダークテーマに合わせて配色を切り替えています。
OSやブラウザ側の表示モードを変更した場合は、アプリを再読み込みしてください。

## アンインストール

作成した仮想環境を削除します。

```sh
rm -rf .venv
```

Windowsでは `.venv` フォルダを削除します。

## 関連文書

- [操作ガイド](user-guide.md)
- [v1.19.7 更新案内](release-notes-v1.19.7.md)
- [出力ファイル仕様](output-format.md)


## PhaseEQ本体の更新

公式公開ReleaseのZIPから展開したPhaseEQは、起動時にGitHubの正式Releaseを確認します。更新がある場合は案内を表示します。オフラインや確認失敗の場合も現在の版で作業を続けられます。

1. PhaseEQの「設定」から「PhaseEQ本体の更新」を開きます。「最新版を確認」で再確認できます。
2. 更新内容を確認し、「更新をダウンロードして予約」を押します。公開ZIP全体とファイル別のSHA-256を検証してから予約します。
3. 編集中の設定を保存し、PhaseEQとマルチウェイの両アプリを終了します。ブラウザーのタブを閉じるだけではサーバーは終了しません。起動元のターミナルで終了してください。
4. `run_PhaseEQ_mac_linux.sh`または`run_PhaseEQ_windows.bat`から起動し直します。起動前に更新を適用し、必要なPython依存関係を既存の起動処理で確認します。

`data/`内の設定・測定DB・ユーザープリセット、出力データ、仮想環境は本体更新で上書きしません。旧ソースと廃止された配布ファイルは`.phaseeq-updates/backup-*`へ退避されます。適用中に失敗した場合は旧ソースへ戻し、更新予約を保持して起動を停止します。次回起動時は中断記録を確認して復元してから再試行します。

本体ファイルの更新が成功すると、今回作成したバックアップと、前回の成功更新で記録したバックアップを直ちに削除します。成功後の旧版本体は保持しません。失敗・中断時は復元用バックアップを保持し、予約取り消しでは既存バックアップを削除しません。削除できない場合は起動元のターミナルに削除未完了を表示し、新版の適用は維持します。残った削除対象は次の更新成功時に再試行します。更新処理が成功記録に残していないフォルダーは自動削除しません。

この成功判定は本体ファイルの置き換え完了を意味し、新版の起動やDB読み込みの成功確認は含みません。本体バックアップにユーザーDBは含まれません。DBは引き継がれ、新版が開く際に必要なテーブルや列を追加する場合があります。

Git管理中の開発用チェックアウト、書き込みできないインストール先、`PUBLIC_FILES.sha256`がない配布物は本体更新の対象外です。配布ソースを利用者が変更した場合も更新を停止します。公式ZIPを別の場所へ展開して移行してください。更新処理はGitHub認証情報を要求しません。

ランチャーの既存の`--update`はPython依存パッケージの更新用です。本体の更新予約とは別の操作です。


予約後、アプリを終了する前なら「更新予約を取り消す」で取り消せます。ローカル変更などでランチャーが更新を停止した場合は、アプリのフォルダーで`python runtime/app_update.py --cancel`を実行すると、ダウンロードを保持したまま予約を解除できます。中断記録が残っている場合は、先にランチャーからの復元を行ってください。
