# PhaseEQ v1.20.37 更新案内

PhaseEQ v1.20.37では、Standaloneと各ウェイチャンネルのIIR EQ／FIR EQの復帰処理を整理し、Stereo Linkの共有対象からTarget設定を外しました。

## 修正内容

- 全ウェイチャンネルで共通の復帰処理を使用し、編集対象とEQの所有者を確認してから設定を適用します。未保存の個別EQを保持し、別のチャンネルやStandaloneのEQを初期値へ流用する経路を修正しました。競合・復元失敗時は切替元の編集を保持します。
- Standalone専用の保存先を設け、最後に保存したチャンネルのEQが新規Standaloneへ入る経路を修正しました。専用保存がない既存環境は従来設定から開始します。以前に失われた編集を自動復旧するものではありません。
- Stereo Linkで共有する設定をIIR EQ／FIR EQ・Auto EQ設定に限定しました。Targetは既存のMultiway側の設定経路を使用し、StandaloneのTargetは単独編集の設定として保持します。
- 旧Target共有は読込み時に外します。IIR／FIRの共有関係とEQを保持し、Targetだけを共有していたグループは独立編集へ移します。DB更新前には旧snapshotを復旧用に保存します。
- リンク変更は、変更案の作成、利用者による確認、競合検証、確定に分けました。解除時はグループ全員の現在の実効EQを独立したコピーとして保持します。StandaloneのExportにはMultiway返却欄を表示しません。

## 検証範囲

軽量品質チェックと配布契約を検査しています。隔離したChromiumで、96 kHz・15,231 tapsの全8チャンネルとStandaloneを2周切り替え、設定保持、新規Standaloneの起動、返却Working Sessionを確認しました。Target共有廃止は96 kHz・351 tapsで両アプリの選択肢、旧データの移行、キャンセル、解除・再リンク、StandaloneのIIR／FIR／Target保持を確認しました。NTM4混在の往復受渡しでは実描画とDSP係数の一致を照合しました。

実利用セッションは検証のために操作していません。実機DSPへの取込と音響測定は今回の確認に含みません。更新後は使用するチャンネルの設定を確認し、機器への取込後に実測応答を確認してください。

## 更新後の再起動

1. 両アプリの編集内容を保存します。
2. 起動に使った各ターミナルで`Ctrl+C`を押して終了します。ブラウザーのタブを閉じるだけではアプリは終了しません。
3. macOS／Linuxは`run_PhaseEQ_mac_linux.sh`、Windowsは`run_PhaseEQ_windows.bat`で起動します。
4. 表示版がv1.20.37になったことを確認します。

操作の詳細は[利用ガイド](user-guide.md)と[Stereo Linkの手順](stereo-link.md)を参照してください。
