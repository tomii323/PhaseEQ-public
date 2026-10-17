# PhaseEQ v1.20.29 更新案内

PhaseEQ v1.20.29では、マルチウェイの合成結果にリバースヌルを重ねて表示できるようになりました。表示専用の計算で、編集データ・保存設定・DSP出力を変更せずにクロス付近の谷を比較できます。

## 使い方

1. 合成結果の表示対象を選び、Graphを「Magnitude」にします。2チャンネル以上が必要です。
2. 「リバースヌルを重ねて表示」をONにし、通常のSumと破線のReverse Nullを比較します。低い帯域から追加極性を交互に反転します。3WayではLowを維持、Midを反転、Highを維持します。
3. 「リバースヌルの使い方」で谷の見方と調整手順を確認します。確認後は表示をOFFにします。谷の深さに一律の合格基準はありません。実機では合成応答を再測定してください。

数値計算・表示データへの変換・UI操作を独立したモジュールへ分離しました。日本語・英語のヘルプ、利用ガイド、画像付き操作マニュアルを同期し、画面画像をダークモードへ更新しています。14ページのマルチウェイ簡易操作マニュアルも公開・同梱します。

## 更新後の再起動

1. 編集内容を保存します。
2. 起動に使った各ターミナルで`Ctrl+C`を押して終了します。ブラウザーのタブを閉じるだけではアプリは終了しません。
3. macOS／Linuxは`run_PhaseEQ_mac_linux.sh`、Windowsは`run_PhaseEQ_windows.bat`で起動します。
4. 表示版がv1.20.29になったことを確認します。

操作の詳細は[利用ガイド](user-guide.md)、[画像付き操作マニュアル](manual/PhaseEQ_v1.20.29_画像付き操作マニュアル_日本語.pdf)、[マルチウェイ簡易操作マニュアル](manual/PhaseEQ_Multiway_QuickGuide_14p_Dark_JA.pdf)を参照してください。

## Restart after updating

1. Save your edits.
2. Press `Ctrl+C` in every launcher terminal. Closing the browser tab does not stop the app.
3. Start `run_PhaseEQ_mac_linux.sh` on macOS/Linux or `run_PhaseEQ_windows.bat` on Windows.
4. Confirm that the displayed version is v1.20.29.

自動テストと隔離環境での画面操作で、追加極性・表示条件・元データの保持を確認しています。実機DSPへの取込・実音響での検証は今回の範囲に含めていません。
