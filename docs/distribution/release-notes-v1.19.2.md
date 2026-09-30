# PhaseEQ v1.19.2 更新案内

PhaseEQ v1.19.2では、Multiway StudioのFullrange+SUBとStereo SUB表示で発生していた応答経路の不具合を修正しました。

## 主な修正

### Fullrange+SUBの相補帯域分割

SUBのLow-passだけでなく、Fullrangeへ同じ境界のHigh-passを生成します。Targetを含めない帯域分割合成が基準帯域で平坦になる構成へ修正しました。FIR方式とIIR方式のどちらでも、同じSUB／Fullrange境界を使用します。

### 片側SUBだけを使用するStereo表示

StereoでLeft SUBなど片側Channelだけを`Compositeへ含める`にした場合も、表示をMonoやMainへ変更しません。`L / R / L+R`を維持し、有効なSUBの実応答とGroup Target参照線を別系列として表示します。TargetはChannel応答やSystem Sumへ乗算しません。

### Shared SUBと左右独立SUB

Shared SUBはChannel線を1系列だけ表示し、Left／RightのSystem Sumへ各1回含めます。左右独立SUBは別Channelとして扱い、左右のSUBを互いに加算しません。

### 全Channel無効時の表示

すべての`Compositeへ含める`をOFFにした場合は、空の物理Groupと空グラフを安全に表示します。インパルス応答などが空配列を参照して停止する問題を修正しました。

## 更新後の確認

1. Multiway Studioで`Fullrange+SUB`を選びます。
2. `Channel layout`をStereoへ変更します。
3. Left SUBだけを`Compositeへ含める`にしても`L / R / L+R`が表示されることを確認します。
4. `Show Group Target`を切り替えてもSUBとSystem SumのLevelが変化しないことを確認します。
5. 必要に応じてShared SUBをONにし、L+RでSUB線が1系列、System Sumが左右2系列であることを確認します。

## 互換性

v1.19.1以前の設定、Multiway System、Target、Assignment、Resume、DSP Exportを継続して読み込めます。Fullrange+SUBを再計算すると、新しい相補High-pass仕様が適用されます。
