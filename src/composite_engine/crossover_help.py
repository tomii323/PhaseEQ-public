"""Shared comparison prose; independent of UI state and filter design."""

FILTER_COMPARISON = {
    "ja": """### 帯域分割フィルターの使い分け

帯域分割フィルター単体の性能による比較です。合成Gainは、重ね代0・設計条件と時間基準が揃った場合を示します。

| 方式 | 合成Gain | 合成Phase | 帯域分離・調整の特徴 | タップ数・負荷・遅延 | 選ぶ目的 |
|---|---|---|---|---|---|
| カイザー窓FIR | 平坦性を保つ設計 | 直線位相 | 約24〜192 dB/octが目安。周期数・β値で減衰と遷移幅を調整 | 必要な性能へタップ数を配分できる。急峻・深い減衰には長さが必要 | 直線位相を保ち、LR4より急峻な分離や用途に合う分割形状を実現したい |
| LR2 FIR | 平坦性を保つ設計 | 直線位相 | 約12 dB/octで、隣接帯域を緩やかにつなぐ | 緩やかな形状と必要長のバランスを取りやすい | 緩やかな接続と直線位相を両立したい |
| LR4 FIR | 平坦性を保つ設計 | 直線位相 | 約24 dB/octで、LR2より帯域を分離 | 強い分離と直線位相のため、タップ数・遅延の余裕が必要 | LR形状で帯域分離を強め、直線位相も保ちたい |
| LR2 IIR | 極性補償後に平坦 | 周波数に応じて位相回転 | 約12 dB/octの緩やかな分割 | 分割段は1バイクアッド。FIRタップ不要 | 軽い処理で緩やかに分割したい |
| LR4 IIR | 平坦 | 周波数に応じて位相回転 | 約24 dB/octで、LR2より帯域を分離 | 分割段は2バイクアッド。FIRタップ不要 | 低負荷と帯域分離を両立したい |
| NTM4 IIR | 平坦 | 周波数に応じて位相回転 | ノッチ後の阻止帯域ピークを−20〜−80 dBで調整 | LP/HP各2バイクアッド。3Way以上は自動適用ONで補償APを追加 | ノッチとLobe Heightで帯域外の特性を調整したい |
| スルー | 追加分割なし | 追加分割による位相変化なし | 外部ネットワークの分割特性を使用 | 追加の分割用タップ・バイクアッド不要 | 外部で完成した帯域分割を活用したい |

カイザー窓の約24〜192 dB/octは、96 kHz・クロス1 kHz・重ね代0、−20〜−100 dB間の平均減衰スロープの目安です。LRの固定次数による傾きとは異なり、全区間で一定ではありません。

FIRのタップ数は、必要な減衰・再現精度と、DSP容量・許容遅延の釣り合いで選びます。IIRの隣接帯域が同相でつながることは、合成Phaseが直線になることを意味しません。

LR4 IIRは、広い帯域外を単調に減衰させる標準の比較基準です。NTM4 IIRは、ノッチ付近の共振を抑えたい場合に比較します。Lobe Heightを下げるとノッチはクロスから遠ざかり、再上昇が小さくなります。ノッチ後と遠方の減衰、合成特性も確認してください。
""",
    "en": """### Choosing a crossover filter

This compares the crossover filters themselves. Summed gain assumes zero overlap and matching design conditions and time references.

| Method | Summed gain | Summed phase | Separation and adjustment | Taps, load and delay | Choose it to |
|---|---|---|---|---|---|
| Kaiser-window FIR | Designed for flat summation | Linear phase | Approx. 24–192 dB/oct as a guide. Adjust cycles and beta for attenuation and transition width | Allocate taps to the required performance; steep transitions and deep attenuation need length | Keep linear phase with steeper separation than LR4 or a tailored crossover shape |
| LR2 FIR | Designed for flat summation | Linear phase | Approx. 12 dB/oct for a gentle transition | Balance a gentle shape against the required length | Combine a gentle transition with linear phase |
| LR4 FIR | Designed for flat summation | Linear phase | Approx. 24 dB/oct for stronger separation than LR2 | Allow sufficient taps and delay for separation with linear phase | Strengthen separation with an LR shape while keeping linear phase |
| LR2 IIR | Flat after polarity compensation | Frequency-dependent phase rotation | Approx. 12 dB/oct for gentle separation | One biquad per crossover section; no FIR taps | Obtain gentle separation with low processing load |
| LR4 IIR | Flat | Frequency-dependent phase rotation | Approx. 24 dB/oct for stronger separation than LR2 | Two biquads per crossover section; no FIR taps | Balance low processing load with band separation |
| NTM4 IIR | Flat | Frequency-dependent phase rotation | Adjust the post-notch stopband peak from −20 to −80 dB | Two biquads per LP/HP; AP compensation for three or more Ways when automatic application is ON | Tune stopband behavior using notches and Lobe Height |
| Through | No added crossover | No phase change from an added crossover | Use the external crossover response | No additional crossover taps or biquads | Use a crossover already implemented externally |

The Kaiser estimate of 24–192 dB/oct is the average slope between −20 and −100 dB at 96 kHz, a 1 kHz crossover and zero overlap. Unlike fixed-order LR slopes, it is not constant throughout the response.

Choose FIR length by balancing attenuation and accuracy against DSP capacity and acceptable delay. Adjacent IIR bands adding in phase does not mean the summed phase is linear.

Use LR4 IIR as the reference for monotonic attenuation across a broad stopband. Compare NTM4 IIR when a notch can suppress an unwanted resonance. Lower Lobe Height moves the notch farther from the crossover and reduces the post-notch peak. Check attenuation beyond the notch and in the far stopband, as well as the summed response.
""",
}


def filter_comparison_text(language: str | None = None) -> str:
    return FILTER_COMPARISON.get(language, FILTER_COMPARISON["ja"])


SETUP_GUIDE = {
    "ja": """### 帯域分割の設定手順

FIRは、音量の周波数バランスや位相、帯域の分割を調整するフィルターです。本アプリの帯域分割FIRは直線位相を使います。直線位相は、各周波数成分を共通の遅延で扱う特性です。補正用FIRでは位相も調整できるため、FIRがすべて直線位相という意味ではありません。

タップ数はFIRの長さを表す数値です。長くすると必要な特性を作る余裕が増えますが、DSPの処理量や遅延も増えます。必要な特性とDSP上限に合わせて選びます。FIR ONはFIR補正を有効にする操作です。補正内容は手動EQまたはAuto Gain／Auto Phaseで設定します。

**設定の流れ**

> 出力先DSPの対応・上限を確認 → 分割方式を選ぶ → クロス周波数を設定\\
> → 初期値で再計算 → 各帯域と合成グラフを確認\\
> → 必要な箇所を一つだけ変更 → 最終タップ数・遅延を確認して保存

1. **方式を選びます。** FIR非対応、または負荷・遅延を抑えたい場合はIIRを選びます。緩やかにつなぐ場合はLR2 IIR、帯域分離を強める場合はLR4 IIRを選びます。FIRでは、緩やかな接続はLR2 FIR、LR形状で分離を強める場合はLR4 FIR、急峻さ・減衰を調整したい場合はカイザー窓を選びます。外部で帯域分割済みの場合はスルーを使います。
2. **クロス周波数を設定します。** 各スピーカーの測定で信頼できる帯域と使用可能範囲に合わせます。周期数・β値・タップ数は初期値を使い、重ね代は0で確認します。LR／NTM4方式で通常の分割を出力する場合は「音響ターゲットに反映」をOFFにします。実測特性を選択した目標へ補正する場合はONにし、別の「音響ターゲット補正の使い方」に沿ってEQを設定します。
3. **グラフで確認します。** 各帯域の分離・帯域外の減衰、合成のクロス付近の山や谷を確認します。PhaseEQでは自動表示更新がOFFなら「更新」、マルチウェイでは「再計算」を押します。条件は一つずつ変更し、同じ表示条件で比較します。
4. **出力前に確認します。** 最終FIRタップ数をDSPの上限と比べます。必要タップ数の目安は分割段の長さ、最終FIRタップ数は補正・合成・クロップ後に保存する係数の長さです。映像との同期やリアルタイム用途では、機器側の遅延も含めて確認します。保存した係数を対応DSPへ適用し、実測で最終確認します。

帯域分割フィルター単体の合成Gainが平坦でも、スピーカーを含む実測特性まで平坦になるとは限りません。
""",
    "en": """### Crossover setup procedure

FIR filters adjust frequency balance, phase and band separation. This application's crossover FIR filters use linear phase: frequency components share a common delay. Correction FIR filters can also adjust phase, so not every FIR filter is linear phase.

Tap count describes FIR length. More length gives room to realize the required response, but also increases DSP processing and delay. Choose the length to meet the required response and DSP limits. FIR ON enables FIR correction. Set the correction using manual EQ or Auto Gain / Auto Phase.

**Setup flow**

> Check DSP support and limits → Choose a method → Set crossover frequency\\
> → Recalculate with initial settings → Inspect bands and summed response\\
> → Change one setting at a time → Check final taps and delay, then save

1. **Choose a method.** Use IIR if FIR is unsupported or low processing load and delay matter: LR2 IIR for gentle transitions, LR4 IIR for stronger separation. With FIR, use LR2 for gentle transitions, LR4 for stronger separation with an LR shape, or Kaiser to adjust steepness and attenuation. Use Through for an external crossover already in place.
2. **Set the crossover frequency.** Use each speaker's reliable measured and usable range. Begin with initial cycles, beta and taps, and zero overlap. For normal LR crossover output, turn Apply to acoustic target OFF. To correct the measured response toward an LR target, turn it ON and set EQ using the acoustic-target guide.
3. **Inspect the graphs.** Check band separation, out-of-band attenuation and peaks or dips around the summed crossover. Click Update in PhaseEQ when automatic updates are off, or Recalculate in Multiway. Change one setting at a time and compare under the same display conditions.
4. **Check before output.** Compare final FIR taps with the DSP limit. The estimated crossover taps describe the split stage; final FIR taps describe the saved coefficients after correction, composition and cropping. For video synchronization or real-time use, include device delay in your check. Apply saved coefficients in a compatible DSP and verify the result by measurement.

Flat summed gain of the crossover filters alone does not guarantee a flat measured speaker response.
""",
}

KAISER_ADJUSTMENT = {
    "ja": """**初期値での確認**

周期数・β値は初期値、重ね代は0から始めます。周期数はFIRの長さ、β値は帯域外減衰の設計目安です。周期数を増やすと生成長が増えます。β値を変えると減衰と遷移幅が変わるため、周期数表も確認します。

> 初期値で再計算 → グラフで確認 → 下の表で次の操作を選ぶ\\
> → 一つだけ変更して再計算 → 最終タップ数・遅延を確認して保存

| 確認した結果 | 次の操作 |
|---|---|
| 希望する減衰スロープに合わせたい | 周期数表でスロープとβ値を選び、交点の周期数を入力欄へ直接入力します |
| 帯域外の減衰を深くしたい | β値の「目標減衰」を確認して調整し、周期数表を見直します |
| タップ数・遅延が大きすぎる | 周期数を少し下げ、必要な分離・減衰が保たれるか再確認します |
| 合成グラフのクロス付近に山・谷がある | 隣接帯域の設定、極性、位相・時間整合を先に確認します |
| 必要な特性とDSP上限を満たしている | その設定を保存します |

一度に複数の値を変更せず、変更前後を同じグラフ条件で比較します。
""",
    "en": """**Check initial settings**

Start with initial cycles and beta, and zero overlap. Cycles determine nominal FIR length; beta is the design guide for out-of-band attenuation. More cycles increase generated length. Changing beta affects attenuation and transition width, so revisit the cycle table.

> Recalculate with initial settings → Inspect graphs → Choose the next action below\\
> → Change one setting and recalculate → Check final taps and delay, then save

| What you see | Next action |
|---|---|
| A particular roll-off slope is needed | Select the slope and beta in the cycle table, then type the intersecting cycle value directly into the input |
| Deeper out-of-band attenuation is needed | Adjust beta using Target attenuation, then revisit the cycle table |
| Too many taps or too much delay | Reduce cycles slightly and check that required separation and attenuation remain |
| Peaks or dips around the summed crossover | Check adjacent band settings, polarity, phase and time alignment first |
| Required response and DSP limits are met | Save these settings |

Change one value at a time and compare using the same graph conditions.
""",
}

ACOUSTIC_FLOW = {
    "ja": {
        "studio": """**ONにするだけでは補正は完了しません。**

> ON → PhaseEQへ送信 → 各Wayの測定を適用・レベル合わせ\\
> → 手動EQまたはAuto Gain／Auto Phaseを設定\\
> → マルチウェイへ返却 → 再計算して合成確認

FIR OFFの場合はIIRイコライザーで補正します。方式・クロス周波数・Lobe Height・ON／OFFを変更したら、再送信して各Wayの補正を更新します。
""",
        "phaseeq": """**ONにするだけでは補正は完了しません。**

> LR／NTM4方式・クロス周波数・Lobe Heightを設定してON → 振幅・位相を含む測定を適用\\
> → 入力をTargetへレベル合わせ → 手動EQまたはAuto Gain／Auto Phaseを設定\\
> → 更新して最終特性を確認 → 保存・出力

自動表示更新がOFFのときは「更新」を押します。FIR OFFの場合はIIRイコライザーで補正します。マルチウェイ連携中は、境界条件をマルチウェイ側で設定し、各Wayの補正を返して合成を確認します。
""",
    },
    "en": {
        "studio": """**Turning ON alone does not complete correction.**

> Turn ON → Send to PhaseEQ → Apply each Way's measurement and match levels\\
> → Set manual EQ or Auto Gain / Auto Phase\\
> → Return to Multiway → Recalculate and inspect the summed response

With FIR OFF, correct using IIR EQ. After changing method, crossover frequency, Lobe Height or ON/OFF, resend and update each Way's correction.
""",
        "phaseeq": """**Turning ON alone does not complete correction.**

> Set LR or NTM4 method and crossover (and NTM4 Lobe Height), then turn ON → Apply magnitude and phase measurement\\
> → Match input level to Target → Set manual EQ or Auto Gain / Auto Phase\\
> → Update and inspect the final response → Save and export

Click Update when automatic display updates are off. With FIR OFF, correct using IIR EQ. When linked to Multiway, set boundary conditions there, return each Way's correction and check the summed response.
""",
    },
}


def setup_guide_text(language: str | None = None) -> str:
    return SETUP_GUIDE.get(language, SETUP_GUIDE["ja"])


def kaiser_adjustment_text(language: str | None = None) -> str:
    return KAISER_ADJUSTMENT.get(language, KAISER_ADJUSTMENT["ja"])


def acoustic_flow_text(*, studio: bool, language: str | None = None) -> str:
    return ACOUSTIC_FLOW.get(language, ACOUSTIC_FLOW["ja"])["studio" if studio else "phaseeq"]


METHOD_USAGE = {
    "ja": {
        "Kaiser FIR": "カイザー窓FIR：平坦な合成Gainと直線位相を保つ設計。周期数・β値で減衰と遷移幅を調整し、LR4より急峻な分離も狙えます。約24〜192 dB/octは96 kHz・クロス1 kHz・重ね代0、−20〜−100 dB間の平均スロープの目安です。必要な性能とタップ数・遅延の釣り合いを確認します。",
        "Linear-phase LR2 FIR": "LR2 FIR：平坦な合成Gainと直線位相を保つ設計。約12 dB/octで隣接帯域を緩やかにつなぎたい場合に選びます。必要な再現精度とFIR長・遅延の釣り合いを確認します。",
        "Linear-phase LR4 FIR": "LR4 FIR：平坦な合成Gainと直線位相を保つ設計。約24 dB/octでLR2より帯域を分離したい場合に選びます。強い分離と直線位相にはタップ数・遅延の余裕が必要です。",
        "LR2": "LR2 IIR：極性補償後に平坦な合成Gainを得られます。約12 dB/oct、分割段は1バイクアッドで、分割用FIRタップは不要です。軽い処理で緩やかに分割したい場合に選びます。合成Phaseは周波数に応じて回転します。",
        "NTM4": "NTM4 IIR：LP/HP各2バイクアッド。LR4を基準に、ノッチ付近の共振を抑えたい場合に比較します。Lobe Heightでノッチ位置と再上昇ピークを調整し、遠方の減衰も確認します。重ね代0・追加補正なしの相補IIR構成は、必要な構造補償APを含めて0 dBに合成されます。合成位相は回転します。",
        "LR4": "LR4 IIR：平坦な合成Gainを得られます。約24 dB/oct、分割段は2バイクアッドで、分割用FIRタップは不要です。広い帯域外を単調に減衰させる標準の比較基準です。合成Phaseは周波数に応じて回転します。",
        "Through": "スルー：外部で完成した帯域分割を活用します。追加の分割によるGain・Phaseの変化や、分割用タップ・バイクアッドを避けられます。必要な分割は外部ネットワーク側で実現してください。",
    },
    "en": {
        "Kaiser FIR": "Kaiser-window FIR: designed for flat summed gain and linear phase. Tune cycles and beta for attenuation and transition width, including steeper separation than LR4. The 24–192 dB/oct guide is an average between −20 and −100 dB at 96 kHz, a 1 kHz crossover and zero overlap. Balance performance against taps and delay.",
        "Linear-phase LR2 FIR": "LR2 FIR: designed for flat summed gain and linear phase. Choose its approx. 12 dB/oct shape for a gentle transition between bands. Balance accuracy against FIR length and delay.",
        "Linear-phase LR4 FIR": "LR4 FIR: designed for flat summed gain and linear phase. Choose its approx. 24 dB/oct shape for stronger separation than LR2. Allow sufficient taps and delay.",
        "LR2": "LR2 IIR: flat summed gain after polarity compensation. Approx. 12 dB/oct with one biquad per crossover section and no crossover FIR taps. Choose gentle separation with low processing load. Summed phase rotates with frequency.",
        "NTM4": "NTM4 IIR: two biquads per LP/HP. Compare against LR4 when a notch can suppress an unwanted resonance. Lobe Height adjusts notch position and the post-notch peak; also check far-stopband attenuation. Complementary IIR banks with zero overlap and no extra correction sum to 0 dB with the required structural AP compensation. Summed phase rotates.",
        "LR4": "LR4 IIR: flat summed gain. Approx. 24 dB/oct with two biquads per crossover section and no crossover FIR taps. Use as the reference for monotonic attenuation across a broad stopband. Summed phase rotates with frequency.",
        "Through": "Through: use an external crossover already in place. Avoid gain/phase changes and tap/biquad usage from an added crossover. The required separation must be provided by the external network.",
    },
}


def method_usage_text(method: str, language: str | None = None) -> str:
    return METHOD_USAGE.get(language, METHOD_USAGE["ja"]).get(method, "")


METHOD_SELECTOR_SUMMARY = {
    "ja": {
        "Kaiser FIR": "**カイザー窓FIR**：平坦な合成Gain・直線位相。周期数とβ値で減衰・遷移幅を調整。急峻な分離にはタップ数・遅延の余裕が必要です。",
        "Linear-phase LR2 FIR": "**LR2 FIR**：平坦な合成Gain・直線位相。約12 dB/octで緩やかにつなぐ場合に選びます。",
        "Linear-phase LR4 FIR": "**LR4 FIR**：平坦な合成Gain・直線位相。約24 dB/octで帯域分離を強める場合に選びます。",
        "LR2": "**LR2 IIR**：極性補償後に平坦な合成Gain。約12 dB/oct・1バイクアッド。軽い処理で緩やかにつなぐ場合に選びます。合成Phaseは周波数に応じて回転します。",
        "NTM4": "**NTM4 IIR**：LP/HP各2バイクアッド。ノッチ付近の共振抑制をLR4と比較します。Lobe Heightでノッチ位置・再上昇を調整し、遠方の減衰も確認します。3Way以上は自動適用ONで補償APを追加します。",
        "LR4": "**LR4 IIR**：平坦な合成Gain。約24 dB/oct・2バイクアッド。広い帯域外を単調に減衰させる標準の比較基準です。合成Phaseは周波数に応じて回転します。",
        "Through": "**スルー**：追加分割なし。外部で完成した帯域分割を使用します。",
    },
    "en": {
        "Kaiser FIR": "**Kaiser-window FIR**: flat summed gain and linear phase. Tune cycles and beta for attenuation and transition width. Steep separation requires sufficient taps and delay.",
        "Linear-phase LR2 FIR": "**LR2 FIR**: flat summed gain and linear phase. Choose approx. 12 dB/oct for gentle transitions.",
        "Linear-phase LR4 FIR": "**LR4 FIR**: flat summed gain and linear phase. Choose approx. 24 dB/oct for stronger separation.",
        "LR2": "**LR2 IIR**: flat summed gain after polarity compensation. Approx. 12 dB/oct, one biquad. Choose gentle transitions with low processing load. Summed phase rotates with frequency.",
        "NTM4": "**NTM4 IIR**: two biquads per LP/HP. Compare with LR4 to suppress resonance near a notch. Lobe Height adjusts notch position and the post-notch peak; also check far-stopband attenuation. AP compensation for three or more Ways when automatic application is ON.",
        "LR4": "**LR4 IIR**: flat summed gain. Approx. 24 dB/oct, two biquads. Use as the reference for monotonic attenuation across a broad stopband. Summed phase rotates with frequency.",
        "Through": "**Through**: no added crossover. Use an external crossover already in place.",
    },
}


def method_selector_help(methods, language: str | None = None) -> str:
    condition = ("Summed gain assumes zero overlap and matching design conditions and time references."
                 if language == "en" else "合成Gainは、重ね代0・設計条件と時間基準が揃った場合です。")
    summaries = METHOD_SELECTOR_SUMMARY.get(language, METHOD_SELECTOR_SUMMARY["ja"])
    return condition + "\n\n" + "\n\n".join(summaries[method] for method in methods if method in summaries)
