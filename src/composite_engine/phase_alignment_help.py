"""Static, bilingual structural-phase guidance, separate from DSP and rendering."""
from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralPhaseExample:
    crossovers_hz: tuple[float, float]
    phase_off_deg: float
    gain_off_db: float
    gain_on_db: float
    minimum_gain_off_db: float
    maximum_abs_gain_on_db: float


# Reference calculations: 96 kHz, 3Way LR4, zero overlap, unity speakers,
# no EQ or added Delay, direct IIR output. Values are not live diagnostics.
STRUCTURAL_PHASE_EXAMPLES = (
    StructuralPhaseExample((300., 3000.), -16.207, -.0885, .0004, -.1177, .0008),
    StructuralPhaseExample((1000., 3000.), -55.707, -1.2277, .0522, -2.2513, .0855),
)

STRUCTURAL_PHASE_HELP = {
    'ja': {
        'title': '位相構造補償の使い方',
        'tooltip': 'IIRを含む3Way以上で、クロス由来の位相差と合成Gainの山谷を抑えます。ONではWay・分割方式・周波数に合わせて自動更新します。ONからOFFへ切り替えると自動補償を解除し、Delayと手動極性は維持します。数値例と必要なリソースは、すぐ下の「位相構造補償の使い方」で確認できます。',
        'purpose': '''**クロス付近の位相差と、合成Gainの山谷を抑える機能です。**

主にIIR LR2／LR4／NTM4を含む3Way以上で使用します。3WayのMidには上下両方のクロスが作用するため、隣接Wayと比べると別のクロスによる位相回転が残ります。不足する位相回転を、各Wayの振幅を保つオールパスで補います。

**設定の流れ**

1. 分割方式・周波数を設定し、「位相構造補償を自動適用」をONにします。理論配置は測定や適用ボタンの操作なしで更新します。
2. ON／OFFを切り替え、クロス付近の合成Gainと隣接Wayの位相差を比較します。
3. 必要に応じて測定に基づく時間整合を適用します。自動適用がONの間、適用ボタンは時間整合とDelayだけを更新します。
4. 合成波、Delayとバイクアッド使用数を確認し、保存・DSP出力します。ONからOFFへの切替では自動オールパスを解除し、Delay・手動極性・Gainは維持します。

**クロス付近の変動例**

96 kHz・3Way・全境界IIR LR4・重ね代0、スピーカー特性・EQ・追加Delayなし、位相・時間整合の音響ターゲットOFFでの計算例です。実測値ではなく、各クロス点で同じ値となる例を示します。''',
        'table_header': '| クロス周波数 | 隣接Wayの位相差 OFF → ON | 合成Gain OFF → ON |',
        'effects': '''表のGainは0 dBからの差です。1,000／3,000 Hzの例では、各クロスの上下1オクターブにある最大約−2.25 dBの落ち込みが、ON時は約＋0.09 dB以内の変動になりました。300／3,000 Hzの例では、同範囲の変動はON時に0.001 dB未満です。

クロス間隔が近い場合は、補償による違いが大きくなる場合があります。補償後も多Wayの合成Gainには小さな変動が残ります。各WayのGainを直接増減せず、位相の関係を整えて合成時の落ち込みを抑えます。

合成PhaseにはIIR由来の位相回転が残ります。スピーカー固有の位相、設置距離、室内反射は、測定と時間整合・EQで別に調整します。補償前後の測定と試聴で効果を確認してください。''',
        'resources': '''**必要なリソース**

直接IIRで適用する場合、追加オールパス1段はSOS／バイクアッドの1枠です。全境界がIIRの場合の、1チャンネル当たりの最大追加段数を示します。

| 構成 | 最大追加段数／チャンネル |
|---|---:|
| 2Way | 0段 |
| 3Way | 1段 |
| 3Way＋SUB／4Way | 2段 |
| 4Way＋SUB | 3段 |

3WayのLR構成ではLowとHighに各1段、Midは追加なしです。NTM4を含む構成では前段HPを共有するため、Lowに後段クロスのAPを追加し、QはそのLobe Heightに連動します。FIR・スルーの境界には自動構造補償用の段を生成しません。直接IIRで適用する場合、この機能だけで追加FIRタップは不要です。出力先のバイクアッド枠とオールパスへの対応を確認してください。

このチェックは設定済みDelayを変更しません。オールパスには周波数ごとの群遅延があり、LR4用1段の中心1 kHz付近では約0.45 ms（96 kHz時）です。一定時間を全周波数へ加えるDelayとは区別して確認します。

「位相・時間整合：音響ターゲットに反映」をONにすると、オールパスを直接出力へ追加せず、PhaseEQの目標へ渡します。測定を適用し、補正を設定・更新して返却してください。この経路では補正FIRの長さと実現精度も確認します。''',
    },
    'en': {
        'title': 'Using structural phase compensation',
        'tooltip': 'Reduces crossover-related phase differences and summed-gain dips or peaks in 3Way or larger systems containing IIR crossovers. ON updates with Way, method and frequency changes. Switching ON to OFF removes automatic compensation while retaining Delay and manual polarity. See the guide immediately below for examples and resource requirements.',
        'purpose': '''**Use this to reduce phase differences and summed-gain dips or peaks around crossovers.**

This mainly applies to 3Way or larger systems containing IIR LR2/LR4. In 3Way, Mid passes through both crossover boundaries, so the other boundary adds phase rotation relative to each adjacent Way. All-pass sections supply the missing rotation while preserving each Way's magnitude.

**Procedure**

1. Set the crossover methods and frequencies, then enable Automatically apply structural phase compensation. The theoretical placement updates without measurements or pressing Apply.
2. Compare OFF and ON using summed gain and adjacent-Way phase differences around the crossovers.
3. Apply measurement-based time alignment as needed. While automatic compensation is ON, Apply updates only time alignment and Delay.
4. Check the sum, Delay and biquad count, then save and export to DSP. Switching ON to OFF removes automatic all-pass sections while retaining Delay, manual polarity and gain.

**Examples around the crossovers**

Calculated at 96 kHz, 3Way, IIR LR4 at both boundaries, zero overlap, unity speakers, no EQ or added Delay, and the phase/time alignment acoustic target OFF. These are calculations, not measurements; both crossover points have the same values in each example.''',
        'table_header': '| Crossover frequencies | Adjacent-Way phase difference OFF → ON | Summed gain OFF → ON |',
        'effects': '''Gain is relative to 0 dB. In the 1,000/3,000 Hz example, the largest dip within one octave above and below each crossover changes from approximately −2.25 dB to a variation within approximately +0.09 dB when ON. In the 300/3,000 Hz example, ON variation in these ranges is below 0.001 dB.

Closer crossover spacing can produce a larger change. Small variations can remain in the multi-Way sum. The function adjusts phase relationships to reduce cancellation without directly changing each Way's gain.

The sum retains IIR phase rotation. Adjust driver-specific phase, placement distance and room reflections separately using measurements, time alignment and EQ. Check the effect with measurements and listening before and after compensation.''',
        'resources': '''**Resource requirements**

For direct IIR output, each added all-pass section uses one SOS/biquad slot. The maximum additional section count per channel below assumes IIR at every boundary.

| Configuration | Maximum added sections per channel |
|---|---:|
| 2Way | 0 |
| 3Way | 1 |
| 3Way+SUB / 4Way | 2 |
| 4Way+SUB | 3 |

In a 3Way LR bank, Low and High each receive one section; Mid receives none. Banks containing NTM4 share preceding HP stages, so Low receives the later crossover AP with Q derived from its Lobe Height. FIR and Through boundaries generate no automatic structural-compensation sections. Direct IIR application requires no additional FIR taps for this function. Check the output device's biquad capacity and all-pass support.

This checkbox does not change configured Delay. All-pass sections introduce frequency-dependent group delay: approximately 0.45 ms near a 1 kHz center for one LR4 compensation section at 96 kHz. Distinguish this from adding a constant delay to all frequencies.

Enabling Phase/time alignment: include in acoustic target sends the all-pass definition to the PhaseEQ target instead of adding it directly to the output. Apply measurements, configure and update correction, then return it. Check correction FIR length and realized accuracy for this route.''',
    },
}


def structural_phase_labels(language='ja'):
    return STRUCTURAL_PHASE_HELP['en' if language == 'en' else 'ja']


def structural_phase_guide_text(language='ja'):
    text = structural_phase_labels(language)
    rows = [text['table_header'], '|---|---:|---:|']
    for example in STRUCTURAL_PHASE_EXAMPLES:
        frequencies = ' / '.join(f'{value:,.0f}' for value in example.crossovers_hz)
        rows.append(f'| {frequencies} Hz | {example.phase_off_deg:.1f}° → ≈0° | '
                    f'{example.gain_off_db:+.2f} dB → {example.gain_on_db:+.2f} dB |')
    return '\n\n'.join((text['purpose'], '\n'.join(rows), text['effects'], text['resources']))
