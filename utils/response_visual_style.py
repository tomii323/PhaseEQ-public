"""Shared response appearance policy; independent of DSP, UI state and renderers."""
from dataclasses import dataclass, replace
import hashlib

from utils.ui_semantic_colors import semantic_chart_colors

STYLE_REVISION = 1

# token, width, opacity, legend priority, drawing priority
ROLE_STYLES = {
    'input': ('input', 1.6, .85, 30, 20),
    'target': ('target', 2.4, 1., 10, 80),
    'result': ('result', 2.6, 1., 20, 90),
    'correction': ('correction', 1.8, .9, 40, 40),
    'sum': ('sum', 2.8, 1., 20, 90),
    'reference': ('reference', 1.2, .4, 70, 10),
    'way': ('bands', 1.6, .9, 50, 30),
    'calibration': ('calibration', 1.4, .75, 60, 20),
    'calculation_input': ('input', 1.2, .6, 60, 15),
    'gain_error': ('gain_error', 2., 1., 10, 60),
    'phase_error': ('phase_error', 2., 1., 20, 60),
    'reverse_null': ('reference', 1.8, .8, 75, 25),
    'auxiliary': ('reference', .8, .5, 80, 5),
}

PHASEEQ_SERIES_ROLES = {
    'Speaker/Input': 'input', 'IIR Result': 'result', 'IIR Filter': 'correction',
    'FIR Output': 'correction', 'IIR Output': 'correction', 'IIR Band Split': 'correction',
    'Target + Band Split': 'target', 'FIR + IIR Output': 'correction',
    'Target Response': 'target', 'Selected Target': 'reference',
    'Edited Target (saved response)': 'target', 'Correction Filter': 'correction',
    'Correction Filter Realized': 'correction',
    'Correction Filter Realized + IIR Crossover': 'correction',
    'Speaker + Realized': 'result', 'Mic Calibration': 'calibration',
    'Current page delta': 'correction', 'Gain Error': 'gain_error', 'Phase Error': 'phase_error',
    'Auto EQ Calculation Input': 'calculation_input', 'Auto IIR Preview': 'correction',
    'Near-field woofer': 'input', 'Near-field port': 'input',
    '現在の割り当て': 'input', '選択候補（未適用）': 'reference',
    'Impedance Ω': 'input', 'Impedance phase': 'input',
}

# Stable semantic Way IDs, not display order or editable channel names.
WAY_COLOR_INDICES = {'High': 0, 'Mid': 1, 'Low': 2, 'SUB': 3, 'Full': 0,
                     'Fullrange': 0, 'MidHigh': 4, 'MidLow': 1}


def way_color_index(way_id):
    if way_id is None or not str(way_id).strip():
        raise ValueError('Way style requires a stable Way ID')
    return WAY_COLOR_INDICES.get(str(way_id), int.from_bytes(
        hashlib.sha256(str(way_id).encode()).digest()[:4], 'big') % 5)


@dataclass(frozen=True)
class ResponseStyle:
    color: str
    width: float
    opacity: float
    dash: str
    legend_priority: int
    drawing_priority: int

    @property
    def vega_dash(self):
        return [8, 5] if self.dash == 'dash' else [1, 0]


def response_style(role, quantity='gain', *, theme='light', way_id=None, source_id=None):
    if role not in ROLE_STYLES:
        raise ValueError(f'Unregistered response role: {role}')
    if quantity not in {'gain', 'phase', 'time', 'delay', 'impedance'}:
        raise ValueError(f'Unregistered response quantity: {quantity}')
    token, width, opacity, legend, drawing = ROLE_STYLES[role]
    colors = semantic_chart_colors(theme)
    color = colors['bands'][way_color_index(way_id)] if role == 'way' else colors[token]
    if role == 'sum' and source_id == 'Right':
        color = colors['sum_secondary']
    return ResponseStyle(str(color), width, opacity, 'dash' if quantity == 'phase' or role == 'reverse_null' else 'solid',
                         legend, drawing)


def phaseeq_response_styles(names, *, quantity='gain', theme='light', roles=None,
                           reference_series=None, highlight_series=None):
    registered = PHASEEQ_SERIES_ROLES | dict(roles or {})
    result = {}
    for name in names:
        if name not in registered:
            raise ValueError(f'Response series requires an explicit role: {name}')
        role = 'reference' if name == reference_series else registered[name]
        item_quantity = 'phase' if name == 'Phase Error' else quantity
        result[name] = response_style(role, item_quantity, theme=theme)
        if name == highlight_series and role == 'correction':
            result[name] = replace(result[name], width=3., opacity=1., drawing_priority=95)
    return result


def response_style_help():
    """One localized help source for both applications."""
    from utils.ui_localization import display_text
    return display_text('入力は青、目標は橙、実現応答は緑、補正は紫、元目標などの参照線は灰色です。ゲインは実線、位相は破線です。各Wayの色は固定し、全体合成は黒／白の太線で表示します。色に加えて凡例名・線種・単位で区別します。')


def composite_response_styles(series, projection, configured_channels, *, theme='light'):
    """Resolve final Composite series from explicit source IDs and channel ownership."""
    ways = {str(row.get('channel_id') or f"{row.get('group')}:{row.get('name')}"): str(row['way'])
            for row in configured_channels if row.get('way')}
    channel_ways = {source.series_id: ways[source.channel_id] for source in projection.channel_sources}
    sums = {source.series_id: source.physical_group for source in projection.sum_sources}
    resolved = {}
    for item in series:
        role = {'Channel': 'way', 'SystemSum': 'sum', 'Target': 'target',
                'ReverseNull': 'reverse_null'}[item.role]
        quantity = 'phase' if item.kind == 'Phase' else 'gain'
        resolved[item.name] = response_style(role, quantity, theme=theme,
            way_id=channel_ways.get(item.series_id), source_id=sums.get(item.series_id))
    return resolved
