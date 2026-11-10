"""Prepare response appearance before caching, preserving numeric series IDs."""
from copy import deepcopy
import json
import re

from utils.ui_localization import display_text


def response_series_layers(frame, *, styles, x, y, legend, tooltips=()):
    """Build styled layers before serialization; resolve labels per series."""
    import altair as alt

    names = sorted(frame['series'].unique(), key=lambda name: styles[name].drawing_priority)
    if not names:
        return []
    domain = sorted(styles, key=lambda name: (styles[name].legend_priority, name))
    labels = {name: response_label(name) for name in domain}
    expression = 'datum.label'
    for name in domain:
        if labels[name] != name:
            expression = (f'datum.label === {json.dumps(name)} ? '
                          f'{json.dumps(labels[name], ensure_ascii=False)} : ({expression})')
    legend = deepcopy(legend)
    legend.labelExpr = expression
    for attribute, getter in (('symbolDash', lambda style: style.vega_dash),
                              ('symbolStrokeWidth', lambda style: style.width),
                              ('symbolOpacity', lambda style: style.opacity)):
        symbol = json.dumps(getter(styles[domain[0]]))
        for name in domain:
            symbol = (f'datum.value === {json.dumps(name)} ? '
                      f'{json.dumps(getter(styles[name]))} : ({symbol})')
        setattr(legend, attribute, alt.ExprRef(expr=symbol))
    translated_tooltips = deepcopy(list(tooltips))
    for field in translated_tooltips:
        if getattr(field, 'shorthand', None) == 'series:N':
            field.shorthand = 'series_label:N'
            field.title = display_text('Series')
    layers = []
    for name in names:
        style = styles[name]
        data = frame.loc[frame['series'] == name].copy()
        data['series_label'] = labels[name]
        layers.append(alt.Chart(data).mark_line(
            clip=True, strokeWidth=style.width, strokeDash=style.vega_dash,
            opacity=style.opacity,
        ).encode(
            x=x, y=y, detail=alt.Detail('line_group:N'),
            color=alt.Color('series:N', title=None, scale=alt.Scale(
                domain=domain, range=[styles[item].color for item in domain]), legend=legend),
            tooltip=translated_tooltips,
        ))
    return layers


def response_label(value):
    text = str(value)
    translated = display_text(text)
    if translated != text:
        return translated
    # Numbered IIR series are assembled from application-owned enum labels.
    match = re.fullmatch(r"(IIR \d+ · )(Manual|Auto) (.+)", text)
    if match:
        prefix, origin, kind = match.groups()
        return prefix + display_text(origin) + " " + display_text(kind)
    return text


def localized_response_spec(spec, *, styles=None):
    """Reference projection for stored specs/tests; live charts build directly."""
    result = deepcopy(spec)
    styles = styles or {}
    names = set()

    def collect(node):
        if isinstance(node, dict):
            if isinstance(node.get("series"), str):
                names.add(node["series"])
                node["series_label"] = response_label(node["series"])
            for value in list(node.values()):
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(result)
    expression = "datum.label"
    for name in sorted(names):
        label = response_label(name)
        if name != label:
            expression = (f"datum.label === {json.dumps(name)} ? "
                          f"{json.dumps(label, ensure_ascii=False)} : ({expression})")

    def localize(node):
        if isinstance(node, dict):
            if isinstance(node.get("title"), str):
                node["title"] = response_label(node["title"])
            encoding = node.get("encoding", {})
            for channel in ("color", "strokeDash", "strokeWidth"):
                field = encoding.get(channel, {})
                if field.get("field") == "series" and field.get("legend", {}) is not None:
                    legend = field.setdefault("legend", {})
                    legend["labelExpr"] = expression
                    if channel == "color" and styles:
                        for attribute, getter in (("symbolDash", lambda style: style.vega_dash),
                                                  ("symbolStrokeWidth", lambda style: style.width),
                                                  ("symbolOpacity", lambda style: style.opacity)):
                            symbol = json.dumps(getter(next(iter(styles.values()))))
                            for name, style in styles.items():
                                symbol = (f"datum.value === {json.dumps(name)} ? "
                                          f"{json.dumps(getter(style))} : ({symbol})")
                            legend[attribute] = {"expr": symbol}
                scale = field.get("scale") or {}
                if field.get('field') == 'series' and styles:
                    domain = scale.get('domain', sorted(names))
                    domain = sorted(domain, key=lambda name: (styles[name].legend_priority, name))
                    field['scale'] = dict(domain=domain, range=[
                        {'color': styles[name].color, 'strokeWidth': styles[name].width,
                         'strokeDash': styles[name].vega_dash}[channel] for name in domain])
            if styles and encoding.get("color", {}).get("field") == "series" and "value" not in encoding.get("opacity", {}):
                encoding['opacity'] = {'condition': [
                    {'test': f'datum.series === {json.dumps(name)}', 'value': styles[name].opacity}
                    for name in names], 'value': 1}
            mark = node.get('mark')
            dataset = result.get('datasets', {}).get(node.get('data', {}).get('name'), [])
            local_names = {row['series'] for row in dataset if 'series' in row}
            if isinstance(mark, dict) and len(local_names) == 1 and styles:
                style = styles[next(iter(local_names))]
                if mark.get('type') == 'line':
                    mark.update(color=style.color, strokeWidth=style.width,
                                strokeDash=style.vega_dash, opacity=style.opacity)
                elif mark.get('type') == 'area':
                    mark.update(color=style.color)
                color = encoding.get('color', {})
                if 'value' in color:
                    color['value'] = style.color
            tooltips = encoding.get("tooltip", [])
            if isinstance(tooltips, dict):
                tooltips = [tooltips]
            for field in tooltips:
                if field.get("field") == "series":
                    field["field"] = "series_label"
                    field["title"] = display_text("Series")
            for value in list(node.values()):
                localize(value)
        elif isinstance(node, list):
            for value in node:
                localize(value)

    def order_layers(node, inherited_data=None):
        if not isinstance(node, dict):
            return
        inherited_data = node.get('data', inherited_data or {})
        if 'layer' in node:
            layers = []
            for layer in node['layer']:
                order_layers(layer, inherited_data)
                encoding = layer.get('encoding', {})
                mark = layer.get('mark', {})
                if isinstance(mark, dict) and mark.get('type') == 'line' and encoding.get('color', {}).get('field') == 'series' and styles:
                    dataset = result.get('datasets', {}).get(layer.get('data', inherited_data).get('name'), [])
                    available = {row['series'] for row in dataset if 'series' in row}
                    for name in sorted(available, key=lambda name: styles[name].drawing_priority):
                        selected = deepcopy(layer)
                        selected.setdefault('transform', []).append({'filter': {'field': 'series', 'equal': name}})
                        style = styles[name]
                        selected['encoding']['strokeWidth'] = {'value': style.width}
                        selected['encoding']['strokeDash'] = {'value': style.vega_dash}
                        selected['encoding']['opacity'] = {'value': style.opacity}
                        layers.append(selected)
                else:
                    layers.append(layer)
            node['layer'] = layers

    order_layers(result)
    localize(result)
    return result


def localized_response_figure(figure, *, styles=None):
    import plotly.graph_objects as go

    result = go.Figure(figure)
    styles = styles or {}
    legend_ranks = {name: rank for rank, name in enumerate(
        sorted(styles, key=lambda name: (styles[name].legend_priority, name)))}
    for axis in (*result.select_xaxes(), *result.select_yaxes()):
        if axis.title.text:
            axis.title.text = response_label(axis.title.text)
    for trace in result.data:
        if trace.name in styles:
            style = styles[trace.name]
            if trace.fill == 'tozeroy':
                from response_display.plotly_charts import _rgba
                trace.fillcolor = _rgba(style.color, .18)
            if trace.line.width != 0:
                trace.line.update(color=style.color, width=style.width, dash=style.dash)
                trace.opacity = style.opacity
            trace.legendrank = legend_ranks[trace.name]
        if trace.name:
            trace.name = response_label(trace.name)
    result.data = tuple(sorted(result.data, key=lambda trace: (
        styles[trace.legendgroup or trace.name].drawing_priority
        if (trace.legendgroup or trace.name) in styles else 0)))
    return result
