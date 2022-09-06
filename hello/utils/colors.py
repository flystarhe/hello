# https://plotly.com/python-api-reference/generated/plotly.colors.html#module-plotly.colors
import plotly.express as px

_swatches = set(["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24"])


hex_to_rgb = px.colors.hex_to_rgb


def get_colors(labels, template=None):
    if template is None:
        template = "Plotly"

    assert template in _swatches

    _colors = getattr(px.colors.qualitative, template)
    _n_colors = len(_colors)

    return {name: _colors[i % _n_colors] for i, name in enumerate(labels)}


def get_colors_rgb(labels, template=None):
    _data = get_colors(labels, template)

    return {k: hex_to_rgb(v) for k, v in _data.items()}


def get_colors_bgr(labels, template=None):
    _data = get_colors_rgb(labels, template)

    return {k: v[::-1] for k, v in _data.items()}
