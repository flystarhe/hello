# https://plotly.com/python-api-reference/generated/plotly.colors.html#module-plotly.colors
# https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
# pip install plotly
import cv2 as cv
import numpy as np
import plotly.express as px

_swatches = set(["Plotly", "D3", "G10", "T10", "Alphabet", "Dark24", "Light24"])


hex_to_rgb = px.colors.hex_to_rgb


def get_colors(labels, template=None):
    if template is None:
        template = "Plotly"

    assert template in _swatches

    _colors = getattr(px.colors.qualitative, template)
    _n_colors = len(_colors)

    return {l: _colors[i % _n_colors] for i, l in enumerate(labels)}


def get_colors_rgb(labels, template=None):
    _data = get_colors(labels, template)

    return {k: hex_to_rgb(v) for k, v in _data.items()}


def get_colors_bgr(labels, template=None):
    _data = get_colors_rgb(labels, template)

    return {k: v[::-1] for k, v in _data.items()}


def gen_palette(labels, template=None, size=80, out_file="palette.png"):
    data = get_colors_rgb(labels, template)
    colors = [data[l] for l in labels]

    n = len(labels)

    a, b = divmod(n, 8)
    c = a + 1 if b > 0 else a

    img_rows = []
    for i in range(c):
        img_row = []
        for j in range(8):
            index = min(i * 8 + j, n - 1)
            name, color = labels[index], colors[index]
            block = np.full((size, size, 3), color, dtype="uint8")
            for k, word in enumerate(name.split(), 1):
                cv.putText(block, word, (5, 15 * k), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            img_row.append(block)
        img_rows.append(np.concatenate(img_row, axis=1))
    img = np.concatenate(img_rows, axis=0)
    cv.imwrite(out_file, img[..., ::-1])

    return list(zip(labels, colors))
