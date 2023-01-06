import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def simplify(data):
    res = dict()
    for k, v in data.items():
        if k == "iter" or k == "epoch":
            res[k] = np.max(v)
        else:
            res[k] = np.mean(v)
    return res


def load_json_log(json_log, schedules=["iter", "lr", "loss_cls", "loss_bbox"], metrics=["bbox_mAP", "bbox_mAP_50"]):
    with open(json_log, "r") as f:
        lines = [l.strip() for l in f.read().splitlines()]

    lines = [l for l in lines if l and not l.startswith("#")]

    x_labels = set(["iter", "epoch"])

    log_dict = defaultdict(list)
    cache = defaultdict(list)
    for row in lines:
        log = json.loads(row)

        mode = log.pop("mode", None)

        if mode == "train":
            for k, v in log.items():
                if k in schedules or k in x_labels:
                    cache[k].append(v)
        elif mode == "val":
            for k, v in log.items():
                if k in metrics:
                    log_dict[k].append(v)

            for k, v in simplify(cache).items():
                log_dict[k].append(v)

            cache = defaultdict(list)
    return log_dict


def plotting_log_dicts(log_dicts, out_dir, schedules, metrics, format):
    cache = defaultdict(list)
    for exp_name, log_dict in log_dicts:
        for k, v in log_dict.items():
            cache[k].extend(v)
        cache["exp_name"].extend([exp_name] * len(v))
    cache = pd.DataFrame(cache)

    columns = set(cache.columns)
    assert "iter" in columns and "epoch" in columns

    cache["iter"] = cache["iter"] * cache["epoch"]

    x_label, schedules = schedules[0], schedules[1:]
    y_labels = schedules + metrics

    for y_label in y_labels:
        out_file = str(out_dir / f"images/{y_label}{format}")
        plotting_metrics(cache, x_label, y_label, out_file)

    for exp_name, data in cache.groupby(by="exp_name"):
        out_file = str(out_dir / f"images/{exp_name}{format}")
        plotting_schedules(data, x_label, y_labels, out_file)


def plotting_metrics(cache, x_label, y_label, out_file):
    fig = go.Figure()

    exp_names = []
    for exp_name, data in cache.groupby(by="exp_name"):
        fig.add_trace(
            go.Scatter(
                x=data[x_label],
                y=data[y_label],
                showlegend=True,
                mode="lines",
                name=exp_name.split("_", maxsplit=1)[0]
            )
        )
        exp_names.append(exp_name)

    fig.update_yaxes(title_text=y_label)
    fig.update_xaxes(title_text=x_label)

    title_text = "<br>".join([
        "Analyze MMDetection Training Json Log",
        f"{exp_names[0]},..."
    ])
    fig.update_layout(title_text=title_text)

    if Path(out_file).suffix == ".html":
        fig.write_html(out_file)
    else:
        fig.write_image(out_file)

    return fig


def plotting_schedules(cache, x_label, y_labels, out_file):
    n_plot = len(y_labels)

    fig = make_subplots(
        rows=n_plot, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=[[{"type": "scatter"}] for _ in range(n_plot)]
    )

    for i, y_label in enumerate(y_labels, 1):
        fig.add_trace(
            go.Scatter(
                x=cache[x_label],
                y=cache[y_label],
                showlegend=False,
                mode="lines",
                name=y_label
            ),
            row=i, col=1
        )
        fig.update_yaxes(title_text=y_label, row=i, col=1)

    fig.update_xaxes(title_text=x_label, row=n_plot, col=1)

    title_text = "<br>".join([
        "Analyze MMDetection Training Json Log",
        str(y_labels)
    ])
    fig.update_layout(height=300*n_plot, title_text=title_text)

    if Path(out_file).suffix == ".html":
        fig.write_html(out_file)
    else:
        fig.write_image(out_file)

    return fig


def func(json_logs, out_dir, schedules, metrics, format):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=False)

    if Path(json_logs[0]).is_dir():
        json_logs = [str(f) for f in Path(json_logs[0]).glob("*/*.log.json")]

    log_dicts = []
    for json_log in json_logs:
        log_dict = load_json_log(json_log, schedules, metrics)
        log_dicts.append([Path(json_log).parent.name, log_dict])

    plotting_log_dicts(log_dicts, out_dir, schedules, metrics, format)
    return str(out_dir)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("json_logs", type=str, nargs='+',
                        help="path of train log in json format or runs dir")
    parser.add_argument("-o", dest="out_dir", type=str,
                        help="save plotting curves to the dir")
    parser.add_argument("-s", dest="schedules", type=str, nargs='+',
                        default=["iter", "lr", "loss_cls", "loss_bbox"],
                        help="the schedule that you want to plot")
    parser.add_argument("-m", dest="metrics", type=str, nargs='+',
                        default=["bbox_mAP", "bbox_mAP_50"],
                        help="the metric that you want to plot")
    parser.add_argument("-f", dest="format", type=str, default=".html",
                        choices=[".png", ".svg", ".pdf", ".html"],
                        help="image save format")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
