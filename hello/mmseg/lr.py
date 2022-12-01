r"""LRFinder(https://docs.fast.ai/callback.schedule.html#lrfinder)

.. math::
    \begin{align}
    & lr_i = lr_{min} \times q^i , \quad i \in [0, N-1]\\
    & lr_{min} = 1e-07 , \quad lr_{max} = 10\\
    & q = \lgroup \frac{lr_{max}}{lr_{min}} \rgroup ^ {\frac{1}{N-1}}
    \end{align}
"""
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_pattern = re.compile(r"Iter \[(\d+)/(\d+)\]")


def valley(lrs: list, losses: list):
    """Suggests a learning rate from the longest valley and returns its index.

    https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py

    Args:
        lrs (list): _description_
        losses (list): _description_
    """
    n = len(losses)
    max_start, max_end = 0, 0

    # find the longest valley
    lds = [1] * n
    for i in range(1, n):
        for j in range(0, i):
            if (losses[i] < losses[j]) and (lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
            if lds[max_end] < lds[i]:
                max_end = i
                max_start = max_end - lds[max_end]

    sections = (max_end - max_start) / 3
    idx = max_start + int(sections) + int(sections/2)

    return lrs[idx], losses[idx]


def to_dict(text):
    data = {}
    for sub_text in text.strip().split(","):
        k, v = sub_text.strip().split(":", maxsplit=1)
        data[k.strip()] = v.strip()
    return data


def load_text_log(text_log):
    log_dict = defaultdict(list)
    with open(text_log, "r") as log_file:
        for line in log_file:
            log = line.strip()

            match = _pattern.search(log)
            if match:
                str_iter = match.group(1)
                log_dict["iter"].append(int(str_iter))

                data = to_dict(log[match.end():])
                for k, v in data.items():
                    log_dict[k].append(v)

    return log_dict


def plot_lr_loss(log_dict, lr_field="lr", loss_field="loss", reduction="mean"):
    df = pd.DataFrame({
        "iter": log_dict["iter"],
        "label": log_dict[lr_field],
        lr_field: [float(i) for i in log_dict[lr_field]],
        loss_field: [float(i) for i in log_dict[loss_field]]
    })

    valley_suggestion, valley_loss = valley(df[lr_field].tolist(), df[loss_field].tolist())

    data = df.groupby(by=["label"], as_index=False, sort=False).mean()

    idx = data[loss_field].argmin()
    minimum_suggestion, minimum_loss = data[lr_field][idx] / 10 / 0.75, data[loss_field][idx]

    if reduction == "mean":
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}]]
        )

        fig.add_trace(
            go.Scatter(
                x=data[lr_field],
                y=data[loss_field],
                mode="lines",
                name=lr_field
            ),
            row=2, col=1
        )
        fig.update_xaxes(title_text=lr_field, row=2, col=1)
        fig.update_yaxes(title_text=loss_field, row=2, col=1)
    else:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            specs=[[{"type": "table"}],
                   [{"type": "scatter"}],
                   [{"type": "scatter"}]]
        )

        fig.add_trace(
            go.Scatter(
                x=df["iter"],
                y=df[lr_field],
                mode="lines",
                name=lr_field
            ),
            row=3, col=1
        )
        fig.update_xaxes(title_text="iter", row=3, col=1)
        fig.update_yaxes(title_text=lr_field, row=3, col=1)

        fig.add_trace(
            go.Scatter(
                x=df["iter"],
                y=df[loss_field],
                mode="lines",
                name=loss_field
            ),
            row=2, col=1
        )
        fig.update_yaxes(title_text=loss_field, row=2, col=1)  # share xaxes

    data[lr_field] = data[lr_field].apply(lambda x: f"{x:.3e}")
    data[loss_field] = data[loss_field].apply(lambda x: f"{x:.4f}")

    fig.add_trace(
        go.Table(
            header=dict(
                values=["iter", lr_field, loss_field],
                font=dict(size=10),
                align="center"
            ),
            cells=dict(
                values=[data[k].tolist() for k in ["iter", lr_field, loss_field]],
                align="right"
            )
        ),
        row=1, col=1
    )

    title_text = "<br>".join([
        "Training with exponentially growing learning rate",
        f"valley suggestion: lr={valley_suggestion:.3e}, loss={valley_loss:.4f}",
        f"minimum suggestion: lr={minimum_suggestion:.3e}, loss={minimum_loss:.4f}"
    ])
    fig.update_layout(height=800, showlegend=False, title_text=title_text)

    return fig


def func(text_logs, out_dir, lr_field="lr", loss_field="loss", reduction="mean", format=".png"):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=False)

    if Path(text_logs[0]).is_dir():
        text_logs = [str(f) for f in Path(text_logs[0]).glob("*.log")]

    for text_log in text_logs:
        log_dict = load_text_log(text_log)
        fig = plot_lr_loss(log_dict, lr_field=lr_field, loss_field=loss_field, reduction=reduction)
        out_file = str(out_dir / f"images/lr_loss_{Path(text_log).stem}{format}")
        if format == ".html":
            fig.write_html(out_file)
        else:
            fig.write_image(out_file)
    return str(out_dir)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("text_logs", type=str, nargs='+',
                        help="path of train log in text format or runs dir")
    parser.add_argument("-o", dest="out_dir", type=str,
                        help="save plotting curves to the dir")
    parser.add_argument("-a", dest="lr_field", type=str, default="lr",
                        help="lr field name")
    parser.add_argument("-b", dest="loss_field", type=str, default="loss",
                        help="loss field name")
    parser.add_argument("-r", dest="reduction", type=str, default="none",
                        choices=["none", "mean"],
                        help="the method used to reduce the loss")
    parser.add_argument("-f", dest="format", type=str, default=".png",
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
