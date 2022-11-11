import json
import shutil
import sys
import time
from pathlib import Path
from string import Template

import hello.fiftyone.dataset_segmentations as hod

tmpl_readme = """\
# README
- `$date`

---

[TOC]

## Metrics
$aggregate_metrics

```json
$report
```
"""
tmpl_readme = Template(tmpl_readme)


def format_kv(k, v):
    if isinstance(v, float):
        v = f"{v:.5f}"

    return f"- `{k}: {v}`"


def make_dataset(dataset_dir, info_py="info.py", data_path="data", preds_path="predictions/", labels_path="labels/"):
    A = hod.load_dataset(dataset_dir, info_py=info_py, data_path=data_path, labels_path=preds_path, field_name="predictions")
    B = hod.load_dataset(dataset_dir, info_py=info_py, data_path=data_path, labels_path=labels_path, field_name="ground_truth")

    dataset = hod.merge_samples([A, B])
    return dataset


def save_plot(plot, html_file):
    if hasattr(plot, "_widget"):
        plot = plot._widget

    if hasattr(plot, "write_html"):
        plot.write_html(html_file)
    elif hasattr(plot, "save"):
        plot.save(html_file)


def func(dataset_dir, info_py="info.py", data_path="data", preds_path="predictions/", labels_path="labels/", output_dir=None, **kwargs):
    dataset = make_dataset(dataset_dir, info_py, data_path, preds_path, labels_path)

    params = dict(
        gt_field="ground_truth",
        eval_key="eval",
        mask_targets=dataset.default_mask_targets,
        method="simple",
        bandwidth=None,
        average="micro",
    )
    params.update(**kwargs)

    results = dataset.evaluate_segmentations("predictions", **params)
    results.print_report()

    if output_dir is not None:
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        (output_dir).mkdir(parents=True, exist_ok=False)

        tmpl_mapping = {
            "date": time.strftime(r"%Y-%m-%d %H:%M"),
            "aggregate_metrics": "\n".join([format_kv(k, v) for k, v in results.metrics().items()]),
            "report": json.dumps(results.report(), indent=4),
        }
        readme_str = tmpl_readme.safe_substitute(tmpl_mapping)
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_str)

        html_file = str(output_dir / "plot_confusion_matrix.html")
        plot = results.plot_confusion_matrix()
        save_plot(plot, html_file)
    return "\n[END]"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset_dir", type=str,
                        help="base dir")
    parser.add_argument("--info", dest="info_py", type=str, default="info.py",
                        help="which the info.py")
    parser.add_argument("--data", dest="data_path", type=str, default="data",
                        help="which the images")
    parser.add_argument("--preds", dest="preds_path", type=str, default="predictions/",
                        help="which the predictions file or dir")
    parser.add_argument("--labels", dest="labels_path", type=str, default="labels/",
                        help="which the ground_truth file or dir")
    parser.add_argument("--out", dest="output_dir", type=str, default=None,
                        help="save results to output dir")
    parser.add_argument("--bandwidth", dest="bandwidth", type=int, default=None,
                        help="evaluate only along the contours of the ground truth masks")
    parser.add_argument("--average", dest="average", type=str, default="micro",
                        choices=["micro", "macro", "weighted", "samples"],
                        help="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
