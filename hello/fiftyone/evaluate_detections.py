import json
import re
import shutil
import sys
import time
from pathlib import Path
from string import Template

from fiftyone import ViewField as F

import hello.fiftyone.dataset as hod

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


def make_dataset(dataset_dir, info_py="info.py", data_path="data", preds_path="predictions.txt", labels_path="labels.json"):
    dataset_dir = Path(dataset_dir or ".")

    with open(dataset_dir / info_py, "r") as f:
        codestr = f.read()

    info = eval(re.split(r"info\s*=\s*", codestr)[1])

    dataset_name = dataset_dir.name
    dataset_type = "detection"
    version = "001"

    classes = info.get("classes", [])
    mask_targets = info.get("mask_targets", {})

    hod.delete_datasets([dataset_name])
    dataset = hod.create_dataset(dataset_name, dataset_type, classes, mask_targets)

    dataset.info["version"] = version

    hod.add_images_dir(dataset, dataset_dir / data_path, None)

    hod.add_detection_labels(dataset, "predictions", dataset_dir / preds_path, classes, mode="text")
    hod.add_detection_labels(dataset, "ground_truth", dataset_dir / labels_path, classes, mode="coco")

    return dataset


def save_plot(plot, html_file):
    if hasattr(plot, "_widget"):
        plot = plot._widget

    if hasattr(plot, "write_html"):
        plot.write_html(html_file)
    elif hasattr(plot, "save"):
        plot.save(html_file)


def func(dataset_dir, info_py="info.py", data_path="data", preds_path="predictions.txt", labels_path="labels.json", output_dir=None, **kwargs):
    dataset = make_dataset(dataset_dir, info_py, data_path, preds_path, labels_path)

    classes = dataset.default_classes
    filter_exp = F("confidence") > kwargs.pop("score_thr", 0.3)
    view = dataset.filter_labels("predictions", filter_exp, only_matches=False)

    params = dict(
        gt_field="ground_truth",
        eval_key="eval",
        classes=None,
        missing=None,
        method="coco",
        iou=0.5,
        classwise=True,
        compute_mAP=False,
    )
    params.update(**kwargs)

    results = view.evaluate_detections("predictions", **params)
    results.print_report(classes=classes)

    compute_mAP = kwargs.get("compute_mAP", False)

    mAP = -1
    if compute_mAP:
        mAP = results.mAP()
        print(f"*** {mAP=:.5f}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        (output_dir).mkdir(parents=True, exist_ok=False)

        tmpl_mapping = {
            "date": time.strftime(r"%Y-%m-%d %H:%M"),
            "aggregate_metrics": format_kv("mAP", mAP),
            "report": json.dumps(results.report(classes=classes), indent=4),
        }
        readme_str = tmpl_readme.safe_substitute(tmpl_mapping)
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_str)

        if compute_mAP:
            html_file = str(output_dir / "plot_confusion_matrix.html")
            plot = results.plot_confusion_matrix(classes=classes)
            save_plot(plot, html_file)
            html_file = str(output_dir / "plot_pr_curves.html")
            plot = results.plot_pr_curves(classes=classes)
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
    parser.add_argument("--preds", dest="preds_path", type=str, default="predictions.txt",
                        help="which the predictions file")
    parser.add_argument("--labels", dest="labels_path", type=str, default="labels.json",
                        help="which the ground_truth file")
    parser.add_argument("--out", dest="output_dir", type=str, default=None,
                        help="save results to output dir")
    parser.add_argument("--score_thr", type=float, default=0.3,
                        help="minimum score of bboxes to compute")
    parser.add_argument("--iou", dest="iou", type=float, default=0.5,
                        help="the IoU threshold")
    parser.add_argument("--classwise", dest="classwise", action="store_false",
                        help="allow matches between classes")
    parser.add_argument("--mAP", dest="compute_mAP", action="store_true",
                        help="mAP and PR curves")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
