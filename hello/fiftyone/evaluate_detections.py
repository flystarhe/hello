import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from string import Template

import fiftyone as fo
from utils.importer import load_from_file

dataset_doc_str = """
    <dataset_name>/
    ├── README.md  # 按照Markdown标准扩展信息
    ├── data
    │   ├── 000000000030.jpg
    │   ├── 000000000036.jpg
    │   └── 000000000042.jpg
    ├── info.py
    ├── labels.json  # ground_truth
    └── predictions.txt  # predictions

    ground_truth/predictions:
        - *.json: COCO format
        - *.txt: An inference result saves a row
            filepath,height,width,x1,y1,x2,y2,confidence,label,x1,y1,x2,y2,confidence,label
"""

tmpl_readme = """# README
- `$date`

---

[TOC]

## Metrics
- `$mAP`

```json
$report
```
"""
tmpl_readme = Template(tmpl_readme)


def parse_text_line(line):
    """
    text format:
        filepath,height,width,x1,y1,x2,y2,confidence,label,x1,y1,x2,y2,confidence,label
    """
    vals = line.split(",")

    assert len(vals) >= 3, "filepath,height,width,..."

    filepath = vals[0].strip()
    height = int(vals[1])
    width = int(vals[2])
    data = vals[3:]

    group_size = 6
    total_size = len(data)
    assert total_size % group_size == 0

    def _parse(args):
        x1, y1, x2, y2 = [float(v) for v in args[:4]]
        confidence = float(args[4])
        label = args[5].strip()
        return x1, y1, x2, y2, confidence, label

    detections = []
    for i in range(0, total_size, group_size):
        x1, y1, x2, y2, confidence, label = _parse(data[i:(i + group_size)])

        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        bounding_box = [x / width, y / height, w / width, h / height]

        detections.append(
            fo.Detection(
                label=label,
                bounding_box=bounding_box,
                confidence=confidence,
            )
        )

    return filepath, detections


def detections_from_text(text_file):
    with open(text_file, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    lines = [l for l in lines if l and not l.startswith("#")]

    data = {}
    for text in lines:
        filepath, detections = parse_text_line(text)
        detections = fo.Detections(detections=detections)
        data[Path(filepath).name] = (filepath, detections)

    return data


def detections_from_coco(json_file):
    with open(json_file, "r") as f:
        coco = json.load(f)

    imgs = {img["id"]: img for img in coco["images"]}
    cats = {cat["id"]: cat for cat in coco["categories"]}

    group_anns = defaultdict(list)
    for ann in coco["annotations"]:
        _img = imgs[ann["image_id"]]

        filepath = _img["file_name"]
        height = _img["height"]
        width = _img["width"]

        _cat = cats[ann["category_id"]]

        label = _cat["name"]

        x, y, w, h = ann["bbox"]
        bounding_box = [x / width, y / height, w / width, h / height]

        confidence = 1.0

        group_anns[filepath].append(
            fo.Detection(
                label=label,
                bounding_box=bounding_box,
                confidence=confidence,
            )
        )

    data = {}
    for _img in coco["images"]:
        filepath = _img["file_name"]
        detections = group_anns[filepath]
        detections = fo.Detections(detections=detections)
        data[Path(filepath).name] = (filepath, detections)

    return data


def make_dataset(info_py="info.py", data_path="data", labels_path="labels.json", preds_path="predictions.txt"):
    assert labels_path is not None and preds_path is not None
    assert Path(labels_path).is_file() and Path(preds_path).is_file()

    dataset = fo.Dataset()

    if info_py is not None and Path(info_py).is_file():
        info = load_from_file("info_py", info_py)
    else:
        info = {
            "dataset_name": "evaluate_detections",
            "dataset_type": "detection",
            "version": "0.01",
            "classes": [],
            "num_samples": {},
            "tail": {},
        }

    dataset.default_classes = info.pop("classes", [])
    dataset.info = info
    dataset.save()

    flag = False
    if data_path is not None:
        data_path = Path(data_path)
        if data_path.is_dir():
            flag = True

    detections_gt = detections_from_coco(labels_path)
    detections_dt = detections_from_text(preds_path)

    gt_imgs, dt_imgs = set(detections_gt.keys()), set(detections_dt.keys())
    print(f"{len(gt_imgs)=},{len(dt_imgs)=},{(gt_imgs - dt_imgs)=}")

    for filepath, predictions in detections_dt.values():
        filepath, ground_truth = detections_gt[Path(filepath).name]

        if flag:
            filepath = str(data_path / filepath)

        sample = fo.Sample(
            filepath=filepath,
            predictions=predictions,
            ground_truth=ground_truth,
        )
        dataset.add_sample(sample)

    # Populate the `metadata` field
    dataset.compute_metadata()

    return dataset


def save_plot(plot, html_file):
    if hasattr(plot, "_widget"):
        plot = plot._widget

    if hasattr(plot, "write_html"):
        plot.write_html(html_file)
    elif hasattr(plot, "save"):
        plot.save(html_file)


def func(dataset_dir, info_py="info.py", data_path="data", labels_path="labels.json", preds_path="predictions.txt", output_dir=None, **kwargs):
    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)

        if info_py is not None:
            info_py = dataset_dir / info_py

        if data_path is not None:
            data_path = dataset_dir / data_path

        if labels_path is not None:
            labels_path = dataset_dir / labels_path

        if preds_path is not None:
            preds_path = dataset_dir / preds_path

    dataset = make_dataset(info_py, data_path, labels_path, preds_path)

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

    results = dataset.evaluate_detections("predictions", **params)
    results.print_report()

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
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "mAP": f"{mAP=:.5f}",
            "report": json.dumps(results.report(), indent=4),
        }
        readme_str = tmpl_readme.safe_substitute(tmpl_mapping)
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_str)

        if compute_mAP:
            html_file = str(output_dir / "plot_confusion_matrix.html")
            plot = results.plot_confusion_matrix()
            save_plot(plot, html_file)
            html_file = str(output_dir / "plot_pr_curves.html")
            plot = results.plot_pr_curves()
            save_plot(plot, html_file)


def parse_args(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Evaluating Predictions")

    parser.add_argument("--root", dest="dataset_dir", type=str, default=None,
                        help="dataset root dir")
    parser.add_argument("--info", dest="info_py", type=str, default=None,
                        help="path of info.py")
    parser.add_argument("--data", dest="data_path", type=str, default=None,
                        help="which the images")
    parser.add_argument("--labels", dest="labels_path", type=str, default=None,
                        help="which the labels file")
    parser.add_argument("--preds", dest="preds_path", type=str, default=None,
                        help="which the predictions file")
    parser.add_argument("--out", dest="output_dir", type=str, default=None,
                        help="save evaluation results to output dir")
    parser.add_argument("--iou", dest="iou", type=float, default=0.5,
                        help="the IoU threshold")
    parser.add_argument("--classwise", dest="classwise", action="store_false",
                        help="allow matches between classes")
    parser.add_argument("--mAP", dest="compute_mAP", action="store_true",
                        help="mAP and PR curves")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    print(dataset_doc_str)
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
