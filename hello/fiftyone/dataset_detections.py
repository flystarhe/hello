import code
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fiftyone as fo

from hello.fiftyone.core import merge_samples

dataset_doc_str = """\
tips:

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
            filepath,height,width,x1,y1,x2,y2,s,l,x1,y1,x2,y2,s,l

    **Basic Usage**
    - To open a dataset in the App, simply set the `session.dataset` property.
    - To load a specific view into your dataset, simply set the `session.view` property.
    - Use `session.refresh()` to refresh the App if you update a dataset outside of the App.
    - Use `session.selected` to retrieve the IDs of the currently selected samples in the App.
    - Use `session.selected_labels` to retrieve the IDs of the currently selected labels in the App.
    - Use `export_dataset()` to exports the dataset or view to disk, or `help(export_dataset)`.
    - Use `dataset.select()/dataset.exclude()` selects the samples with `session.selected`.
"""


def load_coco_dataset(info, data_path, labels_path, field_name):
    dataset = fo.Dataset()

    dataset.default_classes = info.pop("classes", [])
    dataset.info = info
    dataset.save()

    data_path = Path(data_path)

    with open(labels_path, "r") as f:
        coco = json.load(f)

    imgs = {img["id"]: img for img in coco["images"]}
    cats = {cat["id"]: cat for cat in coco["categories"]}

    db = defaultdict(list)
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

        db[filepath].append(
            fo.Detection(
                bounding_box=bounding_box,
                confidence=confidence,
                label=label,
            )
        )

    for img in coco["images"]:
        filepath = img["file_name"]
        detections = db[filepath]  # default=[]
        params = {
            "filepath": str(data_path / filepath),
            field_name: fo.Detections(detections=detections),
        }
        dataset.add_sample(fo.Sample(**params))

    # Populate the `metadata` field
    dataset.compute_metadata()

    return dataset


def load_text_dataset(info, data_path, labels_path, field_name):
    dataset = fo.Dataset()

    dataset.default_classes = info.pop("classes", [])
    dataset.info = info
    dataset.save()

    data_path = Path(data_path)

    with open(labels_path, "r") as f:
        lines = [l.strip() for l in f.read().splitlines()]

    lines = [l for l in lines if l and not l.startswith("#")]

    for row in lines:
        filepath, detections = _parse_text_row(row)
        params = {
            "filepath": str(data_path / filepath),
            field_name: fo.Detections(detections=detections),
        }
        dataset.add_sample(fo.Sample(**params))

    # Populate the `metadata` field
    dataset.compute_metadata()

    return dataset


def _parse_text_row(row):
    """\
    row format:
        ``filepath,height,width,x1,y1,x2,y2,s,l,x1,y1,x2,y2,s,l``
    """
    row_vals = row.split(",")

    assert len(row_vals) >= 3, "filepath,height,width,..."

    filepath = row_vals[0]
    height = int(row_vals[1])
    width = int(row_vals[2])
    data = row_vals[3:]

    group_size = 6
    total_size = len(data)
    assert total_size % group_size == 0

    def _parse(args):
        x1, y1, x2, y2 = [float(v) for v in args[:4]]
        return x1, y1, x2, y2, float(args[4]), args[5]

    detections = []
    for i in range(0, total_size, group_size):
        x1, y1, x2, y2, confidence, label = _parse(data[i:(i + group_size)])

        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        bounding_box = [x / width, y / height, w / width, h / height]

        detections.append(
            fo.Detection(
                bounding_box=bounding_box,
                confidence=confidence,
                label=label,
            )
        )

    return filepath, detections


def load_dataset(dataset_dir, info_py="info.py", data_path="data", labels_path="labels.json", field_name="ground_truth"):
    dataset_dir = Path(dataset_dir or ".")

    if dataset_dir.is_dir():
        info_py = dataset_dir / info_py
        data_path = dataset_dir / data_path
        labels_path = dataset_dir / labels_path
    else:
        info_py = Path(info_py)
        data_path = Path(data_path)
        labels_path = Path(labels_path)

    assert data_path.is_dir() and labels_path.is_file()

    info = {
        "dataset_name": "dataset-name",
        "dataset_type": "detection",
        "version": "001",
        "classes": [],
        "num_samples": {},
        "tail": {},
    }

    if info_py.is_file():
        with open(info_py, "r") as f:
            codestr = f.read()

        _info = eval(re.split(r"info\s*=\s*", codestr)[1])
        info.update(_info)

    suffix = labels_path.suffix

    if suffix == ".json":
        dataset = load_coco_dataset(info, data_path, labels_path, field_name)
    elif suffix == ".txt":
        dataset = load_text_dataset(info, data_path, labels_path, field_name)
    elif suffix == ".log":
        dataset = load_text_dataset(info, data_path, labels_path, field_name)
    else:
        raise NotImplementedError

    return dataset


def func(dataset_dir, info_py="info.py", data_path="data", labels_path="labels.json", preds_path=None):
    dataset = load_dataset(dataset_dir, info_py, data_path, labels_path, "ground_truth")

    if preds_path is not None:
        _dataset = load_dataset(dataset_dir, info_py, data_path, preds_path, "predictions")
        dataset = merge_samples([dataset, _dataset])

    return dataset


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset_dir", type=str,
                        help="base dir")
    parser.add_argument("--info", dest="info_py", type=str, default="info.py",
                        help="which the info.py")
    parser.add_argument("--data", dest="data_path", type=str, default="data",
                        help="which the images")
    parser.add_argument("--labels", dest="labels_path", type=str, default="labels.json",
                        help="which the ground_truth file")
    parser.add_argument("--preds", dest="preds_path", type=str, default=None,
                        help="which the predictions file")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    print(dataset_doc_str)
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    dataset = func(**kwargs)
    session = fo.launch_app(dataset, port=5151, address="0.0.0.0", remote=True)

    banner = "Use quit() or Ctrl-Z plus Return to exit"
    code.interact(banner=banner, local=locals(), exitmsg="End...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
