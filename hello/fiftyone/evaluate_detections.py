import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

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


def parse_from_dict(preds):
    """
    preds = {
        "filepath": filepath,
        "height": height,
        "width": width,
        "data": [(x1, y1, x2, y2, confidence, label)]
    }
    """
    filepath = preds["filepath"]
    height = preds["height"]
    width = preds["width"]

    detections = []
    for x1, y1, x2, y2, confidence, label in preds["data"]:
        bounding_box = [x1 / width, y1 / height,
                        (x2 - x1) / width, (y2 - y1) / height]
        detections.append(
            fo.Detection(
                label=label,
                bounding_box=bounding_box,
                confidence=confidence,
            )
        )

    return filepath, fo.Detections(detections=detections)


def parse_from_text(text):
    """
    text format:
        filepath,height,width,x1,y1,x2,y2,confidence,label,x1,y1,x2,y2,confidence,label
    """
    vals = text.split(",")

    assert len(vals) >= 3, "filepath,height,width,..."

    filepath = vals[0].strip()
    height = int(vals[1])
    width = int(vals[2])
    data = vals[3:]

    group_size = 6
    total_size = len(data)
    assert total_size % group_size == 0

    def _func(args):
        x1, y1, x2, y2 = [int(v) for v in args[:4]]
        confidence = float(args[4])
        label = args[5].strip()
        return x1, y1, x2, y2, confidence, label

    preds = {
        "filepath": filepath,
        "height": height,
        "width": width,
        "data": [_func(data[i:(i + group_size)]) for i in range(0, total_size, group_size)]
    }

    return parse_from_dict(preds)


def detections_from_text(text_file):
    if text_file is None:
        return None

    if not Path(text_file).is_file():
        return None

    with open(text_file, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    lines = [l for l in lines if not l.startswith("#")]

    data = {}
    for text in lines:
        filepath, detections = parse_from_text(text)
        data[Path(filepath).name] = (filepath, detections)

    return data


def detections_from_coco(json_file):
    if json_file is None:
        return None

    if not Path(json_file).is_file():
        return None

    with open(json_file, "r") as f:
        coco = json.load(f)

    imgs = {x["id"]: x for x in coco["images"]}
    cats = {x["id"]: x for x in coco["categories"]}

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
    for filepath, detections in group_anns.items():
        detections = fo.Detections(detections=detections)
        data[Path(filepath).name] = (filepath, detections)

    return data


def make_dataset(info_py="info.py", data_path="data", labels_path="labels.json", detections_dt="predictions.txt"):
    assert labels_path is not None and detections_dt is not None

    dataset = fo.Dataset()

    if info_py is not None and Path(info_py).is_file():
        info = load_from_file("info_py", info_py)
    else:
        info = {
            "dataset_name": "evaluate_detections",
            "dataset_type": "detection",
            "version": "0.01",
            "classes": ["person", "animal"],
            "num_samples": {},
            "tail": {},
        }

    dataset.default_classes = info.pop("classes", [])
    dataset.info = info
    dataset.save()

    detections_gt = detections_from_coco(labels_path)
    detections_dt = detections_from_text(detections_dt)

    for filepath, detections in detections_dt.values():
        sample = fo.Sample(
            filepath=filepath,
            predictions=detections,
            ground_truth=detections_gt[Path(filepath)][1],
        )
        dataset.add_sample(sample)

    # Populate the `metadata` field
    dataset.compute_metadata()

    return dataset


def func(dataset_dir, info_py="info.py", data_path="data", labels_path="labels.json", predictions=None, output_dir=None, launch=False):
    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir)

        if info_py is not None:
            info_py = dataset_dir / info_py

        if data_path is not None:
            data_path = dataset_dir / data_path

        if labels_path is not None:
            labels_path = dataset_dir / labels_path

        if predictions is not None:
            predictions = dataset_dir / predictions

    dataset = make_dataset(info_py, data_path, labels_path, predictions)

    if output_dir is not None:
        output_dir = Path(output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)
        (output_dir / "data").mkdir(parents=True, exist_ok=False)
        if data_path is not None:
            print("save: images, ground_truth, predictions")
        else:
            print("save: ground_truth, predictions")


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
    parser.add_argument("--preds", dest="predictions", type=str, default=None,
                        help="which the predictions file")
    parser.add_argument("-o", dest="output_dir", type=str, default=None,
                        help="output dir")
    parser.add_argument("--launch", action="store_true",
                        help="launch the FiftyOne App")

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
