import code
import sys
from pathlib import Path

from hello.fiftyone.core import merge_samples
from hello.utils import importer

import fiftyone as fo
from fiftyone.utils.labels import (objects_to_segmentations,
                                   segmentations_to_detections)

dataset_doc_str = """\
tips:

    <dataset_name>/
    ├── README.md  # 按照Markdown标准扩展信息
    ├── data
    │   ├── 000000000030.jpg
    │   ├── 000000000036.jpg
    │   └── 000000000042.jpg
    ├── labels  # ground_truth
    │   ├── 000000000030.png
    │   ├── 000000000036.png
    │   └── 000000000042.png
    ├── predictions  # predictions
    │   ├── 000000000030.png
    │   ├── 000000000036.png
    │   └── 000000000042.png
    └── info.py

    ground_truth/predictions:
        - the png file type as uint8
        - 0 means background, 255 means others

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
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        label_types=["segmentations"],
        data_path=data_path,
        labels_path=labels_path,
        label_field=f"{field_name}_coco",
    )

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    objects_to_segmentations(dataset, f"{field_name}_coco", field_name, mask_targets=dataset.default_mask_targets, thickness=1)

    return dataset


def load_png_dataset(info, data_path, labels_path, field_name):
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageSegmentationDirectory,
        data_path=data_path,
        labels_path=labels_path,
        label_field=field_name,
    )

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    segmentations_to_detections(dataset, field_name, f"{field_name}_coco", mask_targets=dataset.default_mask_targets, mask_types="stuff")

    return dataset


def load_dataset(dataset_dir, info_py="info.py", data_path="data", labels_path="labels/", field_name="ground_truth"):
    dataset_dir = Path(dataset_dir or ".")

    if dataset_dir.is_dir():
        info_py = dataset_dir / info_py
        data_path = dataset_dir / data_path
        labels_path = dataset_dir / labels_path
    else:
        info_py = Path(info_py)
        data_path = Path(data_path)
        labels_path = Path(labels_path)

    assert data_path.is_dir() and labels_path.exists()

    info = {
        "dataset_name": "dataset-name",
        "dataset_type": "segmentation",
        "version": "0.01",
        "classes": [],
        "mask_targets": {},
        "num_samples": {},
        "tail": {},
    }

    if info_py.is_file():
        info.update(importer.load_from_file("info_py", info_py).info)

    suffix = labels_path.suffix

    if labels_path.is_dir():
        dataset = load_png_dataset(info, data_path, labels_path, field_name)
    elif suffix == ".json":
        dataset = load_coco_dataset(info, data_path, labels_path, field_name)
    else:
        raise NotImplementedError

    return dataset


def func(dataset_dir, info_py="info.py", data_path="data", labels_path="labels/", preds_path=None):
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
    parser.add_argument("--labels", dest="labels_path", type=str, default="labels/",
                        help="which the ground_truth file or dir")
    parser.add_argument("--preds", dest="preds_path", type=str, default=None,
                        help="which the predictions file or dir")

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
