from pathlib import Path
from string import Template

import fiftyone as fo
import fiftyone.utils.yolo as fouy
from fiftyone.utils.labels import segmentations_to_detections

from hello.fiftyone.core import count_values, merge_samples
from hello.fiftyone.dataset_detections import load_dataset as _load_detection_dataset
from hello.fiftyone.dataset_segmentations import load_dataset as _load_segmentation_dataset

tmpl_info = """\
info = {
    'dataset_name': '$dataset_name',
    'dataset_type': '$dataset_type',
    'version': '$version',
    'classes': $classes,
    'mask_targets': $mask_targets,
    'num_samples': $num_samples,
    'tail': $tail,
}
"""
tmpl_info = Template(tmpl_info)


def add_mmdet_labels(dataset, label_field, labels_path, classes=None, include_missing=False):
    classes = classes or dataset.default_classes

    return dataset


def add_yolov5_labels(dataset, label_field, labels_path, classes=None, include_missing=False):
    classes = classes or dataset.default_classes

    fouy.add_yolo_labels(
        dataset,
        label_field,
        labels_path,
        classes,
        include_missing,
    )

    return dataset


def load_images_dir(dataset_dir, dataset_name=None, dataset_type=None, classes=[], mask_targets={}):
    # `dataset_type` (None) - a string. The possible values are: `detection`, `segmentation`.
    dataset = fo.Dataset.from_images_dir(dataset_dir)

    if dataset_name:
        dataset.name = dataset_name
        dataset.persistent = True

    info = {
        "dataset_name": dataset_name if dataset_name else "dataset-name",
        "dataset_type": dataset_type if dataset_type else "unknown",
        "version": "0.01",
        "classes": classes,
        "mask_targets": mask_targets,
        "num_samples": {},
        "tail": {},
    }

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    return dataset


def load_detection_dataset(dataset_dir, info_py="info.py", data_path="data", labels_path="labels.json", field_name="ground_truth", splits=None):
    dataset_dir = Path(dataset_dir)

    if splits is None:
        dataset = _load_detection_dataset(str(dataset_dir), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
        dataset.tag_samples("train")
    else:
        _datasets = []
        for s in splits:
            _dataset = _load_detection_dataset(str(dataset_dir / s), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
            _dataset.tag_samples(s)
            _datasets.append(_dataset)
        dataset = merge_samples(_datasets)

    return dataset


def load_segmentation_dataset(dataset_dir, info_py="info.py", data_path="data", labels_path="labels/", field_name="ground_truth", splits=None):
    dataset_dir = Path(dataset_dir)

    if splits is None:
        dataset = _load_segmentation_dataset(str(dataset_dir), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
        dataset.tag_samples("train")
    else:
        _datasets = []
        for s in splits:
            _dataset = _load_segmentation_dataset(str(dataset_dir / s), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
            _dataset.tag_samples(s)
            _datasets.append(_dataset)
        dataset = merge_samples(_datasets)

    return dataset


def export_detection_dataset(export_dir, dataset, label_field):
    return export_dataset(export_dir, dataset, label_field=label_field)


def export_segmentation_dataset(export_dir, dataset, label_field, mask_types="stuff"):
    return export_dataset(export_dir, dataset, mask_label_field=label_field, mask_types=mask_types)


def export_dataset(export_dir, dataset, label_field=None, mask_label_field=None, mask_types="stuff"):
    # mask_types: "stuff"(amorphous regions of pixels), "thing"(connected regions, each representing an instance)
    assert label_field is not None or mask_label_field is not None

    dataset.save()
    info = dataset.info
    classes = dataset.default_classes
    mask_targets = dataset.default_mask_targets
    info["num_samples"] = count_values(dataset, "tags")

    if label_field is None:
        label_field = "detections"
        print("todo: segmentations_to_detections()")
        dataset = dataset.select_fields(mask_label_field).clone()
        segmentations_to_detections(dataset, mask_label_field, label_field, mask_targets=dataset.default_mask_targets, mask_types=mask_types)

    splits = dataset.distinct("tags")

    if not splits:
        splits = ["train"]
        dataset.tag_samples("train")

    for split in splits:
        print(f"\n[{split}]\n")
        view = dataset.match_tags(split)
        curr_dir = Path(export_dir) / split

        view.export(
            export_dir=str(curr_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
            classes=classes,
        )

        if mask_label_field is not None:
            view.export(
                dataset_type=fo.types.ImageSegmentationDirectory,
                labels_path=str(curr_dir / "labels"),
                label_field=mask_label_field,
                mask_targets=mask_targets,
            )

        info["tail"].update(count_label=count_values(view, f"{label_field}.detections.label"))

        info_py = tmpl_info.safe_substitute(info,
                                            classes=classes,
                                            mask_targets=mask_targets)

        with open(curr_dir / "info.py", "w") as f:
            f.write(info_py)

    return export_dir
