from pathlib import Path
from string import Template

import fiftyone as fo
from fiftyone.utils.labels import segmentations_to_detections

from .core import count_values
from .dataset_detections import load_dataset as _load_detection_dataset
from .dataset_segmentations import load_dataset as _load_segmentation_dataset

tmpl_info = """info = {
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


def load_images_dir(dataset_dir, dataset_name=None, dataset_type=None, classes=[], mask_targets={}):
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


load_detection_dataset = _load_detection_dataset


load_segmentation_dataset = _load_segmentation_dataset


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
