import json
import shutil
from pathlib import Path
from string import Template

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.utils.coco as fouc
import fiftyone.utils.yolo as fouy
from fiftyone.utils.labels import segmentations_to_detections

from hello.fiftyone.core import count_values, merge_samples
from hello.fiftyone.dataset_detections import \
    load_dataset as _load_detection_dataset
from hello.fiftyone.dataset_segmentations import \
    load_dataset as _load_segmentation_dataset
from hello.fiftyone.utils import load_predictions

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


def add_coco_labels(dataset, label_field, labels_path):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.utils.coco.html#fiftyone.utils.coco.add_coco_labels
    assert Path(labels_path).suffix == ".json"

    with open(labels_path, "r") as f:
        coco = json.load(f)

    assert "categories" in coco and "images" in coco and "annotations" in coco

    classes = [cat["name"] for cat in coco["categories"]]

    db = {Path(img["file_name"]).stem: img["id"] for img in coco["images"]}
    coco_ids = [db.get(Path(filepath).stem, -1) for filepath in dataset.values("filepath")]

    coco_id_field = "coco_id"
    dataset.set_values(coco_id_field, coco_ids)

    fouc.add_coco_labels(
        dataset,
        label_field,
        coco["annotations"],
        classes,
        label_type="detections",
        coco_id_field=coco_id_field,
    )


def add_yolo_labels(dataset, label_field, labels_path, classes):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.utils.yolo.html#fiftyone.utils.yolo.add_yolo_labels
    assert isinstance(classes, list)

    fouy.add_yolo_labels(
        dataset,
        label_field,
        labels_path,
        classes,
    )


def add_detection_labels(dataset, label_field, labels_path, classes, mode="text"):
    """Adds detection labels to the dataset.

    .. note::
        if ``mode=text``, a text row corresponds to a sample prediction result.
        row format: ``filepath,height,width,x1,y1,x2,y2,confidence,label,x1,y1,x2,y2,confidence,label``.

        if ``mode=yolo``, a txt file corresponds to a sample prediction result.
        row format: ``target,xc,yc,w,h,s``.

        if ``mode=coco``, a standard COCO format json file.
        from https://cocodataset.org/#format-data.

    Args:
        dataset (_type_): _description_
        label_field (_type_): _description_
        labels_path (_type_): _description_
        classes (_type_): _description_
        mode (str, optional): _description_. Defaults to "text".
    """
    assert mode in {"text", "yolo", "coco"}

    if mode == "coco":
        add_coco_labels(dataset, label_field, labels_path)
        return

    filepaths, ids = dataset.values(["filepath", "id"])
    id_map = {Path(k).stem: v for k, v in zip(filepaths, ids)}

    db = load_predictions(labels_path, classes=classes, mode=mode)

    stems_adds = set(db.keys())
    stems_base = set(id_map.keys())

    bad_stems = stems_adds - stems_base
    if bad_stems:
        print(f"Ignoring {len(bad_stems)} nonexistent images\n  (eg {bad_stems[:6]})")

    stems = sorted(stems_adds & stems_base)
    matched_ids = [id_map[stem] for stem in stems]
    view = dataset.select(matched_ids, ordered=True)

    labels = []
    for stem in stems:
        detections = [fol.Detection(**detection) for detection in db[stem]]
        labels.append(fol.Detections(detections=detections))

    view.set_values(label_field, labels)


def add_images_dir(images_dir, tags=None, recursive=True):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.add_images_dir
    raise NotImplementedError


def add_dataset_dir(dataset_dir, data_path=None, labels_path=None, label_field=None, tags=None):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.add_dir
    raise NotImplementedError


def load_images_dir(dataset_dir, dataset_name=None, dataset_type=None, classes=[], mask_targets={}):
    # `dataset_type` (None) - a string. The possible values are: `detection`, `segmentation`.
    dataset = fo.Dataset.from_images_dir(dataset_dir)

    if dataset_name:
        dataset.name = dataset_name
        dataset.persistent = True

    info = {
        "dataset_name": dataset_name if dataset_name else "dataset-name",
        "dataset_type": dataset_type if dataset_type else "unknown",
        "version": "001",
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
    shutil.rmtree(export_dir, ignore_errors=True)

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
