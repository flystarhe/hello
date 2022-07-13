import random
import shutil
import time
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.utils.labels import (segmentations_to_detections,
                                   segmentations_to_polylines)


def gen_name(suffix=".png"):
    n = random.randrange(0, 10000)
    return f"{int(time.time())}{n:04d}{suffix}"


def get_classes(dataset, field_or_expr="detections.detections.label"):
    return sorted(set(dataset.distinct(field_or_expr)) - set(["other"])) + ["other"]


def clone_sample_field(dataset, field_name, new_field_name):
    # dataset: Dataset or DatasetView
    dataset.save()
    dataset = dataset.clone()

    sample_fields = dataset.get_field_schema()
    if field_name in sample_fields:
        dataset.clone_sample_field(field_name, new_field_name)
    else:
        print(f"not found: {field_name}\n{sample_fields.keys()}")
    return dataset


def delete_sample_field(dataset, field_name, error_level=0):
    # dataset:  Dataset or DatasetView
    dataset.save()
    dataset = dataset.clone()

    sample_fields = dataset.get_field_schema()
    if field_name in sample_fields:
        dataset.delete_sample_field(field_name, error_level)
    else:
        print(f"not found: {field_name}\n{sample_fields.keys()}")
    return dataset


def rename_sample_field(dataset, field_name, new_field_name):
    # dataset:  Dataset or DatasetView
    dataset.save()
    dataset = dataset.clone()

    sample_fields = dataset.get_field_schema()
    if field_name in sample_fields:
        dataset.rename_sample_field(field_name, new_field_name)
    else:
        print(f"not found: {field_name}\n{sample_fields.keys()}")
    return dataset


def map_labels(dataset, mapping, field_name="ground_truth"):
    classes = dataset.distinct(f"{field_name}.detections.label")

    _map = {}
    for label in classes:
        if label in mapping:
            _map[label] = mapping[label]
        elif "*" in mapping:
            _map[label] = mapping["*"]
    print(f"map_labels:\n{_map}")

    sample_fields = dataset.get_field_schema()
    if field_name in sample_fields:
        dataset = dataset.map_labels(field_name, _map)
    else:
        print(f"not found: {field_name}\n{sample_fields.keys()}")
    return dataset


def filter_labels(dataset, classes, field_name="ground_truth"):
    view = dataset.filter_labels(field_name, F("label").is_in(classes))
    return view


def filter_samples(dataset, classes, field_name="ground_truth"):
    # tagged_view = dataset.match_tags("requires_annotation")
    match = F("label").is_in(classes)
    label_field = F(f"{field_name}.detections")
    view = dataset.match(label_field.filter(match).length() > 0)
    return view


def merge_samples(A, B, **kwargs):
    A.save()
    A = A.clone()

    def key_fcn(sample):
        return Path(sample.filepath).name

    A.merge_samples(B, key_fcn=key_fcn, **kwargs)

    return A


def merge_datasets(name, classes, mask_targets, info, datasets, tmp_dir="/tmp"):
    dataset = fo.Dataset(name=name, overwrite=True)
    dataset.default_classes = classes
    dataset.default_mask_targets = mask_targets
    dataset.info = info
    dataset.save()

    tmp_dir = Path(tmp_dir) / str(name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    num = 0
    for _dataset in datasets:
        info = _dataset.info
        assert "version" in info
        assert "dataset_name" in info
        prefix = f"{info['dataset_name']}:{info['version']}"
        for sample in _dataset.clone():
            num += 1
            filepath = Path(sample.filepath)
            sample["from"] = f"{prefix}/{filepath.name}"
            tempfile = tmp_dir / f"{num:012d}{filepath.suffix}"
            shutil.copyfile(filepath, tempfile)
            sample.filepath = str(tempfile)
            dataset.add_sample(sample)
    return dataset


def segmentations_to(dataset, in_field, out_field, function="detections", mask_targets=None, mask_types="stuff", tolerance=2):
    # mask_types: "stuff"(amorphous regions of pixels), "thing"(connected regions, each representing an instance)
    if mask_targets is None:
        mask_targets = dataset.default_mask_targets

    assert function in ("polylines", "detections")

    if function == "polylines":
        segmentations_to_polylines(dataset, in_field, out_field,
                                   mask_targets=mask_targets, mask_types=mask_types, tolerance=tolerance)
    elif function == "detections":
        segmentations_to_detections(dataset, in_field, out_field,
                                    mask_targets=mask_targets, mask_types=mask_types)
    else:
        raise NotImplementedError

    return dataset


def split_dataset(dataset, splits, limit=500, field_name="ground_truth"):
    dataset.untag_samples(dataset.distinct("tags"))
    dataset = dataset.shuffle()

    if splits is None:
        splits = {"val": 0.1, "train": 0.9}

    val_ids, train_ids = [], []
    for label in dataset.default_classes:
        match = (F("label") == label)
        label_field = F(f"{field_name}.detections")
        view = dataset.match(label_field.filter(match).length() > 0)
        ids = view.take(limit).values("id")

        pos_val = splits.get("val", 0.1)
        pos_train = splits.get("train", 0.9)
        if isinstance(pos_val, float):
            num_samples = len(ids)
            pos_val = int(1 + pos_val * num_samples)
            pos_train = int(1 + pos_train * num_samples)
        ids = ids[:(pos_val+pos_train)]

        val_ids.extend(ids[:pos_val])
        train_ids.extend(ids[pos_val:])

    val_ids = set(val_ids)
    train_ids = set(train_ids) - val_ids
    dataset.select(train_ids).tag_samples("train")
    dataset.select(val_ids).tag_samples("validation")
    dataset.exclude(train_ids | val_ids).tag_samples("test")

    return dataset


def export_dataset(export_dir, dataset, label_field=None, mask_label_field=None):
    assert label_field is not None or mask_label_field is not None
    mask_targets = dataset.default_mask_targets
    splits = dataset.distinct("tags")

    if label_field is None:
        label_field = "detections"
        print("todo: segmentations_to_detections()")
        dataset = delete_sample_field(dataset, label_field)
        dataset = segmentations_to(dataset, mask_label_field, label_field)

    for split in splits:
        view = dataset.match_tags(split)
        curr_dir = str(Path(export_dir) / split)

        view.export(
            export_dir=curr_dir,
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
        )

        if mask_label_field is not None:
            view.export(
                dataset_type=fo.types.ImageSegmentationDirectory,
                labels_path=f"{curr_dir}/labels",
                label_field=mask_label_field,
                mask_targets=mask_targets,
            )

    return dataset.count_values("tags")


def load_dataset(dataset_dir, det_labels="labels.json", seg_labels="labels/"):
    # field_name: detections, segmentations, ground_truth, predictions

    # COCODetectionDataset
    data_path = str(Path(dataset_dir) / "data/")
    labels_path = str(Path(dataset_dir) / det_labels)
    A = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        label_types=["segmentations"],
        data_path=data_path,
        labels_path=labels_path,
        label_field="detections",
    )

    # ImageSegmentationDataset
    data_path = str(Path(dataset_dir) / "data/")
    labels_path = str(Path(dataset_dir) / seg_labels)
    B = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageSegmentationDirectory,
        data_path=data_path,
        labels_path=labels_path,
        label_field="segmentations",
    )

    dataset = merge_samples(A, B)
    tag = Path(dataset_dir).name
    dataset.tag_samples(tag)
    return dataset
