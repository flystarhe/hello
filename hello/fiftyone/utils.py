import random
import shutil
import time
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F


def gen_name(suffix=".png"):
    a = int(time.time())
    b = random.randrange(10000)
    return f"{a}{b:04d}{suffix}"


def get_classes(dataset, field_or_expr="detections.detections.label"):
    # sorted(dataset.distinct("predictions.detections.label"))
    labels = dataset.values(field_or_expr, missing_value="__NONE", unwind=True)
    return sorted(set(labels) - set(["__NONE"]))


def map_labels(dataset, mapping):
    _map = {}
    for label in dataset.default_classes:
        if label in mapping:
            _map[label] = mapping[label]
        elif "*" in mapping:
            _map[label] = mapping["*"]

    print(f"\nmap_labels:\n{_map}\n")
    sample_fields = dataset.get_field_schema()
    if "detections" in sample_fields:
        dataset = dataset.map_labels("detections", _map)
    if "segmentations" in sample_fields:
        dataset = dataset.map_labels("segmentations", _map)
    if "ground_truth" in sample_fields:
        dataset = dataset.map_labels("ground_truth", _map)
    if "predictions" in sample_fields:
        dataset = dataset.map_labels("predictions", _map)
    return dataset


def merge_datasets(name, classes, info, datasets, tmp_dir="/tmp"):
    dataset = fo.Dataset(name=name, overwrite=True)
    dataset.default_classes = classes
    dataset.info = info

    tmp_dir = Path(tmp_dir) / str(name)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    num = 0
    tmp_dir.mkdir(parents=True)
    for _dataset in datasets:
        for sample in _dataset.clone():
            num += 1
            filepath = Path(sample.filepath)
            tempfile = tmp_dir / f"{num:012d}{filepath.suffix}"
            shutil.copyfile(filepath, tempfile)
            sample.filepath = str(tempfile)
            dataset.add_sample(sample)
    return dataset


def split_dataset(dataset, splits, limit=500, field="ground_truth"):
    if splits is None:
        splits = {"train": 0.8, "val": 0.1}

    train_ids, val_ids = [], []
    label_field = f"{field}.detections"
    for label in dataset.default_classes:
        match = (F("label") == label)
        view = dataset.match(
            F(label_field).filter(match).length() > 0
        ).take(limit)

        ids = view.values("id")
        pos_val = splits.get("val", 0.1)
        pos_train = splits.get("train", 0.8)
        if isinstance(pos_train, float):
            num_samples = len(ids)
            pos_val = int(pos_val * num_samples)
            pos_train = int(pos_train * num_samples)
        ids = ids[:pos_val + pos_train]

        train_ids.extend(ids[:pos_train])
        val_ids.extend(ids[pos_train:])

    train_ids = set(train_ids)
    val_ids = set(val_ids) - train_ids

    dataset_train = dataset.select(train_ids)
    dataset_val = dataset.select(val_ids)

    train_val_ids = train_ids | val_ids
    dataset_test = dataset.exclude(train_val_ids)

    return dataset_train, dataset_val, dataset_test


def export_dataset(export_dir, dataset, splits, label_field="ground_truth"):
    for split in splits:
        split_view = dataset.match_tags(split)
        split_view.export(
            export_dir=f"{export_dir}/{split}",
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field)
    return export_dir


def get_started():
    # wget -q https://github.com/flystarhe/containers/releases/download/v0.2.0/coco2017_cat_dog.tar
    # tar -xf coco2017_cat_dog.tar

    dataset_type = fo.types.COCODetectionDataset
    dataset_dir = "coco2017_cat_dog"

    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=dataset_type,
    )
