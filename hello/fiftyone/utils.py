import random
import shutil
import time
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F


def gen_name(suffix=".png"):
    n = random.randrange(0, 10000)
    return f"{int(time.time())}{n:04d}{suffix}"


def get_classes(dataset, field_or_expr="detections.detections.label"):
    return sorted(dataset.distinct(field_or_expr))


def map_labels(dataset, mapping, field_name="detections"):
    # field_name: detections, segmentations, ground_truth, predictions
    _map = {}
    for label in dataset.default_classes:
        if label in mapping:
            _map[label] = mapping[label]
        elif "*" in mapping:
            _map[label] = mapping["*"]

    print(f"\nmap_labels:\n{_map}\n")
    sample_fields = dataset.get_field_schema()
    if field_name in sample_fields:
        dataset = dataset.map_labels(field_name, _map)
    return dataset


def merge_datasets(name, classes, info, datasets, tmp_dir="/tmp"):
    dataset = fo.Dataset(name=name, overwrite=True)
    dataset.default_classes = classes
    dataset.info = info

    tmp_dir = Path(tmp_dir) / str(name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    num = 0
    for _dataset in datasets:
        for sample in _dataset:
            num += 1
            filepath = Path(sample.filepath)
            tempfile = tmp_dir / f"{num:012d}{filepath.suffix}"
            shutil.copyfile(filepath, tempfile)
            sample.filepath = str(tempfile)
            dataset.add_sample(sample)
    return dataset


def split_dataset(dataset, splits, limit=500, field_name="ground_truth"):
    dataset.untag_samples(dataset.distinct("tags"))
    dataset.tag_samples("test")

    if splits is None:
        splits = {"train": 0.8, "val": 0.1}

    train_ids, val_ids = [], []
    label_field = f"{field_name}.detections"
    for label in dataset.default_classes:
        match = (F("label") == label)
        view = dataset.match(
            F(label_field).filter(match).length() > 0
        ).take(limit)

        ids = view.values("id")
        pos_val = splits.get("val", 0.1)
        pos_train = splits.get("train", 0.8)
        if isinstance(pos_val, float):
            num_samples = len(ids)
            pos_val = int(pos_val * num_samples)
            pos_train = int(pos_train * num_samples)

        ids = ids[:(pos_val + pos_train)]
        train_ids.extend(ids[pos_val:])
        val_ids.extend(ids[:pos_val])

    train_ids = set(train_ids)
    val_ids = set(val_ids) - train_ids
    dataset.select(train_ids).tag_samples("train")
    dataset.select(val_ids).tag_samples("validation")
    dataset.select(train_ids | val_ids).untag_samples("test")

    return dataset


def export_dataset(export_dir, dataset, label_field="ground_truth"):
    for split in dataset.distinct("tags"):
        split_view = dataset.match_tags(split)
        split_view.export(export_dir=f"{export_dir}/{split}",
                          dataset_type=fo.types.COCODetectionDataset,
                          label_field=label_field)
    return export_dir
