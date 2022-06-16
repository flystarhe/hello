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


def clone_sample_field(dataset, field_name="detections", new_field_name="ground_truth"):
    # dataset: Dataset or DatasetView
    dataset = dataset.clone()
    dataset.clone_sample_field(field_name, new_field_name)
    return dataset


def delete_sample_field(dataset, field_name, error_level=0):
    # dataset:  Dataset or DatasetView
    dataset = dataset.clone()
    dataset.delete_sample_field(field_name, error_level)
    return dataset


def rename_sample_field(dataset, field_name, new_field_name):
    # dataset:  Dataset or DatasetView
    dataset = dataset.clone()
    dataset.rename_sample_field(field_name, new_field_name)
    return dataset


def map_labels(dataset, mapping, field_name="ground_truth"):
    # field_name: detections, segmentations, ground_truth, predictions
    _map = {}
    for label in dataset.default_classes:
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


def filter_samples(dataset, classes, field_name="ground_truth"):
    # tagged_view = dataset.match_tags("requires_annotation")
    match = F("label").is_in(classes)
    label_field = F(f"{field_name}.detections")
    view = dataset.match(label_field.filter(match).length() > 0)
    return view


def filter_labels(dataset, classes, field_name="ground_truth"):
    view = dataset.filter_labels(field_name, F("label").is_in(classes))
    return view


def merge_datasets(name, classes, info, datasets, tmp_dir="/tmp"):
    dataset = fo.Dataset(name=name, overwrite=True)
    dataset.default_classes = classes
    dataset.info = info

    tmp_dir = Path(tmp_dir) / str(name)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    num = 0
    for _dataset in datasets:
        info = _dataset.info
        assert "version" in info
        assert "dataset_name" in info
        prefix = f"{info['dataset_name']}:{info['version']}"
        for sample in _dataset:
            num += 1
            filepath = Path(sample.filepath)
            sample["from"] = f"{prefix}/{filepath.name}"
            tempfile = tmp_dir / f"{num:012d}{filepath.suffix}"
            shutil.copyfile(filepath, tempfile)
            sample.filepath = str(tempfile)
            dataset.add_sample(sample)
    return dataset


def split_dataset(dataset, splits, limit=500, field_name="ground_truth"):
    dataset.untag_samples(dataset.distinct("tags"))
    dataset = dataset.shuffle()

    if splits is None:
        splits = {"train": 0.8, "val": 0.1}

    train_ids, val_ids = [], []
    label_field = F(f"{field_name}.detections")
    for label in dataset.default_classes:
        match = (F("label") == label)
        view = dataset.match(label_field.filter(match).length() > 0)
        ids = view.take(limit).values("id")

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
    dataset.exclude(train_ids | val_ids).tag_samples("test")

    return dataset


def export_dataset(export_dir, dataset, label_field="ground_truth"):
    splits = dataset.distinct("tags")
    for split in splits:
        view = dataset.match_tags(split)
        view.export(export_dir=f"{export_dir}/{split}",
                    dataset_type=fo.types.COCODetectionDataset,
                    label_field=label_field)
    return dataset.count_values("tags")


def load_dataset(dataset_dir, label_field="ground_truth"):
    dataset = fo.Dataset.from_dir(dataset_dir=dataset_dir,
                                  dataset_type=fo.types.COCODetectionDataset,
                                  label_field=label_field)
    dataset.tag_samples(Path(dataset_dir).name)
    return dataset
