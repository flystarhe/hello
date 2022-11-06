import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from string import Template

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.utils.labels import (segmentations_to_detections,
                                   segmentations_to_polylines)

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


def gen_name(suffix=".png"):
    n = random.randrange(0, 10000)
    return f"{int(time.time())}{n:04d}{suffix}"


def get_classes(dataset, field_or_expr="detections.detections.label"):
    return sorted(set(dataset.distinct(field_or_expr)) - set(["other"])) + ["other"]


def clone_sample_field(dataset, field_name, new_field_name):
    # dataset: Dataset or DatasetView
    dataset.save()
    dataset = dataset.clone()

    if dataset.has_sample_field(field_name):
        dataset.clone_sample_field(field_name, new_field_name)
    else:
        print(f"not found: {field_name}")

    return dataset


def delete_sample_field(dataset, field_name, error_level=0):
    # dataset:  Dataset or DatasetView
    dataset.save()
    dataset = dataset.clone()

    if dataset.has_sample_field(field_name):
        dataset.delete_sample_field(field_name, error_level)
    else:
        print(f"not found: {field_name}")

    return dataset


def rename_sample_field(dataset, field_name, new_field_name):
    # dataset:  Dataset or DatasetView
    dataset.save()
    dataset = dataset.clone()

    if dataset.has_sample_field(field_name):
        dataset.rename_sample_field(field_name, new_field_name)
    else:
        print(f"not found: {field_name}")

    return dataset


def add_sample_field(dataset, field_name, field_type, expression):
    # field_name = "num_objects"; field_type = fo.IntField
    # expression = F("detections.detections").length()
    if dataset.has_sample_field(field_name):
        print(f"overwrite [{field_name=}], already exists")
    else:
        dataset.add_sample_field(field_name, field_type)

    view = dataset.set_field(field_name, expression)
    return view


def map_labels(dataset, mapping, field_name="ground_truth"):
    classes = dataset.distinct(f"{field_name}.detections.label")

    _map = {}
    for label in classes:
        if label in mapping:
            _map[label] = mapping[label]
        elif "*" in mapping:
            _map[label] = mapping["*"]
    print(f"[mapping]: {_map}\n")

    if dataset.has_sample_field(field_name):
        dataset = dataset.map_labels(field_name, _map)
    else:
        print(f"not found: {field_name}")

    return dataset


def filter_labels(dataset, classes, field_name="ground_truth"):
    view = dataset.filter_labels(field_name, F("label").is_in(classes))
    return view


def count_values_label(dataset, field_name="ground_truth", ordered=True):
    count_label = dataset.count_values(f"{field_name}.detections.label")
    count_label = [(k, v) for k, v in count_label.items()]

    if ordered:
        count_label = sorted(count_label, key=lambda x: x[1])

    return count_label


def filter_samples(dataset, classes, field_name="ground_truth"):
    # tagged_view = dataset.match_tags("requires_annotation")
    match = F("label").is_in(classes)
    label_field = F(f"{field_name}.detections")
    view = dataset.match(label_field.filter(match).length() > 0)
    return view


def merge_samples(datasets, **kwargs):
    A = datasets[0]

    A.save()
    A = A.clone()

    def key_fcn(sample):
        return Path(sample.filepath).name

    for B in datasets[1:]:
        B.save()
        B = B.clone()

        A.merge_samples(B, key_fcn=key_fcn, **kwargs)

    return A


def merge_datasets(info, datasets, field_name="ground_truth", tmp_dir="/tmp"):
    dataset = fo.Dataset()

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    tmp_dir = Path(tmp_dir) / info["dataset_name"]
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)

    num = 1
    for _dataset in datasets:
        _dataset.save()
        info = _dataset.info
        assert "version" in info
        assert "dataset_name" in info
        prefix = f"{info['dataset_name']}:{info['version']}"
        for sample in _dataset.select_fields(field_name).clone():
            filepath = Path(sample.filepath)
            sample["from"] = f"{prefix}/{filepath.name}"
            tempfile = tmp_dir / f"{num:012d}{filepath.suffix}"
            shutil.copyfile(filepath, tempfile)
            sample.filepath = str(tempfile)
            sample.save()
            num += 1
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


def prune_dataset(dataset, classes=None, field_name="ground_truth", per_class=1000, max_objects=5):
    expr = F(f"{field_name}.detections").length()
    dataset = add_sample_field(dataset, "num_objects", fo.IntField, expr)

    subset = dataset.match(F("num_objects") <= max_objects)

    if classes is None:
        classes = subset.distinct(f"{field_name}.detections.label")

    ids = set()
    for label in classes:
        match = (F("label") == label)
        label_field = F(f"{field_name}.detections")
        view = subset.match(label_field.filter(match).length() > 0)
        ids.update(view.sort_by("num_objects")[:per_class].values("id"))
    results = subset.select(ids)
    return results


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
    print(dataset.count_values("tags"))
    return dataset


def export_dataset(export_dir, dataset, label_field=None, mask_label_field=None, mask_types="stuff"):
    assert label_field is not None or mask_label_field is not None

    dataset.save()
    info = dataset.info
    classes = dataset.default_classes
    mask_targets = dataset.default_mask_targets
    info["num_samples"] = dataset.count_values("tags")

    if label_field is None:
        label_field = "detections"
        print("todo: segmentations_to_detections()")
        dataset = dataset.select_fields(mask_label_field).clone()
        dataset = segmentations_to(dataset, mask_label_field, label_field,
                                   mask_types=mask_types)

    splits = dataset.distinct("tags")

    if not splits:
        splits = ["train"]
        dataset.tag_samples("train")

    for split in splits:
        print(f"\n[{split}]\n")
        view = dataset.match_tags(split)
        curr_dir = Path(export_dir) / split
        counts = count_values_label(view, label_field)

        view.export(
            export_dir=str(curr_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
        )

        if mask_label_field is not None:
            view.export(
                dataset_type=fo.types.ImageSegmentationDirectory,
                labels_path=str(curr_dir / "labels"),
                label_field=mask_label_field,
                mask_targets=mask_targets,
            )

        tail = info["tail"]
        tail["count_label"] = counts
        info["tail"] = tail

        info_py = tmpl_info.safe_substitute(info,
                                            classes=classes,
                                            mask_targets=mask_targets)

        with open(curr_dir / "info.py", "w") as f:
            f.write(info_py)

    return export_dir


def load_dataset(dataset_dir, det_labels="labels.json", seg_labels="labels/"):
    # field_name: detections, segmentations, ground_truth, predictions
    dataset_dir = Path(dataset_dir)

    # COCODetectionDataset
    data_path = str(dataset_dir / "data/")
    labels_path = str(dataset_dir / det_labels)
    A = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        label_types=["segmentations"],
        data_path=data_path,
        labels_path=labels_path,
        label_field="detections",
    )

    # ImageSegmentationDataset
    data_path = str(dataset_dir / "data/")
    labels_path = str(dataset_dir / seg_labels)
    B = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageSegmentationDirectory,
        data_path=data_path,
        labels_path=labels_path,
        label_field="segmentations",
    )

    dataset = merge_samples([A, B])

    filepath = (dataset_dir / "info.py")
    if not filepath.is_file():
        return dataset

    with open(filepath, "r") as f:
        info = eval(f.read())

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    return dataset


def create_dataset(dataset_dir, info, predictions=None, relative=False):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.ImageDirectory,
        compute_metadata=True,
    )

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    if predictions is None:
        predictions = defaultdict(list)

    for sample in dataset:
        metadata = sample.metadata
        filename = Path(sample.filepath).name
        im_w, im_h = metadata.width, metadata.height

        detections = []
        # bounding_box: [<top-left-x>, <top-left-y>, <width>, <height>] \in [0, 1]
        for x1, y1, x2, y2, confidence, label in predictions[filename]:
            if relative:
                rel_box = [x1, y1, (x2 - x1), (y2 - y1)]
            else:
                rel_box = [x1 / im_w, y1 / im_h,
                           (x2 - x1) / im_w, (y2 - y1) / im_h]
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=rel_box,
                    confidence=confidence,
                )
            )
        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

    return dataset
