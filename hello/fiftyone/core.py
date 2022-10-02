from copy import deepcopy
from pathlib import Path

import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.utils.labels import segmentations_to_detections


def _map_detections(field_data, mapping):
    detections = field_data.detections

    new_detections = []
    for _detection in detections:
        label = _detection.label
        if label in mapping:
            _detection.label = mapping[label]
        elif "*" in mapping:
            _detection.label = mapping["*"]
        new_detections.append(_detection)

    return fo.Detections(detections=new_detections)


def _map_segmentation(field_data, mapping):
    mask = field_data.mask

    new_mask = mask.copy()
    for _old, _new in mapping.items():
        new_mask[mask == _old] = _new

    return fo.Segmentation(mask=new_mask)


def map_labels(dataset, mapping, field_name="ground_truth"):
    for sample in dataset:
        field_data = sample[field_name]
        if field_data:
            if isinstance(field_data, fo.Detections):
                field_data = _map_detections(field_data, mapping)
            elif isinstance(field_data, fo.Segmentation):
                field_data = _map_segmentation(field_data, mapping)
            else:
                raise NotImplementedError
            sample[field_name] = field_data
            sample.save()
    return dataset


def map_default_classes(dataset, mapping, background="background"):
    classes = dataset.default_classes

    new_classes = []
    for label in classes:
        if label in mapping:
            label = mapping[label]
        elif "*" in mapping:
            label = mapping["*"]
        new_classes.append(label)

    sorted_key = classes + list(mapping.values())
    distinct_labels = set(new_classes) - set([background])
    new_classes = sorted(distinct_labels, key=lambda x: sorted_key.index(x))

    new_classes.append(background)
    dataset.default_classes = new_classes
    return dataset


def map_default_mask_targets(dataset, mapping, background=255):
    mask_targets = dataset.default_mask_targets

    new_mask_targets = deepcopy(mask_targets)
    for _old, _new in mapping.items():
        new_mask_targets[_new] = mask_targets[_old]

    for key in new_mask_targets.keys():
        if key not in mapping and key in mask_targets:
            new_mask_targets[key] = mask_targets[key]

    for key in (set(mapping.keys()) - set(mapping.values())):
        del new_mask_targets[key]

    new_mask_targets[background] = "background"
    dataset.default_mask_targets = new_mask_targets
    return dataset


def filter_detections_dataset(dataset, mapping=None, field_name="ground_truth", background="background"):
    dataset.save()
    dataset = dataset.clone()

    if mapping is not None:
        dataset = map_labels(dataset, mapping, field_name=field_name)
        dataset = map_default_classes(dataset, mapping, background=background)

    dataset = dataset.filter_labels(field_name, F("label") != background).clone()

    return dataset


def filter_segmentation_dataset(dataset, mapping=None, field_name="ground_truth", background=255):
    dataset.save()
    dataset = dataset.clone()

    def _check_sample(field_data):
        if field_data:
            return (field_data.mask != background).sum() > 0
        return False

    if mapping is not None:
        dataset = map_labels(dataset, mapping, field_name=field_name)
        dataset = map_default_mask_targets(dataset, mapping, background=background)

    dataset = dataset.select([s.id for s in dataset if _check_sample(s[field_name])]).clone()

    return dataset


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


def count_values(dataset, field_or_expr, ordered=True):
    # field_or_expr: "tags" or "ground_truth.detections.label"
    count_label = dataset.count_values(field_or_expr)
    count_label = [(k, v) for k, v in count_label.items()]

    if ordered:
        count_label = sorted(count_label, key=lambda x: x[1])

    return count_label


def split_dataset(dataset, splits=None, limit=3000, field_name="ground_truth", from_field=None):
    dataset.untag_samples(dataset.distinct("tags"))
    dataset = dataset.shuffle()

    if from_field is not None:
        print("todo: segmentations_to_detections()")
        segmentations_to_detections(dataset, from_field, field_name, mask_targets=dataset.default_mask_targets, mask_types="stuff")

    if splits is None:
        splits = {"val": 0.1, "train": 0.9}

    val_ids, train_ids = [], []
    for label, _ in count_values(dataset, f"{field_name}.detections.label", ordered=True):
        _detections = F(f"{field_name}.detections").filter(F("label") == label)
        view = dataset.exclude(val_ids + train_ids).match(_detections.length() > 0)

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

    dataset.select(val_ids).tag_samples("val")
    dataset.select(train_ids).tag_samples("train")
    dataset.exclude(val_ids + train_ids).tag_samples("test")
    print(count_values(dataset, "tags", ordered=True))
    return dataset
