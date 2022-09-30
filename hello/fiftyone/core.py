from pathlib import Path

from fiftyone import ViewField as F


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
    dataset.select(val_ids).tag_samples("val")
    dataset.select(train_ids).tag_samples("train")
    dataset.exclude(train_ids | val_ids).tag_samples("test")
    print(dataset.count_values("tags"))
    return dataset
