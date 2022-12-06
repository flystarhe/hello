import shutil
from pathlib import Path

import cv2 as cv
import fiftyone as fo
from fiftyone import ViewField as F
from fiftyone.utils.labels import segmentations_to_detections
from prettytable import PrettyTable
from tqdm import tqdm


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
        if _old != _new:
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


def map_default_classes(dataset, classes, background="background"):
    new_classes = [c[0] if isinstance(c, list) else c for c in classes]
    new_classes = new_classes[:-1] + [background]

    dataset.default_classes = new_classes
    return dataset


def map_default_mask_targets(dataset, classes, ignore_index=255):
    new_classes = [c[0] if isinstance(c, list) else c for c in classes]
    new_mask_targets = {i: c for i, c in enumerate(new_classes[:-1])}
    new_mask_targets[ignore_index] = new_classes[-1]

    dataset.default_mask_targets = new_mask_targets
    dataset.default_classes = new_classes
    return dataset


def gen_label_mapping(old_classes, new_classes):
    """Generate detections mapping.

    Args:
        old_classes (list): ``['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'background']``
        new_classes (list): ``['c0', 'c1', 'c2', ['c3', 'c4', 'c5'], 'background']``

    Returns:
        a :class:`dict`
    """
    old_classes, new_classes = old_classes[:-1], new_classes[:-1]

    mapping = {}
    for names in new_classes:
        names = [names] if isinstance(names, str) else names
        assert isinstance(names, list)
        label = names[0]
        for name in names:
            assert name not in mapping
            mapping[name] = label

    for name in old_classes:
        assert name in mapping

    return mapping


def gen_mask_mapping(old_classes, new_classes):
    """Generate segmentation mapping.

    Args:
        old_classes (list): ``['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'be ignored']``
        new_classes (list): ``['c0', 'c1', 'c2', ['c3', 'c4', 'c5'], 'be ignored']``

    Returns:
        a :class:`dict`
    """
    old_classes, new_classes = old_classes[:-1], new_classes[:-1]

    remap = {}
    for i, names in enumerate(new_classes):
        names = [names] if isinstance(names, str) else names
        assert isinstance(names, list)
        for name in names:
            assert name not in remap
            remap[name] = i

    mapping = {}
    for i, name in enumerate(old_classes):
        assert isinstance(name, str)
        mapping[i] = remap[name]

    return mapping


def filter_detections_dataset(dataset, new_classes=None, field_name="ground_truth", background="background"):
    """Steps: map labels -> check dataset.classes -> filter valid samples

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        new_classes (None): refer to :func:`gen_label_mapping`
        field_name (str, optional): defaults to "ground_truth"
        background (str, optional): defaults to "background"

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    dataset.save()
    dataset = dataset.clone()

    if new_classes is not None:
        old_classes = dataset.default_classes
        mapping = gen_label_mapping(old_classes, new_classes)
        dataset = map_labels(dataset, mapping, field_name=field_name)
        dataset = map_default_classes(dataset, new_classes, background=background)

    dataset = dataset.filter_labels(field_name, F("label") != background).clone()

    return dataset


def filter_segmentation_dataset(dataset, new_classes=None, field_name="ground_truth", ignore_index=255):
    """Steps: map labels -> check dataset.mask_targets -> filter valid samples

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        new_classes (None): refer to :func:`gen_mask_mapping`
        field_name (str, optional): defaults to "ground_truth"
        ignore_index (int, optional): defaults to 255

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    dataset.save()
    dataset = dataset.clone()

    def _check_sample(field_data):
        if field_data:
            mask = field_data.mask
            return mask[mask < ignore_index].sum() > 0
        return False

    if new_classes is not None:
        old_classes = dataset.default_classes
        mapping = gen_mask_mapping(old_classes, new_classes)
        dataset = map_labels(dataset, mapping, field_name=field_name)
        dataset = map_default_mask_targets(dataset, new_classes, ignore_index)

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

    table_data = PrettyTable()
    table_data.field_names = ["label", "count"]
    table_data.add_rows(count_label)
    print(table_data)

    return count_label


def split_dataset(dataset, splits=None, limit=3000, seed=51, field_name="ground_truth", from_field=None):
    """Adds the split tags to all samples in this dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        splits (dict, optional): defaults to None
        limit (int, optional): defaults to 3000
        seed (int, optional): defaults to 51
        field_name (str, optional): defaults to "ground_truth"
        from_field (str, optional): defaults to None

    Returns:
        a :class:`DatasetView`
    """
    view = dataset.shuffle(seed=seed)
    view.untag_samples(["train", "val", "test"])

    if from_field is not None:
        print("todo: segmentations_to_detections()")
        segmentations_to_detections(view, from_field, field_name, mask_targets=view.default_mask_targets, mask_types="stuff")

    if splits is None:
        splits = {"val": 0.1, "train": 0.9}

    val_ids, train_ids = [], []
    for label, _ in count_values(view, f"{field_name}.detections.label", ordered=True):
        _detections = F(f"{field_name}.detections").filter(F("label") == label)
        subset = view.exclude(val_ids + train_ids).match(_detections.length() > 0)

        ids = subset.take(limit).values("id")

        pos_val = splits.get("val", 0.1)
        pos_train = splits.get("train", 0.9)
        if isinstance(pos_val, float):
            num_samples = len(ids)
            pos_val = int(1 + pos_val * num_samples)
            pos_train = int(1 + pos_train * num_samples)
        ids = ids[:(pos_val+pos_train)]

        val_ids.extend(ids[:pos_val])
        train_ids.extend(ids[pos_val:])

    view.select(val_ids).tag_samples("val")
    view.select(train_ids).tag_samples("train")
    view.exclude(val_ids + train_ids).tag_samples("test")
    print(count_values(view, "tags", ordered=True))
    return view


def filter_segmentation_samples(out_dir, data_root, classes, mask_targets, threshold=0.05, splits=["train", "val"],
                                img_dir="data", ann_dir="labels", img_suffix=".jpg", seg_map_suffix=".png"):
    """Filter samples, based on area of interest ratio.

    >>> <data_root>/
    >>> ├── objectInfo150.txt
    >>> ├── sceneCategories.txt
    >>> ├── train
    >>> │   ├── data
    >>> │   └── labels
    >>> └── val
    >>>     ├── data
    >>>     └── labels

    Args:
        out_dir (str): _description_
        data_root (str): _description_
        classes (list[str]): _description_
        mask_targets (dict[int, str]): _description_
        threshold (float, optional): _description_. Defaults to 0.05
        splits (list, optional): _description_. Defaults to ["train", "val"]
        img_dir (str, optional): _description_. Defaults to "data"
        ann_dir (str, optional): _description_. Defaults to "labels"
        img_suffix (str, optional): _description_. Defaults to ".jpg"
        seg_map_suffix (str, optional): _description_. Defaults to ".png"

    Returns:
        a :class:`str`
    """
    _mapping = {name: index for index, name in mask_targets.items()}
    _labels = [_mapping[name] for name in classes]

    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    for split in splits:
        (out_dir / f"{split}/data").mkdir(parents=True, exist_ok=False)
        (out_dir / f"{split}/labels").mkdir(parents=True, exist_ok=False)

    data_root = Path(data_root)
    img_files, seg_map_files = [], []
    for split in splits:
        info_py = data_root / f"{split}/info.py"
        if info_py.is_file():
            shutil.copyfile(info_py, out_dir / f"{split}/info.py")
        img_files.extend(data_root.glob(f"{split}/{img_dir}/*{img_suffix}"))
        seg_map_files.extend(data_root.glob(f"{split}/{ann_dir}/*{seg_map_suffix}"))
        print(f"[INFO] add [{split}]: img={len(img_files)}, ann={len(seg_map_files)}")

    img_files = {f.stem: f for f in sorted(img_files)}
    seg_map_files = {f.stem: f for f in sorted(seg_map_files)}
    print(f"[INFO] unique: img={len(img_files)}, ann={len(seg_map_files)}")

    data = []
    for stem, f in seg_map_files.items():
        if stem in img_files:
            mask = cv.imread(str(f), flags=0)
            p = sum([(mask == l).mean() for l in _labels])
            if p >= threshold:
                data.extend((img_files[stem], f))
    print(f"[INFO] samples: {len(data)//2}")

    for f in tqdm(data):
        tempfile = f.relative_to(data_root)
        shutil.copyfile(f, out_dir / tempfile)
    return out_dir
