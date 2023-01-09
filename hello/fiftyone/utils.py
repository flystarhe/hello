import json
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import fiftyone.core.utils as fou
import fiftyone.utils.iou as foui


def equal_list(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False

    for _ai, _bi in zip(a, b):
        if _ai != _bi:
            return False

    return True


def equal_dict(a: dict, b: dict) -> bool:
    if len(a) != len(b):
        return False

    for k, v in a.items():
        if v != b.get(k):
            return False

    return True


def _parse_text_slice(vals):
    assert len(vals) == 6
    x1, y1, x2, y2 = vals[:4]

    bounding_box = [
        float(x1),
        float(y1),
        float(x2) - float(x1),
        float(y2) - float(y1),
    ]

    confidence = float(vals[4])

    label = vals[5]

    return bounding_box, confidence, label


def _parse_text_row(row):
    """\
    row format:
        ``filepath,height,width,x1,y1,x2,y2,s,l,x1,y1,x2,y2,s,l``
    """
    row_vals = row.split(",")

    assert len(row_vals) >= 3, "filepath,height,width,..."

    filepath = Path(row_vals[0])
    height = int(row_vals[1])
    width = int(row_vals[2])
    data = row_vals[3:]

    group_size = 6
    total_size = len(data)
    assert total_size % group_size == 0

    detections = []
    _scale = [width, height, width, height]
    for i in range(0, total_size, group_size):
        bounding_box, confidence, label = _parse_text_slice(data[i:(i + group_size)])

        bounding_box = [a / b for a, b in zip(bounding_box, _scale)]

        detection = dict(
            bounding_box=bounding_box,
            confidence=confidence,
            label=label,
        )

        detections.append(detection)

    return filepath, detections


def _parse_yolo_row(row, classes):
    row_vals = row.split()
    target, xc, yc, w, h = row_vals[:5]

    try:
        label = classes[int(target)]
    except:
        label = str(target)

    bounding_box = [
        (float(xc) - 0.5 * float(w)),
        (float(yc) - 0.5 * float(h)),
        float(w),
        float(h),
    ]

    if len(row_vals) > 5:
        confidence = float(row_vals[5])
    else:
        confidence = 1.0

    detection = dict(
        bounding_box=bounding_box,
        confidence=confidence,
        label=label,
    )
    return detection


def _parse_yolo_annotations(filepath, classes):
    """\
    row format:
        ``target,xc,yc,w,h,s``
    """
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.read().splitlines()]

    lines = [l for l in lines if l and not l.startswith("#")]

    detections = []
    for row in lines:
        detections.append(_parse_yolo_row(row, classes))
    return detections


def load_text_predictions(labels_path):
    with open(labels_path, "r") as f:
        lines = [l.strip() for l in f.read().splitlines()]

    lines = [l for l in lines if l and not l.startswith("#")]

    db = {}
    for row in lines:
        filepath, detections = _parse_text_row(row)
        db[filepath.stem] = detections
    return db


def load_yolo_predictions(labels_path, classes):
    db = {}
    for filepath in Path(labels_path).glob("*.txt"):
        detections = _parse_yolo_annotations(filepath, classes)
        db[filepath.stem] = detections
    return db


def load_coco_predictions(labels_path):
    with open(labels_path, "r") as f:
        coco = json.load(f)

    assert "categories" in coco and "images" in coco and "annotations" in coco

    imgs = {img["id"]: img for img in coco["images"]}
    cats = {cat["id"]: cat for cat in coco["categories"]}

    db = defaultdict(list)
    for ann in coco["annotations"]:
        _img = imgs[ann["image_id"]]

        filepath = _img["file_name"]
        height = _img["height"]
        width = _img["width"]

        _cat = cats[ann["category_id"]]

        label = _cat["name"]

        x, y, w, h = ann["bbox"]
        bounding_box = [x / width, y / height, w / width, h / height]

        confidence = ann.get("score", 1.0)

        detection = dict(
            bounding_box=bounding_box,
            confidence=confidence,
            label=label,
        )
        db[Path(filepath).stem].append(detection)

    stems = [Path(_img["file_name"]).stem for _img in imgs.values()]
    return {stem: db[stem] for stem in stems}


def load_predictions(labels_path, classes=None, mode="text"):
    if mode == "text":
        return load_text_predictions(labels_path)
    elif mode == "yolo":
        assert isinstance(classes, list)
        return load_yolo_predictions(labels_path, classes)
    elif mode == "coco":
        return load_coco_predictions(labels_path)
    else:
        raise NotImplementedError


def find_duplicate_labels(dataset, label_field, iou_thresh=0.999, method="simple", iscrowd=None, classwise=False):
    """Returns IDs of duplicate labels in the given field of the dataset, as defined as labels with an IoU greater than a chosen threshold with another label in the field.

    >>> dup_ids = find_duplicates()
    >>> dataset.untag_labels("duplicate")
    >>> dataset.select_labels(ids=dup_ids).tag_labels("duplicate")
    >>> print(dataset.count_label_tags())
    >>> dataset.delete_labels(tags="duplicate")
    >>> # dataset.delete_labels(ids=dup_ids)  <- best

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        label_field: a label field of type :class:`fiftyone.core.labels.Detections` or :class:`fiftyone.core.labels.Polylines`
        iou_thresh (0.999): the IoU threshold to use to determine whether labels are duplicates
        method ("simple"): supported values are ``("simple", "greedy")``
        iscrowd (None): an optional name of a boolean attribute
        classwise (False): different label values as always non-overlapping
    """
    dup_ids = foui.find_duplicates(dataset, label_field, iou_thresh=iou_thresh, method=method, iscrowd=iscrowd, classwise=classwise)
    return dup_ids


def find_duplicate_images(filepaths, leave_one_out=False):
    db = []
    for filepath in filepaths:
        filehash = fou.compute_filehash(filepath)
        db.append([filepath, filehash])

    groups = defaultdict(list)
    for filepath, filehash in db:
        groups[filehash].append([filepath, filehash])

    results = []
    for vals in groups.values():
        if len(vals) > 1:
            if leave_one_out:
                results.extend(vals[1:])
            else:
                results.extend(vals)

    return results


def read_mask(mask_path, remap=None):
    mask = cv.imread(mask_path, 0)

    if remap is not None:
        new_mask = mask.copy()
        for _old, _new in remap.items():
            if _old != _new:
                new_mask[mask == _old] = _new
        mask = new_mask

    return mask


def gen_mask_remap(dataset_mask_targets, label_mask_targets, ignore_index=255):
    if equal_dict(dataset_mask_targets, label_mask_targets):
        return None

    label2index = {label: index for index, label in dataset_mask_targets.items()}

    remap = {}
    for index, label in label_mask_targets.items():
        new_index = label2index.get(label, ignore_index)
        if index != new_index:
            remap[index] = new_index

    remap = remap if remap else None
    return remap


def load_segmentation_masks(labels_path, remap=None, mode="png"):
    labels_path = Path(labels_path)

    if mode == "png":
        assert labels_path.is_dir()
    else:
        raise NotImplementedError

    db = {}
    for mask_path in sorted(labels_path.glob("*.png")):
        mask = read_mask(str(mask_path), remap)
        db[mask_path.stem] = mask

    return db
