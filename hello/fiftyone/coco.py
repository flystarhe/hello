import os
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.core.utils as fou
import numpy as np
from skimage import measure

from hello.fiftyone.core import count_values, save_tags
from hello.fiftyone.dataset import (add_detection_labels, add_images_dir,
                                    tmpl_info)
from hello.io.utils import save_json

mask_utils = fou.lazy_import("pycocotools.mask", callback=lambda: fou.ensure_import("pycocotools"))


class CocoDataset(object):

    def __init__(self, name, classes=[], mask_targets={}):
        dataset = fo.Dataset(name, persistent=True, overwrite=True)

        info = {
            "dataset_name": name,
            "dataset_type": "detection",
            "version": "001",
            "classes": classes,
            "mask_targets": mask_targets,
            "num_samples": [],
            "tail": {},
        }

        dataset.info = info
        dataset.default_classes = classes
        dataset.default_mask_targets = mask_targets
        dataset.save()

        self.dataset = dataset

    def add_sample(self, filepath, annotations, frame_size):
        """Adds a sample to the dataset.

        Args:
            filepath: the path to the image on disk.
            annotations: a list of ``(label, bbox, mask, confidence, iscrowd)`` tuple
            frame_size: the ``(width, height)`` of the image.
        """
        sample = fo.Sample(filepath=filepath)

        width, height = frame_size

        detections = []
        for label, bbox, mask, confidence, iscrowd in annotations:
            x, y, w, h = bbox
            bounding_box = [x / width, y / height, w / width, h / height]
            mask = mask[int(round(y)):int(round(y + h)), int(round(x)):int(round(x + w))]
            detections.append(fo.Detection(
                label=label,
                bounding_box=bounding_box,
                mask=mask.astype(bool),
                confidence=confidence,
                iscrowd=iscrowd,
            ))

        sample["ground_truth"] = fo.Detections(detections=detections)
        self.dataset.add_sample(sample)


def coco_add_samples(dataset, dataset_dir=None, data_path=None, labels_path=None, label_field=None, splits=None, tags=None):
    if dataset_dir is None and data_path is None and labels_path is None:
        raise ValueError(
            "At least one of `dataset_dir`, `data_path`, and "
            "`labels_path` must be provided"
        )

    dataset_dirs = [dataset_dir]
    if dataset_dir is not None and splits is not None:
        dataset_dir = Path(dataset_dir)
        if splits == "auto":
            dataset_dirs = [f for f in dataset_dir.glob("*") if f.is_dir()]
        elif isinstance(splits, list):
            dataset_dirs = [dataset_dir / split for split in splits]
        else:
            raise ValueError(f"Not supported `{splits=}`")

    label_field = label_field or "ground_truth"

    for dataset_dir in dataset_dirs:
        images_dir = parse_data_path(dataset_dir, data_path, "data/")

        if images_dir is not None:
            add_images_dir(dataset, images_dir, tags, recursive=False)

        coco_json = parse_labels_path(dataset_dir, labels_path, "labels.json")

        if coco_json is not None:
            add_detection_labels(dataset, label_field, coco_json, mode="coco")


def coco_export(export_dir, dataset, label_field, splits=None):
    export_dir = Path(export_dir)
    shutil.rmtree(export_dir, ignore_errors=True)

    dataset.save()
    dataset = dataset.clone()

    info = dataset.info
    info["classes"] = dataset.default_classes
    info["mask_targets"] = dataset.default_mask_targets
    info["num_samples"] = count_values(dataset, "tags", "label")

    _tags = set(dataset.distinct("tags"))
    if splits is None:
        splits = ["train", "val", "test"]
    elif splits == "auto":
        splits = sorted(_tags)

    assert isinstance(splits, list)
    splits = [s for s in splits if s in _tags]

    if not splits:
        splits = ["train"]
        dataset.tag_samples(splits)

    for split in splits:
        print(f"\n[{split}]\n")
        view = dataset.match_tags(split)

        curr_dir = export_dir / split
        (curr_dir / "data/").mkdir(parents=True, exist_ok=True)

        info["tail"].update(count_label=count_values(view, f"{label_field}.detections.label", "label"))

        coco_export_info(info, curr_dir / "info.py")
        coco_export_images(view, curr_dir / "data/")
        coco_export_labels(view, label_field, curr_dir / "labels.json")

    save_tags(dataset, export_dir / "tags.json")

    with open(export_dir / "README.md", "w") as f:
        f.write("# README\n\n## Data Processing\n\n")

    return str(export_dir)


def coco_export_info(info, info_path):
    info_py = tmpl_info.safe_substitute(info)
    with open(info_path, "w") as f:
        f.write(info_py)


def coco_export_images(dataset_or_view, data_path):
    data_path = Path(data_path)
    for sample in dataset_or_view:
        filepath = Path(sample.filepath)
        shutil.copyfile(filepath, data_path / filepath.name)


def coco_export_labels(dataset_or_view, label_field, labels_path):
    cats, idx = [], 1
    for name in dataset_or_view.default_classes:
        cats.append({"id": idx, "name": name, "supercategory": "root"})
        idx += 1

    imgs, idx = [], 1
    for filepath, width, height in zip(*dataset_or_view.values(["filepath", "metadata.width", "metadata.height"])):
        imgs.append({"id": idx, "file_name": Path(filepath).name, "width": width, "height": height})
        idx += 1

    anns, idx = [], 1
    cat_info = {cat["name"]: cat for cat in cats}
    img_info = {img["file_name"]: img for img in imgs}
    for filepath, detections in zip(*dataset_or_view.values(["filepath", f"{label_field}.detections"])):
        img = img_info[Path(filepath).name]
        image_id, width, height = img["id"], img["width"], img["height"]
        for detection in detections:
            category_id = cat_info[detection.label]["id"]

            x, y, w, h = detection.bounding_box
            bbox = [x * width, y * height, w * width, h * height]

            segmentation = None
            if hasattr(detection, "mask") and detection.mask is not None:
                segmentation = mask_to_coco_segmentation(detection.mask, bbox, (width, height))

            score = detection.confidence if hasattr(detection, "confidence") else 1.0

            area = bbox[2] * bbox[3]

            iscrowd = 1 if hasattr(detection, "iscrowd") and detection.iscrowd else 0

            anns.append({"id": idx, "image_id": image_id, "category_id": category_id, "bbox": bbox, "segmentation": segmentation, "score": score, "area": area, "iscrowd": iscrowd})
            idx += 1

    return save_json({"categories": cats, "images": imgs, "annotations": anns}, labels_path)


def parse_data_path(dataset_dir=None, data_path=None, default=None):
    if data_path is None:
        if dataset_dir is not None:
            data_path = default

    if data_path is not None:
        data_path = os.path.expanduser(data_path)

        if not os.path.isabs(data_path) and dataset_dir is not None:
            dataset_dir = os.path.abspath(dataset_dir)
            data_path = os.path.join(dataset_dir, data_path)
        else:
            data_path = os.path.abspath(data_path)

    return data_path


def parse_labels_path(dataset_dir=None, labels_path=None, default=None):
    if labels_path is None:
        if dataset_dir is not None:
            labels_path = default

    if labels_path is not None:
        labels_path = os.path.expanduser(labels_path)

        if not os.path.isabs(labels_path) and dataset_dir is not None:
            dataset_dir = os.path.abspath(dataset_dir)
            labels_path = os.path.join(dataset_dir, labels_path)
        else:
            labels_path = os.path.abspath(labels_path)

    return labels_path


def mask_to_coco_segmentation(mask, bbox, frame_size):
    """Returns a RLE object.

    Args:
        mask: an boolean numpy array defining the object mask
        bbox: a bounding box for the object in ``[xmin, ymin, width, height]`` format
        frame_size: the ``(width, height)`` of the image
    """
    width, height = frame_size
    img_mask = np.zeros((height, width), dtype=bool)

    x1, y1 = int(round(bbox[0] * width)), int(round(bbox[1] * height))

    mask_h, mask_w = mask.shape

    x2, y2 = min(x1 + mask_w, width), min(y1 + mask_h, height)

    img_mask[y1:y2, x1:x2] = mask[:y2 - y1, :x2 - x1]
    return mask_utils.encode(np.asfortranarray(img_mask))


def coco_segmentation_to_mask(segmentation, bbox, frame_size):
    """Returns a COCO segmentation mask.

    Args:
        segmentation: segmentation mask for the object.
        bbox: a bounding box for the object in ``[xmin, ymin, width, height]`` format.
        frame_size: the ``(width, height)`` of the image.
    """
    x, y, w, h = bbox
    width, height = frame_size

    if isinstance(segmentation, list):
        # Polygon -- a single object might consist of multiple parts, so merge
        # all parts into one mask RLE code
        rle = mask_utils.merge(
            mask_utils.frPyObjects(segmentation, height, width)
        )
    elif isinstance(segmentation["counts"], list):
        # Uncompressed RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        # RLE
        rle = segmentation

    mask = mask_utils.decode(rle).astype(bool)

    return mask[
        int(round(y)):int(round(y + h)),
        int(round(x)):int(round(x + w)),
    ]


def ndarray_to_rle(mask):
    """Returns a RLE object.

    Args:
        mask (np.ndarray): a segmentation mask.
    """
    return mask_utils.encode(np.asfortranarray(mask))


def mask_to_polygons(mask, tolerance):
    if tolerance is None:
        tolerance = 2

    # Pad mask to close contours of shapes which start and end at an edge
    padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)

    contours = measure.find_contours(padded_mask, 0.5)
    contours = [c - 1 for c in contours]  # undo padding

    polygons = []
    for contour in contours:
        contour = _close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue

        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()

        # After padding and subtracting 1 there may be -0.5 points
        segmentation = [0 if i < 0 else i for i in segmentation]

        polygons.append(segmentation)

    return polygons


def _close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))

    return contour
