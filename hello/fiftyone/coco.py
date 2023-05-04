from pathlib import Path

import fiftyone as fo
import fiftyone.core.utils as fou
import numpy as np
from skimage import measure

from hello.io.utils import save_json

from .dataset import export_dataset

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
            "num_samples": {},
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

    def export_dataset(self, export_dir, splits=None):
        return export_dataset(export_dir, self.dataset, "ground_truth", splits=splits)

    def export_json(self, labels_path):
        dataset = self.dataset

        # Populate the `metadata` field
        dataset.compute_metadata()

        cats, idx = [], 0
        for name in dataset.default_classes:
            cats.append({"id": idx, "name": name, "supercategory": "root"})
            idx += 1

        imgs, idx = [], 1
        for filepath, width, height in zip(*dataset.values(["filepath", "metadata.width", "metadata.height"])):
            imgs.append({"id": idx, "file_name": Path(filepath).name, "width": width, "height": height})
            idx += 1

        anns, idx = [], 1
        cat_info = {cat["name"]: cat for cat in cats}
        img_info = {img["file_name"]: img for img in imgs}
        for filepath, detections in zip(*dataset.values(["filepath", "ground_truth.detections"])):
            img = img_info[Path(filepath).name]
            image_id, width, height = img["id"], img["width"], img["height"]
            for detection in detections:
                category_id = cat_info[detection.label]["id"]

                x, y, w, h = detection.bounding_box
                bbox = [x * width, y * height, w * width, h * height]

                segmentation = None
                if hasattr(detection, "mask") and detection.mask is not None:
                    mask_h, mask_w = detection.mask.shape
                    y1, x1 = int(round(bbox[1])), int(round(bbox[0]))
                    y2, x2 = min(y1 + mask_h, height), min(x1 + mask_w, width)

                    bg = np.zeros((height, width), dtype="uint8")
                    bg[y1:y2, x1:x2] = detection.mask[:y2 - y1, :x2 - x1]
                    segmentation = ndarray_to_rle(bg)

                score = detection.confidence if hasattr(detection, "confidence") else 1.0

                area = bbox[2] * bbox[3]

                iscrowd = 1 if hasattr(detection, "iscrowd") and detection.iscrowd else 0

                anns.append({"id": idx, "image_id": image_id, "category_id": category_id, "bbox": bbox, "segmentation": segmentation, "score": score, "area": area, "iscrowd": iscrowd})
                idx += 1

        return save_json({"categories": cats, "images": imgs, "annotations": anns}, labels_path)


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
