from itertools import groupby

import fiftyone.core.utils as fou
import numpy as np
from skimage import measure

mask_utils = fou.lazy_import("pycocotools.mask", callback=lambda: fou.ensure_import("pycocotools"))


def mask_to_coco_segmentation(mask, bbox, frame_size, mask_type="polygons", tolerance=1):
    """Returns a RLE object.

    Args:
        mask: an boolean numpy array defining the object mask
        bbox: a bounding box for the object in ``[xmin, ymin, width, height]`` format
        frame_size: the ``(width, height)`` of the image
    """
    assert mask_type in ("polygons", "rle", "rle-uncompressed", "rle-compressed")

    width, height = frame_size
    img_mask = np.zeros((height, width), dtype=bool)

    x1, y1 = int(round(bbox[0])), int(round(bbox[1]))

    mask_h, mask_w = mask.shape

    x2, y2 = min(x1 + mask_w, width), min(y1 + mask_h, height)

    img_mask[y1:y2, x1:x2] = mask[:y2 - y1, :x2 - x1]

    if mask_type == "polygons":
        segmentation = mask_to_polygons(img_mask, tolerance)
    elif mask_type == "rle" or mask_type == "rle-uncompressed":
        segmentation = mask_to_rle_uncompressed(img_mask)
    elif mask_type == "rle-compressed":
        segmentation = mask_utils.encode(np.asfortranarray(img_mask))
        if isinstance(segmentation["counts"], bytes):
            segmentation["counts"] = segmentation["counts"].decode("utf8")
    else:
        segmentation = None

    return segmentation


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
    elif isinstance(segmentation["counts"], str):
        # RLE(counts: str -> bytes) -- bytes saved as str
        segmentation["counts"] = segmentation["counts"].encode("utf8")
        rle = segmentation
    else:
        # RLE
        rle = segmentation

    mask = mask_utils.decode(rle).astype(bool)

    return mask[
        int(round(y)):int(round(y + h)),
        int(round(x)):int(round(x + w)),
    ]


def mask_to_rle_uncompressed(mask):
    counts = []
    for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)

        counts.append(len(list(elements)))

    return {"counts": counts, "size": list(mask.shape)}


def mask_to_polygons(mask, tolerance):
    if tolerance is None:
        tolerance = 2

    # Pad mask to close contours of shapes which start and end at an edge
    padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)

    contours = measure.find_contours(padded_mask, 0.5)
    contours = [c - 1 for c in contours]  # undo padding

    polygons = []
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue

        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()

        # After padding and subtracting 1 there may be -0.5 points
        segmentation = [0 if i < 0 else i for i in segmentation]

        polygons.append(segmentation)

    return polygons


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))

    return contour
