import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm


def extract_patch(out_dir, dataset, class_names, field_name="segmentations", prefix=None, mode=None):
    # TODO
    pass


def from_coco_instance(out_dir, dataset, class_names, field_name="segmentations", prefix=None):
    _db = defaultdict(list)

    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=False)

    if prefix is None:
        prefix = time.strftime(r"%y%m%d_%H%M%S")

    class_names = set(class_names)
    for sample in tqdm(dataset):
        img = cv.imread(sample.filepath, 1)

        img_h, img_w, _ = img.shape
        assert sample.metadata["width"] == img_w
        assert sample.metadata["height"] == img_h

        for obj in sample[field_name]["detections"]:
            label = obj.label

            if label not in class_names:
                continue

            _box = obj.bounding_box  # [x, y, w, h] / s
            x, y = round(_box[0] * img_w), round(_box[1] * img_h)

            mask_h, mask_w = obj.mask.shape
            w, h = min(mask_w, img_w - x), min(mask_h, img_h - y)

            mask = obj.mask.astype("uint8") * 255
            mask = np.stack((mask, mask, mask), axis=-1)

            patch_mask = mask[:h, :w]
            patch = img[y:y + h, x:x + w]

            _db[label].append({
                "image": img_file,
                "mask": mask_file,
                "box": [x, y, w, h],
                "whole": (img_h, img_w),
            })
            img_file = f"data/{label}/{prefix}_{len(_db):06d}.jpg"
            mask_file = f"data/{label}/{prefix}_{len(_db):06d}.png"

            cv.imwrite(str(out_dir / img_file), patch)
            cv.imwrite(str(out_dir / mask_file), patch_mask)

    with open(out_dir / "db.json", "w") as f:
        json.dump(_db, f)
    return str(out_dir)
