import json
import shutil
import time
from pathlib import Path

import cv2 as cv
from tqdm import tqdm


def extract_patch(out_dir, dataset, class_names, field_name="segmentations", prefix=None, mode=None):
    # TODO
    pass


def from_coco_instance(out_dir, dataset, class_names, field_name="segmentations", prefix=None):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=False)

    if class_names is not None:
        class_names = set(class_names)

    if prefix is None:
        prefix = time.strftime(r"%y%m%d_%H%M%S")

    _db = {}
    index = 0
    for sample in tqdm(dataset):
        img = cv.imread(sample.filepath, 1)

        img_h, img_w, _ = img.shape
        assert sample.metadata["width"] == img_w
        assert sample.metadata["height"] == img_h

        try:
            detections = sample[field_name]["detections"]
        except:
            detections = []

        for obj in detections:
            label = obj.label

            if class_names is not None and label not in class_names:
                continue

            _box = obj.bounding_box  # [x, y, w, h] / s
            x, y = round(_box[0] * img_w), round(_box[1] * img_h)

            mask_h, mask_w = obj.mask.shape
            w, h = min(mask_w, img_w - x), min(mask_h, img_h - y)

            mask = obj.mask.astype("uint8") * 255

            retval = cv.connectedComponents(mask)[0]
            if retval == 2:
                continue

            index += 1
            patch_mask = mask[:h, :w]
            patch = img[y:y + h, x:x + w]

            file_stem = f"{prefix}_{index:06d}"
            patch_file = f"data/{label}/{file_stem}.jpg"
            mask_file = f"data/{label}/{file_stem}.png"

            _db[file_stem] = {
                "label": label,
                "box": [x, y, w, h],
                "imgsz": [img_h, img_w],
                "patch": patch_file,
                "mask": mask_file,
            }

            cv.imwrite(str(out_dir / patch_file), patch)
            cv.imwrite(str(out_dir / mask_file), patch_mask)

    with open(out_dir / "db.json", "w") as f:
        json.dump(_db, f, indent=4)
    return str(out_dir)
