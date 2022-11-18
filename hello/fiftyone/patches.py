import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2 as cv
from fiftyone import ViewField as F
from tqdm import tqdm


def extract_patch(out_dir, dataset, class_names, field_name="segmentations", min_bbox_size=32, prefix=None, mode=None):
    # TODO
    pass


def from_coco_instance(out_dir, dataset, class_names, field_name="segmentations", min_bbox_size=32, prefix=None):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=False)

    if class_names is not None:
        dataset = dataset.filter_labels(field_name, F("label").is_in(class_names))
    else:
        dataset = dataset.match(F(f"{field_name}.detections").length() > 0)

    if prefix is None:
        prefix = time.strftime(r"%y%m%d_%H%M%S")

    _db = {}
    counts = defaultdict(int)
    for sample in tqdm(dataset):
        img = cv.imread(sample.filepath, 1)

        img_h, img_w, _ = img.shape
        assert sample.metadata["width"] == img_w
        assert sample.metadata["height"] == img_h

        for obj in sample[field_name]["detections"]:
            label = obj.label

            _box = obj.bounding_box  # [x, y, w, h] / s
            x, y = round(_box[0] * img_w), round(_box[1] * img_h)

            mask_h, mask_w = obj.mask.shape
            w, h = min(mask_w, img_w - x), min(mask_h, img_h - y)

            if w < min_bbox_size or h < min_bbox_size:
                continue

            mask = obj.mask.astype("uint8") * 255

            retval = cv.connectedComponents(mask)[0]
            if retval != 2:
                continue

            patch_mask = mask[:h, :w]
            patch = img[y:y + h, x:x + w]

            counts[label] += 1

            sub_dir = "_".join(label.split())

            _curr_dir = out_dir / f"data/{sub_dir}"
            if not _curr_dir.is_dir():
                _curr_dir.mkdir(parents=True, exist_ok=False)

            file_stem = f"{prefix}_{sub_dir}_{counts[label]:06d}"
            patch_file = f"data/{sub_dir}/{file_stem}.jpg"
            mask_file = f"data/{sub_dir}/{file_stem}.png"

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
    print(f"[INFO] {counts=}")
    return str(out_dir)
