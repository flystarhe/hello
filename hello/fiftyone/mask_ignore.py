import re
import shutil
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from hello.fiftyone.coco import coco_add_samples, coco_export
from hello.fiftyone.dataset import create_dataset


def mask_ignore_sample(sample, field_name="segmentations", ignore_label="ignore", color=(128, 0, 128)):
    try:
        detections = sample[field_name]["detections"]
    except:
        detections = []

    boxes = []
    for obj in detections:
        if ignore_label == obj.label:
            boxes.append(obj.bounding_box)

    if boxes:
        img = cv.imread(sample.filepath, 1)
        height, width, _ = img.shape
        for x, y, w, h in boxes:
            x, y = int(round(x * width)), int(round(y * height))
            w, h = int(round(w * width)), int(round(h * height))

            rx, ry = w // 2, h // 2
            cx, cy = x + rx, y + ry

            cv.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, -1)
        cv.imwrite(sample.filepath, img)


def mask_ignore_dataset(dataset, field_name="segmentations", ignore_label="ignore", color=(128, 0, 128)):
    for sample in tqdm(dataset):
        mask_ignore_sample(sample, field_name, ignore_label, color)


def mask_ignore_from_dir(dataset_dir, splits="auto", field_name="segmentations", ignore_label="ignore", color=(128, 0, 128)):
    dataset_dir = Path(dataset_dir)

    tmp_dir = f"{dataset_dir.name}_tmp"
    tmp_dir = dataset_dir.with_name(tmp_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    export_dir = f"{dataset_dir.name}_masked"
    export_dir = dataset_dir.with_name(export_dir)
    shutil.rmtree(export_dir, ignore_errors=True)

    shutil.copytree(dataset_dir, tmp_dir)

    info_py = sorted(tmp_dir.glob("**/info.py"))[0]
    with open(info_py, "r") as f:
        codestr = f.read()
    info = eval(re.split(r"info\s*=\s*", codestr)[1])

    classes = info["classes"]
    mask_targets = info["mask_targets"]
    dataset = create_dataset("mask_ignore_from_dir", "detection", "001", classes, mask_targets)

    coco_add_samples(dataset, tmp_dir, label_field=field_name, splits=splits)

    mask_ignore_dataset(dataset, field_name, ignore_label, color)

    coco_export(export_dir, dataset, field_name, splits=splits, mask_type="polygons", tolerance=1)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return str(export_dir)
