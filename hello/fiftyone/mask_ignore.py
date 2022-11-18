import shutil
from pathlib import Path

import cv2 as cv
from tqdm import tqdm

from hello.fiftyone.dataset import load_detection_dataset


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
        img_h, img_w, _ = img.shape
        for x, y, w, h in boxes:
            x, y = round(x * img_w), round(y * img_h)
            w, h = round(w * img_w), round(h * img_h)

            rx, ry = w // 2, h // 2
            cx, cy = x + rx, y + ry

            cv.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, -1)
        cv.imwrite(sample.filepath, img)


def mask_ignore_dataset(dataset, field_name="segmentations", ignore_label="ignore", color=(128, 0, 128)):
    for sample in tqdm(dataset):
        mask_ignore_sample(sample, field_name, ignore_label, color)


def mask_ignore_from_dir(dataset_dir, splits=None, field_name="segmentations", ignore_label="ignore", color=(128, 0, 128)):
    dataset_dir = Path(dataset_dir)
    target_dir = f"{dataset_dir.name}_mask"
    target_dir = dataset_dir.with_name(target_dir)

    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(dataset_dir, target_dir)

    dataset = load_detection_dataset(target_dir, splits=splits)
    mask_ignore_dataset(dataset, field_name, ignore_label, color)
    return str(target_dir)
