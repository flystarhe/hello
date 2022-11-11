import shutil
from pathlib import Path

import cv2 as cv
import numpy as np
from fiftyone import ViewField as F


def imshow(img):
    from IPython.display import display
    from PIL import Image
    display(Image.fromarray(img, mode="RGB"))


def from_coco_segmentations(out_dir, dataset, crop_size=(200, 200), notebook=False):
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=False)

    for index, label in enumerate(dataset.default_classes):
        view = dataset.filter_labels("segmentations", F("label") == label)
        if len(view) < 5:
            print(f"id={index}, name={label}, skip")
            continue

        patches = []
        for sample in view.take(5):
            img = cv.imread(sample.filepath, 1)

            img_h, img_w, _ = img.shape
            assert sample.metadata["width"] == img_w
            assert sample.metadata["height"] == img_h

            for obj in sample.segmentations.detections:
                if obj.label == label:
                    _box = obj.bounding_box  # [x, y, w, h] / s
                    x, y = round(_box[0] * img_w), round(_box[1] * img_h)

                    mask_h, mask_w = obj.mask.shape
                    w, h = min(mask_w, img_w - x), min(mask_h, img_h - y)

                    mask = obj.mask.astype("uint8") * 255
                    if mask.ndim == 2:
                        mask = np.stack((mask, mask, mask), axis=-1)
                    mask = cv.resize(mask[:h, :w], crop_size, interpolation=cv.INTER_LINEAR)
                    patch = cv.resize(img[y:y + h, x:x + w], crop_size, interpolation=cv.INTER_LINEAR)

                    patches.append(np.concatenate((patch, mask), axis=0))
                    break
        out_file = str(out_dir / f"images/{index:02d}_{label}.png")
        newimg = np.concatenate(patches, axis=1)
        cv.imwrite(out_file, newimg)
        if notebook:
            print(f"id={index}, name={label}")
            imshow(newimg[..., ::-1])
    return str(out_dir)
