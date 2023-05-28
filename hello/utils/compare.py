import shutil
from pathlib import Path

import cv2 as cv
import numpy as np


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


def compare_image_dir(out_dir, base_dir=None, image_dirs=None):
    assert base_dir is not None or image_dirs is not None

    out_dir = Path(out_dir)

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=False)

    if base_dir is not None:
        base_dir = Path(base_dir)
        if image_dirs is not None:
            image_dirs = [base_dir / image_dir for image_dir in image_dirs]
        else:
            image_dirs = sorted([f for f in base_dir.glob("*") if f.is_dir()])

    data = [{f.name: str(f) for f in Path(image_dir).glob("images/*.jpg")}
            for image_dir in image_dirs]

    names = set(data[0].keys())
    for v in data[1:]:
        names = names & set(v.keys())

    for name in sorted(names):
        imgs = [cv.imread(v[name]) for v in data]
        cv.imwrite(str(out_dir / name), np.concatenate(imgs, axis=0))

    return out_dir
