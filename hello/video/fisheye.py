# pip install opencv-python
# -i https://pypi.tuna.tsinghua.edu.cn/simple
# imwrite()
# - cv.IMWRITE_PNG_COMPRESSION: default 3
# - cv.IMWRITE_JPEG_QUALITY: default 95
import json
import shutil
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

suffix_set = set(".jpg,.png".split(","))


def find_images(input_dir):
    image_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            image_paths.append(f.as_posix())

    return image_paths


def to_unwarp(image_paths, output_dir, fisheye, format, prefix):
    if fisheye is not None:
        with open(fisheye, "r") as f:
            params = json.load(f)

        fisheye_K = np.array(params["fisheye_camera_K"]).reshape(3, 3)
        fisheye_D = np.array(params["fisheye_dist"])
        img_shape = params["fisheye_image_size"]

        map1, map2 = cv.fisheye.initUndistortRectifyMap(
            fisheye_K,
            fisheye_D,
            np.eye(3),
            fisheye_K,
            img_shape,
            cv.CV_32FC1
        )

    for i, image_path in enumerate(image_paths):
        frame = cv.imread(image_path)

        if prefix is not None:
            name = f"{prefix}_{i:06d}"
        else:
            name = Path(image_path).stem

        if fisheye is not None:
            frame = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)

        filename = f"data/{name}{format}"
        cv.imwrite(str(output_dir / filename), frame)


def func(input_dir, output_dir, fisheye, format, prefix):
    input_dir = Path(input_dir)

    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=False)

    if fisheye is not None:
        if not Path(fisheye).is_file():
            fisheye = str(input_dir / fisheye)

        if not Path(fisheye).is_file():
            fisheye = None

    if prefix is not None:
        if prefix == "date":
            prefix = time.strftime(r"%Y%m%d_%H%M%S")

    image_paths = find_images(input_dir)
    print(f"[INFO] find images: {len(image_paths)}")
    to_unwarp(image_paths, output_dir, fisheye, format, prefix)

    with open(output_dir / "README.md", "w") as f:
        f.write("# README\n\n## Data Processing\n\n")

    return f"\n[OUTDIR]\n{output_dir}"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("output_dir", type=str,
                        help="output dir")
    parser.add_argument("-e", "--fisheye", type=str, default=None,
                        help="fisheye parameter file path")
    parser.add_argument("--format", type=str, default=".jpg",
                        choices=[".png", ".jpg"])
    parser.add_argument("--prefix", type=str, default=None,
                        help="'date' or variable name")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
