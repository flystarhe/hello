# pip install opencv-python
# pip install pyomniunwarp>=0.2.4
# -i https://pypi.tuna.tsinghua.edu.cn/simple
# imwrite()
# - cv.IMWRITE_PNG_COMPRESSION: default 3
# - cv.IMWRITE_JPEG_QUALITY: default 95
import shutil
import sys
import time
from pathlib import Path

import cv2 as cv
from pyomniunwarp import OmniUnwarp

suffix_set = set(".jpg,.png".split(","))


def find_images(input_dir):
    image_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            image_paths.append(f.as_posix())

    return image_paths


def to_unwarp(image_paths, output_dir, cal_file, version, mode, fov, rois, format, prefix):
    if cal_file is not None:
        kwargs = {
            "calib_results_path": cal_file,
            "version": version,
            "mode": mode,
            "FOV": fov,
        }
        unwarper = OmniUnwarp(**kwargs)
    else:
        unwarper = None

    for i, image_path in enumerate(image_paths):
        frame = cv.imread(image_path)

        if prefix is not None:
            name = f"{prefix}_{i:06d}"
        else:
            name = Path(image_path).stem

        if unwarper is None:
            filename = f"data/{name}{format}"
            cv.imwrite(str(output_dir / filename), frame)
        else:
            imgs, masks, labels = unwarper.rectify(frame)
            for img, mask, label in zip(imgs, masks, labels):
                if label in rois:
                    filename = f"data/{name}_roi_{label}{format}"
                    cv.imwrite(str(output_dir / filename), img)
                    filename = f"mask/{name}_roi_{label}.png"
                    cv.imwrite(str(output_dir / filename), mask)


def func(input_dir, output_dir, cal_file, version, mode, fov, rois, format, prefix):
    input_dir = Path(input_dir)

    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=False)
    (output_dir / "mask").mkdir(parents=True, exist_ok=False)

    if cal_file is not None:
        if not Path(cal_file).is_file():
            cal_file = str(input_dir / cal_file)

        if not Path(cal_file).is_file():
            cal_file = None

    if prefix is not None:
        if prefix == "date":
            prefix = time.strftime(r"%Y%m%d_%H%M%S")

    image_paths = find_images(input_dir)
    print(f"[INFO] find images: {len(image_paths)}")
    to_unwarp(image_paths, output_dir, cal_file, version, mode, fov, rois, format, prefix)

    return f"\n[OUTDIR]\n{output_dir}"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("output_dir", type=str,
                        help="output dir")
    parser.add_argument("--cal_file", type=str, default=None,
                        help="calibrated model file path")
    parser.add_argument("--version", type=str, default="0.2.2",
                        help="set the kernel version")
    parser.add_argument("--mode", type=str, default="cuboid",
                        help="set the unwarp mode")
    parser.add_argument("--fov", type=int, default=90,
                        help="set the fov")
    parser.add_argument("--rois", type=str, nargs="+",
                        default=["front", "left", "right", "front-left", "front-right"],
                        choices=["front", "left", "back", "right", "front-left", "front-right"])
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
