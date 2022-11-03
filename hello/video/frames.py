# pip install opencv-python
# pip install pyomniunwarp>=0.2.4
# -i https://pypi.tuna.tsinghua.edu.cn/simple
# imwrite()
# - cv.IMWRITE_PNG_COMPRESSION: default 3
# - cv.IMWRITE_JPEG_QUALITY: default 95
import shutil
import sys
from pathlib import Path

import cv2 as cv
from pyomniunwarp import OmniUnwarp

suffix_set = set(".avi,.mp4,.MOV,.mkv".split(","))


def find_videos(input_dir):
    video_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            video_paths.append(f.as_posix())

    return video_paths


def to_frames(video_path, output_dir, fps, cal_file, version, mode, fov, rois, format):
    cap = cv.VideoCapture(video_path)

    cap_fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    step_size = max(1, cap_fps // fps)

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

    prefix = Path(video_path).stem

    index = 0
    while index < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        a, b = divmod(index, cap_fps)
        index += 1

        if (b % step_size) != 0:
            continue

        if unwarper is None:
            filename = f"data/{prefix}_time_{a:06d}_{b:03d}{format}"
            cv.imwrite(str(output_dir / filename), frame)
        else:
            imgs, masks, labels = unwarper.rectify(frame)
            for img, mask, label in zip(imgs, masks, labels):
                if label in rois:
                    filename = f"data/{prefix}_time_{a:06d}_{b:03d}_roi_{label}{format}"
                    cv.imwrite(str(output_dir / filename), img)
                    filename = f"mask/{prefix}_time_{a:06d}_{b:03d}_roi_{label}.png"
                    cv.imwrite(str(output_dir / filename), mask)

    cap.release()


def func(input_dir, output_dir, fps, cal_file, version, mode, fov, rois, format):
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

    for video_path in find_videos(input_dir):
        to_frames(video_path, output_dir, fps, cal_file, version, mode, fov, rois, format)

    return f"\n[OUTDIR]\n{output_dir}"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("output_dir", type=str,
                        help="output dir")
    parser.add_argument("--fps", type=int, default=5,
                        help="sample the frames")
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

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
