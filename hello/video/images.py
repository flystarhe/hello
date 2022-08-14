import shutil
import sys
from pathlib import Path

import cv2 as cv

suffix_set = set(".avi,.mp4".split(","))


def find_videos(input_dir):
    video_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            video_paths.append(f.as_posix())

    return video_paths


def to_images(video_path, output_dir, format):
    cap = cv.VideoCapture(video_path)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    prefix = Path(video_path).stem

    index = 0
    while index < count:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        a, b = divmod(index, fps)
        filename = f"data/{prefix}_time_{a:06d}_{b:03d}{format}"
        cv.imwrite(str(output_dir / filename), frame)
        index += 1


def func(input_dir, output_dir, format):
    input_dir = Path(input_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        new_name = f"{input_dir.name}_images"
        output_dir = input_dir.with_name(new_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "data").mkdir(parents=True)

    video_paths = find_videos(input_dir)
    for video_path in video_paths:
        to_images(video_path, output_dir, format)

    return output_dir.as_posix()


def parse_args(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Clip Videos")

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="output dir")
    parser.add_argument("-f", "--format", type=str, default=".jpg",
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
