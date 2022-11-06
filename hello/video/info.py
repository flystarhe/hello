import sys
from pathlib import Path

import cv2 as cv

suffix_set = set(".avi,.mp4,.MOV,.mkv".split(","))


def find_videos(input_dir):
    video_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            video_paths.append(f.as_posix())

    return video_paths


def print_info(video_path):
    cap = cv.VideoCapture(video_path)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    video_info = ", ".join(Path(video_path).parts[-2:])
    print(f"{video_info}, seconds={count//fps:04d}, size={width}x{height}, {fps=}")

    cap.release()


def func(input_dir):
    input_dir = Path(input_dir)

    if input_dir.is_file():
        video_paths = [input_dir.as_posix()]
        input_dir = input_dir.parent
    else:
        video_paths = find_videos(input_dir)

    for video_path in video_paths:
        print_info(video_path)

    return "\n[END]"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=str,
                        help="videos dir or file path")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
