import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2 as cv

suffix_set = set(".avi,.mp4,.MOV,.mkv".split(","))


def find_videos(input_dir):
    video_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            video_paths.append(f.as_posix())

    return video_paths


def split_video(video_path, duration, out_dir):
    cap = cv.VideoCapture(video_path)
    cap_fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration_s = duration * 60
    total_s = frame_count // cap_fps

    suffix = Path(video_path).suffix
    for i, progress_s in enumerate(range(0, total_s, duration_s), 1):
        i_s = progress_s
        i_m, i_s = divmod(i_s, 60)
        i_h, i_m = divmod(i_m, 60)
        curr_start = f"{i_h:02d}:{i_m:02d}:{i_s:02d}"

        curr_input = video_path

        remaining_s = total_s - progress_s
        if remaining_s < 60:
            continue

        i_s = min(remaining_s, duration_s)
        i_m, i_s = divmod(i_s, 60)
        i_h, i_m = divmod(i_m, 60)
        curr_duration = f"{i_h:02d}:{i_m:02d}:{i_s:02d}"

        prefix = time.strftime("%Y%m%d_%H%M%S")
        curr_output = str(Path(out_dir) / f"{prefix}_{i:02d}{suffix}")

        command_line = f"ffmpeg -ss {curr_start} -i {curr_input} -t {curr_duration} -c copy {curr_output}"
        result = subprocess.run(command_line, shell=True, stdout=subprocess.PIPE)
        if result.returncode != 0:
            print(f"[ERR]\n  {command_line}")


def func(input_dir, output_dir, duration):
    input_dir = Path(input_dir)

    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=False)

    out_dir = str(output_dir / "videos")
    for video_path in find_videos(input_dir):
        split_video(video_path, duration, out_dir)

    return str(out_dir)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("output_dir", type=str,
                        help="output dir")
    parser.add_argument("--duration", type=int, default=5,
                        help="minutes per segment")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
