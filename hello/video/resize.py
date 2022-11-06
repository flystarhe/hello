import shutil
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


def resize_video(video_path, output_dir, factor, fps, nosave):
    cap = cv.VideoCapture(video_path)

    cap_fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if nosave:
        print(f"\n{video_path}\nfps={cap_fps},size={frame_width}x{frame_height}")
        return 0

    outfile = Path(output_dir) / f"data/{Path(video_path).stem}.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    step_size = max(1, cap_fps // fps)
    fps = cap_fps // step_size

    new_width, new_height = frame_width, frame_height
    if factor is not None:
        new_width = int(frame_width * factor)
        new_height = int(frame_height * factor)

    out = cv.VideoWriter(str(outfile), fourcc, fps, (new_width, new_height))

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

        if factor is not None:
            if factor < 0.5:  # shrinking
                frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)
            else:  # zooming
                frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_LINEAR)
        out.write(frame)

    cap.release()
    out.release()


def func(input_dir, output_dir, factor, fps, nosave):
    input_dir = Path(input_dir)

    if input_dir.is_file():
        video_paths = [input_dir.as_posix()]
        input_dir = input_dir.parent
    else:
        video_paths = find_videos(input_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        new_name = f"{input_dir.name}_resize"
        output_dir = input_dir.with_name(new_name)

    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=False)

    for video_path in video_paths:
        resize_video(video_path, output_dir, factor, fps, nosave)

    return f"\n[OUTDIR]\n{output_dir}"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("input_dir", type=str,
                        help="videos dir or file path")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="output dir")
    parser.add_argument("-f", "--factor", type=float, default=None,
                        help="resize factor")
    parser.add_argument("--fps", type=int, default=5,
                        help="sample the frames")
    parser.add_argument("--nosave", action="store_true",
                        help="only print video info")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
