import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

suffix_set = set(".avi,.mp4".split(","))


help_doc_str = """
Press the left mouse button to start marking area.
- press `esc` to exit
- press `space` change mode
- press `u` speed up
- press `d` slow down
- press `n` go further
- press `r` take a step back
- press `k` freeze or not
"""


def make_mp4(output, frames, fps=None, size=None, fourcc=None):
    if fps is None:
        fps = 30

    if size is None:
        size = (600, 800)

    height, width = size

    # .mp4: mp4v|mpeg
    # .avi: XVID|MJPG
    if fourcc is None:
        suffix = Path(output).suffix
        if suffix == ".mp4":
            fourcc = "mpeg"
        elif suffix == ".avi":
            fourcc = "XVID"

    fourcc = cv.VideoWriter_fourcc(*fourcc)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out = cv.VideoWriter(str(output), fourcc, fps, (width, height))
    for i in range(frames):
        a, b = divmod(i + 1, fps)
        frame = np.zeros((height, width, 3), dtype="uint8")
        cv.rectangle(frame, (5, 5), (225, 35), (0, 0, 255), -1)
        cv.putText(frame, f"{a:06d}.{b:03d}", (15, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
        out.write(frame)
    out.release()

    return output


def find_videos(input_dir):
    video_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            video_paths.append(f.as_posix())

    return video_paths


def tag_video(video_path):
    cap = cv.VideoCapture(video_path)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    keep, step_size, curr_pos, next_pos, freeze = 0, 5, -1, 0, 0

    tags = []
    tags_bar = np.zeros((35, count, 3), dtype="uint8")
    while True:
        if next_pos >= count:
            break

        if not freeze:
            tags_bar[curr_pos:next_pos] = keep

        if curr_pos != next_pos:
            cap.set(cv.CAP_PROP_POS_FRAMES, next_pos)
            _, frame = cap.read()
            curr_pos = next_pos

        txt = f"{keep=}, {step_size=}, {curr_pos=}, {next_pos=}, {freeze=}"
        cv.rectangle(frame, (5, 5), (width, 35), (0, 0, 255), -1)
        cv.putText(frame, txt, (15, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

        view_bar = cv.resize(tags_bar, (width, 30),
                             interpolation=cv.INTER_NEAREST)
        cv.circle(view_bar, (50, 15), 5, (0, 0, 255), -1)
        frame[height - 30:, :] = view_bar
        cv.imshow(video_path, frame)

        key = cv.waitKey(0)

        if key == 27:  # esc
            break
        elif key == 32:  # space
            keep = int(not keep)
        elif key == ord("u"):
            step_size += step_size // 2
        elif key == ord("d"):
            step_size -= step_size // 2
        elif key == ord("n"):
            next_pos += step_size * fps
        elif key == ord("r"):
            next_pos -= step_size * fps
        elif key == ord("k"):
            freeze = int(not freeze)

        step_size = max(1, step_size)

    cv.destroyAllWindows()
    cap.release()
    return tags


def clip_video(tags, video_path, output_dir):
    pass


def func(input_dir, output_dir):
    input_dir = Path(input_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        new_name = f"{input_dir.name}_clip"
        output_dir = input_dir.with_name(new_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    print(help_doc_str)

    video_paths = find_videos(input_dir)
    for video_path in video_paths:
        tags = tag_video(video_path)
        clip_video(tags, video_path, output_dir)

    return output_dir.as_posix()


def parse_args(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Clip Videos")

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("-out", "--output_dir", type=str, default=None,
                        help="output dir")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
