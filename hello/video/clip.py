import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

suffix_set = set(".avi,.mp4".split(","))

help_doc_str = """
- press `esc` to exit
- press `space` change mode
- press `u` speed up
- press `d` slow down
- press `n` go further
- press `b` take a step back
- press `f` freeze or not
"""


def find_videos(input_dir):
    video_paths = []

    for f in sorted(Path(input_dir).glob("**/*")):
        if f.suffix in suffix_set:
            video_paths.append(f.as_posix())

    return video_paths


def tag_video(video_path):
    cap = cv.VideoCapture(video_path)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    tag_frames = np.full((30, count, 3), (255, 0, 0), dtype="uint8")
    curr_pos, step_size, freeze, keep = 0, 5, 0, 0

    while curr_pos < count:
        this_pos = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        if this_pos != curr_pos:
            cap.set(cv.CAP_PROP_POS_FRAMES, curr_pos)
            this_pos = curr_pos

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        banner = np.full((30, width, 3), (0, 0, 255), dtype="uint8")
        txt = f"{curr_pos=}, {step_size=}, {freeze=}, {keep=}"
        cv.putText(banner, txt, (15, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

        tag_bar = cv.resize(tag_frames, (width, 30),
                            interpolation=cv.INTER_NEAREST)
        center = (int(curr_pos / count * width), 15)
        cv.circle(tag_bar, center, 5, (255, 255, 255), -1)

        cv.imshow(video_path, np.concatenate((banner, frame, tag_bar)))

        key = cv.waitKey(0)
        if key == 27:  # esc
            break
        elif key == 32:  # space
            keep = int(not keep)
        elif key == ord("u"):
            step_size = step_size * 2
        elif key == ord("d"):
            step_size = step_size // 2
            step_size = max(1, step_size)
        elif key == ord("n"):
            curr_pos = this_pos + step_size * fps
            if freeze == 0:
                if keep == 1:
                    tag_frames[:, this_pos:curr_pos] = (0, 255, 0)
                else:
                    tag_frames[:, this_pos:curr_pos] = (0, 0, 255)
        elif key == ord("b"):
            curr_pos = this_pos - step_size * fps
            curr_pos = max(0, curr_pos)
        elif key == ord("f"):
            freeze = int(not freeze)

    cv.destroyAllWindows()
    cap.release()

    return tag_frames[0, :, 1]


def clip_video(tag_frames, video_path, output_dir):
    outfile = (Path(output_dir) / Path(video_path).name).with_suffix(".mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    cap = cv.VideoCapture(video_path)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter(str(outfile), fourcc, fps, (width, height))

    curr_pos = 0
    while curr_pos < count:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if tag_frames[curr_pos] > 0:
            out.write(frame)
        curr_pos += 1

    out.release()
    cap.release()


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
        tag_frames = tag_video(video_path)
        clip_video(tag_frames, video_path, output_dir)

    return output_dir.as_posix()


def parse_args(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Clip Videos")

    parser.add_argument("input_dir", type=str,
                        help="input dir")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
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
