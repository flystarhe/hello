import re
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

pattern_decimal = re.compile(r"\d+(\.\d+)?")

suffix_set = set(".avi,.mp4,.MOV,.mkv".split(","))

help_doc_str = """\
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


def tag_video(video_path, factor):
    cap = cv.VideoCapture(video_path)

    cap_fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    tag_frames = np.full((30, frame_count, 3), (255, 0, 0), dtype="uint8")

    curr_pos, step_size, freeze, keep = 0, 1, 0, 1
    while curr_pos < frame_count:
        this_pos = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        if this_pos != curr_pos:
            cap.set(cv.CAP_PROP_POS_FRAMES, curr_pos)
            this_pos = curr_pos

        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        banner = np.full((30, frame_width, 3), (0, 0, 255), dtype="uint8")
        txt = f"{curr_pos=}/{frame_count}, {step_size=}*{cap_fps}, {freeze=}, {keep=}"
        cv.putText(banner, txt, (15, 25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        tag_bar = cv.resize(tag_frames, (frame_width, 30), interpolation=cv.INTER_NEAREST)
        center = (int(curr_pos / frame_count * frame_width), 15)
        cv.circle(tag_bar, center, 5, (255, 255, 255), -1)

        image = np.concatenate((banner, frame, tag_bar))
        if factor is not None:
            image = cv.resize(image, None, fx=factor, fy=factor, interpolation=cv.INTER_NEAREST)
        cv.imshow(video_path, image)

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
            curr_pos = this_pos + step_size * cap_fps
            if freeze == 0:
                if keep == 1:
                    tag_frames[:, this_pos:curr_pos] = (0, 255, 0)
                else:
                    tag_frames[:, this_pos:curr_pos] = (0, 0, 255)
        elif key == ord("b"):
            curr_pos = this_pos - step_size * cap_fps
            curr_pos = max(0, curr_pos)
        elif key == ord("f"):
            freeze = int(not freeze)

    cv.destroyAllWindows()
    cap.release()

    return tag_frames[0, :, 1]


def clip_video(video_path, tag_frames, output_dir):
    txt_file = Path(video_path).with_suffix(".txt")
    clip_text_file(txt_file, tag_frames, output_dir)

    outfile = Path(output_dir) / f"data/{Path(video_path).stem}.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    cap = cv.VideoCapture(video_path)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    out = cv.VideoWriter(str(outfile), fourcc, fps, (width, height))

    frame_count = tag_frames.size
    print(f"[{outfile}] Saving ...")

    curr_pos = 0
    while curr_pos < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if tag_frames[curr_pos] > 0:
            out.write(frame)

        curr_pos += 1

    cap.release()
    out.release()


def clip_text_file(infile, tag_frames, output_dir):
    if not Path(infile).is_file():
        return None

    frame_count = tag_frames.size

    curr_pos = 0
    head, data = [], []
    with open(infile, "r") as f:
        lines = [l.strip() for l in f.readlines()]
        lines = [l for l in lines if l]
        for line in lines:
            if curr_pos >= frame_count:
                break

            if pattern_decimal.match(line):
                if tag_frames[curr_pos] > 0:
                    data.append(line)
                curr_pos += 1
            else:
                head.append(line)

    outfile = Path(output_dir) / f"data/{Path(infile).name}"
    with open(outfile, "w") as f:
        f.write("\n".join(head))
        f.write("\n")
        f.write("\n".join(data))
    return outfile


def func(input_dir, output_dir, factor):
    input_dir = Path(input_dir)

    if input_dir.is_file():
        video_paths = [input_dir.as_posix()]
        input_dir = input_dir.parent
    else:
        video_paths = find_videos(input_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        new_name = f"{input_dir.name}_clip"
        output_dir = input_dir.with_name(new_name)

    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=False)

    for video_path in video_paths:
        tag_frames = tag_video(video_path, factor)
        if tag_frames.max() > 0:
            clip_video(video_path, tag_frames, output_dir)

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

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    print(help_doc_str)
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
