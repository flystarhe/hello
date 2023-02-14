import json
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

help_doc_str = """\
- press `esc` to exit
- press `space` change mode
- press `u` speed up
- press `d` slow down
- press `n` go further
- press `b` take a step back
- press `f` freeze or not
"""


suffix_set = set(".avi,.mp4".split(","))


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

    curr_pos, step_size, freeze, keep = 0, cap_fps, 0, 1
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
        txt = f"{curr_pos=}/{frame_count}:{cap_fps}, {step_size=}, {freeze=}, {keep=}"
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
            curr_pos = this_pos + step_size
            if freeze == 0:
                if keep == 1:
                    tag_frames[:, this_pos:curr_pos] = (0, 255, 0)
                else:
                    tag_frames[:, this_pos:curr_pos] = (0, 0, 255)
        elif key == ord("b"):
            curr_pos = this_pos - step_size
            curr_pos = max(0, curr_pos)
        elif key == ord("f"):
            freeze = int(not freeze)

    cv.destroyAllWindows()
    cap.release()

    return tag_frames[0, :, 1]


def clip_video(video_path, tag_frames, output_dir, interval, fisheye):
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

    index = 0
    keep_frames = 0
    limit = tag_frames.size
    prefix = Path(video_path).stem
    cap = cv.VideoCapture(video_path)

    while index < limit:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if tag_frames[index] > 0:
            if keep_frames % interval == 0:
                if fisheye is not None:
                    frame = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR)

                filename = f"data/{prefix}_i{index:06d}.jpg"
                cv.imwrite(str(output_dir / filename), frame)

            keep_frames += 1

        index += 1

    cap.release()


def func(input_dir, output_dir, factor, interval, fisheye):
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

    if fisheye is not None:
        if not Path(fisheye).is_file():
            fisheye = str(input_dir / fisheye)

        if not Path(fisheye).is_file():
            fisheye = None

    shutil.rmtree(output_dir, ignore_errors=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=False)

    for video_path in video_paths:
        tag_frames = tag_video(video_path, factor)
        if tag_frames.max() > 0:
            clip_video(video_path, tag_frames, output_dir, interval, fisheye)

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
    parser.add_argument("-i", "--interval", type=int, default=5,
                        help="sample the frames")
    parser.add_argument("-e", "--fisheye", type=str, default=None,
                        help="fisheye parameter file path")

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
