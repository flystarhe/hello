from pathlib import Path

import cv2 as cv
import numpy as np

from fiftyone.utils.video import sample_video as _sample_video


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


def sample_video(video_path, output_path="/tmp", fps=10, original_frame_numbers=True):
    output_patt = str(Path(output_path) / Path(video_path).stem / "%012d.png")
    return _sample_video(video_path, output_patt, fps=fps, original_frame_numbers=original_frame_numbers)


def from_tar():
    """
    cn_courtyard_lfi_20220527_v0.01.tar
    #  cn_courtyard_lfi_20220527/
    #  ├── calib_results.txt
    #  ├── customize.json
    #  ├── libprocess.so
    #  └── data
    #      ├── 20220527_113755.mp4
    #      ├── 20220527_113803.mp4
    #      ├── 20220527_113805.png
    #      └── 20220527_113814.png
    """

    return 0
