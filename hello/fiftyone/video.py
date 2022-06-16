from pathlib import Path

import cv2 as cv
import numpy as np

from fiftyone.utils.video import sample_video as _sample_video


def make_mp4(output, frames, fps=None, size=None):
    if fps is None:
        fps = 30

    if size is None:
        size = (600, 800)

    height, width = size
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out = cv.VideoWriter(str(output), fourcc, fps, (width, height))
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype="uint8")
        cv.putText(frame, f"{i+1:06d}", (15, 35),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        out.write(frame)
    out.release()

    return output


def sample_video(video_path, output_patt="images/%012d.png", fps=10, original_frame_numbers=True):
    return _sample_video(video_path, output_patt, fps=fps, original_frame_numbers=original_frame_numbers)
