from pathlib import Path

import cv2 as cv
import numpy as np


def make_mp4(output, frames, fps=None, size=None):
    if fps is None:
        fps = 30

    if size is None:
        size = (600, 800)

    height, width = size
    # mp4:mp4v|mpeg; avi:XVID|MJPG
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
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
