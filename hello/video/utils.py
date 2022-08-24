from pathlib import Path

import cv2 as cv
import numpy as np


def make_mp4(outfile, frames, fps=None, size=None, fourcc=None):
    if fps is None:
        fps = 30

    if size is None:
        size = (1080, 1920)

    if fourcc is None:
        suffix = Path(outfile).suffix
        if suffix == ".mp4":
            fourcc = "mp4v"
        elif suffix == ".avi":
            fourcc = "XVID"

    height, width = size
    fourcc = cv.VideoWriter_fourcc(*fourcc)
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    out = cv.VideoWriter(str(outfile), fourcc, fps, (width, height))
    for index in range(frames):
        frame = np.full((height, width, 3), (127, 127, 127), dtype="uint8")
        cv.rectangle(frame, (5, 5), (225, 35), (0, 0, 255), -1)
        a, b = divmod(index, fps)
        cv.putText(frame, f"{a:06d}_{b:03d}", (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    return outfile
