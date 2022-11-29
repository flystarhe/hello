import sys
import time
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np


class Queue:

    def __init__(self, size) -> None:
        self.size = size
        self.data = []

    def append(self, value):
        self.data.append(value)
        self.data = self.data[-self.size:]

    def info(self):
        data = self.data
        n = len(data)
        if n > 1:
            v = (n - 1) / (data[-1] - data[0])
            return f"{v:.1f}"
        return "none"


class Video:

    def __init__(self, rtsp_url) -> None:
        self.time_window = 30  # 30ms
        self.rtsp_url = rtsp_url
        self.cap = None
        self.out = None

    def get_cap(self):
        if self.cap is None:
            self.cap = cv.VideoCapture(self.rtsp_url)

        if not self.cap.isOpened():
            self.cap.open(self.rtsp_url)

        return self.cap

    def get_info(self):
        cap = self.get_cap()

        fps, width, height = 0, 0, 0

        if cap is not None:
            fps = int(cap.get(cv.CAP_PROP_FPS))
            width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        return fps, width, height

    def read(self):
        cap = self.get_cap()

        frame = None
        errstr = None

        if not cap.isOpened():
            errstr = f"Failed to open: {self.rtsp_url}"
            return frame, errstr

        try:
            ret, frame = cap.read()
            if not ret:
                errstr = "Can't receive frame"
                frame = None
        except:
            print(traceback.format_exc())
            errstr = "read exception"
            frame = None

        return frame, errstr

    def get_out(self):
        if self.out is None:
            fps, width, height = self.get_info()

            out = None
            if min(fps, width, height) > 0:
                fourcc = cv.VideoWriter_fourcc(*"XVID")
                out_file = time.strftime(r"%Y%m%d_%H%M%S")
                Path("data").mkdir(parents=True, exist_ok=True)
                out = cv.VideoWriter(f"data/{out_file}.avi", fourcc, fps, (width, height))
                self.time_window = 1000 // fps

            self.out = out

        return self.out

    def write(self, frame):
        if frame is None:
            return False

        out = self.get_out()

        if out is None:
            return False

        out.write(frame)

        return True

    def save(self):
        if self.out is not None:
            try:
                self.out.release()
            except:
                print("`out.release()` exception")

        self.out = None

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                print("`cap.release()` exception")

        if self.out is not None:
            try:
                self.out.release()
            except:
                print("`out.release()` exception")

        self.cap = None
        self.out = None


def func(rtsp_url, view="front"):
    _queue = Queue(10)
    _video = Video(rtsp_url)

    flag = "watch"
    # watch, recording

    time_loc = time.time()
    while True:
        frame, errstr = _video.read()

        if flag == "recording":
            if _video.write(frame):
                n_frame += 1
            msgstr = f"{flag} frame={n_frame} vfps={_queue.info()}"
        else:
            msgstr = f"{flag} vfps={_queue.info()}"

        if frame is None:
            frame = np.zeros((1080, 1920, 3), dtype="uint8")

        if view == "front":
            frame = frame[:, frame.shape[1]//2:]
            frame = cv.flip(cv.transpose(frame), 0)

        if errstr is None:
            errstr = "hotkey: n (recording), s (save), q (quit)"

        cv.putText(frame, msgstr, (15, 25), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv.putText(frame, errstr, (15, 55), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv.imshow("RTSP STREAM", frame)

        latency = int((time.time() - time_loc) * 1000)
        sleep = max(1, _video.time_window - latency)

        key = cv.waitKey(sleep)
        if key == ord("q"):
            cv.destroyAllWindows()
            _video.release()
            break
        elif key == ord("n"):
            if flag != "recording":
                n_frame = 0
                flag = "recording"
        elif key == ord("s"):
            _video.save()
            flag = "watch"

        time_loc = time.time()
        _queue.append(time_loc)


# python rtsp_pull.py rtsp://localhost:8554/mystream
# python -m hello.video.rtsp_pull rtsp://localhost:8554/mystream front
if __name__ == "__main__":
    args = sys.argv[1:]

    assert len(args) > 0

    # base_url = "rtsp://192.168.0.145/<filename>"
    base_url = args[0]

    view = "full"
    if len(args) > 1:
        view = args[1]

    rtsp_url = base_url.replace("<filename>", "lfi_stream.live")
    func(rtsp_url, view=view)
