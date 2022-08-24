import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np

# conda install -c conda-forge ffmpeg
#
# url:
# - https://github.com/topics/rtsp-server
# - https://github.com/aler9/rtsp-simple-server
#
# steps:
# 1. launch rtsp server: rtsp-simple-server.exe
# 2. push: `ffmpeg -re -stream_loop -1 -i test.mp4 -c copy -f rtsp rtsp://localhost:8554/mystream`
# 3. pull: `ffmpeg -i rtsp://localhost:8554/mystream -c copy output.mp4`


def rtsp_push(rtsp_url="rtsp://localhost:8554/video", height=1080, width=1920, fps=20, **kwargs):
    command = ["ffmpeg",
               "-y",
               "-f", "rawvideo",
               "-vcodec", "rawvideo",
               "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}",
               "-r", f"{fps}",
               "-i", "-",
               "-c:v", "libx264",
               "-pix_fmt", "yuv420p",
               "-preset", "ultrafast",
               "-f", "rtsp",
               rtsp_url]

    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv.putText(frame, f"{time.time():.3f}", (15, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        pipe.stdin.write(frame.tobytes())
        cv.imshow("rtsp push", frame)

        key = cv.waitKey(30)
        if key == ord("q"):
            break

    cv.destroyAllWindows()
    return 0


def rtsp_pull(rtsp_url, frames=900, save=False, **kwargs):
    cap = cv.VideoCapture(rtsp_url)

    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {fps=}, {width=}, {height=}")

    delay = 1000 // fps

    if save:
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        Path("tmp").mkdir(parents=True, exist_ok=True)
        out = cv.VideoWriter("tmp/rtsp_video.avi", fourcc, fps, (width, height))

    while True:
        time_loc = time.time()

        if cap.isOpened():
            try:
                frame = cap.read()[1]

                if save:
                    out.write(frame)

                cv.imshow("rtsp stream", frame)
            except Exception as e:
                print(f"\n[ERROR]\n{traceback.format_exc()}")
                cap = cv.VideoCapture(rtsp_url)
                time.sleep(1)
        else:
            print(f"[INFO] can't open <{rtsp_url=}>")

        frames -= 1
        if frames < 1:
            break

        latency = (time.time() - time_loc) * 1000
        sleep = max(1, delay - latency)
        key = cv.waitKey(sleep)
        if key == ord("q"):
            break

    cv.destroyAllWindows()

    if save:
        out.release()

    cap.release()
    return 0


def func(mode="pull", **kwargs):
    if mode == "pull":
        rtsp_pull(**kwargs)
    else:
        rtsp_push(**kwargs)

    return "\n[END]"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("mode", type=str,
                        choices=["pull", "push"])
    parser.add_argument("-i", dest="rtsp_url", type=str,
                        help="pull/push rtsp stream url")
    parser.add_argument("-height", dest="height", type=int, default=1080,
                        help="set video properties: height")
    parser.add_argument("-width", dest="width", type=int, default=1920,
                        help="set video properties: width")
    parser.add_argument("-fps", dest="fps", type=int, default=30,
                        help="set video properties: fps")
    parser.add_argument("-l", dest="frames", type=int, default=900,
                        help="set max frames")
    parser.add_argument("-save", dest="save", action="store_true",
                        help="save video")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
