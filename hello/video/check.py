import cv2 as cv
from moviepy import editor as mpe
import numpy as np


def check(a, b, rtol=1e-3, atol=1e-5):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def get_image(video_path, n):
    cap = cv.VideoCapture(video_path)
    print(cap.get(cv.CAP_PROP_POS_FRAMES))
    cap.set(cv.CAP_PROP_POS_FRAMES, n)
    print(cap.get(cv.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    cap.release()
    return ret, frame


def clip_video(infile, t_start, t_end, outfile, codec=None):
    clip = mpe.VideoFileClip(infile, audio=False).subclip(t_start, t_end)
    clip.write_videofile(outfile, codec=codec)


def get_clip():
    pass
