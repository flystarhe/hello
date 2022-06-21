import cv2 as cv
import numpy as np


def gen_bgr(h, w, input_dtype="uint8"):
    bgr = np.zeros((h, w, 3), dtype=input_dtype)
    return bgr


def gen_nv12(h, w, input_dtype="uint8"):
    nv12 = np.random.randint(256, size=(int(h*1.5), w)).astype(input_dtype)
    return nv12.flatten()


def bgr_to_nv12(bgr):
    # nv12 shape (h*1.5,w)
    # YUV.NV12: Y(h*w)UV(u,v:h*w/4)
    # YUV.I420: Y(h*w)U(h*w/4)V(h*w/4)
    h, w = bgr.shape[:2]
    area = h * w

    yuv420p = cv.cvtColor(bgr, cv.COLOR_BGR2YUV_I420)
    yuv420p = yuv420p.reshape((area * 3 // 2,))

    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:area] = y
    nv12[area:] = uv_packed
    return nv12


def cam_to_bgr(origin_image, h=1080, w=1920):
    # YUV420SP(NV12): YYYYYYYY UVUV
    # x3m mipi: origin_image = cam.get_img(2, 1920, 1080)
    origin_nv12 = np.frombuffer(
        origin_image, dtype=np.uint8).reshape((int(h*1.5), w))
    origin_bgr = cv.cvtColor(origin_nv12, cv.COLOR_YUV420SP2BGR)
    return origin_bgr
