import cv2 as cv
import numpy as np


def gen_bgr(h, w, input_dtype="uint8"):
    bgr = np.zeros((h, w, 3), dtype=input_dtype)
    return bgr


def gen_nv12(h, w, input_dtype="uint8"):
    nv12 = np.random.randint(256, size=(int(h*1.5), w)).astype(input_dtype)
    return nv12.flatten()


def bgr_to_nv12(bgr):
    # NV12 shape as (h*1.5,w)
    # YUV_NV12: Y(y:h*w)UV(u,v:h*w/4)
    # YUV_I420: Y(y:h*w)U(u:h*w/4)V(v:h*w/4)
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
    return nv12.reshape((-1, w))


def bgr_to_nv21(bgr):
    # NV21 shape as (h*1.5,w)
    # YUV_NV21: Y(y:h*w)VU(v,u:h*w/4)
    # YUV_YV12: Y(y:h*w)V(v:h*w/4)U(u:h*w/4)
    h, w = bgr.shape[:2]
    area = h * w

    yuv420p = cv.cvtColor(bgr, cv.COLOR_BGR2YUV_YV12)
    yuv420p = yuv420p.reshape((area * 3 // 2,))

    y = yuv420p[:area]
    vu_planar = yuv420p[area:].reshape((2, area // 4))
    vu_packed = vu_planar.transpose((1, 0)).reshape((area // 2,))

    nv21 = np.zeros_like(yuv420p)
    nv21[:area] = y
    nv21[area:] = vu_packed
    return nv21.reshape((-1, w))


def nv12_to_bgr(nv12, width=None):
    if nv12.ndim == 1:
        assert width is not None
        nv12 = nv12.reshape((-1, width))

    assert nv12.ndim == 2, "shape as (height*1.5, width)"

    bgr = cv.cvtColor(nv12, cv.COLOR_YUV2BGR_NV12)
    return bgr


def nv21_to_bgr(nv21, width=None):
    if nv21.ndim == 1:
        assert width is not None
        nv21 = nv21.reshape((-1, width))

    assert nv21.ndim == 2, "shape as (height*1.5, width)"

    bgr = cv.cvtColor(nv21, cv.COLOR_YUV2BGR_NV21)
    return bgr
