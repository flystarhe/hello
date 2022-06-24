# ai_toolchain/horizon_model_convert_sample/01_common/python/data/transformer.py
import cv2 as cv
import numpy as np


class Transformer(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return self.run_transform(data)

    def run_transform(self, data):
        return data


class TransposeTransformer(Transformer):
    def __init__(self, order):
        self.order = order
        super(TransposeTransformer, self).__init__()

    def run_transform(self, data):
        data[0] = np.transpose(data[0], self.order)
        return data


class HWC2CHWTransformer(Transformer):
    def __init__(self):
        self.transformer = TransposeTransformer((2, 0, 1))

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class CHW2HWCTransformer(Transformer):
    def __init__(self):
        self.transformer = TransposeTransformer((1, 2, 0))

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class PadCropTransformer(Transformer):
    def __init__(self, target_size, pad_value=127.):
        self.rng = np.random.default_rng(41)
        self.target_size = target_size
        self.pad_value = pad_value
        super(PadCropTransformer, self).__init__()

    def run_transform(self, data):
        image = data[0]
        image_h, image_w, _ = image.shape
        target_h, target_w = self.target_size

        _top = self.rng.integers(1 + max(0, image_h - target_h))
        _left = self.rng.integers(1 + max(0, image_w - target_w))
        resize_image = image[_top: _top + target_h, _left: _left + target_w]
        new_h, new_w, _ = resize_image.shape

        pad_image = np.full(shape=(target_h, target_w, 3),
                            fill_value=self.pad_value).astype(image.dtype)
        pad_image[:new_h, :new_w, :] = resize_image

        data[0] = pad_image
        return data


class PadResizeTransformer(Transformer):
    def __init__(self, target_size, pad_value=127.):
        self.target_size = target_size
        self.pad_value = pad_value
        super(PadResizeTransformer, self).__init__()

    def run_transform(self, data):
        image = data[0]
        image_h, image_w, _ = image.shape
        target_h, target_w = self.target_size

        scale = min(target_w * 1.0 / image_w, target_h * 1.0 / image_h)
        new_h, new_w = int(scale * image_h), int(scale * image_w)
        resize_image = cv.resize(image, (new_w, new_h))

        pad_image = np.full(shape=(target_h, target_w, 3),
                            fill_value=self.pad_value).astype(image.dtype)
        pad_image[:new_h, :new_w, :] = resize_image

        data[0] = pad_image
        return data


class _ChannelSwapTransformer(Transformer):
    def __init__(self, order, channel_index=0):
        self.order = order
        self.channel_index = channel_index
        super(_ChannelSwapTransformer, self).__init__()

    def run_transform(self, data):
        image = data[0]
        assert self.channel_index < len(image.shape), \
            "channel index is larger than image.dims"
        assert image.shape[self.channel_index] == len(self.order), \
            "the length of swap order != the number of channel:{}!={}" \
            .format(len(self.order), image.shape[self.channel_index])
        if self.channel_index == 0:
            data[0] = image[self.order, :, :]
        elif self.channel_index == 1:
            data[0] = image[:, self.order, :]
        elif self.channel_index == 2:
            data[0] = image[:, :, self.order]
        else:
            raise ValueError(
                f"channel index: {self.channel_index} error in _ChannelSwapTransformer"
            )
        return data


class BGR2RGBTransformer(Transformer):
    def __init__(self, data_format="HWC"):
        if data_format == "CHW":
            self.transformer = _ChannelSwapTransformer((2, 1, 0))
        elif data_format == "HWC":
            self.transformer = _ChannelSwapTransformer((2, 1, 0), 2)
        else:
            raise ValueError(
                f"unsupported data_format: '{data_format}' in BGR2RGBTransformer"
            )

    def run_transform(self, data):
        return self.transformer.run_transform(data)


class BGR2NV12Transformer(Transformer):
    @staticmethod
    def mergeUV(u, v):
        if u.shape == v.shape:
            uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
            for i in range(0, u.shape[0]):
                for j in range(0, u.shape[1]):
                    uv[i, 2 * j] = u[i, j]
                    uv[i, 2 * j + 1] = v[i, j]
            return uv
        else:
            raise ValueError("size of Channel U is different with Channel V")

    def __init__(self, cvt_mode="bgr_calc"):
        self.cvt_mode = cvt_mode

    def bgr2nv12_calc(self, image):
        if image.ndim == 3:
            b = image[:, :, 0]
            g = image[:, :, 1]
            r = image[:, :, 2]
            y = (0.299 * r + 0.587 * g + 0.114 * b)
            u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
            v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
            uv = self.mergeUV(u, v)
            yuv = np.vstack((y, uv))
            return yuv.astype(np.uint8)
        else:
            raise ValueError("image is not BGR format")

    def bgr2nv12_opencv(self, image):
        if image.ndim == 3:
            image = image.astype(np.uint8)
            height, width = image.shape[0], image.shape[1]
            yuv420p = cv.cvtColor(image, cv.COLOR_BGR2YUV_I420).reshape(
                (height * width * 3 // 2,))
            y = yuv420p[:height * width]
            uv_planar = yuv420p[height * width:].reshape(
                (2, height * width // 4))
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2,))
            nv12 = np.zeros_like(yuv420p)
            nv12[:height * width] = y
            nv12[height * width:] = uv_packed
            return nv12
        else:
            raise ValueError("image is not BGR format")

    def run_transform(self, data):
        image = data[0]
        if self.cvt_mode == "opencv":
            data[0] = self.bgr2nv12_opencv(image)
        else:
            data[0] = self.bgr2nv12_calc(image)
        return data


class ToF32Transformer(Transformer):
    def __init__(self):
        super(ToF32Transformer, self).__init__()

    def run_transform(self, data):
        data[0] = data[0].astype(np.float32)
        return data


class F32ToS8Transformer(Transformer):
    def __init__(self):
        super(F32ToS8Transformer, self).__init__()

    def run_transform(self, data):
        data[0] = data[0].astype(np.int8)
        return data


class F32ToU8Transformer(Transformer):
    def __init__(self):
        super(F32ToU8Transformer, self).__init__()

    def run_transform(self, data):
        data[0] = data[0].astype(np.uint8)
        return data
