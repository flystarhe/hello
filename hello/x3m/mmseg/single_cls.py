import cv2 as cv
import numpy as np


def post_process(outputs, input_shape, infer_scale):
    output = outputs[0]  # (b, h, w, c)
    output = output[0]  # (h, w, num_classes)

    # ('background', 'charging station')
    seg_mask = (output[..., 1] > output[..., 0])

    assert seg_mask.shape == tuple(input_shape)

    return seg_mask[:infer_scale[1], :infer_scale[0]]


def pre_process(image, infer_scale, input_shape, to_rgb=True):
    """For single image inference.

    Examples::

        infer_scale = (960, 540)  # (w/2, h/2)
        input_shape = (544, 960)  # (h, w), divisible by 32
    """
    if isinstance(image, str):
        image = cv.imread(image, 1)  # bgr

    if to_rgb:
        cv.cvtColor(image, cv.COLOR_BGR2RGB, image)  # inplace

    image = cv.resize(image, infer_scale)  # (h, w, c)

    pad_image = np.full(input_shape + (3,), (114, 114, 114), dtype="uint8")
    pad_image[:infer_scale[1], :infer_scale[0], :] = image

    image_data = pad_image[np.newaxis, ...]  # (1, h, w, c)
    return image_data


def show_mask(image, infer_scale, mask):
    from IPython.display import display
    from PIL import Image
    if isinstance(image, str):
        image = cv.imread(image, 1)  # bgr

    image = cv.resize(image, infer_scale)  # (h, w, c)

    bgr_mask = np.zeros(mask.shape + (3,), dtype="uint8")
    bgr_mask[mask] = (0, 0, 255)

    mixed = cv.addWeighted(image, 0.5, bgr_mask, 0.5, 0)

    bgr_image = np.concatenate((image, mixed, bgr_mask), axis=0)
    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    display(Image.fromarray(rgb_image, "RGB"))
