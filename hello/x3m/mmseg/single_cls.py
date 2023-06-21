import cv2 as cv
import numpy as np


def post_process(outputs, input_shape, infer_scale):
    output = outputs[0]  # (b, h, w, c)
    output = output[0]  # (h, w, num_classes)

    # classes: ('background', 'charging station')
    seg_pred = np.argmax(output, axis=-1)
    seg_mask = (seg_pred == 1)

    assert seg_mask.shape == tuple(input_shape)

    return seg_mask[:infer_scale[1], :infer_scale[0]]


def pre_process(image, infer_scale, input_shape, mode="bgr", layout="HWC"):
    """For single image inference.

    Examples::

        infer_scale = (960, 540)  # (w/2, h/2)
        input_shape = (544, 960)  # (h, w), divisible by 32
    """
    if isinstance(image, str):
        image = cv.imread(image, 1)  # bgr

    image = cv.resize(image, infer_scale)  # (h, w, c)

    pad_image = np.full(input_shape + (3,), (114, 114, 114), dtype="uint8")
    pad_image[:infer_scale[1], :infer_scale[0], :] = image

    if mode == "rgb":
        cv.cvtColor(pad_image, cv.COLOR_BGR2RGB, pad_image)  # inplace
    elif mode == "nv12":
        from hello.x3m.transforms import bgr_to_nv12, nv12_to_yuv444
        data, target_size = bgr_to_nv12(pad_image)
        pad_image = nv12_to_yuv444(data, target_size, "HWC")

    if layout == "CHW":
        pad_image = np.transpose(pad_image, (2, 0, 1))

    image_data = pad_image[np.newaxis, ...]
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


def test_post_process():
    output = np.random.rand(1, 2, 8, 8) * 1.0
    ret = post_process([output], (8, 8), (8, 8))
    return ret


def test_notebook():
    """X3M Ai Toolchain Docker Container.

        docker pull openexplorer/ai_toolchain_centos_7_xj3:v2.4.2
        docker run -it --rm -p 7000:9000 --ipc=host -v $(pwd):/workspace openexplorer/ai_toolchain_centos_7_xj3:v2.4.2 bash
        pip install -U hello2 -i https://pypi.org/simple
        nohup jupyter notebook --ip='*' --port=9000 --notebook-dir='/workspace' --NotebookApp.token='hi' --no-browser --allow-root > /workspace/nohup.out 2>&1 &
        # localhost:7000/tree?token=hi
    """
    from horizon_tc_ui import HB_ONNXRuntime

    infer_scale = (960, 540)  # (w/2, h/2)
    input_shape = (544, 960)  # (h, w), divisible by 32
    image_file = "data/20230309_163529_i000246.jpg"
    model_file = "test/model_output/mmseg_544x960_quantized_model.onnx"

    sess = HB_ONNXRuntime(model_file=model_file)
    print(f"{sess.input_names}, {sess.output_names}, {sess.layout}")

    image_data = pre_process(image_file, infer_scale, input_shape, mode="nv12", layout="HWC")
    input_name, output_names = sess.input_names[0], sess.output_names
    outputs = sess.run(output_names, {input_name: image_data}, input_offset=128)
    mask = post_process(outputs, input_shape, infer_scale)

    show_mask(image_file, infer_scale, mask)
    return mask
