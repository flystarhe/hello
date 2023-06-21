import cv2 as cv
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x)
    sum = np.sum(e_x)
    f_x = e_x / sum
    return f_x


def distribution_project(reg_pred, reg_max=7):
    project = np.linspace(0, reg_max, reg_max + 1, dtype=np.float32)
    return np.sum(softmax(reg_pred) * project)


def get_single_level_post_process(output, input_shape, reg_max=7):
    # TODO regress_ranges is `{stride: (stride * 4, stride * 10) for stride in strides}`
    # or set regress_ranges: `{8: (32, 80), 16: (64, 160), 32: (128, 320), 64: (256, 10000)}`
    # `charging station`
    cls_id = 0

    # single level: (b, h, w, c)
    output = output[0]  # (h, w, num_classes+32)

    mat = output[:, :, cls_id]
    i, j = np.unravel_index(np.argmax(mat), mat.shape)

    score = sigmoid(output[i, j, :-32])
    offset_left = distribution_project(output[i, j, -32:-24], reg_max)
    offset_top = distribution_project(output[i, j, -24:-16], reg_max)
    offset_right = distribution_project(output[i, j, -16:-8], reg_max)
    offset_bottom = distribution_project(output[i, j, -8:], reg_max)

    stride = input_shape[0] // output.shape[0]
    cy, cx = i * stride, j * stride

    x1 = np.clip(cx - offset_left * stride, 0, input_shape[1])
    y1 = np.clip(cy - offset_top * stride, 0, input_shape[0])
    x2 = np.clip(cx + offset_right * stride, 0, input_shape[1])
    y2 = np.clip(cy + offset_bottom * stride, 0, input_shape[0])

    bbox = [x1, y1, x2, y2]
    return bbox, score, cls_id


def post_process(outputs, input_shape, reg_max=7):
    # multi-level: outputs is a list
    bbox, score, cls_id = None, None, None
    for output in outputs:
        _bbox, _score, _cls_id = get_single_level_post_process(output, input_shape, reg_max)
        if bbox is None:
            bbox, score, cls_id = _bbox, _score, _cls_id
        elif _score[_cls_id] > score[_cls_id]:
            bbox, score, cls_id = _bbox, _score, _cls_id
    return bbox, score, cls_id


def pre_process(image, infer_scale, input_shape, mode="bgr", layout="HWC"):
    """For single image inference.

    Examples::

        infer_scale = (640, 360)  # (w/3, h/3)
        input_shape = (384, 640)  # (h, w), divisible by 64
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


def show_bbox(bgr_image, infer_scale, bboxes, scores, cls_ids, cls_names=None):
    from IPython.display import display
    from PIL import Image
    if isinstance(bgr_image, str):
        bgr_image = cv.imread(bgr_image, 1)

    f = 1.0
    if bgr_image.shape[0] != infer_scale[1]:
        f = bgr_image.shape[0] / infer_scale[1]  # scale(w, h)

    for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
        cls_name = cls_names[cls_id] if cls_names else f"id:{cls_id}"
        x1, y1, x2, y2 = [int(v * f) for v in bbox]
        cv.rectangle(bgr_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv.putText(bgr_image, f"{score[cls_id]}/{cls_name}", (x1+5, y1+30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
    display(Image.fromarray(rgb_image, "RGB"))


def test_post_process():
    output = np.random.rand(1, 3, 5, 34) * 1.0
    ret = post_process([output], (12, 20))
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

    infer_scale = (640, 360)  # (w/3, h/3)
    input_shape = (384, 640)  # (h, w), divisible by 64
    image_file = "data/20230309_163529_i000246.jpg"
    model_file = "test/model_output/nanodet_384x640_x3_quantized_model.onnx"

    sess = HB_ONNXRuntime(model_file=model_file)
    print(f"{sess.input_names}, {sess.output_names}, {sess.layout}")

    image_data = pre_process(image_file, infer_scale, input_shape, mode="nv12", layout="HWC")
    input_name, output_names = sess.input_names[0], sess.output_names
    outputs = sess.run(output_names, {input_name: image_data}, input_offset=128)
    bbox, score, cls_id = post_process(outputs, input_shape, reg_max=7)

    cls_names = ["charging station", "other object"]
    show_bbox(image_file, infer_scale, [bbox], [score], [cls_id], cls_names)
    return bbox, score, cls_id
