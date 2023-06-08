import numpy as np


def distance2bbox(points, distance, max_shape=None):
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e_x = np.exp(x)
    sum = np.sum(e_x, axis=-1, keepdims=True)
    f_x = e_x / sum
    return f_x


def distribution_project(reg_preds, reg_max=7):
    project = np.linspace(0, reg_max, reg_max + 1, dtype=np.float32)
    reg_preds = softmax(reg_preds.reshape((-1, 4, reg_max + 1)))
    reg_preds = reg_preds * project[None, None, ...]
    reg_preds = np.sum(reg_preds, axis=-1)
    return reg_preds


def get_single_level_center_priors(featmap_size, stride):
    h, w = featmap_size
    y_range = np.arange(h, dtype=np.float32) * stride
    x_range = np.arange(w, dtype=np.float32) * stride
    y, x = np.meshgrid(y_range, x_range)
    y = y.flatten()
    x = x.flatten()
    strides = np.full((x.shape[0],), stride, dtype=np.float32)
    priors = np.stack([x, y, strides], axis=-1)
    return priors


def get_single_level_bboxes(cls_preds, reg_preds, input_shape, reg_max=7):
    h, w, _ = cls_preds.shape
    stride = input_shape[0] // h

    center_priors = get_single_level_center_priors((h, w), stride)
    dis_preds = distribution_project(reg_preds, reg_max) * center_priors[:, 2, None]
    bboxes = distance2bbox(center_priors[..., :2], dis_preds, input_shape)
    scores = sigmoid(cls_preds).reshape((h * w, -1))
    return bboxes, scores


def get_single_level_post_process(output, input_shape, reg_max=7):
    # single level: (b, h, w, c)
    output = output[0]  # (h, w, num_classes+32)
    cls_scores, bbox_preds = np.split(output, [-32], axis=-1)
    bboxes, scores = get_single_level_bboxes(cls_scores, bbox_preds, input_shape, reg_max)
    idx, cls_id = np.argmax(scores[:, 0]), 0  # first class is `charging station`
    return bboxes[idx], scores[idx], cls_id


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


def show_bbox(bgr_image, infer_scale, bboxes, scores, cls_ids, cls_names=None):
    import cv2 as cv
    from IPython.display import display
    from PIL import Image
    if isinstance(bgr_image, str):
        bgr_image = cv.imread(bgr_image, 1)

    f = 1.0
    if bgr_image.shape[:2] != tuple(infer_scale):
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
        nohup jupyter notebook --ip='*' --port=9000 --notebook-dir='/workspace' --NotebookApp.token='hi' --no-browser --allow-root > /workspace/nohup.out 2>&1 &
        # localhost:7000/tree?token=hi
    """
    import cv2 as cv
    from horizon_tc_ui import HB_ONNXRuntime

    infer_scale = (640, 360)  # (w/3, h/3)
    input_shape = (384, 640)  # divisible by 64
    image_file = "data/20230309_163529_i000246.jpg"
    model_file = "test/model_output/nanodet_384x640_x3_bgr_quantized_model.onnx"

    sess = HB_ONNXRuntime(model_file=model_file)
    print(f"{sess.input_names}, {sess.output_names}, {sess.layout}")

    image = cv.imread(image_file, 1)  # bgr
    image = cv.resize(image, infer_scale)  # (w, h, c)
    pad_image = np.full(input_shape + (3,), (114, 114, 114), dtype="uint8")
    pad_image[:360, :640, :] = image

    image_data = pad_image[np.newaxis, ...]  # (1, h, w, c)
    input_name, output_names = sess.input_names[0], sess.output_names
    outputs = sess.run(output_names, {input_name: image_data}, input_offset=128)
    bbox, score, cls_id = post_process(outputs, input_shape, reg_max=7)

    cls_names = ["charging station", "other object"]
    show_bbox(image_file, infer_scale, [bbox], [score], [cls_id], cls_names)

    return bbox, score, cls_id
