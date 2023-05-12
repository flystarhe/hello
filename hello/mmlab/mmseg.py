# mmseg1.x
# pip install openmim
# mim install mmengine
# mim install mmcv==2.0.0
# mim install mmseg==1.0.0
import shutil
import sys
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

from mmseg.apis import inference_model, init_model

suffix_set = set(".avi,.mp4,.MOV,.mkv".split(","))
# ignore warnings when segmentors inference
warnings.filterwarnings("ignore")


def _draw_sem_seg(image, sem_seg, classes, palette):
    num_classes = len(classes)

    sem_seg = sem_seg.cpu().data
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < num_classes
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    colors = [palette[label] for label in labels]

    for label, color in zip(labels, colors):
        pass


def test_image(model, classes, palette, img, out_name, out_dir, add_zero_label=False):
    """Inference image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str/ndarray): Image file or loaded image.
        out_name (str): The file name to save.
        out_dir (pathlib.Path): The directory to save.
        add_zero_label (bool, optional): Defaults to False.
    """
    if isinstance(img, str):
        img = cv.imread(img, flags=cv.IMREAD_COLOR)

    result = inference_model(model, img)

    pred_img_data = img.copy()
    _draw_sem_seg(pred_img_data, result.pred_sem_seg, classes, palette)

    mask = model.show_result(img, result, palette=palette, show=False, out_file=None, opacity=0.5)
    mask_pure = model.show_result(img, result, palette=palette, show=False, out_file=None, opacity=1.0)

    out_file = str(out_dir / "data" / f"{out_name}.jpg")
    cv.imwrite(out_file, img)

    out_file = str(out_dir / "results" / f"{out_name}.jpg")
    cv.imwrite(out_file, np.concatenate((img, mask, mask_pure), axis=0))

    out_file = str(out_dir / "predictions" / f"{out_name}.png")
    if add_zero_label:
        result = [x + 1 for x in result]
    cv.imwrite(out_file, result[0].clip(min=0, max=255).astype("uint8"))


def test_images(model, image_paths, out_dir, add_zero_label=False):
    """Inference images.

    Args:
        model (nn.Module): The loaded segmentor.
        image_paths (list[str]): Image files to inference.
        out_dir (pathlib.Path): The directory to save.
        add_zero_label (bool, optional): Defaults to False.
    """
    if hasattr(model, "module"):
        classes = model.module.dataset_meta["classes"]
        palette = model.module.dataset_meta["palette"]
    else:
        classes = model.dataset_meta["classes"]
        palette = model.dataset_meta["palette"]

    for image_path in tqdm(image_paths):
        out_name = Path(image_path).stem
        test_image(model, classes, palette, image_path, out_name, out_dir, add_zero_label)


def test_videos(model, video_paths, out_dir, add_zero_label=False):
    """Inference videos.

    Args:
        model (nn.Module): The loaded segmentor.
        video_paths (list[str]): Video files to inference.
        out_dir (pathlib.Path): The directory to save.
        add_zero_label (bool, optional): Defaults to False.
    """
    if hasattr(model, "module"):
        classes = model.module.dataset_meta["classes"]
        palette = model.module.dataset_meta["palette"]
    else:
        classes = model.dataset_meta["classes"]
        palette = model.dataset_meta["palette"]

    print(f"[TODO] in development ..")
    for video_path in tqdm(video_paths):
        pass


def func(root, config_file, checkpoint_file, cfg_options, testdata, out_dir, add_zero_label=False):
    """Inference test data.

    Args:
        root (_type_): _description_
        config_file (_type_): _description_
        checkpoint_file (_type_): _description_
        cfg_options (_type_): _description_
        testdata (_type_): _description_
        out_dir (_type_): _description_
        add_zero_label (bool, optional): Defaults to False.
    """
    root = Path(root)
    config_file = str(root / config_file)
    checkpoint_file = str(root / checkpoint_file)
    model = init_model(config_file, checkpoint_file, cfg_options=cfg_options)

    testdata = Path(testdata)
    assert testdata.is_file() or testdata.is_dir()

    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=False)
    (out_dir / "results").mkdir(parents=True, exist_ok=False)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=False)

    if testdata.is_file():
        testdata = [testdata]
    else:
        testdata = sorted(testdata.glob("**/*"))

    image_paths = [str(f) for f in testdata if f.suffix == ".jpg"]
    video_paths = [str(f) for f in testdata if f.suffix in suffix_set]

    if image_paths:
        test_images(model, image_paths, out_dir, add_zero_label)

    if video_paths:
        test_videos(model, video_paths, out_dir, add_zero_label)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("root", type=str,
                        help="the base dir")
    parser.add_argument("config_file", type=str,
                        help="config file path")
    parser.add_argument("checkpoint_file", type=str,
                        help="checkpoint file path")
    parser.add_argument("testdata", type=str,
                        help="image/video(s) path or dir")
    parser.add_argument("-o", dest="out_dir", type=str,
                        help="save results")
    parser.add_argument("-y", dest="add_zero_label", action="store_true",
                        help="add zero label as background")
    parser.add_argument("-e", dest="cfg_options", type=str, default=None,
                        help="to override some settings, string of a python dict")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    cfg_options = kwargs["cfg_options"]
    if cfg_options is not None:
        kwargs["cfg_options"] = eval(cfg_options)

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
