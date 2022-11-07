"""pip install mmsegmentation"""
import shutil
import sys
import warnings
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

from mmseg.apis import inference_segmentor, init_segmentor

suffix_set = set(".avi,.mp4,.MOV,.mkv".split(","))
# ignore warnings when segmentors inference
warnings.filterwarnings('ignore')


def test_image(model, img, palette, out_dir, out_name, add_zero_label=False):
    """Inference image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str/ndarray): Image file or loaded image.
        palette (list[list[int]]): The palette of segmentation map.
        out_dir (pathlib.Path): The directory to save.
        out_name (str): The file name to save.
        add_zero_label (bool, optional): Defaults to False.
    """
    if isinstance(img, str):
        img = cv.imread(img, flags=cv.IMREAD_COLOR)

    result = inference_segmentor(model, img)
    mask = model.show_result(img, result, palette=palette, show=False, out_file=None, opacity=0.9)

    out_file = str(out_dir / "data" / f"{out_name}.jpg")
    cv.imwrite(out_file, img)

    out_file = str(out_dir / "results" / f"{out_name}.jpg")
    cv.imwrite(out_file, np.concatenate((img, mask), axis=0))

    out_file = str(out_dir / "predictions" / f"{out_name}.png")
    if add_zero_label:
        result = [x + 1 for x in result]
    cv.imwrite(out_file, result[0].clip(min=0, max=255).astype("uint8"))


def test_images(model, image_paths, palette, out_dir, add_zero_label=False):
    """Inference images.

    Args:
        model (nn.Module): The loaded segmentor.
        image_paths (list[str]): Image files to inference.
        palette (list[list[int]]): The palette of segmentation map.
        out_dir (pathlib.Path): The directory to save.
        add_zero_label (bool, optional): Defaults to False.
    """
    for img in tqdm(image_paths):
        out_name = f"{Path(img).stem}"
        test_image(model, img, palette, out_dir, out_name, add_zero_label)


def test_video(model, video_path, palette, out_dir, add_zero_label=False):
    """Inference video.

    Args:
        model (nn.Module): The loaded segmentor.
        video_path (str): Video file to inference.
        palette (list[list[int]]): The palette of segmentation map.
        out_dir (pathlib.Path): The directory to save.
        add_zero_label (bool, optional): Defaults to False.
    """
    cap = cv.VideoCapture(video_path)
    cap_fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    prefix = Path(video_path).stem

    for index in tqdm(range(frame_count)):
        ret, img = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if (index % cap_fps) != 0:
            continue

        out_name = f"{prefix}_{index:06d}"
        test_image(model, img, palette, out_dir, out_name, add_zero_label)

    cap.release()


def test_videos(model, video_paths, palette, out_dir, add_zero_label=False):
    """Inference videos.

    Args:
        model (nn.Module): The loaded segmentor.
        video_paths (list[str]): Video files to inference.
        palette (list[list[int]]): The palette of segmentation map.
        out_dir (pathlib.Path): The directory to save.
        add_zero_label (bool, optional): Defaults to False.
    """
    for video_path in video_paths:
        test_video(model, video_path, palette, out_dir, add_zero_label)


def func(root, config_file, checkpoint_file, testdata, out_dir, add_zero_label=False):
    """Inference test data.

    Args:
        root (_type_): _description_
        config_file (_type_): _description_
        checkpoint_file (_type_): _description_
        testdata (_type_): _description_
        out_dir (_type_): _description_
        add_zero_label (bool, optional): Defaults to False.
    """
    root = Path(root)
    config_file = str(root / config_file)
    checkpoint_file = str(root / checkpoint_file)
    model = init_segmentor(config_file, checkpoint_file, device="cuda:0")

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
        testdata = list(testdata.glob("**/*"))

    image_paths = [str(f) for f in testdata if f.suffix == ".jpg"]
    video_paths = [str(f) for f in testdata if f.suffix in suffix_set]

    if image_paths:
        test_images(model, image_paths, model.PALETTE, out_dir, add_zero_label)

    if video_paths:
        test_videos(model, video_paths, model.PALETTE, out_dir, add_zero_label)


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

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
