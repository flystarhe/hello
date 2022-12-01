"""pip install mmdet"""
import shutil
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


class Detector:
    """Instantiate an object detector.

    Args:
        config_file (str): Config file path.
        checkpoint_file (str): Checkpoint file path.
        device (str, optional): Device used for calculating. Defaults to "cuda:0".
        cfg_options (dict, optional): Options to override some settings. Defaults to None.
    """

    def __init__(self, config_file, checkpoint_file, device="cuda:0", cfg_options=None) -> None:
        self.model = init_detector(config_file, checkpoint_file, device, cfg_options)

    def test_images(self, images_or_dir, score_thr=0.3, out_dir=None):
        """Inference images with the detector.

        Args:
            images_or_dir (str or list[str]): Image files.
        """
        if isinstance(images_or_dir, str):
            image_paths = [str(f) for f in Path(images_or_dir).glob("*.jpg")]
        else:
            image_paths = [str(f) for f in images_or_dir]

        assert isinstance(image_paths, list)

        if out_dir is not None:
            out_dir = Path(out_dir)
            shutil.rmtree(out_dir, ignore_errors=True)
            (out_dir / "images").mkdir(parents=True, exist_ok=False)

        results = []
        for image_path in image_paths:
            out_file = None

            if out_dir is not None:
                out_file = str(out_dir / "images" / Path(image_path).name)

            result = self.test_image(image_path, score_thr=score_thr, out_file=out_file)
            results.append(result)

        return results

    def test_image(self, image_path, score_thr=0.3, out_file=None):
        """Inference image with the detector.

        Args:
            img (str): Image file.
        """
        image = cv.imread(image_path, 1)  # cv.IMREAD_COLOR

        bbox_result = inference_detector(self.model, image)

        if out_file is not None:
            show_result_pyplot(self.model, image, bbox_result, score_thr=score_thr, out_file=out_file)

        height, width = image.shape[:2]
        return bbox_result, image_path, height, width

    def to_file(self, results, txt_file):
        """Format results.

        Such as ``filepath,height,width,x1,y1,x2,y2,s,l,x1,y1,x2,y2,s,l``.

        Args:
            results (list[tuple]): The results.
        """
        class_names = self.model.CLASSES

        lines = []
        for bbox_result, image_path, height, width in results:
            bboxes = np.vstack(bbox_result)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            line = [f"{image_path},{height},{width}"]
            for bbox, label in zip(bboxes, labels):
                line.extend([f"{v:.2f}" for v in bbox[:4]])
                line.extend([f"{bbox[4]:.4f}", class_names[label]])
            lines.append(",".join(line))
        text = "\n".join(lines)

        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(txt_file, "w") as f:
            f.write(text)


def func(root, config_file, checkpoint_file, images_dir, score_thr, out_dir, txt_file):
    root = Path(root)

    config_file = str(root / config_file)
    checkpoint_file = str(root / checkpoint_file)

    executor = Detector(config_file, checkpoint_file)
    results = executor.test_images(images_dir, score_thr, out_dir)

    if txt_file is not None:
        if out_dir is not None:
            txt_file = Path(out_dir) / txt_file
        executor.to_file(results, txt_file)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("root", type=str,
                        help="the base dir")
    parser.add_argument("config_file", type=str,
                        help="config file path")
    parser.add_argument("checkpoint_file", type=str,
                        help="checkpoint file path")
    parser.add_argument("images_dir", type=str,
                        help="a directory of images")
    parser.add_argument("-s", dest="score_thr", type=float, default=0.05,
                        help="minimum score of bboxes to be shown")
    parser.add_argument("-o", dest="out_dir", type=str, default=None,
                        help="draw boxes on the image")
    parser.add_argument("-f", dest="txt_file", type=str, default="predictions.txt",
                        help="format and save results to a txt file")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
