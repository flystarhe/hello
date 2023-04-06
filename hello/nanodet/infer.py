"""pip install nanodet"""
import shutil
import sys
from itertools import chain
from pathlib import Path

import cv2 as cv
import torch

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight


class Predictor:
    """Instantiate an predictor.

    >>> cfg_list = ["data.val.input_size", (w, h)]
    >>> cfg_list = ["data.val.keep_ratio", True]

    Args:
        cfg (CfgNode): The configuration tree.
        model_path (str): Checkpoint file path.
        logger (Logger): A logger for `load_model_weight()`.
        device (str, optional): Device used for calculating. Defaults to "cuda:0".
        cfg_list (list, optional): A list. For example, `cfg_list = ['FOO.BAR', 0.5]`.
    """

    def __init__(self, cfg, model_path, logger, device="cuda:0", cfg_list=None) -> None:
        if cfg_list is not None:
            cfg.defrost()
            cfg.merge_from_list(cfg_list)
            cfg.freeze()

        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        """Inference image data.

        Args:
            img (ndarray): A loaded image.
        """
        img_info = {"id": 0, "file_name": None}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=64)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def test_images(self, images_or_dir, score_thr=0.3, out_dir=None):
        """Inference images with the predictor.

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
        """Inference image with the predictor.

        Args:
            img (str): Image file.
        """
        image = cv.imread(image_path, 1)  # cv.IMREAD_COLOR

        dets = self.inference(image)[1][0]  # single

        class_names = self.cfg.class_names

        if out_file is not None:
            result_image = self.model.head.show_result(image, dets, class_names, score_thr, False)
            cv.imwrite(out_file, result_image)

        bbox_result = []
        for label, bbox_list in dets.items():
            label = class_names[label]
            for bbox in bbox_list:
                x1, y1, x2, y2, score = bbox
                bbox_result.append([x1, y1, x2, y2, score, label])
        bbox_result.sort(key=lambda v: v[4], reverse=True)

        height, width = image.shape[:2]
        return bbox_result, image_path, height, width

    def to_file(self, results, txt_file):
        """Format results.

        Such as ``filepath,height,width,x1,y1,x2,y2,s,l,x1,y1,x2,y2,s,l``.

        Args:
            results (list[tuple]): The results.
        """
        lines = []
        for bbox_result, image_path, height, width in results:
            line = [f"{image_path},{height},{width}"]
            for x1, y1, x2, y2, score, label in bbox_result:
                line.extend([f"{v:.2f}" for v in [x1, y1, x2, y2]])
                line.extend([f"{score:.4f}", label])
            lines.append(",".join(line))
        text = "\n".join(lines)

        Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(txt_file, "w") as f:
            f.write(text)


def func(root, config_file, checkpoint_file, cfg_options, images_dir, score_thr, out_dir, txt_file):
    root = Path(root)

    config_file = str(root / config_file)
    checkpoint_file = str(root / checkpoint_file)

    load_config(cfg, config_file)
    model_path = checkpoint_file  # .ckpt
    logger = Logger(0, save_dir="./", use_tensorboard=False)
    cfg_list = list(chain.from_iterable(cfg_options.items())) if cfg_options else None

    executor = Predictor(cfg, model_path, logger, cfg_list=cfg_list)
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
                        help="minimum score of bboxes to display")
    parser.add_argument("-o", dest="out_dir", type=str, default=None,
                        help="draw boxes on the image")
    parser.add_argument("-f", dest="txt_file", type=str, default="predictions.txt",
                        help="format and save results to a txt file")
    parser.add_argument("-e", dest="cfg_options", type=str, default=None,
                        help="to override some settings")

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
