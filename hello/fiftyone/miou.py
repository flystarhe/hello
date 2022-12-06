import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from hello.utils import importer
from prettytable import PrettyTable


class ConfusionMatrix:
    """For segmentation metrics.

    Args:
        class_names (list[str]): the list of class label strings
        reduce_zero_label (bool, optional): defaults to True
    """

    def __init__(self, class_names, reduce_zero_label=True):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reduce_zero_label = reduce_zero_label
        self.mat = None

    def update(self, target, output):
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
            output = torch.from_numpy(output)

        if self.reduce_zero_label:
            target[target == 0] = 255
            target = target - 1
            target[target == 254] = 255

        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.inference_mode():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    @property
    def confusion_matrix(self):
        mat = self.mat.float()

        mat = mat / mat.sum(dim=1, keepdim=True)
        mat = mat - torch.diag_embed(mat.diag())

        top_values, top_indices = mat.topk(2, dim=1)

        n = self.num_classes
        table_data = PrettyTable()
        table_data.add_column("class_name", self.class_names)
        table_data.add_column("index", [f"{v:02d}" for v in range(n)])
        for i in range(top_values.size(1)):
            table_data.add_column(f"top{i+1}_ratio", [f"{v:.2%}" for v in top_values[:, i]], align="r")
            table_data.add_column(f"top{i+1}_class", [self.class_names[v.item()] for v in top_indices[:, i]], align="r")
        return table_data

    @property
    def metrics(self):
        h = self.mat.float()
        precision = torch.diag(h) / h.sum(0) * 100
        recall = torch.diag(h) / h.sum(1) * 100  # diag / sum(gt_label)
        iou = torch.diag(h) / (h.sum(0) + h.sum(1) - torch.diag(h)) * 100
        support = h.sum(1)

        n = self.num_classes
        table_data = PrettyTable()
        table_data.add_column("class_name", self.class_names)
        table_data.add_column("index", [f"{v:02d}" for v in range(n)])
        table_data.add_column("precision", [f"{v:.2f}" for v in precision.tolist()], align="r")
        table_data.add_column("recall", [f"{v:.2f}" for v in recall.tolist()], align="r")
        table_data.add_column("iou", [f"{v:.2f}" for v in iou.tolist()], align="r")
        table_data.add_column("support", [int(v) for v in support.tolist()], align="r")
        table_data.add_column("support_ratio", [f"{v:.2%}" for v in (support / support.sum()).tolist()], align="r")

        table_data.add_row(["macro avg", "-"] + [f"{v.nanmean().item():.2f}" for v in [precision, recall, iou]] + ["-", "-"])

        w = support / support.sum()
        precision, recall, iou = precision * w, recall * w, iou * w
        table_data.add_row(["weighted avg", "-"] + [f"{v.nansum().item():.2f}" for v in [precision, recall, iou]] + ["-", "-"])
        return table_data


def func(true_dir, pred_dir, num_classes, class_names, reduce_zero_label=True):
    true_files = {f.stem: str(f) for f in Path(true_dir).glob("*.png")}
    pred_files = {f.stem: str(f) for f in Path(pred_dir).glob("*.png")}
    common_stems = sorted(set(true_files.keys()) & set(pred_files.keys()))

    print(f"[INFO] number of true files: {len(true_files)}")
    print(f"[INFO] number of pred files: {len(pred_files)}")
    print(f"[INFO] number of common files: {len(common_stems)}")

    assert (num_classes is not None) or (class_names is not None)

    if class_names is None:
        info_py = Path(true_dir).with_name("info.py")
        if info_py.is_file():
            info = importer.load_from_file("info_py", info_py).info
            class_names = info["classes"][:num_classes]

    if class_names is None:
        class_names = [f"c{i:03d}" for i in range(num_classes)]

    confmat = ConfusionMatrix(class_names, reduce_zero_label=reduce_zero_label)
    for stem in common_stems:
        target = cv.imread(true_files[stem], 0)
        output = cv.imread(pred_files[stem], 0)
        confmat.update(target, output)

    print(confmat.metrics.get_string())
    print(confmat.confusion_matrix.get_string())


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("true_dir", type=str,
                        help="the ground truth dir")
    parser.add_argument("pred_dir", type=str,
                        help="the prediction dir")
    parser.add_argument("-n", dest="num_classes", type=int,
                        help="[0, num_classes) to calculate metric")
    parser.add_argument("-c", dest="class_names", type=str, nargs='+',
                        help="if is None, will be generated [c0 ... cn-1]")
    parser.add_argument("-y", dest="reduce_zero_label", action="store_true",
                        help="whether to target zero as ignored")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
