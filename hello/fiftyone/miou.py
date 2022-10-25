import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from prettytable import PrettyTable


class ConfusionMatrix:
    def __init__(self, class_names, reduce_zero_label=True):
        """For segmentation metrics.

        Args:
            class_names (list[str]): _description_
            reduce_zero_label (bool, optional): _description_. Defaults to True.
        """
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
    def pretty_text(self):
        h = self.mat.float()
        precision = torch.diag(h) / h.sum(0) * 100
        recall = torch.diag(h) / h.sum(1) * 100  # diag / sum(gt_label)
        iou = torch.diag(h) / (h.sum(0) + h.sum(1) - torch.diag(h)) * 100
        support = h.sum(1)

        table_data = PrettyTable()
        table_data.add_column("class name", self.class_names)
        table_data.add_column("precision", [f"{i:.2f}" for i in precision.tolist()])
        table_data.add_column("recall", [f"{i:.2f}" for i in recall.tolist()])
        table_data.add_column("iou", [f"{i:.2f}" for i in iou.tolist()])
        table_data.add_column("support", [int(i) for i in support.tolist()])

        table_data.add_row(["macro avg"] + [f"{i.mean().item():.2f}" for i in [precision, recall, iou]] + ["-"])

        w = support / support.sum()
        precision, recall, iou = precision * w, recall * w, iou * w
        table_data.add_row(["weighted avg"] + [f"{i.sum().item():.2f}" for i in [precision, recall, iou]] + ["-"])
        return table_data.get_string()


def func(true_dir, pred_dir, num_classes, class_names, reduce_zero_label=True):
    true_files = {f.stem: str(f) for f in Path(true_dir).glob("*.png")}
    pred_files = {f.stem: str(f) for f in Path(pred_dir).glob("*.png")}
    common_stems = sorted(set(true_files.keys()) & set(pred_files.keys()))

    print(f"[INFO] number of true files: {len(true_files)}")
    print(f"[INFO] number of pred files: {len(pred_files)}")
    print(f"[INFO] number of common files: {len(common_stems)}")

    assert (num_classes is not None) or (class_names is not None)

    if class_names is None:
        class_names = [f"c{i:03d}" for i in range(num_classes)]

    confmat = ConfusionMatrix(class_names, reduce_zero_label=reduce_zero_label)
    for stem in common_stems:
        target = cv.imread(true_files[stem], 0)
        output = cv.imread(pred_files[stem], 0)
        confmat.update(target, output)

    return confmat.pretty_text


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
