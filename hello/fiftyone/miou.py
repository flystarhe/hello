import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, class_names):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.mat = None

    def update(self, target, output):
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
            output = torch.from_numpy(output)

        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.inference_mode():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)  # diag / sum(gt_label)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}").format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )


def testit():
    num_classes = 3

    class_names = [f"c{i:02d}" for i in range(num_classes)]

    confmat = ConfusionMatrix(class_names)

    for _ in range(2):
        target = torch.randint(0, num_classes + 1, (3, 3))
        output = torch.randint(0, num_classes, (3, 3))
        confmat.update(target, output)

        print("*" * 32)
        print(target)
        print(output)

    print(confmat)
