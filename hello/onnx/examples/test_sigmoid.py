import torch
import torch.nn as nn
import torch.nn.init as init
from hello.onnx.export import export_onnx
import numpy as np


def fast_sigmoid(x):
    return (x * 0.08333333 + 0.5).clamp(0., 1.)


def fast_sigmoid_2x(x):
    return (x * 0.16666666 + 1.0).clamp(0., 2.)


class Net(nn.Module):
    def __init__(self, fmsize):
        super().__init__()
        self.fmsize = fmsize

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 85, (5, 5), (1, 1), (2, 2), bias=False)

        self.grid = torch.ones(1, 4, *fmsize)
        self.bn = nn.BatchNorm2d(4)

        self.bn.running_mean.fill_(-6.)
        self.bn.running_var.fill_(12.**2)
        self.bn.eval()

    def forward(self, x):
        # x(1,85,80,80)
        x = self.relu(self.conv1(x))

        xywh = x[:, :4].sigmoid() + self.grid[:, :4]
        x = torch.cat((xywh, x[:, 4:]), 1)

        return x

    def forward_fast(self, x):
        xywh = x[:, :4]
        xywh = fast_sigmoid(xywh) * 2.
        x = torch.cat((xywh, x[:, 4:]), 1)

        return x

    def forward_fast_2x(self, x):
        xywh = x[:, :4]
        xywh = fast_sigmoid_2x(xywh)
        x = torch.cat((xywh, x[:, 4:]), 1)

        return x

    def forward_bn(self, x):
        xywh = x[:, :4].clamp(-6., 6.)
        xywh = self.bn(xywh) * 2.
        x = torch.cat((xywh, x[:, 4:]), 1)
        return x


fmsize = (80, 80)
model = Net(fmsize)
x = torch.randn(1, 85, *fmsize)

with torch.no_grad():
    model.eval()
    y1 = model.forward_fast(x.clone())
    y2 = model.forward_fast_2x(x.clone())
    y3 = model.forward_bn(x.clone())

np.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-05, atol=1e-7)
np.testing.assert_allclose(y1.numpy(), y3.numpy(), rtol=1e-05, atol=1e-7)


fmsize = (16, 16)
model = Net(fmsize)
x = torch.randn(1, 3, *fmsize)
print(export_onnx(model, x, "test_16x16.onnx", device="cpu"))
