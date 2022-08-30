import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv2 = nn.Conv2d(64, 8, (5, 5), (1, 1), (2, 2), bias=False)

        init.ones_(self.conv1.weight)
        init.ones_(self.conv2.weight)

    def forward(self, x):
        y = self.relu(self.conv1(x))

        y = self.conv2(y)
        # set data layout: channel last
        y = y.permute(0, 2, 3, 1).contiguous()
        # Currently observed, it must be the end of convolution + permute

        return y


def select_device(device=""):
    device = str(device).strip().lower().replace("cuda:", "")
    cpu = device == "cpu"

    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device

    cuda = not cpu and torch.cuda.is_available()
    return torch.device("cuda:0" if cuda else "cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def export_onnx(model, x, f, opset=11, device="cpu", simplify=False, rtol=1e-3, atol=1e-5):
    # x = torch.zeros(1, 3, 640, 640)
    # x = torch.randn(1, 3, 224, 224)
    device = select_device(device)
    model = model.to(device)
    x = x.to(device)

    model.eval()

    for _ in range(2):
        y = model(x)  # dry runs

    f = str(Path(f).with_suffix(".onnx"))

    torch.onnx.export(model, x, f,
                      verbose=False,
                      opset_version=opset,
                      input_names=["images"],
                      output_names=["output"])

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    if simplify:
        try:
            import onnxsim
            # pip install onnx-simplifier
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            print(f"simplifier failure: {e}")
    print(f"export success, saved as {f}")

    torch_outs = model(x)
    if isinstance(torch_outs, torch.Tensor):
        torch_outs = [torch_outs]

    cuda = torch.cuda.is_available()
    providers = ["CUDAExecutionProvider",
                 "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(f, providers=providers)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_outs[0]), ort_outs[0],
                               rtol=rtol, atol=atol)
    print("Exported model has been tested.")
    return f


if __name__ == "__main__":
    model = Net()
    x = torch.randn(1, 3, 224, 224)
    export_onnx(model, x, "test.onnx", device="cuda:0")
