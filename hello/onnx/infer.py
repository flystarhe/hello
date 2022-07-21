import logging
from pathlib import Path

import onnxruntime
import torch
import torch.nn as nn
from hello.utils import importer


def set_logging(name=None, verbose=True):
    # Sets level and returns logger
    level = logging.INFO if verbose else logging.ERROR
    log = logging.getLogger(name)
    log.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    log.addHandler(handler)


set_logging()  # run before defining LOGGER
LOGGER = logging.getLogger("onnx-inference")  # define globally


class DetectBackend(nn.Module):

    def __init__(self, root, f="best.onnx", libpath="libprocess.py"):
        root = Path(root)
        f = (root / f).as_posix()
        libpath = (root / libpath).as_posix()

        super().__init__()

        LOGGER.info(f"Loading {f} for ONNX Runtime inference...")
        cuda = torch.cuda.is_available()
        providers = ["CUDAExecutionProvider",
                     "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(f, providers=providers)
        self.output_names = [self.session.get_outputs()[0].name]
        self.input_name = self.session.get_inputs()[0].name

        self.libprocess = importer.load_from_file("libprocess", libpath)
        self.mask_targets = self.libprocess.default_mask_targets()
        self.classes = self.libprocess.default_classes()

    def forward(self, filepath, **kwargs):
        x = self.libprocess.pre_process(filepath, **kwargs)
        y = self.session.run(self.output_names, {self.input_name: x})[0]
        z = self.libprocess.post_process(y, **kwargs)
        return z
