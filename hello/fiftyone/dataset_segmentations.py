import code
import json
import sys
from collections import defaultdict
from pathlib import Path

from hello.fiftyone.core import merge_samples
from hello.utils import importer

import fiftyone as fo

dataset_doc_str = """
    <dataset_name>/
    ├── README.md  # 按照Markdown标准扩展信息
    ├── data
    │   ├── 000000000030.jpg
    │   ├── 000000000036.jpg
    │   └── 000000000042.jpg
    ├── labels  # ground_truth
    │   ├── 000000000030.png
    │   ├── 000000000036.png
    │   └── 000000000042.png
    ├── predictions  # predictions
    │   ├── 000000000030.png
    │   ├── 000000000036.png
    │   └── 000000000042.png
    └── info.py

    ground_truth/predictions:
        - the png file type as uint8
        - 0 means background, 255 means others
"""


def load_dataset(dataset_dir, info_py="info.py", data_path="data", labels_path="labels", field_name="ground_truth"):
    dataset_dir = Path(dataset_dir or ".")

    if dataset_dir.is_dir():
        info_py = dataset_dir / info_py
        data_path = dataset_dir / data_path
        labels_path = dataset_dir / labels_path
    else:
        info_py = Path(info_py)
        data_path = Path(data_path)
        labels_path = Path(labels_path)
