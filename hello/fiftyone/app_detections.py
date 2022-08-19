import code
import sys

import fiftyone as fo

dataset_doc_str = """
    <dataset_name>/
    ├── README.md  # 按照Markdown标准扩展信息
    ├── data
    │   ├── 000000000030.jpg
    │   ├── 000000000036.jpg
    │   └── 000000000042.jpg
    ├── info.py
    ├── labels.json  # ground_truth
    └── predictions.txt  # predictions

    ground_truth/predictions:
        - *.json: COCO format
        - *.txt: An inference result saves a row
            filepath,height,width,x1,y1,x2,y2,confidence,label,x1,y1,x2,y2,confidence,label
"""


def func(**kwargs):
    dataset = None
    session = fo.launch_app(dataset)
    return session


def parse_args(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Evaluating Predictions")

    parser.add_argument("dataset_dir", type=str,
                        help="dataset root dir")
    parser.add_argument("--info", dest="info_py", type=str, default="info.py",
                        help="path of info.py")
    parser.add_argument("--data", dest="data_path", type=str, default="data",
                        help="which the images")
    parser.add_argument("--labels", dest="labels_path", type=str, default="labels.json",
                        help="which the labels file")
    parser.add_argument("--preds", dest="preds_path", type=str, default="predictions.txt",
                        help="which the predictions file")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    print(dataset_doc_str)
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    session = func(**kwargs)

    banner = "Use quit() or Ctrl-Z plus Return to exit"
    code.interact(banner=banner, local=locals(), exitmsg="End...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
