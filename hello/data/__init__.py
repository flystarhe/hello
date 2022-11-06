import sys

help_doc_str = """\
usage: hello-data <command> [options]

Commands:
    coco2yolo
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "coco2yolo":
        from .coco2yolo import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
