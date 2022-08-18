import os
import sys


# Remove "" and current working directory from the first entry of sys.path
if sys.path[0] in ("", os.getcwd(), os.path.dirname(__file__)):
    sys.path.pop(0)


help_doc_str = """
Usage:
    python -m hello.fiftyone <command> [options]

Commands:
    det
    seg
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "det":
        from hello.fiftyone.evaluate_detections import main as _main
        _main(args)
    elif command == "seg":
        from hello.fiftyone.evaluate_segmentations import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0


# develop:
# PYTHONPATH=$(pwd) python hello/fiftyone ...
# runtime:
# python -m hello.fiftyone ...
if __name__ == "__main__":
    sys.exit(main())
