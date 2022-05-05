import os
import sys


# Remove "" and current working directory from the first entry of sys.path
if sys.path[0] in ("", os.getcwd(), os.path.dirname(__file__)):
    sys.path.pop(0)


help_doc_str = """
Usage:
    python -m hello.data <command> [options]

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

    command, *remainder = command.split(".", maxsplit=1)

    if command == "coco2yolo":
        from hello.data.coco2yolo import main as _main
        _main(remainder + args)
    else:
        print(help_doc_str)

    return 0


# develop:
# PYTHONPATH=$(pwd) python hello/data ...
# runtime:
# python -m hello.data ...
if __name__ == "__main__":
    sys.exit(main())
