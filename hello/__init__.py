"""A collection of useful tools!"""
import sys

__version__ = "1.0.6"

help_doc_str = """\
usage: hello [--version] [--help]

shell command:
    hello -h
    hello-data -h
    hello-fiftyone -h
    hello-mmdet -h
    hello-mmseg -h
    hello-video -h
    hello-x3m -h

command-line interface:
    python -m hello -h
    python -m hello.data -h
    python -m hello.fiftyone -h
    python -m hello.video -h
    python -m hello.x3m -h
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "--version":
        print(f"hello version {__version__}")
    else:
        print(help_doc_str)

    return 0
