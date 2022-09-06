"""A collection of useful tools!"""
import sys

__version__ = "0.2.2"

help_doc_str = """
Usage:
    hello <command> [options]

Commands:
    data
    fiftyone
    video
    x3m
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    command, *remainder = command.split(".", maxsplit=1)
    args = ["--help"] if len(args) == 0 else args

    if command == "data":
        from hello.data.__main__ import main as _main
        _main(remainder + args)
    elif command == "fiftyone":
        from hello.fiftyone.__main__ import main as _main
        _main(remainder + args)
    elif command == "video":
        from hello.video.__main__ import main as _main
        _main(remainder + args)
    elif command == "x3m":
        from hello.x3m.__main__ import main as _main
        _main(remainder + args)
    else:
        print(help_doc_str)

    return 0
