import os
import sys


# Remove "" and current working directory from the first entry of sys.path
if sys.path[0] in ("", os.getcwd(), os.path.dirname(__file__)):
    sys.path.pop(0)


help_doc_str = """
Usage:
    python -m hello.x3m <command> [options]

Commands:
    preprocess
    config
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "preprocess":
        from hello.x3m.preprocess import main as _main
        _main(args)
    elif command == "config":
        from hello.x3m.config import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0


# develop:
# PYTHONPATH=$(pwd) python hello/x3m ...
# runtime:
# python -m hello.x3m ...
if __name__ == "__main__":
    sys.exit(main())
