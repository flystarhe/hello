import sys

help_doc_str = """\
usage: hello-x3m <command> [options]

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
        from .preprocess import main as _main
        _main(args)
    elif command == "config":
        from .config import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
