import sys

help_doc_str = """\
usage: hello-mmlab <command> [options]

Commands:
    mmdet3
    mmseg
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "mmdet3":
        from .mmdet3 import main as _main
        _main(args)
    elif command == "mmseg":
        from .mmseg import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
