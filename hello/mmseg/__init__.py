import sys

help_doc_str = """\
usage: hello-mmseg <command> [options]

Commands:
    infer
    log
    lr
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "infer":
        from .infer import main as _main
        _main(args)
    elif command == "log":
        from .log import main as _main
        _main(args)
    elif command == "lr":
        from .lr import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
