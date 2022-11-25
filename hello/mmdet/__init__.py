import sys

help_doc_str = """\
usage: hello-mmdet <command> [options]

Commands:
    export
    flop
    infer
    log
    plot
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "export":
        from .export import main as _main
        _main(args)
    elif command == "flop":
        from .flop import main as _main
        _main(args)
    elif command == "infer":
        from .infer import main as _main
        _main(args)
    elif command == "log":
        from .log import main as _main
        _main(args)
    elif command == "plot":
        from .plot import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
