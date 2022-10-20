import sys

help_doc_str = """usage: hello-mmdet <command> [options]

Commands:
    export
    flop
    infer
    log
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
        from hello.mmdet.export import main as _main
        _main(args)
    elif command == "flop":
        from hello.mmdet.flop import main as _main
        _main(args)
    elif command == "infer":
        from hello.mmdet.infer import main as _main
        _main(args)
    elif command == "log":
        from hello.mmdet.log import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
