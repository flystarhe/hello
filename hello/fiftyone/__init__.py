import sys

help_doc_str = """\
usage: hello-fiftyone <command> [options]

Commands:
    miou
    unique
    det-view
    det-eval
    seg-view
    seg-eval
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "miou":
        from .miou import main as _main
        _main(args)
    elif command == "unique":
        from .unique import main as _main
        _main(args)
    elif command == "det-view":
        from .dataset_detections import main as _main
        _main(args)
    elif command == "det-eval":
        from .evaluate_detections import main as _main
        _main(args)
    elif command == "seg-view":
        from .dataset_segmentations import main as _main
        _main(args)
    elif command == "seg-eval":
        from .evaluate_segmentations import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
