import sys

help_doc_str = """usage: hello-fiftyone <command> [options]

Commands:
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

    if command == "unique":
        from hello.fiftyone.unique import main as _main
        _main(args)
    elif command == "det-view":
        from hello.fiftyone.dataset_detections import main as _main
        _main(args)
    elif command == "det-eval":
        from hello.fiftyone.evaluate_detections import main as _main
        _main(args)
    elif command == "seg-view":
        from hello.fiftyone.dataset_segmentations import main as _main
        _main(args)
    elif command == "seg-eval":
        from hello.fiftyone.evaluate_segmentations import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
