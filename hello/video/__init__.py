# Getting Started with Videos:
# https://docs.opencv.org/4.6.0/dd/d43/tutorial_py_video_display.html
# https://docs.opencv.org/5.x/dd/d43/tutorial_py_video_display.html
import sys

help_doc_str = """\
usage: hello-video <command> [options]

Commands:
    clip
    frames
    info
    resize
    split
    unwarp
"""


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    if len(args) >= 1:
        command, *args = args
    else:
        command, *args = ["--help"]

    args = ["--help"] if len(args) == 0 else args

    if command == "clip":
        from .clip import main as _main
        _main(args)
    elif command == "frames":
        from .frames import main as _main
        _main(args)
    elif command == "info":
        from .info import main as _main
        _main(args)
    elif command == "resize":
        from .resize import main as _main
        _main(args)
    elif command == "split":
        from .split import main as _main
        _main(args)
    elif command == "unwarp":
        from .unwarp import main as _main
        _main(args)
    else:
        print(help_doc_str)

    return 0
