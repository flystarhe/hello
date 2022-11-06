import shutil
import sys
from pathlib import Path

import cv2 as cv

_bgr_image = None
_data_points = None

help_doc_str = """\
- press the left mouse button to mark points
- press `q` to quit
- press `b` to undo
- press `r` to clear
"""


def _text(text):
    global _bgr_image
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.rectangle(_bgr_image, (10, 2), (300, 18), (255, 255, 255), -1)
    cv.putText(_bgr_image, text, (15, 15), font, 0.5, (0, 0, 0), 1)


def _point(x, y, n):
    global _bgr_image
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.circle(_bgr_image, (x, y), 9, (0, 0, 255), 1)
    cv.putText(_bgr_image, f"{n}", (x + 10, y - 5), font, 0.5, (0, 0, 0), 1)


def _on_mouse(event, x, y, flags, param):
    global _bgr_image, _data_points

    if event == cv.EVENT_LBUTTONDOWN:
        _data_points = _data_points or []
        _point(x, y, len(_data_points))
        _data_points.append((x, y))
    elif event == cv.EVENT_MOUSEMOVE:
        _text(f"({x=}, {y=})")


def _save_to(file_stem, out_dir):
    global _bgr_image, _data_points
    tmpl_file = Path(out_dir) / f"data/{file_stem}.png"

    cv.imwrite(str(tmpl_file.with_suffix(".png")), _bgr_image)
    with open(tmpl_file.with_suffix(".txt"), "w") as f:
        f.write(str(_data_points))


def mark_points(image_path, out_dir):
    global _bgr_image, _data_points

    _nparr = cv.imread(image_path, 1)

    _bgr_image = _nparr.copy()
    window_name = Path(image_path).stem

    cv.imshow(window_name, _bgr_image)
    cv.setMouseCallback(window_name, _on_mouse)

    while True:
        cv.imshow(window_name, _bgr_image)

        key = cv.waitKey(30)

        if key == ord("q"):
            break
        elif key == ord("b"):
            _bgr_image = _nparr.copy()
            _data_points = _data_points[:-1]
            for num, xy in enumerate(_data_points, 0):
                _point(xy[0], xy[1], num)
        elif key == ord("r"):
            _data_points = []
            _bgr_image = _nparr.copy()

    _save_to(window_name, out_dir)
    cv.destroyAllWindows()

    return 0


def func(in_dir, out_dir):
    in_dir = Path(in_dir)

    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=False)

    for image_path in sorted(in_dir.glob("*.jpg")):
        mark_points(str(image_path), out_dir)

    return "\n[END]"


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", dest="in_dir", type=str,
                        help="input dir")
    parser.add_argument("-o", dest="out_dir", type=str,
                        help="output dir")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    print(help_doc_str)
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(func(**kwargs))

    return 0


if __name__ == "__main__":
    sys.exit(main())
