import shutil
from pathlib import Path

from hello.x3m.transforms import *

img_formats = set([".bmp", ".jpg", ".jpeg", ".png"])


def regular_preprocess(img_path, out_path, transformers, dtype=np.uint8):
    img_path, out_path = str(img_path), str(out_path)
    img = cv.imread(img_path)

    if img.ndim != 3:
        img = img[..., np.newaxis]
        img = np.concatenate((img, img, img), axis=-1)

    data = [img]
    for t in transformers:
        data = t(data)
    img = data[0]

    img.astype(dtype).tofile(out_path)
    return out_path


def todo(src, dst, imgsz, layout, ext):
    in_dir, out_dir = Path(src), Path(dst)

    imgsz = tuple(int(x) for x in imgsz.split("x"))
    assert len(imgsz) == 2, "format: HxW"
    layout = layout.upper()
    ext = ext.lower()

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)

    imgs = [f for f in in_dir.glob("**/*")
            if f.suffix in img_formats]

    transformers = [
        PadResizeTransformer(target_size=imgsz),
        BGR2RGBTransformer() if ext == "rgb" else Transformer(),
        BGR2NV12Transformer() if ext == "nv12" else Transformer(),
        HWC2CHWTransformer() if layout == "CHW" else Transformer(),
    ]

    dtype = np.float32 if out_dir.name.endswith("_f32") else np.uint8

    for img_path in sorted(imgs):
        out_path = out_dir / f"{img_path.stem}.{ext}"
        regular_preprocess(img_path, out_path, transformers, dtype)

    return 0


def parse_args(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="X3M preprocess")

    parser.add_argument("src", type=str,
                        help="dataset root dir")
    parser.add_argument("dst", type=str,
                        help="output root dir")
    parser.add_argument("--imgsz", type=str, default="640x640",
                        help="HxW")
    parser.add_argument("--layout", type=str, default="CHW",
                        help="CHW/HWC")
    parser.add_argument("--ext", type=str, default="rgb",
                        help="rgb/bgr/nv12")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(todo(**kwargs))

    return 0
