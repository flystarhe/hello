# https://github.com/ultralytics/JSON2YOLO
import shutil
from pathlib import Path

import numpy as np
from hello import io
from tqdm import tqdm

img_formats = set([".bmp", ".jpg", ".jpeg", ".png"])


def coco_to_yolo(coco_dir, json_dir=None, classes=None):
    coco_dir = Path(coco_dir)

    if json_dir is None:
        json_dir = coco_dir
    else:
        json_dir = coco_dir / json_dir

    out_dir = coco_dir.parent / (coco_dir.name + "_yolo")
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "labels").mkdir(parents=True)
    (out_dir / "images").mkdir(parents=True)

    real_path = {f.name: str(f) for f in coco_dir.glob("**/*")
                 if f.suffix in img_formats}
    total_images = len(real_path)

    for json_file in sorted(json_dir.glob("*.json")):
        data = io.load_json(json_file)

        images = {x["id"]: x for x in data["images"]}
        names = classes or [x["name"] for x in data["categories"]]
        names = ["REMAINDER"] + sorted(set(names).difference(["REMAINDER"]))

        name_dict = {s: i for i, s in enumerate(names, 0)}
        cvt_id = {x["id"]: name_dict.get(x["name"], 0)
                  for x in data["categories"]}

        image_path_list = []
        for x in tqdm(data["images"], desc=f"{json_file.stem}"):
            img_name = x["file_name"]
            src_path = real_path[img_name]
            dst_path = out_dir / "images" / img_name
            if not dst_path.exists():
                shutil.copyfile(src_path, dst_path)
            image_path_list.append(f"./images/{img_name}")
        image_path_list = sorted(set(image_path_list))
        n_images = len(image_path_list)

        with open(out_dir / "names.txt", "a") as file:
            file.write("{}: {}\n".format(json_file.stem, names))

        with open(out_dir / (json_file.stem + ".txt"), "w") as file:
            file.write("\n".join(image_path_list))

        for x in tqdm(data["annotations"], desc=f"{json_file.stem} ({n_images}/{total_images})"):
            if x.get("iscrowd"):
                continue

            img = images[x["image_id"]]

            h, w, f = img["height"], img["width"], img["file_name"]

            # format is [top left x, top left y, width, height]
            box = np.array(x["bbox"], dtype=np.float32)
            box[:2] += box[2:] / 2  # to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y

            if (box[2] > 0.) and (box[3] > 0.):  # if w * h > 0
                with open(out_dir / "labels" / (Path(f).stem + ".txt"), "a") as file:
                    file.write("%g %.6f %.6f %.6f %.6f\n" %
                               (cvt_id[x["category_id"]], *box))
    return str(out_dir)


def parse_args(args=None):
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("coco_dir", type=str,
                        help="dataset root dir")
    parser.add_argument("-j", "--json_dir", type=str,
                        help="coco json file dir")
    parser.add_argument("--classes", nargs="+", type=str,
                        help="filter by class: --classes c0 c2 c3")

    args = parser.parse_args(args=args)
    return vars(args)


def main(args=None):
    kwargs = parse_args(args)
    print(f"{__file__}: {kwargs}")

    print(coco_to_yolo(**kwargs))

    return 0
