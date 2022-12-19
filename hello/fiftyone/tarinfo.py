import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np


def get_readme(filename):
    s = None
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            filepath = Path(name)
            if filepath.name.lower() == "readme.md":
                with tar.extractfile(name) as f:
                    s = f.read().decode("utf8")
                break
    return s


def get_image_names(filename, data_path="data"):
    data = []
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            filepath = Path(name)
            if filepath.parent.name == data_path:
                data.append(filepath.name)

    vals = set(data)
    a, b = len(data), len(vals)
    if a != b:
        print(f"[W] {filename}: total {a}, unique {b}")

    return sorted(vals)


def get_image_paths(filename, data_path="data"):
    data = []
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            filepath = Path(name)
            if filepath.parent.name == data_path:
                data.append(name)
    return data


def compare(file1, file2, data_path="data"):
    base_imgs = get_image_paths(file1, data_path)
    base_dict = {Path(f).name: f for f in base_imgs}

    a, b = len(base_imgs), len(base_dict)
    if a != b:
        print(f"[W] {file1}: total {a}, unique {b}")

    side_imgs = get_image_paths(file2, data_path)
    side_dict = {Path(f).name: f for f in side_imgs}

    a, b = len(side_imgs), len(side_dict)
    if a != b:
        print(f"[W] {file2}: total {a}, unique {b}")

    names = sorted(set(base_dict.keys()) & set(side_dict.keys()))

    a, b, c = len(base_dict), len(side_dict), len(names)
    print(f"[I] ({a=}) & ({b=}) => intersect:{c}")

    tar1 = tarfile.open(file1, "r")
    tar2 = tarfile.open(file2, "r")
    eqs = []
    for name in names:
        member = base_dict[name]
        with tar1.extractfile(member) as f:
            nparr = np.frombuffer(f.read(), np.uint8)
            im1 = cv.imdecode(nparr, 1)

        member = side_dict[name]
        with tar2.extractfile(member) as f:
            nparr = np.frombuffer(f.read(), np.uint8)
            im2 = cv.imdecode(nparr, 1)

        if np.array_equal(im1, im2):
            eqs.append(name)
    print(f"[I] {len(names)=}, {len(eqs)=}")
    tar1.close()
    tar2.close()

    return names, eqs


def extract_images(out_dir, files, data_path="data"):
    shutil.rmtree(out_dir, ignore_errors=True)

    if isinstance(files, str):
        files = [files]

    db = {}
    for file in sorted(files):
        for name in get_image_paths(file, data_path):
            db[Path(name).name] = (file, name)

    tasks = defaultdict(list)
    for file, name in db.values():
        tasks[file].append(name)

    tmp_dir = Path(out_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=False)

    for file, names in tasks.items():
        print(f"[I] extract {len(names)} images from <{file}>")
        with tarfile.open(file, "r") as tar:
            members = [tar.getmember(name) for name in names]
            tar.extractall(tmp_dir, members)

    data_dir = Path(out_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=False)

    for src in tmp_dir.glob("**/*"):
        if src.is_file():
            shutil.copy(src, data_dir)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    with open(Path(out_dir) / "README.md", "w") as f:
        f.write(f"# README\n\n## Data Processing\n\n**from**\n\n{files}")

    return out_dir
