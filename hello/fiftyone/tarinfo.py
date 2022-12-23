import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from_pattern = None


def equal_list(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False

    for _ai, _bi in zip(a, b):
        if _ai != _bi:
            return False

    return True


def equal_dict(a: dict, b: dict) -> bool:
    if len(a) != len(b):
        return False

    for k, v in a.items():
        if v != b.get(k):
            return False

    return True


def get_readme(filename):
    docstr = None
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            filepath = Path(name)
            if filepath.name == "README.md":
                with tar.extractfile(name) as f:
                    docstr = f.read().decode("utf8")
                break
    return docstr


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
        print(f"[W] <{filename}>:\n  total {a}, unique {b}")

    return sorted(vals)


def get_image_paths(filename, data_path="data"):
    data = []
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            filepath = Path(name)
            if filepath.parent.name == data_path:
                data.append(name)
    return data


def compare(file1, file2, data_path="data", verbose=True):
    if verbose:
        print(f"Compare data:\n  a: <{file1}>\n  b: <{file2}>")

    base_imgs = get_image_paths(file1, data_path)
    base_dict = {Path(f).name: f for f in base_imgs}

    a, b = len(base_imgs), len(base_dict)
    if a != b:
        print(f"[W] <{file1}>:\n  total {a}, unique {b}")

    side_imgs = get_image_paths(file2, data_path)
    side_dict = {Path(f).name: f for f in side_imgs}

    a, b = len(side_imgs), len(side_dict)
    if a != b:
        print(f"[W] <{file2}>:\n  total {a}, unique {b}")

    names = sorted(set(base_dict.keys()) & set(side_dict.keys()))

    a, b, c = len(base_dict), len(side_dict), len(names)
    if verbose:
        print(f"[I] ({a=}) & ({b=}) => intersect:{c}")

    eqs = []
    tar1 = tarfile.open(file1, "r")
    tar2 = tarfile.open(file2, "r")
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
    tar1.close()
    tar2.close()

    if verbose:
        print(f"[I] {len(names)=}, {len(eqs)=}")

    return names, eqs


def extract_images(out_dir, files, data_path="data", exclude_names=None):
    shutil.rmtree(out_dir, ignore_errors=True)

    if isinstance(files, str):
        files = [files]

    files = sorted(files)

    if exclude_names is not None:
        exclude_names = set([Path(name).name for name in exclude_names])

    db = {}
    for file in files:
        for name in get_image_paths(file, data_path):
            db[Path(name).name] = (file, name)

    tasks = defaultdict(list)
    for file, name in db.values():
        tasks[file].append(name)

    tmp_dir = Path(out_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=False)

    for file, names in tasks.items():
        print(f"[I] extract {len(names)} images from <{file}>")
        if exclude_names is not None:
            names = [name for name in names
                     if Path(name).name not in exclude_names]
            print(f"[-] extract {len(names)} images after exclusion")
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
        from_data = "\n".join([Path(file).name for file in files])
        f.write(f"# README\n\n## Data Processing\n\n**from**\n\n```python\n{from_data}\n```")

    return out_dir


def extract_from_data(filename):
    global from_pattern

    docstr = get_readme(filename)

    if docstr is None:
        print(f"[E] <{filename}> not found `README.md`")
        return None

    if from_pattern is None:
        from_pattern = re.compile(r"\*\*from:\*\*\n+```[a-z]+\n([\(\[\)\]\n\s,'._0-9a-z]+)\n```")

    m = from_pattern.search(docstr)

    if m is None:
        return None

    substr = m.group(1)

    results = re.split(r"[\(\[\)\]\n\s,']+", substr)
    results = [r for r in results if r]
    return results


def extract_info_py(filename):
    data = []
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            filepath = Path(name)
            if filepath.name == "info.py":
                with tar.extractfile(name) as f:
                    codestr = f.read().decode("utf8")
                    data.append([name, re.split(r"info\s*=\s*", codestr)[1]])
    return data


def compare_info_py(file1, file2, keys=None, verbose=True):
    if verbose:
        print(f"Compare info:\n  a: <{file1}>\n  b: <{file2}>")

    if keys is None:
        keys = ["classes", "mask_targets"]

    base_info = eval(extract_info_py(file1)[0][1])
    side_info = eval(extract_info_py(file2)[0][1])

    results = {}
    for key in keys:
        a, b = base_info[key], side_info[key]

        if isinstance(a, str) and isinstance(b, str):
            result = (a == b)
        elif isinstance(a, list) and isinstance(b, list):
            result = equal_list(a, b)
        elif isinstance(a, dict) and isinstance(b, dict):
            result = equal_dict(a, b)
        else:
            result = "Unkown (BadType)"

        if verbose:
            print(f"  <{key}>: {result}")

        results[key] = result

    return results
