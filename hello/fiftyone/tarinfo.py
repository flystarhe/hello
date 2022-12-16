import tarfile
from pathlib import Path


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


def get_imgs(filename):
    data = []
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            filepath = Path(name)
            if filepath.parent.name == "data":
                data.append(filepath.name)

    vals = set(data)
    a, b = len(data), len(vals)
    if a != b:
        print(f"[INFO] {filename}: total {a}, unique {b}")

    return sorted(vals)
