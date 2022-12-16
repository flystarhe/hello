import tarfile
from pathlib import Path


def get_readme(filename):
    s = None
    with tarfile.open(filename, "r") as tar:
        for name in tar.getnames():
            if Path(name).name.lower() == "readme.md":
                with tar.extractfile(name) as f:
                    s = f.read().decode("utf8")
                break
    return s
