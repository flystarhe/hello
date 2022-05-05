import shutil
from pathlib import Path

try:
    import simplejson as json
except ImportError:
    import json


def make_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    return path


def copyfile(src, dst):
    # copies the file src to the file or directory dst
    return shutil.copy(src, dst)


def copyfile2(src, dst):
    # dst must be the complete target file name
    return shutil.copyfile(src, dst)


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    make_dir(Path(json_file).parent)

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    return json_file
