from pathlib import Path

import hello
import hello.fiftyone.core as hoc
import hello.fiftyone.dataset as hod
import hello.fiftyone.tarinfo as hot
import hello.fiftyone.view as hov

print(hello.__version__)


def get_images(root, files, out_dir, exclude_names, include_names):
    sub_dirs = []

    if root is not None:
        root = Path(root) if isinstance(root, str) else root
        files = [str(root / f) for f in files]

    out_dir = Path(out_dir)
    for i, f in enumerate(files, 1):
        kwargs = dict(data_path="data", exclude_names=exclude_names, include_names=include_names)
        sub_dir = hot.extract_images(out_dir / f"patch{i:02d}", [f], **kwargs)
        sub_dirs.append((sub_dir, Path(f).stem))

    return sub_dirs


def get_dataset(dataset_name, dataset_type, version, classes, mask_targets, sub_dirs):
    hod.delete_datasets([dataset_name], non_persistent=False)
    dataset = hod.create_dataset(dataset_name, dataset_type, version, classes, mask_targets)

    for sub_dir, stem in sub_dirs:
        hod.add_images_dir(dataset, Path(sub_dir) / "data", stem)

    counts = hoc.count_values(dataset, "tags")
    return dataset, counts


def get_uniqueness(dataset, counts, n, by=None, model="clip-vit-base32-torch"):
    if isinstance(n, list):
        unique_tag = f"unique_{sum(n)}"
        tasks = [(c[0], n[i]) for i, c in enumerate(counts)]
    elif by == "tag":
        unique_tag = f"unique_{n}x{len(counts)}"
        tasks = [(c[0], int(n)) for c in counts]
    else:
        unique_tag = f"unique_{n}"
        p = n / sum([c[1] for c in counts])
        tasks = [(c[0], int(c[1] * p)) for c in counts]
        tasks[-1] = (tasks[-1][0], tasks[-1][1] + n - sum([c[1] for c in tasks]))

    dataset.untag_samples(unique_tag)
    for tag, count in tasks:
        print(f"[I] unique({tag=}, {count=})")
        for brain_key in dataset.list_brain_runs():
            dataset.delete_brain_run(brain_key)
        view = dataset.match_tags([tag]).sort_by("filepath")
        view = hov.uniqueness(view, count, brain_key="img_sim", model=model)
        view.tag_samples(unique_tag)

    return dataset.match_tags(unique_tag), unique_tag


def test_images():
    dataset_name = "novabot_front_img_20230505_big_test_ver001"
    dataset_type = "unknown"
    version = "001"
    classes = []
    mask_targets = {}

    root = Path("/workspace/data/tarfiles/novabot_front/img_data")
    files = [
        "novabot_front_img_20230516_us_fawndr_ver002.tar",
        "novabot_front_img_20230501_big_test_ver001.tar",
    ]

    # {exclude/include}_names: None or json file path or dict or list
    # for big train dataset, e.g.
    # exclude_names = hot.get_image_names(tar_file_path)
    exclude_names = None
    include_names = None

    sub_dirs = get_images(root, files, f"tmp/{dataset_name}", exclude_names, include_names)
    dataset, counts = get_dataset(dataset_name, dataset_type, version, classes, mask_targets, sub_dirs)

    hod.export_image_dataset(f"exports/{dataset_name}", dataset, splits=None)


def test_uniqueness():
    dataset_name = "novabot_front_det_20230525_zhengshu_batch02_object9_ver001"
    dataset_type = "unknown"
    version = "001"
    classes = []
    mask_targets = {}

    root = Path("/workspace/data/tarfiles/novabot_front/img_data")
    files = [
        "novabot_front_img_20230525_us_afternoon_ver001.tar",
        "novabot_front_img_20230525_us_evening_ver001.tar",
    ]

    # {exclude/include}_names: None or json file path or dict or list
    # for big train dataset, e.g.
    # exclude_names = hot.get_image_names(tar_file_path)
    exclude_names = None
    include_names = None

    sub_dirs = get_images(root, files, f"tmp/{dataset_name}", exclude_names, include_names)
    dataset, counts = get_dataset(dataset_name, dataset_type, version, classes, mask_targets, sub_dirs)

    view, unique_tag = get_uniqueness(dataset, counts, [500] * len(files))
    hod.export_image_dataset(f"exports/{dataset_name}", view, splits=[unique_tag])
