import json
import re
import shutil
from pathlib import Path
from string import Template

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou
import fiftyone.utils.coco as fouc
import fiftyone.utils.iou as foui
import fiftyone.utils.yolo as fouy
from fiftyone.utils.labels import segmentations_to_detections

import hello.fiftyone.core as hoc
import hello.fiftyone.utils as hou
from hello.fiftyone.dataset_detections import \
    load_dataset as _load_detection_dataset
from hello.fiftyone.dataset_segmentations import \
    load_dataset as _load_segmentation_dataset

tmpl_info = """\
info = {
    'dataset_name': '$dataset_name',
    'dataset_type': '$dataset_type',
    'version': '$version',
    'classes': $classes,
    'mask_targets': $mask_targets,
    'num_samples': $num_samples,
    'tail': $tail,
}
"""
tmpl_info = Template(tmpl_info)


def add_classification_labels(dataset, label_field, labels_path):
    # https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#fiftyoneimageclassificationdataset-export
    assert Path(labels_path).suffix == ".json"

    with open(labels_path, "r") as f:
        data = json.load(f)

    assert "classes" in data and "labels" in data

    db = {}
    label_type = None
    for k, v in data["labels"].items():
        if isinstance(v, list):
            if label_type is None:
                label_type = "classifications"
            assert label_type == "classifications"
            classifications = [fol.Classification(**vi) for vi in v]
            db[k] = fol.Classifications(classifications=classifications)
        else:
            if label_type is None:
                label_type = "classification"
            assert label_type == "classification"
            db[k] = fol.Classification(**v)

    filepaths, ids = dataset.values(["filepath", "id"])
    id_map = {Path(k).stem: v for k, v in zip(filepaths, ids)}

    stems_adds = set(db.keys())
    stems_base = set(id_map.keys())

    bad_stems = stems_adds - stems_base
    if bad_stems:
        print(f"<{labels_path}>\n  Ignoring {len(bad_stems)} nonexistent images (eg {list(bad_stems)[:3]})")

    stems = sorted(stems_adds & stems_base)
    matched_ids = [id_map[stem] for stem in stems]
    view = dataset.select(matched_ids, ordered=True)

    labels = [db[stem] for stem in stems]

    view.set_values(label_field, labels)
    print(f"update {len(labels)=}")


def add_coco_labels(dataset, label_field, labels_path, label_type="detections"):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.utils.coco.html#fiftyone.utils.coco.add_coco_labels
    assert label_type in {"detections", "segmentations", "keypoints"}
    assert Path(labels_path).suffix == ".json"

    with open(labels_path, "r") as f:
        coco = json.load(f)

    assert "categories" in coco and "images" in coco and "annotations" in coco

    classes = [cat["name"] for cat in coco["categories"]]

    db = {Path(img["file_name"]).stem: img["id"] for img in coco["images"]}
    coco_ids = [db.get(Path(filepath).stem, -1) for filepath in dataset.values("filepath")]

    coco_id_field = "coco_id"
    dataset.set_values(coco_id_field, coco_ids)

    fouc.add_coco_labels(
        dataset,
        label_field,
        coco["annotations"],
        classes,
        label_type=label_type,
        coco_id_field=coco_id_field,
    )


def add_yolo_labels(dataset, label_field, labels_path, classes):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.utils.yolo.html#fiftyone.utils.yolo.add_yolo_labels
    assert isinstance(classes, list)

    fouy.add_yolo_labels(
        dataset,
        label_field,
        labels_path,
        classes,
    )


def add_detection_labels(dataset, label_field, labels_path, classes=None, mode="text"):
    """Adds detection labels to the dataset.

    .. note::
        if ``mode=text``, a text row corresponds to a sample prediction result.
        row format: ``filepath,height,width,x1,y1,x2,y2,s,l,x1,y1,x2,y2,s,l``.

        if ``mode=yolo``, a txt file corresponds to a sample prediction result.
        row format: ``target,xc,yc,w,h,s``.

        if ``mode=coco``, a standard COCO format json file.
        from https://cocodataset.org/#format-data.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        label_field (str): the label field in which to store the labels
        labels_path (str): the labels load from
        classes (list): the list of class label strings
        mode (str): supported values are ``("text", "yolo", "coco")``
    """
    assert mode in {"text", "yolo", "coco"}

    dataset_classes = dataset.default_classes

    assert classes is None or isinstance(classes, list)
    included_labels = set(dataset_classes)

    filepaths, ids = dataset.values(["filepath", "id"])
    id_map = {Path(k).stem: v for k, v in zip(filepaths, ids)}

    db = hou.load_predictions(labels_path, classes=classes, mode=mode)

    stems_adds = set(db.keys())
    stems_base = set(id_map.keys())

    bad_stems = stems_adds - stems_base
    if bad_stems:
        print(f"<{labels_path}>\n  Ignoring {len(bad_stems)} nonexistent images (eg {list(bad_stems)[:3]})")

    stems = sorted(stems_adds & stems_base)
    matched_ids = [id_map[stem] for stem in stems]
    view = dataset.select(matched_ids, ordered=True)

    labels = []
    for stem in stems:
        detections = [fol.Detection(**detection) for detection in db[stem]
                      if detection["label"] in included_labels]
        labels.append(fol.Detections(detections=detections))

    view.set_values(label_field, labels)
    print(f"update {len(labels)=}")


def add_segmentation_labels(dataset, label_field, labels_path, mask_targets="auto", mode="png"):
    """Adds segmentation labels to the dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        label_field (str): the label field in which to store the labels
        labels_path (str): the labels load from
        mask_targets (dict): a dict mapping pixel values to semantic label strings
        mode (str): supported values are ``("png", "coco")``
    """
    assert mode in {"png", "coco"}

    dataset_mask_targets = dataset.default_mask_targets

    if mask_targets == "auto":
        info_py = Path(labels_path).with_name("info.py")
        with open(info_py, "r") as f:
            codestr = f.read()

        info = eval(re.split(r"info\s*=\s*", codestr)[1])
        mask_targets = info["mask_targets"]

    assert isinstance(mask_targets, dict)
    remap = hou.gen_mask_remap(dataset_mask_targets, mask_targets)

    filepaths, ids = dataset.values(["filepath", "id"])
    id_map = {Path(k).stem: v for k, v in zip(filepaths, ids)}

    db = hou.load_segmentation_masks(labels_path, remap, mode)

    stems_adds = set(db.keys())
    stems_base = set(id_map.keys())

    bad_stems = stems_adds - stems_base
    if bad_stems:
        print(f"<{labels_path}>\n  Ignoring {len(bad_stems)} nonexistent images (eg {list(bad_stems)[:3]})")

    stems = sorted(stems_adds & stems_base)
    matched_ids = [id_map[stem] for stem in stems]
    view = dataset.select(matched_ids, ordered=True)

    labels = []
    for stem in stems:
        mask = db[stem]
        labels.append(fol.Segmentation(mask=mask))

    view.set_values(label_field, labels)
    print(f"update {len(labels)=}")


def add_images_dir(dataset, images_dir, tags=None, recursive=True):
    """Adds the given directory of images to the dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        images_dir (str): a directory of images
        tags (None): an optional tag or iterable of tags to attach to each sample
        recursive (True): whether to recursively traverse subdirectories
    """
    # https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.add_images_dir
    if not recursive:
        image_paths = [str(f) for f in Path(images_dir).glob("*.jpg")]
    else:
        image_paths = [str(f) for f in Path(images_dir).glob("**/*.jpg")]

    stems_base = set([Path(filepath).stem for filepath in dataset.values("filepath")])
    stems_adds = set([Path(filepath).stem for filepath in image_paths])

    bad_stems = stems_base & stems_adds
    if bad_stems:
        print(f"<{images_dir}>\n  Ignoring {len(bad_stems)} existing images (eg {list(bad_stems)[:3]})")

    image_paths = sorted([filepath for filepath in image_paths
                          if Path(filepath).stem not in bad_stems])

    dataset.add_images(image_paths, tags=tags)
    # Populate the `metadata` field
    dataset.compute_metadata()


def delete_duplicate_images(dataset):
    """Delete duplicate images.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
    """
    filepaths, ids = dataset.values(["filepath", "id"])

    unique_ids = []
    filehash_set = set()
    for k, v in zip(filepaths, ids):
        filehash = fou.compute_filehash(k)
        if filehash not in filehash_set:
            filehash_set.add(filehash)
            unique_ids.append(v)

    dup_ids = set(ids) - set(unique_ids)
    if dup_ids:
        print(f"Delete {len(dup_ids)} duplicate images (eg {list(dup_ids)[:3]})")

    dataset.delete_samples(dup_ids)


def delete_duplicate_labels(dataset, label_field, iou_thresh=0.999, method="simple", iscrowd=None, classwise=True):
    """Delete duplicate labels in the given field of the dataset, as defined as labels with an IoU greater than a chosen threshold with another label in the field.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        label_field: a label field of type :class:`fiftyone.core.labels.Detections` or :class:`fiftyone.core.labels.Polylines`
        iou_thresh (0.999): the IoU threshold to use to determine whether labels are duplicates
        method ("simple"): supported values are ``("simple", "greedy")``
        iscrowd (None): an optional name of a boolean attribute
        classwise (True): different label values as always non-overlapping
    """
    dup_ids = foui.find_duplicates(dataset, label_field, iou_thresh=iou_thresh, method=method, iscrowd=iscrowd, classwise=classwise)
    if dup_ids:
        print(f"Delete {len(dup_ids)} duplicate labels (eg {list(dup_ids)[:3]})")

    dataset.delete_labels(ids=dup_ids, fields=label_field)


def add_dataset_dir(dataset_dir, data_path=None, labels_path=None, label_field=None, tags=None):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.add_dir
    raise NotImplementedError


def add_dataset(dataset, skip_existing=True, insert_new=True, fields=None, expand_schema=True):
    # https://voxel51.com/docs/fiftyone/api/fiftyone.core.dataset.html#fiftyone.core.dataset.Dataset.merge_samples
    raise NotImplementedError


def create_dataset(dataset_name, dataset_type, classes=[], mask_targets={}):
    """Create an empty :class:`fiftyone.core.dataset.Dataset` with the name.

    Args:
        dataset_name (str): a name for the dataset
        dataset_type (str): supported values are ``("detection", "segmentation", "unknown")``
        classes (list, optional): defaults to ``[]``
        mask_targets (dict, optional): defaults to ``{}``

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    assert dataset_type in {"detection", "segmentation", "unknown"}
    dataset = fo.Dataset()

    dataset.name = dataset_name
    dataset.persistent = True

    info = {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "version": "001",
        "classes": classes,
        "mask_targets": mask_targets,
        "num_samples": {},
        "tail": {},
    }

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    return dataset


def load_images_dir(dataset_dir, dataset_name, dataset_type, classes=[], mask_targets={}):
    """Create a :class:`fiftyone.core.dataset.Dataset` from the given directory of images.

    Args:
        dataset_dir (str): a directory of images
        dataset_name (str): a name for the dataset
        dataset_type (str): supported values are ``("detection", "segmentation", "unknown")``
        classes (list, optional): defaults to ``[]``
        mask_targets (dict, optional): defaults to ``{}``

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    assert dataset_type in {"detection", "segmentation", "unknown"}
    dataset = fo.Dataset.from_images_dir(dataset_dir)

    dataset.name = dataset_name
    dataset.persistent = True

    info = {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "version": "001",
        "classes": classes,
        "mask_targets": mask_targets,
        "num_samples": {},
        "tail": {},
    }

    dataset.default_classes = info.pop("classes", [])
    dataset.default_mask_targets = info.pop("mask_targets", {})
    dataset.info = info
    dataset.save()

    return dataset


def list_datasets():
    return fo.list_datasets()


def delete_datasets(names=None, non_persistent=True):
    names = set(names or [])

    _vals = set(fo.list_datasets())

    bad_names = names - _vals
    if bad_names:
        print(f"Ignoring {len(bad_names)} nonexistent datasets (eg {list(bad_names)[:3]})")

    for name in sorted(names & _vals):
        fo.delete_dataset(name, verbose=True)

    if non_persistent:
        fo.delete_non_persistent_datasets(verbose=True)


def load_dataset(name):
    """Loads the FiftyOne dataset with the given name.

    Args:
        name (str): the name of the dataset
    """
    return fo.load_dataset(name)


def load_detection_dataset(dataset_dir, info_py="info.py", data_path="data", labels_path="labels.json", field_name="ground_truth", splits=None):
    dataset_dir = Path(dataset_dir)

    if splits is None:
        dataset = _load_detection_dataset(str(dataset_dir), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
        dataset.tag_samples("train")
    else:
        _datasets = []
        for s in splits:
            _dataset = _load_detection_dataset(str(dataset_dir / s), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
            _dataset.tag_samples(s)
            _datasets.append(_dataset)
        dataset = hoc.merge_samples(_datasets)

    return dataset


def load_segmentation_dataset(dataset_dir, info_py="info.py", data_path="data", labels_path="labels/", field_name="ground_truth", splits=None):
    dataset_dir = Path(dataset_dir)

    if splits is None:
        dataset = _load_segmentation_dataset(str(dataset_dir), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
        dataset.tag_samples("train")
    else:
        _datasets = []
        for s in splits:
            _dataset = _load_segmentation_dataset(str(dataset_dir / s), info_py=info_py, data_path=data_path, labels_path=labels_path, field_name=field_name)
            _dataset.tag_samples(s)
            _datasets.append(_dataset)
        dataset = hoc.merge_samples(_datasets)

    return dataset


def export_classification_labels(export_dir, dataset, label_field, splits=None):
    shutil.rmtree(export_dir, ignore_errors=True)

    _tags = set(dataset.distinct("tags"))

    if splits is None:
        splits = ["train", "val", "test"]
    elif splits == "auto":
        splits = sorted(_tags)

    assert isinstance(splits, list)
    splits = [s for s in splits if s in _tags]

    if not splits:
        splits = ["train"]
        dataset.tag_samples(splits)

    for split in splits:
        print(f"\n[{split}]\n")
        view = dataset.match_tags(split)
        curr_dir = Path(export_dir) / split

        view.export(
            dataset_type=fo.types.FiftyOneImageClassificationDataset,
            labels_path=str(curr_dir / "labels.json"),
            label_field=label_field,
            include_confidence=True,
        )

    return export_dir


def export_classification_dataset(export_dir, dataset, label_field, splits=None, export_media=True):
    shutil.rmtree(export_dir, ignore_errors=True)

    _tags = set(dataset.distinct("tags"))

    if splits is None:
        splits = ["train", "val", "test"]
    elif splits == "auto":
        splits = sorted(_tags)

    assert isinstance(splits, list)
    splits = [s for s in splits if s in _tags]

    if not splits:
        splits = ["train"]
        dataset.tag_samples(splits)

    for split in splits:
        print(f"\n[{split}]\n")
        view = dataset.match_tags(split)
        curr_dir = Path(export_dir) / split

        view.export(
            export_dir=str(curr_dir),
            dataset_type=fo.types.FiftyOneImageClassificationDataset,
            export_media=export_media,
            label_field=label_field,
            include_confidence=True,
        )

    return export_dir


def export_detection_dataset(export_dir, dataset, label_field, splits=None):
    return export_dataset(export_dir, dataset, label_field=label_field, splits=splits)


def export_segmentation_dataset(export_dir, dataset, label_field, mask_types="stuff", splits=None):
    return export_dataset(export_dir, dataset, mask_label_field=label_field, mask_types=mask_types, splits=splits)


def export_dataset(export_dir, dataset, label_field=None, mask_label_field=None, mask_types="stuff", splits=None):
    """Exports the samples in the collection to disk.

    Args:
        export_dir: the directory to which to export the samples
        dataset: a :class:`fiftyone.core.collections.SampleCollection`
        label_field: controls the label field(s) to export
        mask_label_field: controls the label field(s) to export
        mask_types ("stuff"): "stuff"(amorphous regions of pixels), "thing"(connected regions, each representing an instance)
        splits (None): a list of strings, respectively, specifying the splits to load. If "auto" will computes the distinct tags
    """
    assert label_field is not None or mask_label_field is not None
    shutil.rmtree(export_dir, ignore_errors=True)

    dataset.save()
    info = dataset.info
    classes = dataset.default_classes
    mask_targets = dataset.default_mask_targets
    info["num_samples"] = hoc.count_values(dataset, "tags")

    if label_field is None:
        label_field = "detections"
        print("todo: segmentations_to_detections()")
        dataset = dataset.select_fields(mask_label_field).clone()
        segmentations_to_detections(dataset, mask_label_field, label_field, mask_targets=dataset.default_mask_targets, mask_types=mask_types)
    else:
        dataset = dataset.clone()

    _tags = set(dataset.distinct("tags"))

    if splits is None:
        splits = ["train", "val", "test"]
    elif splits == "auto":
        splits = sorted(_tags)

    assert isinstance(splits, list)
    splits = [s for s in splits if s in _tags]

    if not splits:
        splits = ["train"]
        dataset.tag_samples(splits)

    for split in splits:
        print(f"\n[{split}]\n")
        view = dataset.match_tags(split)
        curr_dir = Path(export_dir) / split

        view.export(
            export_dir=str(curr_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
            classes=classes,
        )

        if mask_label_field is not None:
            view.export(
                dataset_type=fo.types.ImageSegmentationDirectory,
                labels_path=str(curr_dir / "labels"),
                label_field=mask_label_field,
                mask_targets=mask_targets,
            )

        info["tail"].update(count_label=hoc.count_values(view, f"{label_field}.detections.label"))

        info_py = tmpl_info.safe_substitute(info,
                                            classes=classes,
                                            mask_targets=mask_targets)

        with open(curr_dir / "info.py", "w") as f:
            f.write(info_py)

    return export_dir
