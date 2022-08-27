from pathlib import Path
from string import Template

import fiftyone as fo
from fiftyone.utils.labels import segmentations_to_detections

tmpl_info = """
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


def merge_samples(datasets, **kwargs):
    A = datasets[0]

    A.save()
    A = A.clone()

    def key_fcn(sample):
        return Path(sample.filepath).name

    for B in datasets[1:]:
        B.save()
        B = B.clone()

        A.merge_samples(B, key_fcn=key_fcn, **kwargs)

    return A


def export_dataset(export_dir, dataset, label_field=None, mask_label_field=None, mask_types="thing"):
    assert label_field is not None or mask_label_field is not None

    dataset.save()
    info = dataset.info
    classes = dataset.default_classes
    mask_targets = dataset.default_mask_targets
    info["num_samples"] = dataset.count_values("tags")

    if label_field is None:
        label_field = "detections"
        print("todo: segmentations_to_detections()")
        dataset = dataset.select_fields(mask_label_field).clone()
        segmentations_to_detections(dataset, mask_label_field, label_field, mask_targets=dataset.default_mask_targets, mask_types=mask_types)

    splits = dataset.distinct("tags")

    if not splits:
        splits = ["all"]
        dataset.tag_samples("all")

    for split in splits:
        print(f"\n[{split}]\n")
        view = dataset.match_tags(split)
        curr_dir = Path(export_dir) / split

        view.export(
            export_dir=str(curr_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
        )

        if mask_label_field is not None:
            view.export(
                dataset_type=fo.types.ImageSegmentationDirectory,
                labels_path=str(curr_dir / "labels"),
                label_field=mask_label_field,
                mask_targets=mask_targets,
            )

        info["tail"].update(count_label=count_values(view, f"{label_field}.detections.label"))

        info_py = tmpl_info.safe_substitute(info,
                                            classes=classes,
                                            mask_targets=mask_targets)

        with open(curr_dir / "info.py", "w") as f:
            f.write(info_py)

    return export_dir


def count_values(dataset, field_or_expr, ordered=True):
    count_label = dataset.count_values(field_or_expr)
    count_label = [(k, v) for k, v in count_label.items()]

    if ordered:
        count_label = sorted(count_label, key=lambda x: x[1])

    return count_label
