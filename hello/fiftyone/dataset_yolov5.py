import fiftyone as fo


def load_yolov5_dataset(dataset_dir, dataset_name, label_field="ground_truth", splits=["train", "val"]):
    # https://voxel51.com/docs/fiftyone/user_guide/dataset_creation/datasets.html#yolov5dataset
    dataset = fo.Dataset(dataset_name)

    for split in splits:
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            tags=split,
            split=split,
        )

    return dataset


def export_yolov5_dataset(export_dir, dataset_or_view, label_field):
    # https://voxel51.com/docs/fiftyone/user_guide/export_datasets.html#yolov5dataset
    # or https://docs.ultralytics.com/tutorials/train-custom-datasets/#3-organize-directories
    splits = dataset_or_view.distinct("tags")

    if not splits:
        splits = ["train"]
        dataset_or_view.tag_samples("train")

    for split in splits:
        print(f"\n[{split}]\n")
        split_view = dataset_or_view.match_tags(split)
        split_view.export(
            export_dir=export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            data_path=None,
            labels_path=None,
            label_field=label_field,
            classes=dataset_or_view.default_classes,
            split="val" if split == "validation" else split,
        )

    return export_dir
