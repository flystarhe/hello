import fiftyone as fo


def annotate(dataset_or_view, label_field="ground_truth", label_type="instances",
             url="http://119.23.212.113:6060", username="hejian", password="LFIcvat123",
             task_assignee="hejian", job_assignees=["weiqiaomu", "jiasiyu"]):
    # `label_type` (None) - a string. The possible values are: `classification`, `classifications`, `detections`, `instances`, `segmentation`, `scalar`.
    # `mask_targets` (None) - a dict mapping pixel values to semantic label strings. Only applicable when annotating semantic segmentations.
    anno_key = f"{dataset_or_view.name}_{label_field}_{label_type}"

    # The new attributes that we want to populate
    attributes = True
    if label_type == "detections":
        attributes = {
            "iscrowd": {
                "type": "radio",
                "values": [1, 0],
                "default": 0,
            }
        }

    dataset_or_view.annotate(
        anno_key,
        label_field=label_field,
        label_type=label_type,
        classes=dataset_or_view.default_classes or None,
        attributes=attributes,
        mask_targets=dataset_or_view.default_mask_targets or None,
        launch_editor=False,
        url=url,
        username=username,
        password=password,
        task_size=1500,
        segment_size=50,
        image_quality=95,
        task_assignee=task_assignee,
        job_assignees=job_assignees,
        project_name=anno_key,
    )

    return {"dataset_name": dataset_or_view.name, "anno_key": anno_key}


def load_annotations(dataset_name, anno_key, cleanup=False,
                     url="http://119.23.212.113:6060", username="hejian", password="LFIcvat123"):
    dataset = fo.load_dataset(dataset_name)

    dataset.load_annotations(
        anno_key,
        cleanup=cleanup,
        url=url,
        username=username,
        password=password,
    )

    return dataset
