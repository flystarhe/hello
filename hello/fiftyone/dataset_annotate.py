import fiftyone as fo
import fiftyone.core.view as fov


def annotate(dataset_or_view, label_field="ground_truth", label_type="instances",
             url="http://119.23.212.113:6060", username="hejian", password="LFIcvat123",
             task_size=2000, segment_size=1000, task_assignee="hejian", job_assignees=["weiqiaomu", "jiasiyu"]):
    """Exports the samples to the given annotation backend.

    ``mask_targets`` is a dict mapping pixel values to semantic label strings.
    Only applicable when annotating semantic segmentations. the default is
    ``dataset_or_view.default_mask_targets``.

    Args:
        dataset_or_view: a :class:`fiftyone.core.collections.SampleCollection`
        label_field ("ground_truth"): a string indicating a new or existing label field to annotate
        label_type ("instances"): a string indicating the type of labels to annotate. The possible values are:

            -   ``"classification"``: a single classification stored in
                :class:`fiftyone.core.labels.Classification` fields
            -   ``"classifications"``: multilabel classifications stored in
                :class:`fiftyone.core.labels.Classifications` fields
            -   ``"detections"``: object detections stored in
                :class:`fiftyone.core.labels.Detections` fields
            -   ``"instances"``: instance segmentations stored in
                :class:`fiftyone.core.labels.Detections` fields with their
                :attr:`mask <fiftyone.core.labels.Detection.mask>`
                attributes populated
            -   ``"polylines"``: polylines stored in
                :class:`fiftyone.core.labels.Polylines` fields with their
                :attr:`filled <fiftyone.core.labels.Polyline.filled>`
                attributes set to ``False``
            -   ``"polygons"``: polygons stored in
                :class:`fiftyone.core.labels.Polylines` fields with their
                :attr:`filled <fiftyone.core.labels.Polyline.filled>`
                attributes set to ``True``
            -   ``"keypoints"``: keypoints stored in
                :class:`fiftyone.core.labels.Keypoints` fields
            -   ``"segmentation"``: semantic segmentations stored in
                :class:`fiftyone.core.labels.Segmentation` fields
            -   ``"scalar"``: scalar labels stored in
                :class:`fiftyone.core.fields.IntField`,
                :class:`fiftyone.core.fields.FloatField`,
                :class:`fiftyone.core.fields.StringField`, or
                :class:`fiftyone.core.fields.BooleanField` fields

        url (str, optional): defaults to "http://119.23.212.113:6060"
        username (str, optional): defaults to "hejian"
        password (str, optional): defaults to "LFIcvat123"
        task_size (int, optional): defaults to 2000
        segment_size (int, optional): defaults to 1000
        task_assignee (str, optional): defaults to "hejian"
        job_assignees (list, optional): defaults to ``["weiqiaomu", "jiasiyu"]``
    """
    assert label_type in {"classification", "classifications", "detections", "instances", "polylines", "polygons", "keypoints", "segmentation", "scalar"}

    if isinstance(dataset_or_view, fov.DatasetView):
        dataset_name = dataset_or_view.dataset_name
    else:
        dataset_name = dataset_or_view.name

    anno_key = f"{dataset_name}_{label_field}_{label_type}"

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
        task_size=task_size,
        segment_size=segment_size,
        image_quality=95,
        task_assignee=task_assignee,
        job_assignees=job_assignees,
        project_name=anno_key,
    )

    return {"dataset_name": dataset_name, "anno_key": anno_key}


def load_annotations(dataset_name, anno_key, cleanup=True,
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
