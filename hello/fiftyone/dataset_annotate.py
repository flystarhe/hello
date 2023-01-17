import fiftyone as fo
import fiftyone.core.view as fov


def annotate(batch, samples, label_field="ground_truth", label_type="instances",
             url="http://119.23.212.113:6060", username="hejian", password="LFIcvat123",
             task_size=1500, segment_size=500, task_assignee="hejian", job_assignees=["weiqiaomu", "jiasiyu"]):
    """Exports the samples to the given annotation backend.

    ``mask_targets`` is a dict mapping pixel values to semantic label strings.
    Only applicable when annotating semantic segmentations. the default is
    ``samples.default_mask_targets``.

    Args:
        batch: a name of str or index of int for samples
        samples: a :class:`fiftyone.core.collections.SampleCollection`
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
        task_size (int, optional): defaults to 1500
        segment_size (int, optional): defaults to 500
        task_assignee (str, optional): defaults to "hejian"
        job_assignees (list, optional): defaults to ``["weiqiaomu", "jiasiyu"]``
    """
    assert label_type in {"classification", "classifications", "detections", "instances", "polylines", "polygons", "keypoints", "segmentation", "scalar"}

    if isinstance(samples, fov.DatasetView):
        dataset_name = samples.dataset_name
    else:
        dataset_name = samples.name

    anno_key = f"{dataset_name}_{label_field}_{label_type}_{batch}"

    if anno_key in samples.list_annotation_runs():
        samples.delete_annotation_run(anno_key)

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

    samples.annotate(
        anno_key,
        label_field=label_field,
        label_type=label_type,
        classes=samples.default_classes or None,
        attributes=attributes,
        mask_targets=samples.default_mask_targets or None,
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

    return anno_key


def load_annotations(dataset_name, anno_keys, cleanup=True,
                     url="http://119.23.212.113:6060", username="hejian", password="LFIcvat123"):
    dataset = fo.load_dataset(dataset_name)

    assert isinstance(anno_keys, list)

    for anno_key in anno_keys:
        dataset.load_annotations(
            anno_key,
            cleanup=cleanup,
            url=url,
            username=username,
            password=password
        )

    return dataset
