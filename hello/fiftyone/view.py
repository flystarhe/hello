from pathlib import Path

import fiftyone.brain as fob
import fiftyone.utils.iou as foui
from fiftyone import ViewField as F


def uniqueness(dataset, count, model=None):
    """The uniqueness of a Dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        count (int, float): specific int value or percentage
        model (None): a :class:`fiftyone.core.models.Model` or the name of a model

    Examples::

        >>> model: 'mobilenet-v2-imagenet-torch'
        >>> model: 'resnet50-imagenet-torch', 'resnet101-imagenet-torch', 'resnet152-imagenet-torch'
        >>> model: 'resnext50-32x4d-imagenet-torch', 'resnext101-32x8d-imagenet-torch'

    Returns:
        a :class:`DatasetView`
    """
    num_samples = len(dataset)
    assert num_samples > 0

    if isinstance(count, float):
        count = int(count * num_samples)

    count = min(count, num_samples)

    if model is None:
        view = dataset.sort_by("filepath")

        step = num_samples // count
        sample_ids = view.values("id")
        sample_ids = sample_ids[::step]

        unique_view = view.select(sample_ids, ordered=True)
        return unique_view.limit(count)

    results = fob.compute_similarity(dataset, brain_key="img_sim", model=model)
    results.find_unique(count)

    unique_ids = results.unique_ids
    unique_view = dataset.select(unique_ids)

    return unique_view.limit(count)


def labeled(dataset, field_name="ground_truth"):
    view = dataset.exists(field_name, bool=True)
    return view


def unlabeled(dataset, field_name="ground_truth"):
    view = dataset.exists(field_name, bool=False)
    return view


def match_tags(dataset, tags, bool=None):
    """Returns a view containing the samples in the collection that have (or do not have) any of the given tag(s).

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        tags: the tag or iterable of tags to match
        bool (None): whether to match samples that have (None or True) or do not have (False) the given tags

    Returns:
        a :class:`DatasetView`
    """
    view = dataset.match_tags(tags, bool)
    return view


def sort_by_filename(dataset):
    filepaths, ids = dataset.values(["filepath", "id"])

    data = [(Path(k).name, v) for k, v in zip(filepaths, ids)]
    data = sorted(data, key=lambda x: x[0])
    sorted_ids = [x[1] for x in data]

    view = dataset.select(sorted_ids, ordered=True)

    return view


def select_labels(dataset, labels=None, ids=None, tags=None, fields=None, omit_empty=True):
    """Selects only the specified labels from the collection.

    Args:
        dataset: a :class:`fiftyone.core.collections.SampleCollection`
        labels (None):  a list of dicts specifying the labels to select in the format returned by :func:`fiftyone.core.session.Session.selected_labels()`
        ids (None): an ID or iterable of IDs of the labels to select
        tags (None): a tag or iterable of tags of labels to select
        fields (None): a field or iterable of fields from which to select
        omit_empty (True): whether to omit samples that have no labels after filtering
    """
    view = dataset.select_labels(labels=labels, ids=ids, tags=tags, fields=fields, omit_empty=omit_empty)
    return view


def exclude_labels(dataset, labels=None, ids=None, tags=None, fields=None, omit_empty=True):
    """Excludes the specified labels from the collection.

    Args:
        dataset: a :class:`fiftyone.core.collections.SampleCollection`
        labels (None):  a list of dicts specifying the labels to exclude in the format returned by :func:`fiftyone.core.session.Session.selected_labels()`
        ids (None): an ID or iterable of IDs of the labels to exclude
        tags (None): a tag or iterable of tags of labels to exclude
        fields (None): a field or iterable of fields from which to exclude
        omit_empty (True): whether to omit samples that have no labels after filtering
    """
    view = dataset.exclude_labels(labels=labels, ids=ids, tags=tags, fields=fields, omit_empty=omit_empty)
    return view


def filter_field(dataset, field, a, op, b, only_matches=True):
    """Filters the values of a field or embedded field of each sample in the collection.

    Args:
        dataset: a :class:`fiftyone.core.collections.SampleCollection`
        field: the label field to filter
        only_matches (True): whether to only include samples with at least one label after filtering (True) or include all samples (False)
    """
    a = F() if a is None else F(a)

    if op == "eq":
        view = dataset.filter_field(field, a == b, only_matches)
    elif op == "ge":
        view = dataset.filter_field(field, a >= b, only_matches)
    elif op == "gt":
        view = dataset.filter_field(field, a > b, only_matches)
    elif op == "le":
        view = dataset.filter_field(field, a <= b, only_matches)
    elif op == "lt":
        view = dataset.filter_field(field, a < b, only_matches)
    elif op == "ne":
        view = dataset.filter_field(field, a != b, only_matches)
    else:
        view = dataset.filter_field(field, getattr(a, op)(b), only_matches)

    return view


def filter_labels(dataset, field, expression, only_matches=True):
    """Filters the :class:`fiftyone.core.labels.Label` field of each sample in the collection.

    >>> from fiftyone import ViewField as F
    >>> filter_labels(dataset, "ground_truth", F("label") == "house")
    >>> filter_labels(dataset, "ground_truth", F("label").is_in(["cat", "dog"]))
    >>> filter_labels(dataset, "predictions", F("confidence") > 0.8)

    Args:
        dataset: a :class:`fiftyone.core.collections.SampleCollection`
        field: the label field to filter
        expression: a :class:`fiftyone.core.expressions.ViewExpression`
        only_matches (True): whether to only include samples with at least one label after filtering (True) or include all samples (False)
    """
    if isinstance(expression, str):
        expression = eval(expression)

    view = dataset.filter_labels(field, expression, only_matches=only_matches)
    return view


def filter_samples(dataset, expression):
    """Filters the samples in the collection by the given filter.

    >>> from fiftyone import ViewField as F
    >>> filter_samples(dataset, F("filepath").ends_with(".jpg"))
    >>> filter_samples(dataset, F("predictions.detections").length() >= 2)
    >>> # Only include samples whose `predictions` field contains at least
    >>> # one object with area smaller than 0.2
    >>> bbox = F("bounding_box")
    >>> bbox_area = bbox[2] * bbox[3]
    >>> small_boxes = F("predictions.detections").filter(bbox_area < 0.2)
    >>> view = dataset.match(small_boxes.length() > 0)

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        expression: a :class:`fiftyone.core.expressions.ViewExpression`
    """
    if isinstance(expression, str):
        expression = eval(expression)

    view = dataset.match(expression)
    return view


def filter_duplicate_labels(dataset, label_field, iou_thresh=0.999, method="simple", iscrowd=None, classwise=False):
    """Delete duplicate labels in the given field of the dataset, as defined as labels with an IoU greater than a chosen threshold with another label in the field.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        label_field: a label field of type :class:`fiftyone.core.labels.Detections` or :class:`fiftyone.core.labels.Polylines`
        iou_thresh (0.999): the IoU threshold to use to determine whether labels are duplicates
        method ("simple"): supported values are ``("simple", "greedy")``
        iscrowd (None): an optional name of a boolean attribute
        classwise (False): different label values as always non-overlapping
    """
    dup_ids = foui.find_duplicates(dataset, label_field, iou_thresh=iou_thresh, method=method, iscrowd=iscrowd, classwise=classwise)

    view = dataset.match_labels(ids=dup_ids, fields=label_field)
    return view
