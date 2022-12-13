from pathlib import Path

import fiftyone.brain as fob


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

    p = count if isinstance(count, float) else count / num_samples

    p = min(1.0, p)

    if model is None:
        view = dataset.sort_by("filepath")

        step = int(1.01 / p)
        sample_ids = view.values("id")
        sample_ids = sample_ids[::step]

        view = view.select(sample_ids, ordered=True)
        return view

    results = fob.compute_similarity(dataset, brain_key="img_sim", model=model)
    results.find_unique(int(num_samples * p))

    unique_view = dataset.select(results.unique_ids)
    return unique_view


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


def filter_labels(dataset, field, filter, only_matches=True):
    """Filters the :class:`fiftyone.core.labels.Label` field of each sample in the collection.

    >>> from fiftyone import ViewField as F
    >>> filter_labels(dataset, "ground_truth", F("label") == "house")
    >>> filter_labels(dataset, "ground_truth", F("label").is_in(["cat", "dog"]))
    >>> filter_labels(dataset, "predictions", F("confidence") > 0.8)

    Args:
        dataset: a :class:`fiftyone.core.collections.SampleCollection`
        field: the label field to filter
        filter: a :class:`fiftyone.core.expressions.ViewExpression`
        only_matches (True): whether to only include samples with at least one label after filtering (True) or include all samples (False)
    """
    view = dataset.filter_labels(field, filter, only_matches=only_matches)
    return view


def filter_samples(dataset, filter):
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
        filter: a :class:`fiftyone.core.expressions.ViewExpression`
    """
    view = dataset.match(filter)
    return view


def add_sample_field(dataset, field_name, ftype):
    """Adds a new sample field or embedded field to the dataset, if necessary.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        field_name: the field name or `embedded.field.name`
        ftype: the field type to create. Must be a subclass of :class:`fiftyone.core.fields.Field`
    """
    dataset.add_sample_field(field_name, ftype)


def clear_sample_field(dataset, field_name):
    """Clears the values of the field from all samples in the dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        field_name: the field name or `embedded.field.name`
    """
    dataset.clear_sample_field(field_name)


def clone_sample_field(dataset, field_name, new_field_name):
    """Clones the given sample field into a new field of the dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        field_name: the field name or `embedded.field.name`
        new_field_name: the new field name or `embedded.field.name`
    """
    dataset.clone_sample_field(field_name, new_field_name)


def delete_sample_field(dataset, field_name, error_level=0):
    """Deletes the field from all samples in the dataset.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        field_name: the field name or `embedded.field.name`
        error_level (int, optional): the error level to use
    """
    dataset.delete_sample_field(field_name, error_level=error_level)


def rename_sample_field(dataset, field_name, new_field_name):
    """Renames the sample field to the given new name.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        field_name: the field name or `embedded.field.name`
        new_field_name: the new field name or `embedded.field.name`
    """
    dataset.rename_sample_field(field_name, new_field_name)


def merge_labels(dataset, in_field, out_field):
    """Merges the labels from the given input field into the given output field of the collection.

    If this collection is a dataset, the input field is deleted after the
    merge.

    If this collection is a view, the input field will still exist on the
    underlying dataset but will only contain the labels not present in this
    view.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        in_field (str): the name of the input label field
        out_field (str): the name of the output label field, which will be created if necessary
    """
    dataset.merge_labels(in_field, out_field)
