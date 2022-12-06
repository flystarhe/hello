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

    if isinstance(count, float):
        count = int(num_samples * count) + 1

    count = min(num_samples, count)

    if model is None:
        view = dataset.sort_by("filepath")

        sample_ids = view.values("id")
        sample_ids = sample_ids[::(num_samples//count)]

        view = view.select(sample_ids, ordered=True)
        return view

    results = fob.compute_similarity(dataset, brain_key="img_sim", model=model)
    results.find_unique(count)

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
