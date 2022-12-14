import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.core.view as fov
from fiftyone import ViewField as F


def to_patches(samples, field, **kwargs):
    """Creates a view that contains one sample per object patch in the
    specified field of the collection.

    Fields other than ``field`` and the default sample fields will not be
    included in the returned view. A ``sample_id`` field will be added that
    records the sample ID from which each patch was taken.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        field: the patches field, which must be of type
            :class:`fiftyone.core.labels.Detections` or
            :class:`fiftyone.core.labels.Polylines`

    Returns:
        a :class:`fiftyone.core.patches.PatchesView`
    """
    view = samples.to_patches(field, **kwargs)
    return view


def mistakenness_views(samples, pred_field="predictions", pred_filter=None, label_field="ground_truth", label_filter=None):
    """Create a view containing the currently selected objects.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        pred_field ("predictions"): the label field to filter
        pred_filter (None): a :class:`fiftyone.core.expressions.ViewExpression`
        label_field ("ground_truth"): the label field to filter
        label_filter (None): a :class:`fiftyone.core.expressions.ViewExpression`
    """
    if pred_filter is None:
        pred_filter = F("possible_missing")

    if label_filter is None:
        label_filter = (F("mistakenness") > 0.5) | (F("mistakenness_loc") > 0.5) | F("possible_spurious")

    if isinstance(samples, fov.DatasetView):
        dataset_name = samples.dataset_name
    else:
        dataset_name = samples.name

    dataset = fo.load_dataset(dataset_name)

    pred_field_review = "review_preds"
    if dataset.has_sample_field(pred_field_review):
        dataset.delete_sample_field(pred_field_review)

    label_field_review = "review_labels"
    if dataset.has_sample_field(label_field_review):
        dataset.delete_sample_field(label_field_review)

    pred_view = samples.filter_labels(pred_field, pred_filter)
    pred_view.clone_sample_field(pred_field, pred_field_review)

    label_view = samples.filter_labels(label_field, label_filter)
    label_view.clone_sample_field(label_field, label_field_review)

    return pred_view, label_view


def mistakenness_view(samples, min_mistakenness=0.5):
    """Create a view containing the currently selected samples.

    The following sample-level fields:

    -   **(mistakenness)** The mistakenness field, store the mistakenness value.

    -   **(possible_missing)** The missing field, store per-sample counts of potential missing.

    -   **(possible_spurious)** The spurious field, store per-sample counts of potential spurious.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        min_mistakenness (0.5): the mistakenness field minimum value
    """
    filter = (F("mistakenness") > min_mistakenness) | (F("possible_missing") > 0) | (F("possible_spurious") > 0)
    view = samples.match(filter)
    return view


def classification_hardness(samples, label_field, hardness_field="hardness"):
    """Adds a hardness field to each sample scoring the difficulty that the
    specified label field observed in classifying the sample.

    All classifications must have their
    :attr:`logits <fiftyone.core.labels.Classification.logits>` attributes
    populated in order to use this method.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        label_field: the :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` field to use from
            each sample
        hardness_field ("hardness"): the field name to use to store the
            hardness value for each sample
    """
    return fob.compute_hardness(
        samples,
        label_field,
        hardness_field=hardness_field,
    )


def detection_mistakenness(
    samples,
    pred_field,
    label_field="ground_truth",
    mistakenness_field="mistakenness",
    missing_field="possible_missing",
    spurious_field="possible_spurious",
    use_logits=False,
    copy_missing=False,
):
    """Computes the mistakenness of the labels in the specified
    ``label_field``, scoring the chance that the labels are incorrect.

    Mistakenness is computed based on the predictions in the ``pred_field``,
    through either their ``confidence`` or ``logits`` attributes. This measure
    can be used to detect things like annotation errors and unusually hard samples.

    This method supports both classifications and detections/polylines.

    For classifications, a ``mistakenness_field`` field is populated on each
    sample that quantifies the likelihood that the label in the ``label_field``
    of that sample is incorrect.

    For detections/polylines, the mistakenness of each object in
    ``label_field`` is computed. Three types of mistakes are identified:

    -   **(Mistakes)** Objects in ``label_field`` with a match in
        ``pred_field`` are assigned a mistakenness value in their
        ``mistakenness_field`` that captures the likelihood that the class
        label of the detection in ``label_field`` is a mistake. A
        ``mistakenness_field + "_loc"`` field is also populated that captures
        the likelihood that the detection in ``label_field`` is a mistake due
        to its localization (bounding box).

    -   **(Missing)** Objects in ``pred_field`` with no matches in
        ``label_field`` but which are likely to be correct will have their
        ``missing_field`` attribute set to True. In addition, if
        ``copy_missing`` is True, copies of these objects are *added* to the
        ground truth ``label_field``.

    -   **(Spurious)** Objects in ``label_field`` with no matches in
        ``pred_field`` but which are likely to be incorrect will have their
        ``spurious_field`` attribute set to True.

    In addition, for detections/polylines, the following sample-level fields
    are populated:

    -   **(Mistakes)** The ``mistakenness_field`` of each sample is populated
        with the maximum mistakenness of the objects in ``label_field``

    -   **(Missing)** The ``missing_field`` of each sample is populated with
        the number of missing objects that were deemed missing from
        ``label_field``.

    -   **(Spurious)** The ``spurious_field`` of each sample is populated with
        the number of objects in ``label_field`` that were given deemed
        spurious.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        pred_field: the name of the predicted label field to use from each
            sample. Can be of type
            :class:`fiftyone.core.labels.Classification`,
            :class:`fiftyone.core.labels.Classifications`, or
            :class:`fiftyone.core.labels.Detections`
        label_field ("ground_truth"): the name of the "ground truth" label
            field that you want to test for mistakes with respect to the
            predictions in ``pred_field``. Must have the same type as
            ``pred_field``
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
        missing_field ("possible_missing): the field in which to store
            per-sample counts of potential missing detections/polylines
        spurious_field ("possible_spurious): the field in which to store
            per-sample counts of potential spurious detections/polylines
        use_logits (False): whether to use logits (True) or confidence (False)
            to compute mistakenness. Logits typically yield better results,
            when they are available
        copy_missing (False): whether to copy predicted detections/polylines
            that were deemed to be missing into ``label_field``
    """
    return fob.compute_mistakenness(
        samples,
        pred_field,
        label_field=label_field,
        mistakenness_field=mistakenness_field,
        missing_field=missing_field,
        spurious_field=spurious_field,
        use_logits=use_logits,
        copy_missing=copy_missing)
