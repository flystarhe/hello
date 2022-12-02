import fiftyone as fo
import fiftyone.zoo as foz


def get_dataset(name="_test_det", label_types=["detections"]):
    """Loads the dataset of the given name from the FiftyOne Dataset Zoo.

    Args:
        name (str, optional): a str. Defaults to "_test_det".
        label_types (list, optional): optional values are in ``{'detections', 'segmentations'}``. Defaults to None.

    Returns:
        a :class:`fiftyone.core.dataset.Dataset`
    """
    if label_types is None:
        label_types = ["detections"]

    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
    else:
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            splits=["validation"],
            dataset_name="_test_coco2017",
            label_types=label_types,
            max_samples=50,
        )

    dataset.name = name
    dataset.persistent = True

    return dataset
