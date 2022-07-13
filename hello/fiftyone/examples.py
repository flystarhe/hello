import fiftyone as fo
import fiftyone.zoo as foz
from hello.fiftyone.utils import *

# download and load COCO
label_types = ["detections", "segmentations"]
classes = ["cat", "dog"]
dataset = foz.load_zoo_dataset("coco-2017",
                               label_types=label_types,
                               classes=classes,
                               max_samples=30)
print(dataset)

# cat + dog
cat_dog = dataset.select_fields("segmentations").clone()
cat_dog = clone_sample_field(cat_dog, "segmentations", "ground_truth")
classes = ["cat", "dog"]
mask_targets = {127: "cat", 255: "dog"}
cat_dog.default_classes = classes
cat_dog.default_mask_targets = mask_targets
cat_dog.info = {
    "dataset_name": "cat and dog",
    "version": "0.01",
    "classes": str(classes),
    "mask_targets": str(mask_targets),
}
cat_dog.save()  # when `Dataset.info` is modified
print(cat_dog.count_values("tags"))
print(cat_dog.count_values("ground_truth.detections.label"))

# map labels
mapping = {"cat": "cat", "dog": "dog", "*": "other"}
cat_dog = map_labels(cat_dog, mapping, field_name="ground_truth")
print(cat_dog.count_values("tags"))
print(cat_dog.count_values("ground_truth.detections.label"))

# filter samples
classes = ["cat", "dog"]
cat_dog = filter_samples(cat_dog, classes, field_name="ground_truth")
print(cat_dog.count_values("tags"))
print(cat_dog.count_values("ground_truth.detections.label"))

# merge datasets
classes = ["cat", "dog", "other"]
mask_targets = {70: "cat", 120: "dog", 255: "other"}
info = {
    "dataset_name": "cat and dog",
    "version": "0.01",
    "classes": str(classes),
    "mask_targets": str(mask_targets),
}
big_dataset = merge_datasets("big", classes, mask_targets, info, [cat_dog])
print(big_dataset.count_values("tags"))
print(big_dataset.count_values("ground_truth.detections.label"))

# split dataset
splits = {"val": 0.1, "train": 0.9}
big_dataset = split_dataset(big_dataset, splits=splits,
                            limit=200, field_name="ground_truth")
print(big_dataset.count_values("tags"))
print(big_dataset.count_values("ground_truth.detections.label"))

# export dataset
results = export_dataset("tmp/examples/big", big_dataset,
                         label_field="ground_truth", mask_label_field="ground_truth")
print(results)

# load dataset
data = load_dataset("tmp/examples/big/train")
print(data.count_values("tags"))
session = fo.launch_app(data)
