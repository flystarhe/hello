# %%
import fiftyone as fo

import hello
import hello.fiftyone.core as hoc
import hello.fiftyone.dataset as hod

print(hello.__version__)

# %%
dataset_name = "novabot_front_seg_20230404_test_640_suzhou_mall_ver001"
dataset_type = "segmentation"
version = "map9"
classes = ["unlabeled", "background", "lawn", "road", "terrain", "obstacle", "leaf debris", "faeces", "unknown", "charging station", "ignore"]
mask_targets = {0: "unlabeled", 1: "background", 2: "lawn", 3: "road", 4: "terrain", 5: "obstacle", 6: "leaf debris", 7: "faeces", 8: "unknown", 9: "charging station", 255: "ignore"}

hod.delete_datasets([dataset_name], non_persistent=False)
dataset = hod.create_dataset(dataset_name, dataset_type, version, classes, mask_targets)

# %%
label_mask_targets = dataset.default_mask_targets

from_dir = "/workspace/users/hejian/tmp/Segmentation/train"
hod.add_images_dir(dataset, f"{from_dir}/data", "train")

from_dir = "/workspace/users/hejian/tmp/Segmentation/train"
hod.add_segmentation_labels(dataset, "ground_truth", f"{from_dir}/labels", label_mask_targets, mode="png")

# %%
new_classes = None

dataset = hoc.remap_segmentation_dataset(dataset, new_classes, "ground_truth", ignore_index=255, least_one=True)
print(f"{dataset.default_classes=}\n{dataset.default_mask_targets=}")
print("count-images:", dataset.count("filepath"))

# %%
session = fo.launch_app(dataset, port=20006, address="192.168.0.119", auto=False)

# %%
ret = hoc.count_values(dataset, "tags")

# %%
hoc.random_split(dataset, splits={"val": 0.1, "train": 0.9}, seed=51)
hod.export_dataset(f"exports/{dataset_name}", dataset, mask_label_field="ground_truth", splits=["train", "val", "test"])
