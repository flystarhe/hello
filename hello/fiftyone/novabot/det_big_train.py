# %%
import fiftyone as fo
from fiftyone import ViewField as F

import hello
import hello.fiftyone.brain as hob
import hello.fiftyone.coco as hoco
import hello.fiftyone.core as hoc
import hello.fiftyone.dataset as hod

print(hello.__version__)

# %%
dataset_name = "novabot_front_det_20230531_big_train_object9_ver003b"
dataset_type = "detection"
version = "object9"
classes = ["person", "animal", "shoes", "wheel", "other obstacle", "obstacle", "leaf debris", "faeces", "rock", "charging station", "background"]
mask_targets = {}

hod.delete_datasets([dataset_name], non_persistent=False)
dataset = hod.create_dataset(dataset_name, dataset_type, version, classes, mask_targets)

# %%
label_classes = dataset.default_classes

from_dir = "/workspace/users/hejian/tmp/novabot_front_det_20230314_zhengshu_batch01_object9_ver003/train"
hod.add_images_dir(dataset, f"{from_dir}/data", "train")
hod.add_detection_labels(dataset, "ground_truth", f"{from_dir}/labels.json", label_classes, mode="coco")

from_dir = "/workspace/users/hejian/tmp/novabot_front_det_20230505_zhengshu_batch02_object9_ver003/train"
hod.add_images_dir(dataset, f"{from_dir}/data", "val")
hod.add_detection_labels(dataset, "ground_truth", f"{from_dir}/labels.json", label_classes, mode="coco")

ret = hoc.count_values(dataset, "ground_truth.detections.label")
print("count-images:", dataset.count("filepath"))

# %%
dataset = dataset.filter_labels("ground_truth", ~F("label").is_in(["leaf debris"]), only_matches=False)
print("count-labels:", dataset.count("ground_truth.detections"))

# %%
new_classes = [
    ["person"],
    ["animal"],
    ["shoes"],
    ["wheel"],
    ["other obstacle", "obstacle"],
    ["leaf debris"],
    ["faeces"],
    ["rock"],
    ["charging station"],
    ["background"],
]

dataset = hoc.remap_detections_dataset(dataset, new_classes, "ground_truth", background="background", least_one=True)
print(f"{dataset.default_classes=}\n{dataset.default_mask_targets=}")
print("count-labels:", dataset.count("ground_truth.detections"))
print("count-images:", dataset.count("filepath"))

# %%
dataset.default_classes = dataset.default_classes[:-1]
print(f"{dataset.default_classes=}\n{dataset.default_mask_targets=}")

# %%
hod.delete_duplicate_labels(dataset, "ground_truth", iou_thresh=0.99, method="simple", iscrowd=None, classwise=False)
print("count-labels:", dataset.count("ground_truth.detections"))

# %% [markdown]
# ---
# start

# %%
label_ids = []
range_a, range_b = 0, 64
for sample_detections in dataset.values("ground_truth.detections"):
    for detection in sample_detections:
        x, y, w, h = detection.bounding_box
        area = w * 1920 * h * 1080
        if range_a <= area < range_b:
            label_ids.append(detection.id)
print(f"[{range_a}, {range_b}]: {len(label_ids)=}")
ret = hoc.count_values(dataset, "ground_truth.detections.tags")

# %%
dataset.select_labels(ids=label_ids).tag_labels("ignore")
ret = hoc.count_values(dataset, "ground_truth.detections.tags")

# %%
label_ids = []
range_a, range_b = 64, 16 * 16 * 9
for sample_detections in dataset.values("ground_truth.detections"):
    for detection in sample_detections:
        x, y, w, h = detection.bounding_box
        area = w * 1920 * h * 1080
        if range_a <= area < range_b:
            label_ids.append(detection.id)
print(f"[{range_a}, {range_b}]: {len(label_ids)=}")
ret = hoc.count_values(dataset, "ground_truth.detections.iscrowd")

# %%
dataset.set_label_values("ground_truth.detections.iscrowd", {_id: 1 for _id in label_ids})
ret = hoc.count_values(dataset, "ground_truth.detections.iscrowd")

# %%
label_ids = []
range_a, range_b = 16 * 16 * 9, 32 * 32 * 9
for sample_detections in dataset.values("ground_truth.detections"):
    for detection in sample_detections:
        x, y, w, h = detection.bounding_box
        area = w * 1920 * h * 1080
        if range_a <= area < range_b:
            label_ids.append(detection.id)
print(f"[{range_a}, {range_b}]: {len(label_ids)=}")
ret = hoc.count_values(dataset, "ground_truth.detections.tags")

# %%
dataset.untag_labels("todo")
dataset.tag_labels("train")
dataset.select_labels(ids=label_ids).tag_labels("todo")
dataset.select_labels(tags=["todo"]).untag_labels("train")
ret = hoc.count_values(dataset, "ground_truth.detections.tags")

# %%
hob.compute_similarity(dataset, "ground_truth", "gt_sim", "clip-vit-base32-torch")

# %%
patches = hob.patches_view(dataset, "ground_truth")
session = fo.launch_app(view=patches, port=20001, address="192.168.0.119", auto=False)  # tag label for `issue/ignore`

# %%
dataset.untag_samples("issue")
dataset.select_labels(tags="issue", omit_empty=True).tag_samples("issue")
dataset.match_tags("issue").untag_samples(["train", "val"])
ret = hoc.count_values(dataset, "tags")

# %%
dataset = dataset.exclude_labels(tags="ignore", omit_empty=False)
ret = hoc.count_values(dataset, "ground_truth.detections.label")
print("count-images:", dataset.count("filepath"))

# %% [markdown]
# end
#
# ---

# %%
hoco.coco_export(f"exports/{dataset_name}", dataset, label_field="ground_truth", splits=["train", "val", "issue"])

# %%
hoc.random_split(dataset.match_tags("train"), splits={"val": 0.1, "train": 0.9}, seed=51)
hoco.coco_export(f"exports/{dataset_name}", dataset, label_field="ground_truth", splits=["train", "val", "issue"])
