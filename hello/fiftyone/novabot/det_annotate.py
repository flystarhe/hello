# %%
import fiftyone as fo
from fiftyone import ViewField as F

import hello
import hello.fiftyone.annotate as hoa
import hello.fiftyone.coco as hoco
import hello.fiftyone.core as hoc
import hello.fiftyone.dataset as hod

print(hello.__version__)

# %%
dataset_name = "novabot_front_det_20230314_zhengshu_batch01_object9_ver004a"
dataset_type = "detection"
version = "object9"
classes = ["person", "animal", "shoes", "wheel", "other obstacle", "obstacle", "leaf debris", "faeces", "rock", "charging station", "background"]
mask_targets = {}

hod.delete_datasets([dataset_name], non_persistent=False)
dataset = hod.create_dataset(dataset_name, dataset_type, version, classes, mask_targets)

# %%
label_classes = dataset.default_classes

from_dir = "/workspace/users/hejian/todo/novabot_front_det_20230314_zhengshu_batch01_object9_ver004/train"
hod.add_images_dir(dataset, f"{from_dir}/data", "train")

from_dir = "/workspace/users/hejian/todo/novabot_front_det_20230314_zhengshu_batch01_object9_ver004/train"
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
label_classes = dataset.default_classes

from_dir = ""
hod.add_detection_labels(dataset, "ground_truth_iter", f"{from_dir}/labels.json", label_classes, mode="coco")

ret = hoc.count_values(dataset, "ground_truth_iter.detections.label")
print("count-images:", dataset.count("filepath"))

# %%
dataset.clone_sample_field("ground_truth", "ground_truth_iter")

# %%
bbox_area = (
    F("$metadata.width") * F("bounding_box")[2] *
    F("$metadata.height") * F("bounding_box")[3]
)
view = dataset.filter_labels(
    "ground_truth", (512 <= bbox_area), only_matches=False
)

print("count-labels:", view.count("ground_truth.detections"))

# %%
dataset = view.clone()
dataset.name

# %%
view = dataset

hoa.to_cvat(
    "novabot_front_det_20230314_iter", view,
    label_field="ground_truth",
    label_type="detections",
    url="http://192.168.0.119:8080",
    username="hejian", password="LFIcvat123",
    task_size=1000, segment_size=200, task_assignee="hejian", job_assignees=["hejian"])

# %%
_dataset_name = "2023.05.31.16.21.28"
anno_keys = ["novabot_front_det_20230314_iter"]

dataset = hoa.from_cvat(
    _dataset_name, anno_keys,
    cleanup=False,
    url="http://192.168.0.119:8080",
    username="hejian", password="LFIcvat123")
ret = hoc.count_values(dataset, "ground_truth.detections.label")
print("count-images:", dataset.count("filepath"))

# %%
results = dataset.evaluate_detections(
    "ground_truth",
    gt_field="ground_truth_iter",
    eval_key="eval",
)

# %%
dataset.untag_samples("ng")
view = dataset.match((F("eval_fp") > 0) | (F("eval_fn") > 0))
print("length:", view.count("filepath"))

view.untag_samples("train")
view.tag_samples("issue")

view = view.filter_labels(
    "ground_truth_iter", F("eval") != "tp", only_matches=False
).filter_labels(
    "ground_truth", F("eval") != "tp", only_matches=False
).match(
    (F("ground_truth_iter.detections").length() > 0) | (F("ground_truth.detections").length() > 0)
)
print("count-labels-old:", view.count("ground_truth_iter.detections"))
print("count-labels-new:", view.count("ground_truth.detections"))

session = fo.launch_app(view, port=20002, address="192.168.0.119", auto=False)  # tag sample for `ng`

# %%
view.untag_samples("issue")
view.tag_samples("train")

view.match_tags("ng").untag_samples("train")
view.match_tags("ng").tag_samples("issue")

# %%
view = dataset.match_tags("issue")

hoa.to_cvat(
    "novabot_front_det_20230314_iter2", view,
    label_field="ground_truth",
    label_type="detections",
    url="http://192.168.0.119:8080",
    username="hejian", password="LFIcvat123",
    task_size=1000, segment_size=200, task_assignee="hejian", job_assignees=["hejian"])

# %%
_dataset_name = ""
anno_keys = [""]

dataset = hoa.from_cvat(
    _dataset_name, anno_keys,
    cleanup=False,
    url="http://192.168.0.119:8080",
    username="hejian", password="LFIcvat123")
ret = hoc.count_values(dataset, "ground_truth.detections.label")
print("count-images:", dataset.count("filepath"))

# %%
dataset.untag_samples("ok")
view = dataset.match_tags("issue")
session = fo.launch_app(view, port=20003, address="192.168.0.119", auto=False)  # tag sample for `ok`

# %%
dataset.match_tags("ok").untag_samples("issue")
dataset.match_tags("ok").tag_samples("train")
ret = hoc.count_values(dataset, "tags")

# %%
dataset.untag_samples("ng")
view = dataset.match_tags("train")
session = fo.launch_app(view, port=20004, address="192.168.0.119", auto=False)  # tag sample for `ng`

# %%
dataset.match_tags("ng").untag_samples("train")
dataset.match_tags("ng").tag_samples("issue")
ret = hoc.count_values(dataset, "tags")

# %% [markdown]
# end
#
# ---

# %%
hoco.coco_export(f"exports/{dataset_name}", dataset, label_field="ground_truth", splits=["train", "val", "issue"])

# %%
hoc.random_split(dataset.match_tags("train"), splits={"val": 0.1, "train": 0.9}, seed=51)
hoco.coco_export(f"exports/{dataset_name}", dataset, label_field="ground_truth", splits=["train", "val", "issue"])
