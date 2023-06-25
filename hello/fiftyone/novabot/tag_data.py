# %%
import hello
import hello.fiftyone.annotate as hoa
import hello.fiftyone.core as hoc
import hello.fiftyone.dataset as hod
import hello.fiftyone.view as hov

print(hello.__version__)

# %%
dataset_name = "novabot_front_det_20230525_zhengshu_batch02_object9_ver002a"
dataset_type = "unknown"
version = "001"
classes = ["nothing", "anything", "ok", "ng"]
mask_targets = {}

hod.delete_datasets([dataset_name], non_persistent=False)
dataset = hod.create_dataset(dataset_name, dataset_type, version, classes, mask_targets)

from_dir = "/workspace/users/hejian/todo/novabot_front_det_20230525_zhengshu_batch02_object9_ver002/nothing"
hod.add_images_dir(dataset, f"{from_dir}/data", "train")

print("count-images:", dataset.count("filepath"))

# %%
view = dataset  # 多个Tag默认取最新的

hoa.to_cvat(
    "det_20230525_nothing_anything", view,
    label_field="ground_truth",
    label_type="classification",
    url="http://192.168.0.119:8080",
    username="hejian", password="LFIcvat123",
    task_size=1000, segment_size=200, task_assignee="hejian", job_assignees=["hejian"])

# %%
_dataset_name = "novabot_front_det_20230525_zhengshu_batch02_object9_ver002a"
anno_keys = ["det_20230525_nothing_anything"]

dataset = hoa.from_cvat(
    _dataset_name, anno_keys,
    cleanup=False,
    url="http://192.168.0.119:8080",
    username="hejian", password="LFIcvat123")

ret = hoc.count_values(dataset, "ground_truth.label", "count")

# %%
classes = ["nothing", "anything"]

dataset.untag_samples(classes)
for label in classes:
    hov.filter_labels(dataset, "ground_truth", f"F('label') == '{label}'").tag_samples(label)
ret = hoc.count_values(dataset, "tags", "count")

hod.export_image_dataset(f"exports/{dataset_name}", dataset, splits=classes)

# %%
classes = ["train", "val", "test"]

dataset.untag_samples(classes)
view = hov.filter_labels(dataset, "ground_truth", "F('label') == 'ok'")
hoc.random_split(view, splits={"val": 0.1, "train": 0.9}, seed=51)

hod.export_image_dataset(f"exports/{dataset_name}", dataset, splits=classes)
