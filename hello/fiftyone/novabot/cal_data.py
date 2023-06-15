# %%
import hello
import hello.fiftyone.dataset as hod
import hello.fiftyone.tarinfo as hot
import hello.fiftyone.view as hov

print(hello.__version__)

# %%
dataset_name = "novabot_front_cal_20230606_charging_station_ver001"
dataset_type = "unknown"
version = "001"
classes = []
mask_targets = {}

hod.delete_datasets([dataset_name], non_persistent=False)
dataset = hod.create_dataset(dataset_name, dataset_type, version, classes, mask_targets)

tar_files = [
    "/workspace/users/hejian/tmp/novabot_front_det_20230404_big_train_object9_ver001.tar",
]
from_dir = hot.extract_images(f"tmp/{dataset_name}", tar_files, data_path="data")
hod.add_images_dir(dataset, f"{from_dir}/data", "cal")

print("count-images:", dataset.count("filepath"))

# %%
view = hov.uniqueness(dataset, 100, model="clip-vit-base32-torch")

print("count-images:", view.count("filepath"))

# %%
hod.export_image_dataset(f"exports/{dataset_name}", view, splits="auto")
