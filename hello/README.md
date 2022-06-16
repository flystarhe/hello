# Hello
A collection of useful tools!

```sh
pip install hello2
```

## hello.data
`python -m hello.data coco2yolo -h`
```log
COCO to YOLO

positional arguments:
  coco_dir              dataset root dir

optional arguments:
  -h, --help            show this help message and exit
  -j JSON_DIR, --json_dir JSON_DIR
                        coco json file dir
  --classes CLASSES [CLASSES ...]
                        filter by class: --classes c0 c2 c3
```

## hello.fiftyone
```python
import fiftyone.zoo as foz
from hello.fiftyone.utils import *

# Download and load COCO-2017
label_types = ("detections", "segmentations")
dataset = foz.load_zoo_dataset("coco-2017", label_types=label_types, classes=["cat", "dog"])  # max_samples=100
print(dataset)

# clone sample field
dataset = clone_sample_field(dataset, "detections", "ground_truth")
print(dataset)

# map labels
mapping = {"cat": "CAT", "dog": "DOG", "person": "PERSON", "*": "OTHER"}
cat_dog_person = map_labels(dataset, mapping, "ground_truth")
print(cat_dog_person.count_values("ground_truth.detections.label"))

# filter samples
classes = ["CAT", "DOG", "PERSON"]
cat_dog_person = filter_samples(cat_dog_person, classes, field_name="ground_truth")
print(cat_dog_person.count_values("ground_truth.detections.label"))

# filter labels
classes = ["CAT", "DOG", "PERSON"]
cat_dog_person = filter_labels(cat_dog_person, classes, field_name="ground_truth")
print(cat_dog_person.count_values("ground_truth.detections.label"))

# merge datasets
classes = ["CAT", "DOG", "PERSON", "OTHER"]
info = {"dataset_name": "COCO 2017",
        "version": "1.0"}
big_dataset = merge_datasets("big", classes, info, [cat_dog_person])
print(big_dataset.count_values("tags"))

# split dataset
splits = {"train": 0.8, "val": 0.1}
split_dataset(big_dataset, splits=splits, limit=200, field_name="ground_truth")
print(big_dataset.count_values("tags"))

# export dataset
results = export_dataset("/workspace/tmp/big", big_dataset, "ground_truth")
print(results)

# load dataset
test = load_dataset("/workspace/tmp/big/test", label_field="ground_truth")
print(test.count_values("tags"))
```
