import cv2 as cv
import numpy as np
from fiftyone import ViewField as F


def imshow(img):
    from IPython.display import display
    from PIL import Image
    display(Image.fromarray(img, mode="RGB"))


def from_coco_object(dataset, classes=None, field_name="ground_truth", crop_size=(200, 200)):
    if classes is None:
        classes = dataset.distinct(f"{field_name}.detections.label")

    for index, label in enumerate(classes):
        view = dataset.filter_labels(field_name, F("label") == label)
        if len(view) < 50:
            print(f"## id={index}, name='{label}'\n- skip")
            continue

        patches = []
        for sample in view:
            img = cv.imread(sample.filepath, 1)

            img_h, img_w, _ = img.shape
            assert sample.metadata["width"] == img_w
            assert sample.metadata["height"] == img_h

            for obj in sample[field_name]["detections"]:
                if obj.label == label:
                    _box = obj.bounding_box  # [x, y, w, h] / s
                    x, y = round(_box[0] * img_w), round(_box[1] * img_h)
                    w, h = round(_box[2] * img_w), round(_box[3] * img_h)

                    if w > 64 or h > 64:
                        patch = cv.resize(img[y:y + h, x:x + w], crop_size, interpolation=cv.INTER_LINEAR)
                        patches.append(patch)
                        break

            if len(patches) == 5:
                break

        newimg = np.concatenate(patches, axis=1)
        print(f"## id={index}, name='{label}'")
        imshow(newimg[..., ::-1])


def from_coco_instance(dataset, classes=None, field_name="segmentations", crop_size=(200, 200)):
    if classes is None:
        classes = dataset.distinct(f"{field_name}.detections.label")

    for index, label in enumerate(classes):
        view = dataset.filter_labels(field_name, F("label") == label)
        if len(view) < 50:
            print(f"## id={index}, name='{label}'\n- skip")
            continue

        patches = []
        for sample in view:
            img = cv.imread(sample.filepath, 1)

            img_h, img_w, _ = img.shape
            assert sample.metadata["width"] == img_w
            assert sample.metadata["height"] == img_h

            for obj in sample[field_name]["detections"]:
                if obj.label == label:
                    _box = obj.bounding_box  # [x, y, w, h] / s
                    x, y = round(_box[0] * img_w), round(_box[1] * img_h)
                    w, h = round(_box[2] * img_w), round(_box[3] * img_h)

                    if w > 64 or h > 64:
                        mask = obj.mask.astype("uint8") * 255
                        mask = np.stack((mask, mask, mask), axis=-1)
                        patch_mask = cv.resize(mask[:h, :w], crop_size, interpolation=cv.INTER_NEAREST)
                        patch = cv.resize(img[y:y + h, x:x + w], crop_size, interpolation=cv.INTER_LINEAR)
                        patches.append(np.concatenate((patch, patch_mask), axis=0))
                        break

            if len(patches) == 5:
                break

        newimg = np.concatenate(patches, axis=1)
        print(f"## id={index}, name='{label}'")
        imshow(newimg[..., ::-1])
