# https://voxel51.com/docs/fiftyone/tutorials/evaluate_detections.html
# https://voxel51.com/docs/fiftyone/user_guide/evaluation.html


def evaluate_detections(dataset_dir, predictions="predictions.json", ground_truth="labels.json", eval_key="eval", classes=None, method="coco", iou=0.5, classwise=True, compute_mAP=False):
    dataset = None

    results = dataset.evaluate_detections(
        "predictions",
        gt_field="ground_truth",
        eval_key=eval_key,
        classes=classes,
        method=method,
        iou=iou,
        classwise=classwise,
        compute_mAP=compute_mAP,
    )

    results.print_report(classes=classes)

    return results


def evaluate_segmentations(dataset_dir, predictions="predictions/", ground_truth="labels/", eval_key="eval_simple", mask_targets=None, method="simple", bandwidth=None, average="micro"):
    dataset = None

    results = dataset.evaluate_segmentations(
        "predictions",
        gt_field="ground_truth",
        eval_key=eval_key,
        mask_targets=mask_targets,
        method=method,
        bandwidth=bandwidth,
        average=average,
    )

    results.print_report()

    return results
