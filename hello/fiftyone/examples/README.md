# FiftyOne

## Detection
```sh
python -m hello.fiftyone <command> [options]

# hello/fiftyone/evaluate_detections.py
python -m hello.fiftyone eval.det --preds test/faster-rcnn-resnet50-fpn-coco.txt --labels test/ground_truth.json --out test/reports --mAP
```
