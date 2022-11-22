from pathlib import Path


def _parse_text_slice(vals):
    assert len(vals) == 6
    x1, y1, x2, y2 = vals[:4]

    bounding_box = [
        float(x1),
        float(y1),
        float(x2) - float(x1),
        float(y2) - float(y1),
    ]

    confidence = float(vals[4])

    label = vals[5]

    return bounding_box, confidence, label


def _parse_text_row(row):
    """\
    row format:
        filepath,height,width,x1,y1,x2,y2,confidence,label,x1,y1,x2,y2,confidence,label
    """
    row_vals = row.split(",")

    assert len(row_vals) >= 3, "filepath,height,width,..."

    filepath = row_vals[0]
    height = int(row_vals[1])
    width = int(row_vals[2])
    data = row_vals[3:]

    group_size = 6
    total_size = len(data)
    assert total_size % group_size == 0

    detections = []
    for i in range(0, total_size, group_size):
        bounding_box, confidence, label = _parse_text_slice(data[i:(i + group_size)])

        detection = dict(
            bounding_box=bounding_box,
            confidence=confidence,
            label=label,
        )

        detections.append(detection)

    img_size = None
    if width > 0 and height > 0:
        img_size = (width, height)

    return filepath, img_size, detections


def _parse_yolo_row(row, classes):
    row_vals = row.split()
    target, xc, yc, w, h = row_vals[:5]

    try:
        label = classes[int(target)]
    except:
        label = str(target)

    bounding_box = [
        (float(xc) - 0.5 * float(w)),
        (float(yc) - 0.5 * float(h)),
        float(w),
        float(h),
    ]

    if len(row_vals) > 5:
        confidence = float(row_vals[5])
    else:
        confidence = None

    detection = dict(
        bounding_box=bounding_box,
        confidence=confidence,
        label=label,
    )
    return detection


def load_yolo_annotations(filepath, classes):
    """\
    row format:
        target,xc,yc,w,h,s
    """
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f.read().splitlines()]

    lines = [l for l in lines if l and not l.startswith("#")]

    detections = []
    for row in lines:
        detection = _parse_yolo_row(row, classes)
        detections.append(detection)
    return detections


def load_text_predictions(labels_path):
    with open(labels_path, "r") as f:
        lines = [l.strip() for l in f.read().splitlines()]

    lines = [l for l in lines if l and not l.startswith("#")]

    db = {}
    for row in lines:
        filepath, img_size, detections = _parse_text_row(row)
        db[Path(filepath).stem] = dict(
            detections=detections,
            size=img_size,
        )
    return db


def load_yolo_predictions(labels_path, classes):
    db = {}
    for filepath in Path(labels_path).glob("*.txt"):
        detections = load_yolo_annotations(filepath, classes)
        db[filepath.stem] = dict(
            detections=detections,
        )
    return db


def load_predictions(labels_path, classes=None, mode="text"):
    if mode == "text":
        return load_text_predictions(labels_path)
    elif mode == "yolo":
        assert isinstance(classes, list)
        return load_yolo_predictions(labels_path, classes)
    else:
        raise NotImplementedError
