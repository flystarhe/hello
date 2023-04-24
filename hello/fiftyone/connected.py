from pathlib import Path

import cv2 as cv
from skimage.measure import label
from tqdm import tqdm


def count_areas(image_dir, pattern="*.png"):
    """Count connected regions of an integer array.

    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

    Args:
        image_dir (str): _description_
        pattern (str): _description_
    """
    image_dir = Path(image_dir)

    files = sorted([str(f) for f in image_dir.glob(pattern)])

    counts = []
    for f in tqdm(files):
        x = cv.imread(f, 0)
        labels = label(x, connectivity=x.ndim)
        counts.append((f, labels.max()))

    print(f"{len(counts)=}, {sum([v[1] for v in counts])=}")
    return counts
