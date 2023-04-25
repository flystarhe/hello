from pathlib import Path

import cv2 as cv
from skimage.measure import label, regionprops
from tqdm import tqdm


def count_areas(image_dir, pattern="*.png", limit=None):
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
        if limit is None:
            n = labels.max()
        else:
            props = regionprops(labels)
            n = len([1 for r in props if r.num_pixels >= limit])
        counts.append((f, n))

    print(f"{len(counts)=}, sum(counts)={sum([v[1] for v in counts])}")
    return counts
