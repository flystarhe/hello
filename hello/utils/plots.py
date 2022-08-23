import cv2 as cv
import matplotlib.pyplot as plt
from fiftyone.core.context import is_notebook_context


def imshow(img, mode="BGR"):
    mode = mode.upper()

    if isinstance(img, str):
        img = cv.imread(img, cv.IMREAD_COLOR)
        mode = "BGR"

    if is_notebook_context():
        if mode != "RGB":
            code = getattr(cv, f"COLOR_{mode}2RGB")
            img = cv.cvtColor(img, code)

        plt.imshow(img)
        plt.show()
    else:
        if mode != "BGR":
            code = getattr(cv, f"COLOR_{mode}2BGR")
            img = cv.cvtColor(img, code)

        cv.imshow("imshow", img)
