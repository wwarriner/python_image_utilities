from math import isnan, isinf

import numpy as np
import cv2


# TODO add pre/postcondition checks to functions


def adjust_gamma(image, gamma=1.0):
    """Adjusts image gamma. Works on both float and uint8 as expected. Float
    images must be in the range (0.0, 1.0)."""
    is_float = image.dtype == np.float
    if is_float:
        image = image * 255.0
    igamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** igamma) * 255.0 for i in range(0, 256)])
    adjusted = cv2.LUT(np.array(image, dtype=np.uint8), lut.astype(np.uint8))
    if is_float:
        adjusted = adjusted / 255.0
    return adjusted


def apply_clahe(image):
    """Applies CLAHE equalization to input image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    return clahe.apply(image.astype(np.uint8))


def rescale(image, out_range=(0.0, 1.0), in_range=(float("-inf"), float("+inf"))):
    """Rescales image from their input range to out_range."""
    lo = in_range[0]
    if lo == float("-inf") or isnan(lo):
        lo = np.min(image)

    hi = in_range[1]
    if hi == float("+inf") or isnan(hi):
        hi = np.max(image)

    assert not isinf(lo) and not isnan(lo)
    assert not isinf(hi) and not isnan(hi)
    assert lo < hi

    med = (image - lo) / (hi - lo)
    return med * (out_range[1] - out_range[0]) + out_range[0]


def rgb2gray(rgb_image):
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def standardize(images):
    """Standardizes an (N+1)-D block of N-D images by the usual method, i.e.
    (x-u)/s."""
    s = np.std(images)
    u = np.mean(images)
    standardized = (images - u) / s
    return standardized
