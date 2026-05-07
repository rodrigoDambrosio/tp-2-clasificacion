import json
import cv2
import numpy as np


def preprocess_frame(
    frame,
    invert=False,
    blur_ksize=5,
    method="otsu",
    close_ksize=5,
    dilate_iter=1,
    manual_thresh=127,
):
    """Preprocess an input BGR frame and return (gray, thresh).

    method: 'otsu' (default) uses Otsu thresholding; 'adaptive' uses
    adaptive thresholding; 'manual' uses a fixed threshold; 'canny' runs
    Canny edge detection followed by dilation to close gaps.
    close_ksize: kernel size used for morphological closing/dilation.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    if method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        if invert:
            thresh = cv2.bitwise_not(thresh)
        # close small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return gray, thresh

    if method == "manual":
        _, thresh = cv2.threshold(gray, manual_thresh, 255, cv2.THRESH_BINARY)
        if invert:
            thresh = cv2.bitwise_not(thresh)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return gray, thresh

    if method == "canny":
        # Canny edges then dilate to close small gaps so contours are continuous
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        thresh = cv2.dilate(edges, kernel, iterations=dilate_iter)
        if invert:
            thresh = cv2.bitwise_not(thresh)
        return gray, thresh

    # default: Otsu
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert:
        thresh = cv2.bitwise_not(thresh)
    # closing helps to join boundary gaps and remove small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return gray, thresh


def find_largest_contour(thresh, min_area=800):
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    return largest


def compute_hu(contour, use_log=True):
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    if use_log:
        hu_abs = np.abs(hu)
        hu_log = np.zeros_like(hu)
        mask = hu_abs > 0
        hu_log[mask] = -np.sign(hu[mask]) * np.log10(hu_abs[mask])
        hu = hu_log
    return hu


def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {i + 1: str(label) for i, label in enumerate(data)}
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            try:
                idx = int(key)
            except ValueError:
                continue
            out[idx] = str(value)
        return out
    raise ValueError("labels file must be a list or dict")
