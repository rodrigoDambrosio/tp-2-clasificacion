import json
import cv2
import numpy as np


def preprocess_frame(frame, invert=False, blur_ksize=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if invert:
        thresh = cv2.bitwise_not(thresh)
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
        hu = np.where(hu == 0, 0.0, -np.sign(hu) * np.log10(np.abs(hu)))
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
