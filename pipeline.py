from dataclasses import dataclass

from vision_utils import preprocess_frame, find_largest_contour, compute_hu


@dataclass
class PipelineConfig:
    invert: bool = False
    raw_hu: bool = False
    edges: bool = False
    min_area: int = 800
    morph_size: int = 5
    dilate_iter: int = 1
    method: str | None = None
    manual_thresh: int = 127


def clamp_roi(roi, frame_shape):
    if roi is None:
        return None
    x, y, w, h = roi
    h_frame, w_frame = frame_shape[:2]
    x = max(0, min(x, w_frame - 1))
    y = max(0, min(y, h_frame - 1))
    w = max(1, min(w, w_frame - x))
    h = max(1, min(h, h_frame - y))
    return x, y, w, h


def analyze_frame(frame, config, roi=None):
    method = config.method or ("canny" if config.edges else "otsu")
    roi = clamp_roi(roi, frame.shape) if roi is not None else None

    if roi is not None:
        x, y, w, h = roi
        crop = frame[y : y + h, x : x + w]
        _, thresh = preprocess_frame(
            crop,
            invert=config.invert,
            method=method,
            close_ksize=config.morph_size,
            dilate_iter=config.dilate_iter,
            manual_thresh=config.manual_thresh,
        )
        contour = find_largest_contour(thresh, min_area=config.min_area)
        hu = compute_hu(contour, use_log=not config.raw_hu) if contour is not None else None
        return {
            "roi": roi,
            "crop": crop,
            "thresh": thresh,
            "contour": contour,
            "hu": hu,
        }

    _, thresh = preprocess_frame(
        frame,
        invert=config.invert,
        method=method,
        close_ksize=config.morph_size,
        dilate_iter=config.dilate_iter,
        manual_thresh=config.manual_thresh,
    )
    contour = find_largest_contour(thresh, min_area=config.min_area)
    hu = compute_hu(contour, use_log=not config.raw_hu) if contour is not None else None
    return {
        "roi": None,
        "crop": None,
        "thresh": thresh,
        "contour": contour,
        "hu": hu,
    }
