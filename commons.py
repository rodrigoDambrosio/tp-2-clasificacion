import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_roi_arg(roi_str):
    if not roi_str:
        return None
    parts = [p.strip() for p in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    x, y, w, h = [int(float(p)) for p in parts]
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be positive")
    return x, y, w, h


class RoiSelector:
    def __init__(self):
        self.roi = None  # (x, y, w, h)
        self.roi_mode = False
        self.drawing = False
        self.ix = -1
        self.iy = -1
        self.button_rect = (0, 0, 0, 0)

    def update_button(self, frame_shape, size=(140, 28), margin=10):
        h_frame, w_frame = frame_shape[:2]
        bw, bh = size
        bx = max(margin, w_frame - bw - margin)
        by = margin
        self.button_rect = (bx, by, bw, bh)
        return self.button_rect

    def on_mouse(self, event, x, y, flags, param):
        bx, by, bw, bh = self.button_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.roi_mode = True
                self.drawing = False
                return
            if self.roi_mode:
                self.drawing = True
                self.ix, self.iy = x, y
                return

        if event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x0, y0 = min(self.ix, x), min(self.iy, y)
            x1, y1 = max(self.ix, x), max(self.iy, y)
            w, h = x1 - x0, y1 - y0
            if w > 5 and h > 5:
                self.roi = (x0, y0, w, h)
            self.roi_mode = False

    def get_clamped_roi(self, frame_shape):
        if self.roi is None:
            return None
        h_frame, w_frame = frame_shape[:2]
        x, y, w, h = self.roi
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        return x, y, w, h

    def clear(self):
        self.roi = None
        self.roi_mode = False
        self.drawing = False


def _load_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def overlay_frame(
    frame_bgr,
    button_rect,
    status_lines=None,
    pred_text=None,
    pred_color=(0, 200, 0),
    button_active=False,
):
    if status_lines is None:
        status_lines = []

    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    base = pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(16)

    bx, by, bw, bh = button_rect
    btn_fill = (60, 60, 60, 230) if not button_active else (90, 90, 90, 230)
    draw.rectangle([bx, by, bx + bw, by + bh], fill=btn_fill, outline=(220, 220, 220, 255))
    draw.text((bx + 8, by + 6), "Select ROI", fill=(255, 255, 255, 255), font=font)

    if status_lines:
        pad = 6
        max_w = 0
        line_h = 0
        for line in status_lines:
            w, h = _text_size(draw, line, font)
            max_w = max(max_w, w)
            line_h = max(line_h, h)
        total_h = line_h * len(status_lines) + (len(status_lines) - 1) * 2 + pad * 2
        x0, y0 = 8, 8
        draw.rectangle(
            [x0, y0, x0 + max_w + pad * 2, y0 + total_h],
            fill=(0, 0, 0, 150),
        )
        for idx, line in enumerate(status_lines):
            y = y0 + pad + idx * (line_h + 2)
            draw.text((x0 + pad, y), line, fill=(255, 255, 255, 255), font=font)

    if pred_text:
        draw.text((10, 40), pred_text, fill=pred_color + (255,), font=font)

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)
