import argparse
import cv2
from joblib import load

from vision_utils import preprocess_frame, find_largest_contour, compute_hu, load_label_map


def main():
    parser = argparse.ArgumentParser(description="Classify contours from webcam")
    parser.add_argument("--model", type=str, default="model.joblib")
    parser.add_argument("--labels", type=str, default="labels.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--raw-hu", action="store_true")
    parser.add_argument(
        "--edges",
        action="store_true",
        help="Use Canny edges + dilation for contour detection (more robust)",
    )
    parser.add_argument(
        "--morph-size",
        type=int,
        default=5,
        help="Morphological kernel size used to close gaps (default 5)",
    )
    parser.add_argument(
        "--dilate-iter",
        type=int,
        default=1,
        help="Number of dilation iterations for Canny edges (default 1)",
    )
    args = parser.parse_args()

    clf = load(args.model)
    label_map = load_label_map(args.labels)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("SPACE: save hu | Q or ESC: quit | Click button to select ROI")

    # ROI selection state
    roi = None  # tuple (x,y,w,h)
    roi_mode = False  # True when button clicked and waiting for drag
    drawing = False
    ix = iy = -1

    # shared button rect, updated each frame so mouse callback can read it
    button_rect = [0, 0, 0, 0]

    def on_mouse(event, x, y, flags, param):
        nonlocal roi, roi_mode, drawing, ix, iy
        bx, by, bw, bh = button_rect
        # click inside button toggles ROI mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                roi_mode = True
                drawing = False
                return
            if roi_mode:
                drawing = True
                ix, iy = x, y
                return

        if event == cv2.EVENT_MOUSEMOVE and drawing:
            return

        if event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, y0 = min(ix, x), min(iy, y)
            x1, y1 = max(ix, x), max(iy, y)
            w, h = x1 - x0, y1 - y0
            if w > 5 and h > 5:
                roi = (x0, y0, w, h)
            roi_mode = False

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_frame, w_frame = frame.shape[:2]

        # compute button position (top-right)
        bw, bh = 140, 28
        bx, by = max(10, w_frame - bw - 10), 10
        button_rect[0], button_rect[1], button_rect[2], button_rect[3] = bx, by, bw, bh

        display = frame.copy()

        # draw button
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        cv2.putText(
            display,
            "Select ROI",
            (bx + 8, by + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if roi is not None:
            x, y, w, h = roi
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = max(1, min(w, w_frame - x))
            h = max(1, min(h, h_frame - y))
            crop = frame[y : y + h, x : x + w]
            method = "canny" if args.edges else "otsu"
            _, thresh_crop = preprocess_frame(
                crop, invert=args.invert, method=method, close_ksize=args.morph_size, dilate_iter=args.dilate_iter
            )
            contour = find_largest_contour(thresh_crop, min_area=args.min_area)
            # draw ROI on display
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if contour is not None:
                cv2.drawContours(crop, [contour], -1, (0, 255, 0), 2)
                hu = compute_hu(contour, use_log=not args.raw_hu)
                pred = clf.predict([hu])[0]
                pred_int = int(pred)
                label_text = label_map.get(pred_int, str(pred_int))
                cv2.putText(
                    display,
                    f"Pred: {label_text} ({pred_int})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("crop", crop)
            cv2.imshow("thresh", thresh_crop)
        else:
            method = "canny" if args.edges else "otsu"
            _, thresh = preprocess_frame(
                frame, invert=args.invert, method=method, close_ksize=args.morph_size, dilate_iter=args.dilate_iter
            )
            contour = find_largest_contour(thresh, min_area=args.min_area)
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                hu = compute_hu(contour, use_log=not args.raw_hu)
                pred = clf.predict([hu])[0]
                pred_int = int(pred)
                label_text = label_map.get(pred_int, str(pred_int))
                cv2.putText(
                    display,
                    f"Pred: {label_text} ({pred_int})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "No contour",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow("frame", display)
            cv2.imshow("thresh", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord("i"):
            args.invert = not args.invert
            print(f"Invert set to {args.invert}")
            continue
        if key == ord("-") or key == ord("_"):
            args.min_area = max(1, args.min_area - 100)
            print(f"min_area={args.min_area}")
            continue
        if key == ord("=") or key == ord("+"):
            args.min_area = args.min_area + 100
            print(f"min_area={args.min_area}")
            continue
        if key == ord("r"):
            roi = None
            print("ROI cleared")
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
