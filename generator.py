import argparse
import csv
import os
import cv2

from vision_utils import preprocess_frame, find_largest_contour, compute_hu


def append_row(path, row, write_header=False):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = [f"hu_{i + 1}" for i in range(len(row) - 1)] + ["label"]
            writer.writerow(header)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Capture Hu moments from webcam contours."
    )
    parser.add_argument("--label", type=int, help="Numeric label for output")
    parser.add_argument("--output", type=str, default="", help="CSV output")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Interactively select a region of interest (ROI) before capture",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="",
        help="Set ROI as x,y,w,h (overrides interactive selection)",
    )
    parser.add_argument("--invert", action="store_true", help="Invert threshold")
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument(
        "--raw-hu", action="store_true", help="Disable log transform"
    )
    parser.add_argument(
        "--edges",
        action="store_true",
        help="Use Canny edges + dilation for contour detection (more robust)",
    )
    args = parser.parse_args()

    if args.output and args.label is None:
        parser.error("--label is required when --output is set")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    cv2.namedWindow("frame")

    # runtime state
    invert = args.invert
    min_area = args.min_area

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
            # just update; actual rectangle drawn in main loop
            return

        if event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            x0, y0 = min(ix, x), min(iy, y)
            x1, y1 = max(ix, x), max(iy, y)
            w, h = x1 - x0, y1 - y0
            if w > 5 and h > 5:
                roi = (x0, y0, w, h)
            roi_mode = False

    cv2.setMouseCallback("frame", on_mouse)

    print("SPACE: capture | Q or ESC: quit | Click button to select ROI")

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

        # If ROI exists, process crop
        if roi is not None:
            x, y, w, h = roi
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = max(1, min(w, w_frame - x))
            h = max(1, min(h, h_frame - y))
            crop = frame[y : y + h, x : x + w]
            method = "canny" if args.edges else "otsu"
            _, thresh_crop = preprocess_frame(crop, invert=invert, method=method)
            contour = find_largest_contour(thresh_crop, min_area=min_area)
            # draw ROI on display
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if contour is not None:
                # draw contour on crop image for feedback
                cv2.drawContours(crop, [contour], -1, (0, 255, 0), 2)
            cv2.imshow("crop", crop)
            cv2.imshow("thresh", thresh_crop)
        else:
            # No ROI: use full frame thresh
            method = "canny" if args.edges else "otsu"
            _, thresh = preprocess_frame(frame, invert=invert, method=method)
            contour = find_largest_contour(thresh, min_area=min_area)
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
            cv2.imshow("thresh", thresh)

        # If user is in roi_mode and currently dragging, draw current selection
        if roi_mode and drawing and ix >= 0:
            # get current mouse position via waitKey workaround: not available here,
            # but OpenCV will show selection while dragging via on_mouse updates; skip
            pass

        # show status
        status_text = f"SPACE: capture  Q/ESC: quit  Invert: {'ON' if invert else 'OFF'}  MinArea: {min_area}"
        cv2.putText(
            display,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("frame", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        # runtime controls for debugging detection
        if key == ord("i"):
            invert = not invert
            print(f"Invert set to {invert}")
            continue
        if key == ord("-") or key == ord("_"):
            min_area = max(1, min_area - 100)
            print(f"min_area={min_area}")
            continue
        if key == ord("=") or key == ord("+"):
            min_area = min_area + 100
            print(f"min_area={min_area}")
            continue
        if key == ord("r"):
            roi = None
            print("ROI cleared")
            continue
        if key == ord(" "):
            if roi is not None:
                # compute hu from cropped contour
                # recompute thresh_crop and contour to ensure latest
                x, y, w, h = roi
                crop = frame[y : y + h, x : x + w]
                _, thresh_crop = preprocess_frame(crop, invert=invert)
                contour = find_largest_contour(thresh_crop, min_area=min_area)
                if contour is None:
                    print("No contour found in ROI")
                    continue
                hu = compute_hu(contour, use_log=not args.raw_hu)
            else:
                # use full-frame contour
                if 'contour' not in locals() or contour is None:
                    print("No contour found")
                    continue
                hu = compute_hu(contour, use_log=not args.raw_hu)

            print(hu.tolist())
            if args.output:
                row = list(hu) + [args.label]
                write_header = not os.path.exists(args.output) or os.path.getsize(
                    args.output
                ) == 0
                append_row(args.output, row, write_header=write_header)
                print(f"Saved to {args.output}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
