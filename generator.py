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
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="Process a directory of images in batch (manual labeling)",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default=".png,.jpg,.jpeg",
        help="Comma-separated image extensions to load in batch mode",
    )
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

    # In webcam mode we require --label when saving; in batch mode labels are
    # provided interactively per-image so --label is not required.
    if args.output and args.label is None and not args.input_dir:
        parser.error("--label is required when --output is set (webcam mode)")

    # Prepare windows and resources. In batch mode we don't open camera.
    cap = None
    if not args.input_dir:
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

    # If input_dir provided, process files sequentially
    files = []
    if args.input_dir:
        exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
        for fn in sorted(os.listdir(args.input_dir)):
            if any(fn.lower().endswith(ext) for ext in exts):
                files.append(os.path.join(args.input_dir, fn))
        if not files:
            print("No images found in", args.input_dir)
            return

    def process_frame_and_get_contour(frame):
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
                crop, invert=invert, method=method, close_ksize=args.morph_size, dilate_iter=args.dilate_iter
            )
            contour = find_largest_contour(thresh_crop, min_area=min_area)
            # draw ROI on display
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if contour is not None:
                cv2.drawContours(crop, [contour], -1, (0, 255, 0), 2)
            cv2.imshow("crop", crop)
            cv2.imshow("thresh", thresh_crop)
        else:
            method = "canny" if args.edges else "otsu"
            _, thresh = preprocess_frame(
                frame, invert=invert, method=method, close_ksize=args.morph_size, dilate_iter=args.dilate_iter
            )
            contour = find_largest_contour(thresh, min_area=min_area)
            if contour is not None:
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
            cv2.imshow("thresh", thresh)

        cv2.imshow("frame", display)
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
        return contour

    # Main loop: either iterate camera frames or batch files
    file_idx = 0
    while True:
        if args.input_dir:
            if file_idx >= len(files):
                break
            path = files[file_idx]
            frame = cv2.imread(path)
            if frame is None:
                print("Failed to load", path)
                file_idx += 1
                continue
            contour = process_frame_and_get_contour(frame)
        else:
            ret, frame = cap.read()
            if not ret:
                break
            contour = process_frame_and_get_contour(frame)

        # UI and contour display are handled inside process_frame_and_get_contour

        # If user is in roi_mode and currently dragging, drawing is handled
        # by the mouse callback and the per-frame processor. Just poll keys.
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
        # In webcam mode space still captures (using --label if provided)
        if key == ord(" ") and not args.input_dir:
            if roi is not None:
                x, y, w, h = roi
                crop = frame[y : y + h, x : x + w]
                _, thresh_crop = preprocess_frame(crop, invert=invert, method=("canny" if args.edges else "otsu"))
                contour = find_largest_contour(thresh_crop, min_area=min_area)
                if contour is None:
                    print("No contour found in ROI")
                    continue
                hu = compute_hu(contour, use_log=not args.raw_hu)
            else:
                if contour is None:
                    print("No contour found")
                    continue
                hu = compute_hu(contour, use_log=not args.raw_hu)

            print(hu.tolist())
            if args.output:
                if args.label is None:
                    print("--label required to save in webcam mode")
                    continue
                row = list(hu) + [args.label]
                write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
                append_row(args.output, row, write_header=write_header)
                print(f"Saved to {args.output}")

        # Batch-mode labeling keys
        if args.input_dir:
            # Numeric keys 0-9 assign that label and save
            if ord("0") <= key <= ord("9"):
                label = int(chr(key))
                if contour is None:
                    print("No contour found; skipping save")
                else:
                    hu = compute_hu(contour, use_log=not args.raw_hu)
                    if args.output:
                        write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
                        append_row(args.output, list(hu) + [label], write_header=write_header)
                        print(f"Saved {path} label={label} -> {args.output}")
                file_idx += 1
                continue

            if key == ord("n"):
                # prompt for arbitrary integer label
                try:
                    s = input("Label for image (integer, empty to skip): ")
                except Exception:
                    s = ""
                if s.strip() == "":
                    file_idx += 1
                    continue
                try:
                    label = int(s.strip())
                except ValueError:
                    print("Invalid label")
                    continue
                if contour is None:
                    print("No contour found; skipping save")
                else:
                    hu = compute_hu(contour, use_log=not args.raw_hu)
                    if args.output:
                        write_header = not os.path.exists(args.output) or os.path.getsize(args.output) == 0
                        append_row(args.output, list(hu) + [label], write_header=write_header)
                        print(f"Saved {path} label={label} -> {args.output}")
                file_idx += 1
                continue

            if key == ord("s"):
                print(f"Skipped {path}")
                file_idx += 1
                continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
