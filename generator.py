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
    parser.add_argument("--invert", action="store_true", help="Invert threshold")
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument(
        "--raw-hu", action="store_true", help="Disable log transform"
    )
    args = parser.parse_args()

    if args.output and args.label is None:
        parser.error("--label is required when --output is set")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("SPACE: capture | Q or ESC: quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, thresh = preprocess_frame(frame, invert=args.invert)
        contour = find_largest_contour(thresh, min_area=args.min_area)

        display = frame.copy()
        if contour is not None:
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)

        cv2.putText(
            display,
            "SPACE: capture  Q/ESC: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("frame", display)
        cv2.imshow("thresh", thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord(" "):
            if contour is None:
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
