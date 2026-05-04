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
    args = parser.parse_args()

    clf = load(args.model)
    label_map = load_label_map(args.labels)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Q or ESC: quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, thresh = preprocess_frame(frame, invert=args.invert)
        contour = find_largest_contour(thresh, min_area=args.min_area)

        display = frame.copy()
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
