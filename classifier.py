import argparse
import cv2
from joblib import load

from commons import RoiSelector, overlay_frame
from pipeline import PipelineConfig, analyze_frame
from vision_utils import load_label_map


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
    parser.add_argument("--min-confidence", type=float, default=0.7)
    args = parser.parse_args()

    clf = load(args.model)
    label_map = load_label_map(args.labels)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("SPACE: save hu | Q or ESC: quit | Click button to select ROI")

    roi_selector = RoiSelector()
    config = PipelineConfig(
        invert=args.invert,
        raw_hu=args.raw_hu,
        edges=args.edges,
        min_area=args.min_area,
        morph_size=args.morph_size,
        dilate_iter=args.dilate_iter,
    )

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", roi_selector.on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        button_rect = roi_selector.update_button(frame.shape)
        pred_text = None
        pred_color = (0, 200, 0)
        contour_color = (0, 255, 0)

        roi = roi_selector.get_clamped_roi(frame.shape)
        result = analyze_frame(frame, config, roi=roi)
        contour = result["contour"]

        if result["roi"] is not None:
            x, y, w, h = result["roi"]
            crop = result["crop"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if contour is not None:
                cv2.drawContours(crop, [contour], -1, contour_color, 2)
            cv2.imshow("crop", crop)
            cv2.imshow("thresh", result["thresh"])
        else:
            if contour is not None:
                cv2.drawContours(display, [contour], -1, contour_color, 2)
            cv2.imshow("thresh", result["thresh"])

        if result["hu"] is not None:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba([result["hu"]])[0]
                best_idx = int(proba.argmax())
                best_conf = float(proba[best_idx])
                pred = clf.classes_[best_idx]
                if best_conf < args.min_confidence:
                    pred_text = f"Unknown ({best_conf:.2f})"
                    pred_color = (200, 40, 40)
                    contour_color = (0, 0, 255)
                else:
                    pred_int = int(pred)
                    label_text = label_map.get(pred_int, str(pred_int))
                    pred_text = f"Pred: {label_text} ({pred_int})"
            else:
                pred = clf.predict([result["hu"]])[0]
                pred_int = int(pred)
                label_text = label_map.get(pred_int, str(pred_int))
                pred_text = f"Pred: {label_text} ({pred_int})"
        else:
            pred_text = "No contour"
            pred_color = (200, 40, 40)

        status_lines = [
            "Q/ESC: quit  I: invert  +/-: min area  R: clear ROI",
            f"Invert: {'ON' if args.invert else 'OFF'}  MinArea: {args.min_area}",
        ]
        display = overlay_frame(
            display,
            button_rect,
            status_lines=status_lines,
            pred_text=pred_text,
            pred_color=pred_color,
            button_active=roi_selector.roi_mode,
        )
        cv2.imshow("frame", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord("i"):
            args.invert = not args.invert
            config.invert = args.invert
            print(f"Invert set to {args.invert}")
            continue
        if key == ord("-") or key == ord("_"):
            args.min_area = max(1, args.min_area - 100)
            config.min_area = args.min_area
            print(f"min_area={args.min_area}")
            continue
        if key == ord("=") or key == ord("+"):
            args.min_area = args.min_area + 100
            config.min_area = args.min_area
            print(f"min_area={args.min_area}")
            continue
        if key == ord("r"):
            roi_selector.clear()
            print("ROI cleared")
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
