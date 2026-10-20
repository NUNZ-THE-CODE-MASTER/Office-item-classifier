"""Classify an office item from an image file, a folder of images, or a camera.

Usage:
    python src/predict.py --image path/to/photo.jpg
    python src/predict.py --folder path/to/images --csv results/predictions.csv
    python src/predict.py --camera            # default webcam, press q to quit
    python src/predict.py --camera 1          # second camera

Output for one image (example):
    photo.jpg: mug (0.93)
      top-3: mug 0.93 | bottle 0.04 | mouse 0.01
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

import config


def load_model(path=config.BEST_MODEL):
    """Load trained weights, with a clear message if training was not run."""
    if not Path(path).exists():
        raise SystemExit(f"Model {path} not found. Train first: python src/train.py")
    return YOLO(str(path))


def classify(model, source, min_conf=config.UNKNOWN_THRESHOLD):
    """Classify one image (path or BGR numpy frame).

    Returns (label, confidence, top3) where top3 is [(name, prob), ...].
    A prediction below `min_conf` is reported as "unknown".
    """
    # No imgsz here: the model remembers the size it was trained with.
    result = model.predict(source, verbose=False)[0]
    probs = result.probs
    top3 = [(model.names[int(i)], float(probs.data[int(i)])) for i in probs.top5[:3]]
    label, conf = top3[0]
    if conf < min_conf:
        label = "unknown"
    return label, conf, top3


def format_top3(top3):
    return "top-3: " + " | ".join(f"{n} {p:.2f}" for n, p in top3)


def run_image(model, path, min_conf):
    label, conf, top3 = classify(model, str(path), min_conf)
    print(f"{Path(path).name}: {label} ({conf:.2f})")
    print(f"  {format_top3(top3)}")


def run_folder(model, folder, min_conf, csv_path=None):
    images = config.image_files(Path(folder))
    if not images:
        raise SystemExit(f"No images found in {folder}")
    rows, counts = [], Counter()
    for p in images:
        label, conf, _ = classify(model, str(p), min_conf)
        print(f"{p.name}: {label} ({conf:.2f})")
        rows.append((p.name, label, round(conf, 4)))
        counts[label] += 1
    print("\nSummary:", ", ".join(f"{k}: {v}" for k, v in counts.most_common()))
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "predicted", "confidence"])
            w.writerows(rows)
        print(f"Saved {csv_path}")


def run_camera(model, index, min_conf, show=True, max_frames=None):
    """Live classification from a standard camera. Press q to quit."""
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. "
                         "See 'Troubleshooting' in the README.")
    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            label, conf, _ = classify(model, frame, min_conf)
            text = f"{label} ({conf:.2f})"
            if show:
                color = (0, 200, 0) if conf >= config.HIGH_CONFIDENCE else (0, 165, 255)
                if label == "unknown":
                    color = (0, 0, 255)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, color, 2)
                cv2.imshow("Office item classifier (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(text)
            frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to one image")
    src.add_argument("--folder", help="folder of images")
    src.add_argument("--camera", nargs="?", const=0, type=int, metavar="INDEX",
                     help="camera index (default 0)")
    ap.add_argument("--model", default=str(config.BEST_MODEL))
    ap.add_argument("--min-conf", type=float, default=config.UNKNOWN_THRESHOLD,
                    help="below this confidence the answer is 'unknown'")
    ap.add_argument("--csv", help="(folder mode) save predictions to this CSV")
    ap.add_argument("--no-window", action="store_true",
                    help="(camera mode) print results instead of opening a window")
    ap.add_argument("--max-frames", type=int, help="(camera mode) stop after N frames")
    args = ap.parse_args()

    model = load_model(args.model)
    if args.image:
        if not Path(args.image).is_file():
            raise SystemExit(f"Image not found: {args.image}")
        run_image(model, args.image, args.min_conf)
    elif args.folder:
        run_folder(model, args.folder, args.min_conf, args.csv)
    else:
        run_camera(model, args.camera, args.min_conf,
                   show=not args.no_window, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
