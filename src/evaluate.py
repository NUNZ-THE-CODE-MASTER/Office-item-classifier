"""Evaluate the trained model on the held-out test split.

Usage:
    python src/evaluate.py

Writes to results/:
    metrics.json              accuracy, macro-F1, per-class scores, top confusions
    classification_report.txt precision / recall / F1 per class
    confusion_matrix.csv/.png counts, rows = true class, columns = predicted
    misclassified.csv         every wrong prediction, for the error analysis
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                        # no display needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from ultralytics import YOLO

import config


def collect_test_images(test_dir):
    """Return ([paths], [true class names]) from data/split/test/<class>/."""
    paths, truth = [], []
    for class_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        for p in config.image_files(class_dir):
            paths.append(p)
            truth.append(class_dir.name)
    return paths, truth


def predict_all(model, paths, batch=32):
    """Return ([predicted names], [confidences]) for a list of image paths."""
    names, confs = [], []
    for i in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[i:i + batch]]
        # No imgsz here: the model remembers the size it was trained with.
        for r in model.predict(chunk, verbose=False):
            names.append(model.names[int(r.probs.top1)])
            confs.append(float(r.probs.top1conf))
    return names, confs


def save_confusion_png(cm, labels, path):
    fig, ax = plt.subplots(figsize=(0.9 * len(labels) + 2, 0.8 * len(labels) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (test set)")
    top = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > top else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(config.BEST_MODEL))
    ap.add_argument("--test-dir", default=str(config.SPLIT_DIR / "test"))
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise SystemExit(f"{test_dir} not found. Run prepare_dataset.py and train.py first.")
    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} not found. Run: python src/train.py")

    model = YOLO(args.model)
    labels = [model.names[i] for i in sorted(model.names)]
    paths, truth = collect_test_images(test_dir)
    pred, conf = predict_all(model, paths)

    acc = accuracy_score(truth, pred)
    macro_f1 = f1_score(truth, pred, average="macro", labels=labels, zero_division=0)
    report = classification_report(truth, pred, labels=labels, zero_division=0)
    report_dict = classification_report(truth, pred, labels=labels,
                                        zero_division=0, output_dict=True)
    cm = confusion_matrix(truth, pred, labels=labels)

    wrong = [(str(p), t, q, c) for p, t, q, c in zip(paths, truth, pred, conf) if t != q]
    pairs = Counter((t, q) for _, t, q, _ in wrong).most_common(5)

    out = config.RESULTS_DIR
    out.mkdir(exist_ok=True)
    (out / "classification_report.txt").write_text(report)
    np.savetxt(out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(labels), comments="")
    save_confusion_png(cm, labels, out / "confusion_matrix.png")
    with open(out / "misclassified.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "predicted", "confidence"])
        w.writerows(wrong)
    (out / "metrics.json").write_text(json.dumps({
        "test_images": len(paths),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": {c: round(report_dict[c]["f1-score"], 4) for c in labels},
        "most_confused": [{"true": t, "predicted": q, "count": n} for (t, q), n in pairs],
    }, indent=2))

    print(report)
    print(f"Test images : {len(paths)}")
    print(f"Accuracy    : {acc:.3f}")
    print(f"Macro-F1    : {macro_f1:.3f}")
    if pairs:
        print("Most confused (true -> predicted):")
        for (t, q), n in pairs:
            print(f"  {t} -> {q}: {n}")
    print(f"\nFiles written to {out}")


if __name__ == "__main__":
    main()
