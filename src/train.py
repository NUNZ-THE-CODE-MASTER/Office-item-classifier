"""Fine-tune a pretrained YOLO11 classifier on data/split.

Usage:
    python src/train.py
    python src/train.py --epochs 50 --imgsz 320

The best weights are copied to models/best.pt so predict.py, evaluate.py and
gui.py can find them without knowing the run folder.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

import config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=config.BASE_WEIGHTS,
                    help="starting weights (downloaded automatically)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--imgsz", type=int, default=config.IMGSZ)
    ap.add_argument("--batch", type=int, default=config.BATCH)
    ap.add_argument("--name", default="office_items", help="run folder name")
    args = ap.parse_args()

    if not (config.SPLIT_DIR / "train").exists():
        raise SystemExit("data/split/train not found. "
                         "Run: python src/prepare_dataset.py")

    model = YOLO(args.weights)
    model.train(
        data=str(config.SPLIT_DIR),          # folder holding train/ and val/
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=config.PATIENCE,            # stop early if val stops improving
        seed=config.SEED,
        project=str(config.MODELS_DIR / "runs"),
        name=args.name,
        exist_ok=True,                       # overwrite the previous run
    )

    best = Path(model.trainer.best)
    config.MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, config.BEST_MODEL)
    print(f"\nBest weights copied to {config.BEST_MODEL}")
    print("Next: python src/evaluate.py")


if __name__ == "__main__":
    main()
