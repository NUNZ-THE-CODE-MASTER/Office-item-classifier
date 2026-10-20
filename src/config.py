"""Shared settings for the office-item classifier.

Change paths or hyper-parameters here, not in the other scripts, so every
script (prepare, train, evaluate, predict, gui) stays consistent.
"""
from pathlib import Path

# ---- folders ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"        # you add: data/raw/<class_name>/*.jpg
SPLIT_DIR = ROOT / "data" / "split"    # created by prepare_dataset.py
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
BEST_MODEL = MODELS_DIR / "best.pt"    # copied here by train.py

# ---- dataset ---------------------------------------------------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_RATIOS = (0.8, 0.1, 0.1)         # train / validation / test
SEED = 42                              # fixed so splits are reproducible
MIN_CLASSES, MAX_CLASSES = 6, 10       # the brief asks for 6-10 classes
MIN_IMAGES_PER_CLASS = 100             # warn below this (aim for 150-200+)

# ---- model / training ------------------------------------------------------
BASE_WEIGHTS = "yolo11n-cls.pt"        # small ImageNet-pretrained classifier
EPOCHS = 30
IMGSZ = 224
BATCH = 32
PATIENCE = 10                          # early stopping

# ---- inference -------------------------------------------------------------
UNKNOWN_THRESHOLD = 0.50               # below this confidence -> "unknown"
HIGH_CONFIDENCE = 0.85                 # used to colour the GUI / overlay


def image_files(folder: Path):
    """Return a sorted list of image files directly inside `folder`."""
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
