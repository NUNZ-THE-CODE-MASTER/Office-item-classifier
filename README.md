# Office Item Classifier (PDE3802 Assessment 1, Part B)

Recognises an office item from a **single image** or a **live camera frame** and prints the predicted class with a confidence score. It is the perception module for the office-organiser robot described in our Part A report.

- **Classes ([N], between 6 and 10):** [class 1, class 2, class 3, class 4, class 5, class 6, class 7]
- **Model:** YOLO11n-cls (Ultralytics), fine-tuned from ImageNet weights
- **Inputs:** an image file, a folder of images, or a standard webcam
- **Output:** predicted class + confidence (and top-3 scores)

## Quick start

```bash
git clone [repo URL]
cd office-item-classifier
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

python src/predict.py --image examples/[some_image].jpg
```

The trained model `models/best.pt` is included, so you can run predictions without training.

## Dataset

All images used for training, validation and testing are in `data/raw/<class_name>/`
([or download from: LINK — make sure it is shared with all tutors]).
See [`DATASET_CARD.md`](DATASET_CARD.md) for sources, counts, labelling and limitations.

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| [fill from `results/dataset_stats.csv`] | | | | |

## How to run

1. **Split the data** (80/10/10, reproducible with a fixed seed):
   ```bash
   python src/prepare_dataset.py                    # or add --group-by-prefix
   ```
   Expected output: a table of images per class and split, saved to `results/dataset_stats.csv`.
2. **Train:**
   ```bash
   python src/train.py                              # options: --epochs 50 --imgsz 320
   ```
   Expected output: Ultralytics training log; best weights copied to `models/best.pt`.
3. **Evaluate on the held-out test set:**
   ```bash
   python src/evaluate.py
   ```
   Expected output (example): per-class precision/recall/F1, then
   `Accuracy : 0.xxx`, `Macro-F1 : 0.xxx`, the most confused class pairs, and files in `results/`.
4. **Predict:**
   ```bash
   python src/predict.py --image path/to/photo.jpg
   python src/predict.py --folder path/to/images --csv results/predictions.csv
   python src/predict.py --camera                   # press q to quit
   python src/gui.py                                # simple desktop app
   ```
   Expected output (example):
   ```
   photo.jpg: mug (0.93)
     top-3: mug 0.93 | bottle 0.04 | mouse 0.01
   ```
   A prediction with confidence below 0.50 is shown as `unknown` (change with `--min-conf`).

## Results (held-out test set)

Fill these in from `results/metrics.json` after running `evaluate.py`.

| Metric | Value |
|---|---|
| Test images | [N] |
| Top-1 accuracy | [0.xxx] |
| Macro-F1 | [0.xxx] |

![Confusion matrix](results/confusion_matrix.png)

### Error analysis

Use `results/misclassified.csv` and look at the actual wrong images.

- **Most confused pairs:** [A -> B (n times), C -> D (n times)]
- **Why:** [e.g. similar shape, similar colour, small in frame, reflective surface, clutter in background]
- **Evidence:** [name 2-3 example files from misclassified.csv and what you saw in them]
- **Class balance / test size:** [mention any class with few test images, because its score is noisy]
- **Robustness checks:** [dim light, partial occlusion, new background: what happened to F1?]
- **What we would try next:** [more images of the weak class, higher image size, different camera angles]

## Project structure

```
data/raw/          images, one folder per class (committed or linked)
data/split/        train/val/test copy made by prepare_dataset.py (not committed)
models/best.pt     trained weights
results/           metrics, confusion matrix, misclassified list
src/config.py      paths and hyper-parameters (edit here)
src/prepare_dataset.py   split the data
src/train.py       fine-tune the model
src/evaluate.py    accuracy, macro-F1, confusion matrix, error list
src/predict.py     command-line prediction (image, folder, camera)
src/gui.py         desktop app
tests/             pytest tests for the split
TASKS.md           who does what (delete before submission if you prefer)
```

Run the tests with `pytest`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` (or cv2, sklearn) | Activate the virtual environment, then `pip install -r requirements.txt`. |
| `Model models/best.pt not found` | Run `python src/train.py` first, or pull the repository again (the model file is committed). |
| `No class folders found in data/raw` | Put images in `data/raw/<class_name>/`, one folder per class. |
| `Could not open camera 0` | Close other apps using the camera, try `--camera 1`, and allow camera access in your OS privacy settings. |
| `ModuleNotFoundError: tkinter` (GUI) | Ubuntu/Debian: `sudo apt install python3-tk`. On Windows/macOS reinstall Python with Tk support. |
| Training is very slow | Use fewer epochs (`--epochs 15`) or a smaller `--imgsz`; a GPU is optional but faster. |
| Different results each run | The split and training use a fixed seed; re-run `prepare_dataset.py` before `train.py` if you changed the data. |

## Team contributions

| Member | Student ID | Contribution |
|---|---|---|
| [Name 1] | [ID] | [e.g. dataset collection, dataset card, prepare_dataset.py] |
| [Name 2] | [ID] | [e.g. training, evaluation, error analysis] |
| [Name 3] | [ID] | [e.g. predict.py, gui.py, README, tests, release tag] |

## Releases

The submitted version is tagged `v1.0`:

```bash
git tag -a v1.0 -m "Submitted version"
git push origin v1.0
```
