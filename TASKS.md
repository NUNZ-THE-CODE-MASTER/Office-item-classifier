# Who does what (group of 3)

Part B is marked on code quality (35), testing and evaluation (15), plus the 3-minute video. Split by area so every member has real commits in the history.

| Area | Owner | Files | Done when |
|---|---|---|---|
| **A. Data** | [Name 1] | `data/raw/`, `DATASET_CARD.md`, `src/prepare_dataset.py`, `tests/` | 6-10 classes, 150-200+ images each, card complete, tests pass |
| **B. Model and evaluation** | [Name 2] | `src/train.py`, `src/evaluate.py`, `results/`, README results and error analysis | `models/best.pt` committed, metrics and confusion matrix in README |
| **C. App, docs, release** | [Name 3] | `src/predict.py`, `src/gui.py`, README run and troubleshooting, tag `v1.0`, video | Image, folder and camera modes work; fresh clone runs from the README alone |

## Suggested order

1. **Day 1:** agree the class list; create the repo; each member clones it and makes a first commit. Start collecting photos (split classes between you).
2. **Days 2-3:** finish collection; fill the dataset card; run `prepare_dataset.py`.
3. **Days 3-4:** train a baseline; run `evaluate.py`; read `results/misclassified.csv`.
4. **Day 5:** improve the weak classes (more photos, more epochs); re-evaluate once.
5. **Day 6:** error analysis, README results, GUI polish, tests; test from a fresh clone on a second computer.
6. **Day 7:** record the video; tag `v1.0`; check that every tutor can open the repo, dataset and video links.

## Commit habits

Small, regular commits from each member with clear messages, for example:

- `Add class folders and first 40 mug images`
- `Fix split so photo sessions stay together`
- `Train baseline YOLO11n-cls, 30 epochs`
- `Add confusion matrix and misclassified list to evaluate.py`
- `Document camera troubleshooting in README`

Avoid one giant commit at the end; the rubric asks for "regular commits showing progress".

## Before you submit

- [ ] `pytest` passes
- [ ] Fresh clone: `pip install -r requirements.txt` then `python src/predict.py --image ...` works
- [ ] README results, error analysis, team table and dataset link filled in
- [ ] `models/best.pt`, `results/metrics.json` and `results/confusion_matrix.png` committed
- [ ] Tag `v1.0` pushed
- [ ] Video is 3 minutes or less, concept and architecture first, demo after; link opens for tutors
- [ ] GitHub, dataset and video links are in the separate links PDF
