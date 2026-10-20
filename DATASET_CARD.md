# Dataset Card: Office Items

Replace every [bracket]. Keep it factual: a grader should be able to see exactly where each image came from.

## Summary

[One or two sentences: what the dataset is for, e.g. "Photos of seven common office items for a 7-class image classifier used by an office-organiser robot."]

## Classes and counts

| Class | Images | Train | Val | Test |
|---|---|---|---|---|
| [fill from `results/dataset_stats.csv`] | | | | |

**Total images:** [N]  |  **Image formats:** JPG/PNG  |  **Typical size:** [e.g. 1080x1440 phone photos]

## Collection

- **Who collected:** [names]
- **Devices:** [phone models / webcam]
- **Locations:** [e.g. 3 offices and 2 homes] | **Dates:** [dd/mm/yyyy to dd/mm/yyyy]
- **Variation on purpose:** [angles, distances 0.5 to 2 m, lighting (daylight, lamp, dim), backgrounds, clutter, partial occlusion]
- **Public images used (if any):** [dataset name, URL, licence, number of images, which classes]. Give credit for every outside source.

## Labelling

- Each image shows **one main item**; its label is the folder name.
- **Who labelled / checked:** [names]. **Quality check:** [e.g. a second member reviewed every image; wrong or blurry photos removed]

## Splits

- **Ratio:** 80% train / 10% validation / 10% test, fixed seed 42 (`src/prepare_dataset.py`).
- **Leakage control:** [photos taken in the same session are kept in the same split via `--group-by-prefix` / otherwise random]
- The **test set is never used for training or tuning**.

## Known limitations and biases

- [e.g. all photos taken in a small number of rooms; one brand of mug; few images of class X]
- [Class imbalance: list the smallest and largest class]
- [Not tested on: outdoor scenes, very dark rooms, items inside drawers]

## Intended and out-of-scope use

- **Intended:** coursework demonstration of an office-item classifier.
- **Not suitable for:** safety-critical decisions or recognising people.

## Privacy

[Confirm that no faces or screens with personal information appear in the images, or that they were removed or blurred.]

## Download

[Link to the full dataset if it is too large for the repository, shared with all tutors]
