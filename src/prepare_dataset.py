"""Split data/raw/<class>/*.jpg into data/split/{train,val,test}/<class>/.

Usage:
    python src/prepare_dataset.py
    python src/prepare_dataset.py --group-by-prefix   # keep photo sessions together

--group-by-prefix treats the text before the first "_" in a file name as the
photo session (e.g. s01_001.jpg and s01_002.jpg are one session). All photos of
a session go to the same split, so near-duplicate shots can never leak from
train into test.
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import config

SPLITS = ("train", "val", "test")


def split_groups(groups, ratios, rng):
    """Divide a list of groups into train/val/test lists (by group count).

    Every split gets at least one group when there are 3 or more groups.
    """
    groups = list(groups)
    rng.shuffle(groups)
    n = len(groups)
    n_val = round(n * ratios[1])
    n_test = round(n * ratios[2])
    if n >= 3:
        n_val, n_test = max(1, n_val), max(1, n_test)
    n_train = n - n_val - n_test
    return {
        "train": groups[:n_train],
        "val": groups[n_train:n_train + n_val],
        "test": groups[n_train + n_val:],
    }


def group_images(images, by_prefix):
    """Return {group_key: [paths]}; each image is its own group by default."""
    groups = defaultdict(list)
    for p in images:
        key = p.stem.split("_")[0] if by_prefix else p.name
        groups[key].append(p)
    return list(groups.values())


def build_split(raw_dir, out_dir, ratios=config.SPLIT_RATIOS,
                seed=config.SEED, group_by_prefix=False):
    """Copy images into the split folders. Returns {class: {split: count}}."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    classes = sorted(d.name for d in raw_dir.iterdir()
                     if d.is_dir() and not d.name.startswith("."))
    if not classes:
        raise SystemExit(f"No class folders found in {raw_dir}. "
                         "Expected data/raw/<class_name>/*.jpg")

    rng = random.Random(seed)
    stats = {}
    for cls in classes:
        images = config.image_files(raw_dir / cls)
        groups = group_images(images, group_by_prefix)
        parts = split_groups(groups, ratios, rng)
        stats[cls] = {}
        for split in SPLITS:
            dest = out_dir / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            files = [p for g in parts[split] for p in g]
            for p in files:
                shutil.copy2(p, dest / p.name)
            stats[cls][split] = len(files)
    return stats


def print_and_save_stats(stats, csv_path):
    """Print a class-distribution table and save it as CSV."""
    header = ("class", "train", "val", "test", "total")
    rows = [(c, s["train"], s["val"], s["test"], sum(s.values()))
            for c, s in stats.items()]
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(5)]
    for r in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")

    n = len(stats)
    if not config.MIN_CLASSES <= n <= config.MAX_CLASSES:
        print(f"WARNING: {n} classes found; the brief asks for "
              f"{config.MIN_CLASSES}-{config.MAX_CLASSES}.")
    for c, total in ((r[0], r[4]) for r in rows):
        if total < config.MIN_IMAGES_PER_CLASS:
            print(f"WARNING: class '{c}' has only {total} images "
                  f"(aim for {config.MIN_IMAGES_PER_CLASS}+).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=config.RAW_DIR)
    ap.add_argument("--out", type=Path, default=config.SPLIT_DIR)
    ap.add_argument("--group-by-prefix", action="store_true",
                    help="keep files sharing a name prefix (photo session) together")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete an existing split folder first")
    args = ap.parse_args()

    if args.out.exists() and not args.keep:
        shutil.rmtree(args.out)        # always rebuild so splits never go stale
    stats = build_split(args.raw, args.out, group_by_prefix=args.group_by_prefix)
    print_and_save_stats(stats, config.RESULTS_DIR / "dataset_stats.csv")


if __name__ == "__main__":
    main()
