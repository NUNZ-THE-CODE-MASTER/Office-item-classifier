"""Tests for the dataset split (run with: pytest)."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prepare_dataset import build_split  # noqa: E402


def make_raw(root, classes=6, per_class=20, session_prefix=False):
    """Create tiny dummy images: root/<class>/<name>.jpg."""
    for c in range(classes):
        d = root / f"class{c}"
        d.mkdir(parents=True)
        for i in range(per_class):
            name = f"s{i // 5}_{i}.jpg" if session_prefix else f"img{i}.jpg"
            Image.new("RGB", (8, 8), (c * 30, i * 10, 0)).save(d / name)


def files_in(split_dir, split, cls):
    return {p.name for p in (split_dir / split / cls).iterdir()}


def test_every_image_lands_in_exactly_one_split(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "split"
    make_raw(raw)
    stats = build_split(raw, out)
    for cls, counts in stats.items():
        assert sum(counts.values()) == 20
        assert all(n > 0 for n in counts.values())
        train, val, test = (files_in(out, s, cls) for s in ("train", "val", "test"))
        assert not (train & val) and not (train & test) and not (val & test)


def test_split_is_reproducible(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw)
    a = build_split(raw, tmp_path / "a", seed=1)
    b = build_split(raw, tmp_path / "b", seed=1)
    assert a == b
    assert files_in(tmp_path / "a", "test", "class0") == files_in(tmp_path / "b", "test", "class0")


def test_photo_sessions_are_not_split_across_sets(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "split"
    make_raw(raw, per_class=20, session_prefix=True)
    build_split(raw, out, group_by_prefix=True)
    for c in range(6):
        seen = {}
        for split in ("train", "val", "test"):
            for name in files_in(out, split, f"class{c}"):
                session = name.split("_")[0]
                assert seen.setdefault(session, split) == split
