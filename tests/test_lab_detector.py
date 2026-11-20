"""scripts/lab/train_detector.py: the data file it writes and the box count it reports. Neither
needs Ultralytics, which only the Kaggle job installs."""

import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "train_detector", os.path.join(ROOT, "scripts", "lab", "train_detector.py"))
detector = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(detector)


def test_the_data_file_names_the_splits_and_the_four_classes(tmp_path):
    out = detector.dataset_yaml(str(tmp_path / "data"), str(tmp_path / "data.yaml"))
    text = open(out, encoding="utf-8").read()
    assert "train: train/images" in text and "val: valid/images" in text
    assert "test: test/images" in text
    assert [line.strip() for line in text.splitlines()[-4:]] == [
        "0: ball", "1: goalkeeper", "2: player", "3: referee"]


def test_boxes_are_counted_per_class(tmp_path):
    for split in ("train",):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        (tmp_path / split / "images" / "a.jpg").write_bytes(b"")
        (tmp_path / split / "images" / "b.png").write_bytes(b"")
        (tmp_path / split / "labels" / "a.txt").write_text(
            "2 0.5 0.5 0.1 0.2\n2 0.4 0.4 0.1 0.2\n0 0.3 0.3 0.01 0.01\n", encoding="utf-8")
        (tmp_path / split / "labels" / "b.txt").write_text("3 0.1 0.1 0.1 0.1\n\n",
                                                           encoding="utf-8")
    images, labels, boxes = detector.count_split(str(tmp_path), "train")
    assert (images, labels) == (2, 2)
    assert boxes == {"ball": 1, "goalkeeper": 0, "player": 2, "referee": 1}
