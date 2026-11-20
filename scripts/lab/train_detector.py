"""Lab, after the competition: a detector for players, goalkeepers, referees and the ball.

Open data only. The frames come from the Hugging Face dataset martinjolif/football-player-detection
(CC BY 4.0 on its card; a Roboflow Universe export of 372 frames of football matches with YOLO
labels for ball, goalkeeper, player and referee), pinned to one revision. Nothing here reads the
competition's match data.

COCO-pretrained Ultralytics models are fine-tuned on the train split, with the valid split for
early stopping and checkpoint selection, at 1280 px because the ball is only a few pixels wide.
Every model gets the same settings and seed. Each is then scored once on the test split, which no
choice used: mAP50 and mAP50-95 overall and per class, with precision and recall. A few annotated
test frames are written to the notebook's output for a look.

Ultralytics is AGPL-3.0; this script only calls it. A job for Kaggle with a GPU; kaggle/run.py
installs requirements-cv.txt for it.

    python scripts/lab/train_detector.py [--models yolo11n yolo11s] [--epochs 120] [--imgsz 1280]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

REPO_ID = "martinjolif/football-player-detection"
REVISION = "e8b8cea002692efd74c945fcdad63e729adc5671"
CLASSES = ("ball", "goalkeeper", "player", "referee")
SPLITS = ("train", "valid", "test")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KAGGLE_OUTPUT = "/kaggle/working"


def dataset_yaml(root, out_path):
    """Write an Ultralytics data file pointing at the three splits under root; return out_path."""
    lines = ["path: " + root.replace("\\", "/"), "train: train/images", "val: valid/images",
             "test: test/images", "names:"]
    lines += ["  {}: {}".format(i, name) for i, name in enumerate(CLASSES)]
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def count_split(root, split):
    """(images, label files, boxes per class name) in one split of a YOLO dataset."""
    images = [n for n in os.listdir(os.path.join(root, split, "images"))
              if n.lower().endswith((".jpg", ".jpeg", ".png"))]
    label_dir = os.path.join(root, split, "labels")
    labels = [n for n in os.listdir(label_dir) if n.endswith(".txt")]
    boxes = {name: 0 for name in CLASSES}
    for name in labels:
        with open(os.path.join(label_dir, name), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    boxes[CLASSES[int(parts[0])]] += 1
    return len(images), len(labels), boxes


def fetch(target):
    """The dataset at the pinned revision under target; returns the folder holding the splits."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
                             local_dir=target)
    return os.path.join(path, "data")


def scored(metrics):
    """A JSON-ready summary of an Ultralytics detection validation."""
    box = metrics.box
    per_class = {}
    for i, index in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        per_class[CLASSES[int(index)]] = {"precision": round(float(p), 4),
                                          "recall": round(float(r), 4),
                                          "mAP50": round(float(ap50), 4),
                                          "mAP50-95": round(float(ap), 4)}
    return {"mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
            "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["yolo11n", "yolo11s"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "lab-detector"),
                        help="the report and each model's best weights")
    parser.add_argument("--work-dir", default=os.path.join(tempfile.gettempdir(), "lab-detector"),
                        help="the dataset and the training runs, outside the repository so the "
                             "Kaggle runner does not pack them")
    args = parser.parse_args()

    start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    data = fetch(os.path.join(args.work_dir, "dataset"))
    for split in SPLITS:
        images, labels, boxes = count_split(data, split)
        print("{:<5}: {} images, {} label files, boxes {}".format(split, images, labels, boxes))
    config = dataset_yaml(data, os.path.join(args.work_dir, "data.yaml"))
    print("dataset {} at {} ({:.0f} s)".format(REPO_ID, REVISION[:8], time.time() - start))

    # Kaggle's kernel sets MPLBACKEND to its notebook backend, which this venv lacks; matplotlib
    # (imported by Ultralytics) then refuses to load. Plots here only go to files.
    os.environ["MPLBACKEND"] = "Agg"
    import torch
    from ultralytics import YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    print("torch {} on {}".format(torch.__version__, torch.cuda.get_device_name(0)
                                  if device == 0 else "the CPU (slow: is the GPU on?)"))

    report = {}
    for name in args.models:
        model = YOLO(name + ".pt")
        model.train(data=config, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    patience=args.patience, seed=0, deterministic=True, device=device,
                    project=args.work_dir, name=name, exist_ok=True, plots=False, workers=2,
                    verbose=False)
        shutil.copy(model.trainer.best, os.path.join(args.out_dir, name + "-best.pt"))
        best = YOLO(model.trainer.best)
        valid = scored(best.val(data=config, split="val", imgsz=args.imgsz, batch=args.batch,
                                device=device, plots=False, verbose=False))
        test = scored(best.val(data=config, split="test", imgsz=args.imgsz, batch=args.batch,
                               device=device, plots=False, verbose=False))
        report[name] = {"epochs_run": int(model.trainer.epoch) + 1, "valid": valid, "test": test}
        print("\n{} after {} epochs ({:.0f} s so far)".format(name, report[name]["epochs_run"],
                                                               time.time() - start))
        print("  valid mAP50 {mAP50:.3f} mAP50-95 {mAP50-95:.3f}".format(**valid))
        print("  test  mAP50 {mAP50:.3f} mAP50-95 {mAP50-95:.3f} P {precision:.3f} "
              "R {recall:.3f}".format(**test))
        for cls, m in test["per_class"].items():
            print("    {:<10} mAP50 {mAP50:.3f} mAP50-95 {mAP50-95:.3f} P {precision:.3f} "
                  "R {recall:.3f}".format(cls, **m))
        if os.path.isdir(KAGGLE_OUTPUT):
            test_images = sorted(os.listdir(os.path.join(data, "test", "images")))[:4]
            best.predict([os.path.join(data, "test", "images", n) for n in test_images],
                         imgsz=args.imgsz, device=device, save=True, conf=0.25,
                         project=os.path.join(KAGGLE_OUTPUT, "lab-detector"), name=name,
                         exist_ok=True, verbose=False)
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print("\ndone in {:.0f} s".format(time.time() - start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
