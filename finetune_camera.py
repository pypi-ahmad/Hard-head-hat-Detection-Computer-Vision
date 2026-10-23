"""Train reviewed camera data; evaluate frozen holdouts before publishing a candidate."""

import argparse
import json
import shutil
from pathlib import Path

import yaml

from evaluate import evaluate_checkpoint
from safety.datasets import sha256
from safety.inference import selected_checkpoint
from safety.paths import ARTIFACTS
from safety.profiles import dataset_root
from train import train_safety_model


def prepare_revision(config, output):
    config = Path(config).resolve()
    rows = json.loads((config.parent / "manifest.json").read_text(encoding="utf-8"))
    if not rows or any(row.get("status") != "approved" for row in rows):
        raise ValueError("Only an approved review export can be used for fine-tuning")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if root != config.parent:
        raise ValueError("Revision config must describe its own directory")
    hashes, groups = {}, {}
    for row in rows:
        split = row["split"]
        for mapping, key in ((hashes, row["hash"]), (groups, row["group"])):
            if key in mapping and mapping[key] != split:
                raise ValueError("Cross-split source group or duplicate")
            mapping[key] = split
        image = root / split / "images" / f"{row['id']}.jpg"
        label = root / split / "labels" / f"{row['id']}.txt"
        if sha256(image) != row["hash"] or sha256(label) != row["label_sha256"]:
            raise ValueError("Reviewed revision has changed")
    for split in ("train", "val", "test"):
        expected = {f"{row['id']}.jpg" for row in rows if row["split"] == split}
        actual = {p.name for p in (root / split / "images").iterdir() if p.is_file()}
        if not expected or actual != expected:
            raise ValueError("Empty split or images outside the approved manifest")
    if output.exists():
        raise ValueError("Run directory already exists; preserve previous experiments")
    # Include only the original SHWD training split; camera holdouts remain frozen.
    source = dataset_root("shwd")
    base = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    training_files = []
    for row in base:
        if row["split"] != "train" or row["sha256"] in hashes:
            continue
        image = source / row["image"]
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        if sha256(image) != row["sha256"] or sha256(label) != row["label_sha256"]:
            raise ValueError(f"SHWD source changed since preparation: {image}")
        training_files.append((image, label))
    shutil.copytree(root, output / "data")
    mixed = output / "data"
    for image, label in training_files:
        shutil.copy2(image, mixed / "train/images" / f"shwd-{image.name}")
        shutil.copy2(label, mixed / "train/labels" / f"shwd-{label.name}")
    data["path"] = str(mixed.resolve())
    target = mixed / "data.yaml"
    target.write_text(yaml.safe_dump(data), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({"camera_manifest": sha256(config.parent / "manifest.json"),
        "shwd_manifest": sha256(source / "manifest.json")}, indent=2), encoding="utf-8")
    return target


def run(config, name):
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Use a single run name")
    output = ARTIFACTS / "camera_training" / name
    data = prepare_revision(config, output)
    previous, _ = selected_checkpoint("shwd")
    if previous is None:
        raise ValueError("Existing SHWD checkpoint required")
    reports = []
    def measure(checkpoint, tag, split="val", confidence=None, imgsz=640):
        return evaluate_checkpoint(checkpoint, split, confidence, output / tag / split,
                                   profile="shwd", data=data, imgsz=imgsz)
    baseline = measure(previous, "baseline")
    winner, resolution = previous, 640
    best = baseline["operating_point"]
    for model, tag, size in ((previous, "small640", 640), (previous, "small960", 960),
                             ("yolo26m.pt", "medium960", 960)):
        if tag != "small640" and best["target_recall"] >= .95 and best["target_precision"] >= .8:
            break
        for batch in (4, 2, 1):
            result = train_safety_model(model=model, data=data, profile="shwd", epochs=50,
                patience=10, seed=42, batch=batch, imgsz=size, project=output / "runs", name=tag)
            if result["ok"]:
                break
            if "out of memory" not in result.get("detail", "").lower():
                raise RuntimeError(result)
        if not result["ok"]:
            raise RuntimeError(result)
        report = measure(result["best_model"], tag, imgsz=size)
        reports.append(report)
        point = report["qualifying"]
        if point and (best["target_precision"] < .8 or
            (point["target_recall"], point["target_precision"]) >
                (best["target_recall"], best["target_precision"])):
            winner, best, resolution = result["best_model"], point, size
    old_test = measure(previous, "baseline", "test", baseline["operating_point"]["confidence"])
    candidate_test = measure(winner, "candidate", "test", best["confidence"], resolution)
    point = candidate_test["operating_point"]
    publish = (str(winner) != str(previous) and point["target_precision"] >= .8
               and point["target_recall"] >= max(.95, old_test["operating_point"]["target_recall"]))
    (output / "comparison.json").write_text(json.dumps({"baseline": baseline, "candidates": reports,
        "baseline_test": old_test, "candidate_test": candidate_test, "published": publish}, indent=2), encoding="utf-8")
    if publish:
        selection = ARTIFACTS / "shwd/selected_model.json"
        shutil.copy2(selection, output / "previous_selection.json")
        temporary = selection.with_suffix(".tmp")
        temporary.write_text(json.dumps({"checkpoint": str(winner), "confidence": best["confidence"],
            "imgsz": resolution, "reviewed_revision": str(data)}, indent=2), encoding="utf-8")
        temporary.replace(selection)
    print(f"Evaluation complete. Published replacement: {publish}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    run(args.data, args.name)
