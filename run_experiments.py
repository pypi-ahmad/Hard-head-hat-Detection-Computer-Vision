"""Run the agreed CUDA smoke check and full Small/Medium/Large comparison."""

import argparse
import gc
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from safety.paths import ARTIFACTS, DATASET
from safety.profiles import PROFILES, artifact_root, dataset_root
from train import train_safety_model


def smoke_dataset(profile="ppe"):
    root = artifact_root(profile) / "smoke_data"
    source_config = dataset_root(profile) / "data.yaml"
    data = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    source_root = source_config.parent if "path" not in data else Path(data["path"])
    manifest = []
    for source, target, count in (("train", "train", 8), ("valid", "valid", 8), ("test", "test", 2)):
        for kind in ("images", "labels"):
            (root / target / kind).mkdir(parents=True, exist_ok=True)
        image_dir = source_root / data["val" if source == "valid" else source]
        for image in sorted(image_dir.glob("*"))[:count]:
            shutil.copy2(image, root / target / "images" / image.name)
            label = image_dir.parent / "labels" / (image.stem + ".txt")
            shutil.copy2(label, root / target / "labels" / label.name)
            manifest.append({"image": image.name, "split": target,
                             "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                             "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()})
    config = root / "data.yaml"
    config.write_text(yaml.safe_dump({"train": "train/images", "val": "valid/images",
                                     "test": "test/images", "names": PROFILES[profile]["names"]}), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return config


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full comparison; no CPU fallback")
    ARTIFACTS.mkdir(exist_ok=True)
    journal = ARTIFACTS / "training_results.json"
    results = []
    smoke = train_safety_model(model="yolo26n.pt", data=smoke_dataset(), epochs=1,
                               batch=2, imgsz=320, device=0, name="smoke", workers=0)
    results.append({"model": "smoke", **smoke})
    journal.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not smoke["ok"]:
        raise RuntimeError(smoke)
    from PIL import Image

    from safety.inference import Detector

    detector = Detector(smoke["best_model"], device=0)
    with Image.open(next((DATASET / "css-data/test/images").glob("*.jpg"))) as image:
        detector.predict(image, 0.4)
    del detector
    for scale in "sml":
        for batch in (4, 2, 1):
            gc.collect()
            torch.cuda.empty_cache()
            result = train_safety_model(model=f"yolo26{scale}.pt", device=0,
                                        name=f"compare_yolo26{scale}", batch=batch)
            results.append({"model": f"yolo26{scale}", **result})
            journal.write_text(json.dumps(results, indent=2), encoding="utf-8")
            if result["ok"]:
                break
            if "out of memory" not in result.get("detail", "").lower():
                raise RuntimeError(result)
        else:
            raise RuntimeError(f"YOLO26{scale} does not fit even with batch size 1")
    print("Full training comparison complete", flush=True)
    from evaluate import compare

    compare()
    print("Evaluation and model selection complete", flush=True)


def run_profile(profile):
    """Audit, smoke check, train Small, evaluate, and verify the supplied videos."""
    from audit_dataset import audit
    from evaluate import compare
    from safety.datasets import sha256
    from verify_videos import verify_videos

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for profile training")
    root = artifact_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    profile_status(profile, "auditing")
    config = dataset_root(profile) / "data.yaml"
    report = audit(profile=profile)
    if (not report["decoded_images"] or report["issues"] or report["exact_cross_split_duplicates"]
            or any(split["orphaned_labels"] for split in report["splits"].values())):
        raise RuntimeError(f"Dataset audit failed; inspect {root / 'audit/audit.json'}")
    expected = json.loads((dataset_root(profile) / "manifest.json").read_text(encoding="utf-8"))
    observed = json.loads((root / "audit/manifest.json").read_text(encoding="utf-8"))
    expected_hashes = {row["image"]: (row["sha256"], row["label_sha256"]) for row in expected}
    observed_hashes = {row["path"]: (row["sha256"], row["label_sha256"]) for row in observed}
    if expected_hashes != observed_hashes:
        raise RuntimeError("Converted dataset changed since preparation; inspect the manifests")
    journal = root / "training_results.json"
    results = json.loads(journal.read_text(encoding="utf-8")) if journal.exists() else []
    previous = [row for row in results if row["ok"] and row["model"] == "yolo26s"]
    reusable = previous and previous[-1].get("dataset_config_sha256") == sha256(config)
    reusable = reusable and previous[-1].get("split_manifest_sha256") == sha256(dataset_root(profile) / "manifest.json")
    reusable = reusable and Path(previous[-1]["best_model"]).is_file()
    reusable = reusable and sha256(previous[-1]["best_model"]) == previous[-1]["checkpoint_sha256"]
    if not reusable:
        results = []
        profile_status(profile, "smoke_training")
        smoke = train_safety_model(model="yolo26n.pt", data=smoke_dataset(profile), epochs=1,
                                   batch=2, imgsz=320, device=0, project=root / "runs",
                                   name="smoke", workers=0, profile=profile)
        results.append({"model": "smoke", **smoke})
        journal.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if not smoke["ok"]:
            raise RuntimeError(smoke)
        from PIL import Image

        from safety.inference import Detector

        detector = Detector(smoke["best_model"], device=0)
        if detector.profile != profile:
            raise RuntimeError("Smoke checkpoint has incorrect classes")
        with Image.open(next((config.parent / "yolo/test/images").glob("*"))) as image:
            detector.annotate(image, .4)
        del detector
        for batch in (4, 2, 1):
            profile_status(profile, "training", model="yolo26s", batch=batch)
            gc.collect()
            torch.cuda.empty_cache()
            result = train_safety_model(model="yolo26s.pt", data=config, device=0,
                                        project=root / "runs", name="yolo26s", batch=batch,
                                        profile=profile)
            results.append({"model": "yolo26s", **result})
            journal.write_text(json.dumps(results, indent=2), encoding="utf-8")
            if result["ok"]:
                break
            if "out of memory" not in result.get("detail", "").lower():
                raise RuntimeError(result)
        else:
            raise RuntimeError("YOLO26 Small does not fit even with batch size 1")
    profile_status(profile, "evaluating")
    comparison = compare(device=0, profile=profile)
    profile_status(profile, "verifying_videos")
    verify_videos(profile, comparison["validation"]["checkpoint"],
                  comparison["validation"]["operating_point"]["confidence"])
    profile_status(profile, "complete", qualification=comparison["status"])
    print(f"{profile}: training, evaluation, and video verification complete", flush=True)


def profile_status(profile, state, **fields):
    root = artifact_root(profile)
    root.mkdir(parents=True, exist_ok=True)
    temp = root / "pipeline_status.json.tmp"
    temp.write_text(json.dumps({"state": state, "profile": profile,
        "updated_at": datetime.now(timezone.utc).isoformat(), **fields}, indent=2), encoding="utf-8")
    temp.replace(root / "pipeline_status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=list(PROFILES), default="shwd")
    args = parser.parse_args()
    if args.profile == "ppe":
        main()
    else:
        for profile in [args.profile]:
            try:
                run_profile(profile)
            except Exception as exc:
                profile_status(profile, "failed", detail=str(exc))
                raise
