"""Train a PPE checkpoint and report the files actually produced by Ultralytics."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from safety.paths import ARTIFACTS, DATASET, project_path
from safety.profiles import PROFILES, artifact_root, dataset_root


def validate_dataset(config):
    config = project_path(config)
    if not config.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config}")
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Dataset config must be a YAML mapping")
    root = Path(data.get("path", config.parent))
    if not root.is_absolute():
        raise ValueError("Dataset root must be absolute or omitted; relative roots are ambiguous")
    for split in ("train", "val", "test"):
        location = (root / data[split]).resolve()
        if not location.is_dir() or not any(location.iterdir()):
            raise ValueError(f"Missing or empty {split} image directory: {location}")
    return config


def train_safety_model(model="yolo26l.pt", data=None, epochs=50,
                       batch=4, imgsz=640, device=None, seed=42, patience=10,
                       project=None, name="safety_model", workers=2,
                       profile="ppe"):
    started = time.monotonic()
    data = data or dataset_root(profile) / "data.yaml"
    project = project or artifact_root(profile) / "runs"
    try:
        config = validate_dataset(data)
        names = yaml.safe_load(config.read_text(encoding="utf-8")).get("names")
        names = list(names.values()) if isinstance(names, dict) else names
        if names != PROFILES[profile]["names"]:
            raise ValueError(f"Dataset class order does not match {profile}")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        return {"ok": False, "error": "missing_data_yaml" if not project_path(data).exists()
                else "invalid_dataset", "detail": str(exc)}
    device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    try:
        detector = YOLO(str(project_path(model)))
    except Exception as exc:
        return {"ok": False, "error": "model_load_failed", "detail": str(exc)}
    arguments = dict(data=str(config), epochs=epochs, imgsz=imgsz, batch=batch,
                     device=device, workers=workers, project=str(project_path(project)),
                     name=name, save=True, plots=True, seed=seed, deterministic=True,
                     patience=patience, amp=True)
    try:
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        detector.train(**arguments)
        best = Path(detector.trainer.best).resolve()
        if not best.is_file():
            raise FileNotFoundError(f"Training produced no best checkpoint: {best}")
        manifest = config.parent / "manifest.json"
        if profile == "ppe" and config.parent == DATASET:
            manifest = ARTIFACTS / "audit/manifest.json"
        with best.open("rb") as handle:
            checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        result = {
            "ok": True, "best_model": str(best), "device": device, "profile": profile,
            "save_dir": str(best.parent.parent), "arguments": arguments,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_sha256": checkpoint_hash,
            "dataset_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
            if manifest.is_file() else None,
            "python": platform.python_version(),
            "versions": {package: importlib.metadata.version(package)
                         for package in ("torch", "ultralytics", "numpy", "pillow")},
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()
            if device != "cpu" and torch.cuda.is_available() else 0,
        }
        (best.parent.parent / "run_metadata.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        return {"ok": False, "error": "train_failed", "detail": str(exc),
                "arguments": arguments, "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--data")
    parser.add_argument("--profile", choices=list(PROFILES), default="shwd")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project")
    parser.add_argument("--name", default="safety_model")
    parser.add_argument("--workers", type=int, default=2)
    args = vars(parser.parse_args())
    args["data"] = args["data"] or dataset_root(args["profile"]) / "data.yaml"
    args["project"] = args["project"] or artifact_root(args["profile"]) / "runs"
    result = train_safety_model(**args)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
