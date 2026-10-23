"""Evaluate checkpoints, choose a validation threshold, and measure video throughput."""

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from audit_dataset import inspect_label
from safety.inference import Detector
from safety.paths import ARTIFACTS, BASELINE, DATASET, ROOT, SELECTION, project_path
from safety.profiles import PROFILES, artifact_root, dataset_root


def match_detections(predictions, labels, shape):
    """Confidence-ordered, one-to-one class-aware matches at IoU >= 0.5."""
    width, height = shape
    targets = np.asarray([[cls, (cx - w / 2) * width, (cy - h / 2) * height,
                           (cx + w / 2) * width, (cy + h / 2) * height]
                          for cls, cx, cy, w, h in labels], dtype=float).reshape(-1, 5)
    used, matched = set(), []
    for x1, y1, x2, y2, confidence, cls in sorted(predictions, key=lambda p: -p[4]):
        candidates = [i for i, row in enumerate(targets) if i not in used and row[0] == cls]
        best, best_iou = None, 0.0
        for index in candidates:
            _, tx1, ty1, tx2, ty2 = targets[index]
            intersection = max(0, min(x2, tx2) - max(x1, tx1)) * max(0, min(y2, ty2) - max(y1, ty1))
            union = max(0, x2 - x1) * max(0, y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - intersection
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best, best_iou = index, iou
        correct = best is not None and best_iou >= 0.5
        if correct:
            used.add(best)
        matched.append((int(cls), float(confidence), bool(correct)))
    return matched


def threshold_scores(matches, targets, threshold, profile="ppe"):
    names = PROFILES[profile]["names"]
    target_ids = PROFILES[profile]["targets"]
    predicted, correct = Counter(), Counter()
    for cls, confidence, match in matches:
        if confidence >= threshold:
            predicted[cls] += 1
            correct[cls] += int(match)
    per_class = {names[cls]: {"precision": correct[cls] / predicted[cls] if predicted[cls] else 0,
                              "recall": correct[cls] / targets[cls] if targets[cls] else 0,
                              "targets": targets[cls], "predictions": predicted[cls], "true_positives": correct[cls]}
                 for cls in range(len(names))}
    result = {"confidence": float(threshold), "per_class": per_class,
              "target_precision": float(np.mean([per_class[names[c]]["precision"] for c in target_ids])),
              "target_recall": float(np.mean([per_class[names[c]]["recall"] for c in target_ids]))}
    if profile == "ppe":
        result.update(violation_precision=result["target_precision"], violation_recall=result["target_recall"])
    return result


def benchmark_video(detector, confidence, path, frames=300):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open benchmark video: {path}")
    start_frame = max(0, (int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - frames - 20) // 2)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    samples = []
    use_cuda = detector.device != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        for index in range(frames + 20):
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Benchmark video contains no decodable frames")
            detector.annotate(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), confidence)
            if use_cuda:
                torch.cuda.synchronize()
            if index >= 20:
                samples.append(time.perf_counter() - started)
    finally:
        capture.release()
    return {"frames": len(samples), "mean_seconds": float(np.mean(samples)),
            "_samples": samples,
            "video": path.relative_to(ROOT).as_posix(),
            "start_frame": start_frame, "warmup_frames": 20,
            "p95_seconds": float(np.percentile(samples, 95)), "fps": len(samples) / sum(samples),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated() if use_cuda else 0,
            "device": detector.device,
            "includes": "decode, RGB conversion, prediction, annotation; excludes browser/network"}


def benchmark(detector, confidence, frames=300):
    sources = [ROOT / "custom Youtube video file by Mujtaba1_Part1Trim.mp4"]
    sources.extend(sorted((DATASET / "source_files/source_files").glob("*.mp4")))
    reports = [benchmark_video(detector, confidence, path, frames) for path in sources]
    samples = [value for report in reports for value in report.pop("_samples")]
    return {"frames": len(samples), "mean_seconds": float(np.mean(samples)),
            "p95_seconds": float(np.percentile(samples, 95)), "fps": len(samples) / sum(samples),
            "peak_gpu_bytes": max(report["peak_gpu_bytes"] for report in reports),
            "device": detector.device, "videos": reports,
            "includes": "decode, RGB conversion, prediction, annotation; excludes browser/network"}


def evaluate_checkpoint(checkpoint, split="val", confidence=None, output=None, device=None,
                        profile="ppe", data=None, imgsz=640):
    checkpoint = project_path(checkpoint)
    output = Path(output or artifact_root(profile) / "evaluation" / checkpoint.parent.parent.name / split)
    output.mkdir(parents=True, exist_ok=True)
    device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    detector = Detector(checkpoint, device=device, imgsz=imgsz)
    if detector.profile != profile:
        raise ValueError(f"Checkpoint classes do not match {profile}")
    config_path = Path(data) if data else dataset_root(profile) / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = PROFILES[profile]["names"]
    metrics = detector.model.val(data=str(config_path), split=split,
                                 imgsz=imgsz, batch=4, device=detector.device,
                                 plots=True, project=str(output), name="metrics", verbose=False)
    folder = (Path(config.get("path", config_path.parent)) / config[split]).parent
    matches, target_counts = [], Counter()
    class_ids = {key: names.index(name) for key, name in detector.model.names.items()}
    for path in sorted((folder / "images").glob("*")):
        labels, errors = inspect_label(folder / "labels" / (path.stem + ".txt"), len(names))
        if errors:
            raise ValueError(f"Invalid labels for {path.name}: {errors}")
        target_counts.update(box[0] for box in labels)
        with Image.open(path) as image:
            result = detector.predict(image, 0.001)
            predictions = result.boxes.data.numpy().copy()
            for prediction in predictions:
                prediction[5] = class_ids[int(prediction[5])]
            matches.extend(match_detections(predictions, labels, image.size))
    thresholds = [round(value / 100, 2) for value in range(1, 100)] if confidence is None else [confidence]
    scores = [threshold_scores(matches, target_counts, value, profile) for value in thresholds]
    qualifying = [score for score in scores if score["target_precision"] >= 0.8]
    best = max(qualifying, key=lambda s: (s["target_recall"], s["target_precision"])) if qualifying else None
    chosen = best or max(scores, key=lambda s: (s["target_recall"], s["target_precision"]))
    report = {"checkpoint": str(checkpoint), "split": split, "profile": profile,
              "target_classes": [names[i] for i in PROFILES[profile]["targets"]], "metrics": metrics.results_dict,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "per_class_map50_95": {detector.model.names[index]: float(value)
                                     for index, value in enumerate(metrics.box.maps)},
              "thresholds": scores, "qualifying": best, "operating_point": chosen,
              "benchmark": benchmark(detector, chosen["confidence"]) if split == "val" else None}
    (output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del detector, metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def compare(device=None, profile="ppe"):
    if profile != "ppe":
        return compare_profile(profile, device)
    training = json.loads((ARTIFACTS / "training_results.json").read_text(encoding="utf-8"))
    completed = {row["model"] for row in training if row["ok"]}
    missing = {"yolo26s", "yolo26m", "yolo26l"} - completed
    if missing:
        raise RuntimeError(f"Comparison is incomplete; successful training required for: {', '.join(sorted(missing))}")
    checkpoints = [BASELINE] + [Path(row["best_model"]) for row in training
                               if row["ok"] and row["model"] != "smoke"]
    reports = [evaluate_checkpoint(path, device=device) for path in checkpoints]
    qualified = [row for row in reports if row["qualifying"]]
    selected = max(qualified, key=lambda row: (row["qualifying"]["violation_recall"],
                                              row["qualifying"]["violation_precision"],
                                              -row["benchmark"]["mean_seconds"])) if qualified else None
    tests = []
    if selected:
        selection = {"checkpoint": Path(selected["checkpoint"]).relative_to(ROOT).as_posix(),
                     "confidence": selected["qualifying"]["confidence"],
                     "validation_precision": selected["qualifying"]["violation_precision"],
                     "validation_recall": selected["qualifying"]["violation_recall"],
                     "limitation": "Existing splits preserved; source-group overlap requires review."}
        tests.append(evaluate_checkpoint(selected["checkpoint"], "test", selection["confidence"], device=device))
        if project_path(selected["checkpoint"]) != BASELINE:
            baseline_threshold = reports[0]["operating_point"]["confidence"]
            tests.append(evaluate_checkpoint(BASELINE, "test", baseline_threshold, device=device))
        temporary = SELECTION.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        temporary.replace(SELECTION)
    summary = {"validation": reports, "selected": selected["checkpoint"] if selected else None,
               "test": tests, "criterion": "Maximum macro violation recall with >=80% macro precision",
               "dataset_limitations": "Existing splits retained. Read artifacts/audit/AUDIT.md."}
    (ARTIFACTS / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Model comparison", "", "| Checkpoint | Violation precision | Violation recall | Confidence | FPS | Qualifies |",
             "|---|---:|---:|---:|---:|---|"]
    for row in reports:
        point = row["operating_point"]
        lines.append(f"| {Path(row['checkpoint']).parent.parent.name} | {point['violation_precision']:.3f} | "
                     f"{point['violation_recall']:.3f} | {point['confidence']:.2f} | {row['benchmark']['fps']:.1f} | {bool(row['qualifying'])} |")
    lines += ["", f"Selected: {summary['selected'] or 'None; the existing default remains.'}", "",
              "Thresholds were selected on validation data. Test results are in comparison.json. "
              "Source-group overlap limits independence claims. FPS excludes browser and network time."]
    (ARTIFACTS / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def compare_profile(profile, device=None):
    root = artifact_root(profile)
    journal = json.loads((root / "training_results.json").read_text(encoding="utf-8"))
    completed = [row for row in journal if row["ok"] and row["model"] == "yolo26s"]
    if not completed:
        raise RuntimeError(f"Successful YOLO26 Small training required for {profile}")
    checkpoint = completed[-1]["best_model"]
    validation = evaluate_checkpoint(checkpoint, device=device, profile=profile)
    point = validation["operating_point"]
    test = evaluate_checkpoint(checkpoint, "test", point["confidence"], device=device, profile=profile)
    qualified = validation["qualifying"] is not None
    summary = {"profile": profile, "validation": validation, "test": test,
               "selected": checkpoint if qualified else None,
               "criterion": "Maximum target-class macro recall with >=80% macro precision",
               "status": "qualified" if qualified else "experimental"}
    (root / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# {PROFILES[profile]['title']}: YOLO26 Small", "",
             f"Status: {summary['status']}. Validation-selected confidence: {point['confidence']:.2f}.", "",
             "Selection maximizes recall across " + ", ".join(validation["target_classes"]) +
             " with at least 80% validation precision.", "",
             "| Split | Target precision | Target recall |", "|---|---:|---:|"]
    for split_name, report in (("Validation", validation), ("Test", test)):
        score = report["operating_point"]
        lines.append(f"| {split_name} | {score['target_precision']:.3f} | {score['target_recall']:.3f} |")
    lines += ["", "| Test class | Precision | Recall | Targets | mAP50–95 |",
              "|---|---:|---:|---:|---:|"]
    for name, score in test["operating_point"]["per_class"].items():
        lines.append(f"| {name} | {score['precision']:.3f} | {score['recall']:.3f} | "
                     f"{score['targets']} | {test['per_class_map50_95'][name]:.3f} |")
    lines += ["", f"Video benchmark: {validation['benchmark']['fps']:.1f} FPS over "
              f"{validation['benchmark']['frames']:,} measured frames.",
              "", "Public-split scores do not establish deployment accuracy. Review dataset source overlap "
              "and sparse classes before interpreting these results. Unlabeled videos measure behavior and throughput."]
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if qualified:
        selection = {"profile": profile, "checkpoint": Path(checkpoint).relative_to(ROOT).as_posix(),
                     "confidence": point["confidence"], "validation_precision": point["target_precision"],
                     "validation_recall": point["target_recall"]}
        temp = root / "selected_model.json.tmp"
        temp.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        temp.replace(root / "selected_model.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--profile", choices=list(PROFILES), default="shwd")
    args = parser.parse_args()
    if args.weights:
        result = evaluate_checkpoint(args.weights, args.split, args.confidence, device=args.device, profile=args.profile)
        print(json.dumps(result["metrics"], indent=2))
    else:
        compare(device=args.device, profile=args.profile)
