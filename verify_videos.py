"""Annotate and fully decode the four supplied videos with one profile checkpoint."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

from safety.inference import Detector, selected_checkpoint, selected_resolution
from safety.paths import DATASET, ROOT
from safety.profiles import PROFILES, artifact_root


def verify_video(detector, source, output, confidence):
    from safety.events import StreamDetector

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {source}")
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or expected <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {source}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create {output}")
    count, violations, equipment = 0, Counter(), Counter()
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            source_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if source_time <= 0:
                source_time = count / fps
            annotated, reasons, items = (detector.annotate(image, confidence, source_time)
                if isinstance(detector, StreamDetector) else detector.annotate(image, confidence))
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            violations.update(reasons)
            equipment.update(items)
            if count in (0, expected // 2, expected - 1):
                Image.fromarray(annotated).save(output.with_name(f"{output.stem}_{count:05d}.jpg"))
            count += 1
    finally:
        capture.release()
        writer.release()
        if isinstance(detector, StreamDetector):
            detector.finish()
    elapsed = time.perf_counter() - started
    decoded, check = 0, cv2.VideoCapture(str(output))
    try:
        while check.read()[0]:
            decoded += 1
    finally:
        check.release()
    if decoded != count or count != expected:
        raise RuntimeError(f"Incomplete video: expected {expected}, processed {count}, decoded {decoded}")
    return {"source": str(source), "output": str(output), "frames": count,
            "output_decoded_frames": decoded, "elapsed_seconds": elapsed,
            "processing_fps": count / elapsed, "violation_detections": dict(violations),
            "equipment_detections": dict(equipment)}


def verify_videos(profile="ppe", checkpoint=None, confidence=None, device=0):
    if checkpoint is None:
        checkpoint, default_confidence = selected_checkpoint(profile)
        confidence = default_confidence if confidence is None else confidence
    if checkpoint is None or confidence is None:
        raise ValueError("A trained checkpoint and confidence are required")
    detector = Detector(checkpoint, device=device, imgsz=selected_resolution(profile))
    if detector.profile != profile:
        raise ValueError("Checkpoint does not match the requested profile")
    output = artifact_root(profile) / "videos"
    output.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "custom Youtube video file by Mujtaba1_Part1Trim.mp4"]
    sources += sorted((DATASET / "source_files/source_files").glob("*.mp4"))
    if len(sources) != 4:
        raise ValueError("Expected the custom clip and three source_files videos")
    reports = []
    for source in sources:
        from safety.events import StreamDetector

        processor = StreamDetector(detector, confidence, source=str(source)) if profile == "shwd" else detector
        reports.append(verify_video(processor, source, output / f"{source.stem}.mp4", confidence))
        print(f"{profile}: verified {source.name}, {reports[-1]['frames']} frames", flush=True)
    report = {"profile": profile, "checkpoint": str(checkpoint), "confidence": confidence,
              "device": device, "videos": reports,
              "limitation": "Unlabeled videos: counts repeat across frames and do not measure accuracy or unique workers."}
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=list(PROFILES), default="shwd")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--device", default=0)
    verify_videos(**vars(parser.parse_args()))
