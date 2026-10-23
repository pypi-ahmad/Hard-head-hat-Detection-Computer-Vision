"""Evaluate reviewed full-frame helmet-required video intervals."""

import argparse
import json
import time

import cv2
import numpy as np
from PIL import Image

from safety.datasets import sha256
from safety.event_evaluation import event_scores
from safety.events import StreamDetector
from safety.inference import Detector, selected_checkpoint, selected_resolution
from safety.review import ReviewStore


def evaluate_source(store, source, detector, confidence):
    if not store.events_reviewed(source["hash"]):
        raise ValueError("Review the complete video's event intervals before evaluation")
    if sha256(source["path"]) != source["hash"]:
        raise ValueError("Reviewed video has changed")
    processor = StreamDetector(detector, confidence, (0, 0, 1, 1), source["path"])
    capture = cv2.VideoCapture(source["path"])
    samples, count = [], 0
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or expected <= 0:
            raise ValueError("Invalid video metadata")
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            source_time = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            processor.annotate(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                               confidence, source_time if source_time > 0 else count/fps)
            samples.append(time.perf_counter()-started)
            count += 1
        if count != expected:
            raise ValueError("Incomplete video decode")
        processor.finish()
        result = event_scores(processor.monitor.events, store.intervals(source["hash"]), count/fps, size)
        result.update(source=source, frames=count, confidence=confidence,
            processing_fps=count/sum(samples), p95_seconds=float(np.percentile(samples, 95)),
            imgsz=detector.imgsz, device=str(detector.device),
            scope="Whole frame confirmed helmet-required by reviewer; all frames processed sequentially")
        (processor.directory / "evaluation.json").parent.mkdir(parents=True, exist_ok=True)
        (processor.directory / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hash", required=True)
    parser.add_argument("--whole-frame-helmet-required", action="store_true", required=True)
    args = parser.parse_args()
    store = ReviewStore()
    source = next(s for s in store.sources() if s["hash"] == args.source_hash)
    checkpoint, confidence = selected_checkpoint()
    if checkpoint is None:
        raise ValueError("SHWD checkpoint required")
    print(json.dumps(evaluate_source(store, source,
        Detector(checkpoint, imgsz=selected_resolution()), confidence), indent=2))
