"""Per-stream tracking and timestamp-based helmet-required area events."""

import csv
import io
import json
import time
from dataclasses import asdict, dataclass
from threading import RLock
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from safety.annotations import draw_safety_box
from safety.paths import ARTIFACTS


@dataclass
class Detection:
    box: list[float]
    name: str
    confidence: float
    track_id: int | None = None


class EventMonitor:
    def __init__(self, zone=None, confidence=.12):
        self.zone, self.confidence = zone, confidence
        self.pending, self.events = {}, []
        self.last_time = None

    def update(self, detections, timestamp, size):
        if self.last_time is not None and timestamp < self.last_time:
            raise ValueError("Stream timestamps must be monotonic")
        self.last_time = timestamp
        if self.zone is None:
            return []
        width, height = size
        seen, opened = set(), []
        for detection in detections:
            if detection.track_id is None or detection.confidence < self.confidence:
                continue
            x1, y1, x2, y2 = detection.box
            x, y = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
            inside = self.zone[0] <= x <= self.zone[2] and self.zone[1] <= y <= self.zone[3]
            state = self.pending.get(detection.track_id)
            if detection.name != "bare_head" or not inside:
                if state:
                    state["reason"] = "helmet observed" if inside else "area exited"
                continue
            seen.add(detection.track_id)
            if state and timestamp - state["last"] >= 1:
                self._close(detection.track_id, state["last"] + 1, state["reason"])
                state = None
            if state is None:
                state = {"start": timestamp, "last": timestamp, "evidence": 0.,
                         "event": None, "reason": "track lost", "observed": False}
                self.pending[detection.track_id] = state
            gap = timestamp - state["last"]
            if gap > .5:
                state["evidence"] = 0.
            elif state["observed"]:
                state["evidence"] += gap
            state["observed"] = True
            state["last"], state["reason"] = timestamp, "track lost"
            if state["evidence"] >= 1 and state["event"] is None:
                event = {"id": len(self.events) + 1, "track_id": detection.track_id,
                         "start": state["start"], "confirmed": timestamp, "end": None,
                         "confidence": detection.confidence, "box": detection.box,
                         "reason": None}
                self.events.append(event)
                state["event"] = event
                opened.append(event)
        for track_id, state in list(self.pending.items()):
            if track_id not in seen:
                state["observed"] = False
            if track_id not in seen and timestamp - state["last"] >= 1:
                self._close(track_id, state["last"] + 1, state["reason"])
        return opened

    def _close(self, track_id, timestamp, reason):
        state = self.pending.pop(track_id)
        if state["event"]:
            state["event"].update(end=timestamp, reason=reason)

    def finish(self, timestamp, reason="source ended"):
        for track_id in list(self.pending):
            self._close(track_id, timestamp, reason)


class StreamDetector:
    """Share inference only; each stream owns its tracker, clock, and event journal."""

    def __init__(self, detector, confidence=.12, zone=None, source="stream"):
        self.detector, self.confidence = detector, confidence
        self.monitor = EventMonitor(zone, confidence)
        self.tracker = BYTETracker(SimpleNamespace(track_high_thresh=confidence,
            track_low_thresh=min(.05, confidence), new_track_thresh=confidence,
            track_buffer=30, match_thresh=.8, fuse_score=True))
        self.source = source
        self.directory = ARTIFACTS / "events" / uuid4().hex
        self.detections = []
        self.finished = False
        self.time_reference = None
        self.lock = RLock()

    def restart(self):
        self.finish()
        self.__init__(self.detector, self.confidence, self.monitor.zone, self.source)

    def annotate(self, image, confidence, timestamp=None):
        with self.lock:
            return self._annotate(image, confidence, timestamp)

    def _annotate(self, image, confidence, timestamp):
        if confidence != self.confidence:
            raise ValueError("Restart the stream after changing confidence")
        timestamp = time.monotonic() if timestamp is None else timestamp
        if self.time_reference is None:
            self.time_reference = {"stream_timestamp": timestamp, "unix_timestamp": time.time()}
        result = self.detector.predict(image, min(.05, confidence))
        tracks = self.tracker.update(result.boxes.numpy(), np.asarray(image))
        track_ids = {int(row[-1]): int(row[4]) for row in tracks}
        self.detections = [Detection(list(map(float, box.xyxy[0])),
            result.names[int(box.cls[0])], float(box.conf[0]), track_ids.get(index))
            for index, box in enumerate(result.boxes)]
        before = [(e["id"], e["end"]) for e in self.monitor.events]
        opened = self.monitor.update(self.detections, timestamp, image.size)
        for event in opened:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"event-{event['id']}.jpg"
            image.save(path)
            event["image"] = str(path)
        if before != [(e["id"], e["end"]) for e in self.monitor.events]:
            self.save()
        canvas, heads, helmets = draw_safety_box(image, result, confidence, True)
        for detection in self.detections:
            if detection.track_id is not None and detection.confidence >= confidence:
                x, y = map(int, detection.box[:2])
                cv2.putText(canvas, f"ID {detection.track_id}", (x, max(16, y-22)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        if self.monitor.zone:
            a, b, c, d = self.monitor.zone
            cv2.rectangle(canvas, (int(a*image.width), int(b*image.height)),
                          (int(c*image.width), int(d*image.height)), (255, 200, 0), 2)
        return canvas, heads, helmets

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / "events.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(self.json(), encoding="utf-8")
        temporary.replace(target)

    def finish(self):
        with self.lock:
            self.monitor.finish(self.monitor.last_time or 0)
            self.finished = True
            if self.monitor.events:
                self.save()

    def json(self):
        with self.lock:
            return json.dumps({"source": self.source, "zone": self.monitor.zone,
                           "finished": self.finished,
                           "time_reference": self.time_reference,
                           "events": self.monitor.events,
                           "detections": [asdict(d) for d in self.detections]}, indent=2)

    def csv(self):
        with self.lock:
            return self._csv()

    def _csv(self):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["id", "track_id", "start", "confirmed",
            "end", "confidence", "reason"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.monitor.events)
        return output.getvalue()
