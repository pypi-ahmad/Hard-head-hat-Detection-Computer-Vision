"""Durable human review; prediction drafts are never exported as ground truth."""

import json
import math
import shutil
import sqlite3
from pathlib import Path

import cv2
import yaml
from PIL import Image

from safety.datasets import sha256
from safety.paths import ARTIFACTS

REVIEW_ROOT = ARTIFACTS / "camera_review"


class ReviewStore:
    def __init__(self, root=REVIEW_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sources (
                    hash TEXT PRIMARY KEY, path TEXT, group_id TEXT, split TEXT);
                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY, source TEXT, timestamp REAL, image TEXT,
                    hash TEXT, width INTEGER, height INTEGER, boxes TEXT,
                    status TEXT DEFAULT 'unreviewed', revision INTEGER DEFAULT 0);
                CREATE TABLE IF NOT EXISTS intervals (
                    source TEXT, start REAL, end REAL, box TEXT);
                CREATE TABLE IF NOT EXISTS event_reviews (source TEXT PRIMARY KEY);
            ''')

    def connect(self):
        db = sqlite3.connect(self.root / "review.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def sources(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM sources ORDER BY path")]

    def frames(self):
        with self.connect() as db:
            return [dict(row) | {"boxes": json.loads(row["boxes"])}
                    for row in db.execute("SELECT * FROM frames ORDER BY source, timestamp")]

    def assign(self, source, group, split):
        if not group.strip() or split not in ("train", "val", "test", None):
            raise ValueError("A source group and valid split are required")
        with self.connect() as db:
            db.execute("UPDATE sources SET group_id=?, split=? WHERE hash=?", (group, split, source))

    def extract(self, source, detector=None, timestamps=None):
        source = Path(source).resolve()
        digest = sha256(source)
        capture = cv2.VideoCapture(str(source))
        try:
            fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not capture.isOpened() or fps <= 0 or count <= 0:
                raise ValueError(f"Invalid video: {source}")
            with self.connect() as db:
                db.execute("INSERT OR IGNORE INTO sources VALUES (?, ?, ?, NULL)",
                           (digest, str(source), digest))
            times = timestamps if timestamps is not None else range(math.ceil(count / fps / 2))
            for value in times:
                timestamp = float(value) if timestamps is not None else value * 2.
                if not math.isfinite(timestamp) or not 0 <= timestamp < count / fps:
                    raise ValueError("Timestamp outside video")
                index = round(timestamp * fps)
                key = f"{digest[:16]}-{index:08d}"
                with self.connect() as db:
                    if db.execute("SELECT 1 FROM frames WHERE id=?", (key,)).fetchone():
                        continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode frame {index}")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes = []
                if detector:
                    result = detector.predict(image, .12)
                    boxes = [{"class": result.names[int(b.cls[0])],
                              "xyxy": list(map(float, b.xyxy[0]))} for b in result.boxes]
                folder = self.root / "images"
                folder.mkdir(exist_ok=True)
                path = folder / f"{key}.jpg"
                image.save(path, quality=95)
                with self.connect() as db:
                    db.execute("INSERT OR IGNORE INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unreviewed', 0)",
                        (key, digest, index / fps, str(path), sha256(path), *image.size, json.dumps(boxes)))
        finally:
            capture.release()

    def save(self, frame, boxes, status):
        if status not in ("unreviewed", "approved", "excluded"):
            raise ValueError("Invalid review status")
        validate_boxes(boxes, frame["width"], frame["height"])
        with self.connect() as db:
            cursor = db.execute("UPDATE frames SET boxes=?, status=?, revision=revision+1 "
                "WHERE id=? AND revision=?", (json.dumps(boxes), status, frame["id"], frame["revision"]))
            if cursor.rowcount != 1:
                raise ValueError("Another review changed this frame; reload before saving")

    def save_intervals(self, source, intervals, duration):
        for interval in intervals:
            if not (0 <= interval["start"] < interval["end"] <= duration):
                raise ValueError("Event interval must be within video duration")
            validate_boxes([{"class": "bare_head", "xyxy": interval["box"]}], 1, 1)
        with self.connect() as db:
            db.execute("DELETE FROM intervals WHERE source=?", (source,))
            db.executemany("INSERT INTO intervals VALUES (?, ?, ?, ?)",
                [(source, row["start"], row["end"], json.dumps(row["box"])) for row in intervals])
            db.execute("INSERT OR IGNORE INTO event_reviews VALUES (?)", (source,))

    def events_reviewed(self, source):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM event_reviews WHERE source=?", (source,)).fetchone() is not None

    def intervals(self, source):
        with self.connect() as db:
            return [dict(row) | {"box": json.loads(row["box"])} for row in db.execute(
                "SELECT start, end, box FROM intervals WHERE source=?", (source,))]

    def export(self, destination):
        destination = Path(destination).resolve()
        if destination.exists():
            raise ValueError("Choose a new version directory; existing exports are immutable")
        sources = {s["hash"]: s for s in self.sources()}
        approved = [f for f in self.frames() if f["status"] == "approved"]
        if not approved:
            raise ValueError("No approved annotations; review frames first")
        groups, hashes, splits = {}, {}, set()
        for frame in approved:
            source = sources[frame["source"]]
            split, group = source["split"], source["group_id"]
            if split not in ("train", "val", "test"):
                raise ValueError("Assign every approved source to a split")
            for mapping, key in ((groups, group), (hashes, frame["hash"])):
                if key in mapping and mapping[key] != split:
                    raise ValueError("Related source groups or duplicate images cross splits")
                mapping[key] = split
            if sha256(frame["image"]) != frame["hash"]:
                raise ValueError("Review image changed on disk")
            validate_boxes(frame["boxes"], frame["width"], frame["height"])
            splits.add(split)
        if splits != {"train", "val", "test"}:
            raise ValueError("Approved train, validation, and test sources are required")
        destination.mkdir(parents=True)
        manifest = []
        for frame in approved:
            split = sources[frame["source"]]["split"]
            images, labels = destination / split / "images", destination / split / "labels"
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frame["image"], images / f"{frame['id']}.jpg")
            lines = []
            for box in frame["boxes"]:
                x1, y1, x2, y2 = box["xyxy"]
                w, h = frame["width"], frame["height"]
                lines.append(f"{['helmet', 'bare_head'].index(box['class'])} "
                             f"{(x1+x2)/2/w} {(y1+y2)/2/h} {(x2-x1)/w} {(y2-y1)/h}\n")
            label = labels / f"{frame['id']}.txt"
            label.write_text("".join(lines), encoding="utf-8")
            manifest.append(frame | {"split": split, "group": sources[frame["source"]]["group_id"],
                                     "label_sha256": sha256(label)})
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        config = destination / "data.yaml"
        config.write_text(yaml.safe_dump({"path": str(destination), "train": "train/images",
            "val": "val/images", "test": "test/images", "names": ["helmet", "bare_head"]}), encoding="utf-8")
        return config


def validate_boxes(boxes, width, height):
    for box in boxes:
        if box["class"] not in ("helmet", "bare_head"):
            raise ValueError("Unknown annotation class")
        x1, y1, x2, y2 = box["xyxy"]
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            raise ValueError("Box coordinates must be finite")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Box must have positive area inside the image")
