"""Continuous camera ingestion, bounded previews, and local event retention."""

import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from urllib.parse import urlsplit

import av
import yaml
from PIL import Image

from safety.events import StreamDetector
from safety.paths import ARTIFACTS, project_path

RUNTIME = ARTIFACTS / "cctv"


@dataclass(frozen=True)
class Camera:
    id: str
    name: str
    source: str = field(repr=False)
    zone: tuple | None = None
    file: bool = False


@dataclass(frozen=True)
class Settings:
    cameras: tuple
    confidence: float = .12
    inference_fps: float = 10
    retention_days: float = 7
    storage_mb: float = 2048


def load_settings(path):
    """Only environment references are accepted for private camera URLs."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Camera configuration must be a mapping")
    allowed = {"cameras", "confidence", "inference_fps", "retention_days", "storage_mb"}
    if set(data) - allowed or not isinstance(data.get("cameras", []), list):
        raise ValueError("Unknown configuration option or invalid camera list")
    cameras, ids = [], set()
    for row in data.get("cameras", []):
        if not isinstance(row, dict) or set(row) - {"id", "name", "url_env", "helmet_required_zone"}:
            raise ValueError("Invalid camera entry or unknown camera option")
        camera_id = row.get("id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}", camera_id) or camera_id in ids:
            raise ValueError("Camera IDs must be unique lowercase names, digits, underscores or hyphens")
        ids.add(camera_id)
        name = row.get("name", camera_id)
        if not isinstance(name, str) or not 1 <= len(name) <= 80:
            raise ValueError(f"Invalid display name for camera {camera_id}")
        variable = row.get("url_env", "")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", variable):
            raise ValueError(f"Camera {camera_id} needs a url_env variable name")
        url = os.environ.get(variable, "")
        if not url:
            raise ValueError(f"Camera {camera_id}: required environment variable {variable} is missing")
        try:
            parts = urlsplit(url)
            valid = parts.scheme in ("rtsp", "rtsps", "http", "https") and parts.hostname
            valid = valid and not parts.fragment and (parts.port is None or parts.port > 0)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError(f"Camera {camera_id}: use an RTSP or HTTP(S) camera stream")
        zone = row.get("helmet_required_zone")
        if zone is not None:
            if (not isinstance(zone, list) or len(zone) != 4
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in zone)
                    or not 0 <= zone[0] < zone[2] <= 1 or not 0 <= zone[1] < zone[3] <= 1):
                raise ValueError(f"Camera {camera_id}: zone must be normalized [left, top, right, bottom]")
            zone = tuple(zone)
        cameras.append(Camera(camera_id, name, url, zone))
    values = {}
    for key, default, low, high in (("confidence", .12, .01, 1),
            ("inference_fps", 10, 3, 30), ("retention_days", 7, .01, 365),
            ("storage_mb", 2048, 10, 1048576)):
        value = data.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        values[key] = value
    return Settings(tuple(cameras), **values)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class RuntimeLock:
    """OS releases the lock after a crash; the file contains no credentials."""

    def __init__(self, root=RUNTIME):
        root.mkdir(parents=True, exist_ok=True)
        self.handle = (root / "worker.lock").open("a+b")
        self.handle.write(b"0")
        self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("A CCTV worker is already running for this installation") from None

    def close(self):
        self.handle.close()


class CameraWorker:
    """Decode continuously; inference consumes only the latest received frame."""

    def __init__(self, camera, detector, settings, root=RUNTIME):
        self.camera, self.detector, self.settings, self.root = camera, detector, settings, root
        self.stop = Event()
        self.condition = Condition()
        self.state_lock = Lock()
        self.latest = None
        self.sequence = self.generation = 0
        self.state = {"id": camera.id, "name": camera.name, "status": "connecting",
                      "last_frame": None, "last_inference": None, "frames": 0,
                      "last_processed_frame": None,
                      "bare_heads": 0, "helmets": 0, "reconnects": 0, "error": None,
                      "zone": camera.zone, "fps": 0, "latency_ms": None}
        self.reader = Thread(target=self._read, name=f"capture-{camera.id}", daemon=True)
        self.processor = Thread(target=self._process, name=f"inference-{camera.id}", daemon=True)

    def start(self):
        self.reader.start()
        self.processor.start()

    def update(self, **values):
        with self.state_lock:
            self.state.update(values)

    def snapshot(self):
        with self.state_lock:
            result = dict(self.state)
        if result["status"] == "online" and any(time.time() - (result[key] or 0) > 5
                                               for key in ("last_frame", "last_inference", "last_processed_frame")):
            result["status"] = "stale"
        return result

    def _read(self):
        backoff = 1
        av.logging.set_level(None)
        while not self.stop.is_set():
            try:
                options = {"rtsp_transport": "tcp"} if self.camera.source.startswith("rtsp") else {}
                with av.open(self.camera.source, timeout=(5., 5.), options=options) as container:
                    if not container.streams.video:
                        raise ValueError("No video stream")
                    with self.condition:
                        self.generation += 1
                    self.update(status="connecting", error=None)
                    origin = time.monotonic()
                    for frame in container.decode(video=0):
                        if self.stop.is_set():
                            break
                        if self.camera.file and frame.time is not None:
                            if self.stop.wait(max(0, origin + frame.time - time.monotonic())):
                                break
                        image = frame.to_image()
                        image.thumbnail((1920, 1080))
                        with self.condition:
                            self.sequence += 1
                            self.latest = (self.sequence, self.generation, time.monotonic(), image)
                            self.condition.notify()
                        self.update(last_frame=time.time())
                        backoff = 1
                if not self.stop.is_set():
                    raise EOFError("Camera stream ended")
            except Exception:
                # Decoder exceptions may include the full URL, including credentials.
                with self.condition:
                    self.latest = None
                with self.state_lock:
                    self.state["reconnects"] += 1
                self.update(status="offline", error="Camera unavailable; retrying automatically")
            if self.stop.wait(backoff):
                break
            backoff = min(30, backoff * 2)

    def _process(self):
        processor, generation, sequence, rotated = None, None, 0, 0
        last_preview, last_done = 0., None
        try:
            while not self.stop.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.stop.is_set() or (
                        self.latest is not None and self.latest[0] != sequence), timeout=.5)
                    if self.stop.is_set():
                        break
                    if self.latest is None or self.latest[0] == sequence:
                        continue
                    sequence, current, timestamp, image = self.latest
                now = time.monotonic()
                if (processor is None or current != generation or now - rotated >= 3600
                        or len(processor.monitor.events) >= 1000):
                    if processor:
                        processor.finish()
                    processor = StreamDetector(self.detector, self.settings.confidence,
                        self.camera.zone, source=self.camera.id)
                    processor.directory = self.root / "events" / self.camera.id / processor.directory.name
                    processor.time_reference = {"stream_timestamp": time.monotonic(),
                                                "unix_timestamp": time.time()}
                    generation, rotated = current, now
                result = processor.annotate(image, self.settings.confidence, timestamp)
                done = time.monotonic()
                with self.state_lock:
                    self.state["frames"] += 1
                with self.condition:
                    connected = self.latest is not None and self.generation == current
                self.update(status="online" if connected else "offline", last_inference=time.time(),
                    last_processed_frame=time.time() - (done - timestamp),
                    bare_heads=len(result[1]), helmets=len(result[2]),
                    fps=round(1 / (done - last_done), 2) if last_done else 0,
                    latency_ms=round((done - timestamp) * 1000, 1))
                last_done = done
                if done - last_preview >= .5:
                    preview = Image.fromarray(result[0])
                    preview.thumbnail((1280, 720))
                    target = self.root / f"{self.camera.id}.jpg"
                    temporary = target.with_suffix(".tmp")
                    preview.save(temporary, format="JPEG", quality=80)
                    temporary.replace(target)
                    last_preview = done
                self.stop.wait(max(0, 1 / self.settings.inference_fps - (time.monotonic() - now)))
        except Exception:
            self.update(status="error", error="Inference or storage failed; restart after checking model and disk")
            self.stop.set()
        finally:
            if processor:
                try:
                    processor.finish()
                except OSError:
                    self.update(status="error", error="Event journal could not be saved; check disk")

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        for thread in (self.reader, self.processor):
            if thread.ident is not None:
                thread.join(timeout=12)
                if thread.is_alive():
                    raise RuntimeError(f"Camera {self.camera.id} is still stopping")


def prune_events(root, settings):
    """Delete only closed CCTV sessions, within this worker's event directory."""
    directory = root / "events"
    sessions = []
    for journal in directory.glob("*/*/events.json"):
        try:
            report = json.loads(journal.read_text(encoding="utf-8"))
            if not report.get("finished", False):
                continue
            files = list(journal.parent.iterdir())
            size = sum(p.stat().st_size for p in files if p.is_file())
            sessions.append((journal.stat().st_mtime, size, journal.parent))
        except (OSError, ValueError, KeyError):
            continue
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            # An active writer may have atomically renamed its temporary journal.
            continue
    cutoff = time.time() - settings.retention_days * 86400
    for modified, size, folder in sorted(sessions):
        if modified < cutoff or total > settings.storage_mb * 1024 * 1024:
            if not folder.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Event directory escapes runtime")
            shutil.rmtree(folder)
            total -= size
    return total


def recover_journals(root):
    """Mark previous interrupted sessions closed without inventing event end times."""
    for journal in (root / "events").glob("*/*/events.json"):
        try:
            report = json.loads(journal.read_text(encoding="utf-8"))
            if not report.get("finished", False):
                report.update(finished=True, interrupted=True)
                atomic_json(journal, report)
        except (OSError, ValueError):
            continue


def read_status(root=RUNTIME):
    try:
        report = json.loads((root / "status.json").read_text(encoding="utf-8"))
        report["worker_online"] = report.get("running", False) and time.time() - report["heartbeat"] < 10
        return report
    except (OSError, ValueError, KeyError):
        return {"worker_online": False, "cameras": []}


def runtime_path(value):
    return project_path(value) if value else RUNTIME
