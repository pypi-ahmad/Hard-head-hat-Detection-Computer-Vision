import json
from threading import Lock

from ultralytics import YOLO

from safety.annotations import draw_safety_box
from safety.paths import BASELINE, ROOT, SELECTION, project_path
from safety.profiles import artifact_root, checkpoint_profile

PRESETS = {f"YOLO26 {size}": ROOT / f"yolo26{letter}.pt" for size, letter in
           zip(("Nano", "Small", "Medium", "Large", "Extra Large"), "nsmlx", strict=True)}


def checkpoint_identity(path):
    path = project_path(path)
    stat = path.stat() if path.exists() else None
    return str(path), stat.st_size if stat else 0, stat.st_mtime_ns if stat else 0


def selected_checkpoint(profile="shwd"):
    selection = SELECTION if profile == "ppe" else artifact_root(profile) / "selected_model.json"
    if selection.exists():
        data = json.loads(selection.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("checkpoint"), str):
            raise ValueError("Selection must contain a checkpoint path")
        confidence = float(data["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("Selected confidence must be between 0 and 1")
        path = project_path(data["checkpoint"])
        if path.is_file():
            return path, confidence
    if profile == "ppe":
        return (BASELINE if BASELINE.is_file() else PRESETS["YOLO26 Nano"]), 0.4
    return None, 0.4


def custom_checkpoints(profile="shwd"):
    candidates = list((artifact_root(profile) / "runs").glob("*/weights/best.pt"))
    candidates = [path for path in candidates if "smoke" not in path.parent.parent.name]
    if profile == "ppe":
        candidates += [BASELINE, *ROOT.glob("runs/detect/safety_model*/weights/best.pt")]
    return sorted({path.resolve() for path in candidates if path.is_file()})


def selected_resolution(profile="shwd"):
    selection = artifact_root(profile) / "selected_model.json"
    if not selection.exists():
        return 640
    size = json.loads(selection.read_text(encoding="utf-8")).get("imgsz", 640)
    if not isinstance(size, int) or size not in (640, 960):
        raise ValueError("Unsupported selected inference resolution")
    return size


class Detector:
    """One model and one lock; prediction results leave the lock on CPU."""

    def __init__(self, checkpoint, device=None, imgsz=640):
        self.path = project_path(checkpoint)
        self.model = YOLO(str(self.path))
        self.device = device
        self.imgsz = imgsz
        self.lock = Lock()
        self.profile = checkpoint_profile(self.model.names)
        self.is_custom = self.profile is not None

    def predict(self, image, confidence):
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        image = image.convert("RGB")
        with self.lock:
            result = self.model.predict(image, conf=confidence, device=self.device,
                                        imgsz=self.imgsz, verbose=False)[0].cpu()
        return result

    def annotate(self, image, confidence):
        image = image.convert("RGB")
        return draw_safety_box(image, self.predict(image, confidence), confidence, self.is_custom)
