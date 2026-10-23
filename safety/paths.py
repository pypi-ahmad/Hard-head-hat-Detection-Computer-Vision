from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/safety"
ARTIFACTS = ROOT / "artifacts"
BASELINE = DATASET / "results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt"
SELECTION = ARTIFACTS / "selected_model.json"


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
