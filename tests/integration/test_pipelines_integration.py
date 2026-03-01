from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import app
import data_setup
import train


class FakePredictingYOLO:
    def __init__(self, model_path):
        self.model_path = model_path

    def __call__(self, image):
        class Box:
            def __init__(self):
                self.conf = np.array([0.95], dtype=float)
                self.cls = np.array([0], dtype=float)
                self.xyxy = np.array([[1, 1, 8, 8]], dtype=float)

        class Result:
            boxes = [Box()]
            names = {0: "Hardhat"}

        return [Result()]


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available

    def get_device_name(self, _):
        return "Fake GPU"


class FakeTorch:
    def __init__(self, available):
        self.cuda = FakeCuda(available)


class RecordingTrainYOLO:
    def __init__(self, model_name):
        self.model_name = model_name
        self.kwargs = None

    def train(self, **kwargs):
        self.kwargs = kwargs
        return {"trained": True}


def test_training_pipeline_returns_success_status(monkeypatch, tmp_path):
    project_root = tmp_path
    (project_root / "datasets" / "safety").mkdir(parents=True)
    (project_root / "datasets" / "safety" / "data.yaml").write_text("train: x\nval: y\ntest: z\n")

    def yolo_factory(model_name):
        return RecordingTrainYOLO(model_name)

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(train, "YOLO", yolo_factory)
    monkeypatch.setattr(train, "torch", FakeTorch(available=False))

    result = train.train_safety_model()
    assert result["ok"] is True


def test_inference_pipeline_model_load_predict_and_postprocess(monkeypatch, tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_text("weights")

    monkeypatch.setattr(app, "YOLO", FakePredictingYOLO)

    model, is_custom = app.load_model("Custom Trained Model", str(model_path))

    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    prediction = model(image)[0]
    annotated, violations, safe = app.draw_safety_box(image, prediction, conf_threshold=0.4, is_custom=is_custom)

    assert is_custom is True
    assert annotated.shape == (16, 16, 3)
    assert violations == []
    assert safe == ["Hardhat"]


def test_training_pipeline_reads_expected_dataset_config_path(monkeypatch, tmp_path):
    project_root = tmp_path
    (project_root / "datasets" / "safety").mkdir(parents=True)
    (project_root / "datasets" / "safety" / "data.yaml").write_text("train: x\nval: y\n")

    recorder = {}

    def yolo_factory(model_name):
        instance = RecordingTrainYOLO(model_name)
        recorder["instance"] = instance
        return instance

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(train, "YOLO", yolo_factory)
    monkeypatch.setattr(train, "torch", FakeTorch(available=False))

    train.train_safety_model()

    assert recorder["instance"].kwargs is not None
    assert recorder["instance"].kwargs["data"] == str(Path("datasets/safety/data.yaml"))
    assert recorder["instance"].kwargs["project"] == "runs/detect"
    assert recorder["instance"].kwargs["name"] == "safety_model"


def test_end_to_end_setup_then_train_pipeline(monkeypatch, tmp_path):
    project_root = tmp_path
    dataset_dir = project_root / "datasets" / "safety"
    (dataset_dir / "css-data" / "train" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "train" / "labels").mkdir(parents=True)
    (dataset_dir / "css-data" / "valid" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "valid" / "labels").mkdir(parents=True)
    (dataset_dir / "css-data" / "test" / "images").mkdir(parents=True)
    (dataset_dir / "css-data" / "test" / "labels").mkdir(parents=True)

    monkeypatch.chdir(project_root)
    monkeypatch.setattr(data_setup, "DATASET_DIR", dataset_dir)

    data_setup.setup_data()

    generated = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    assert generated["train"] == "css-data/train/images"

    recorder = {}

    def yolo_factory(model_name):
        instance = RecordingTrainYOLO(model_name)
        recorder["instance"] = instance
        return instance

    monkeypatch.setattr(train, "YOLO", yolo_factory)
    monkeypatch.setattr(train, "torch", FakeTorch(available=False))

    train.train_safety_model()

    assert recorder["instance"].kwargs["data"] == str(Path("datasets/safety/data.yaml"))
