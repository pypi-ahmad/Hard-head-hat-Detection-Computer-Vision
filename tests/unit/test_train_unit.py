import train
from pathlib import Path


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available

    def get_device_name(self, _):
        return "Fake GPU"


class FakeTorch:
    def __init__(self, available):
        self.cuda = FakeCuda(available)


class RecordingYOLO:
    def __init__(self, model_name):
        self.model_name = model_name
        self.train_called = False
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_called = True
        self.train_kwargs = kwargs
        return {"ok": True}


def test_train_safety_model_uses_cpu_when_cuda_unavailable(monkeypatch, tmp_path):
    (tmp_path / "datasets" / "safety").mkdir(parents=True)
    (tmp_path / "datasets" / "safety" / "data.yaml").write_text("train: x\nval: y\n")
    monkeypatch.chdir(tmp_path)

    captured = {}

    def yolo_factory(model_name):
        instance = RecordingYOLO(model_name)
        captured["instance"] = instance
        return instance

    monkeypatch.setattr(train, "torch", FakeTorch(available=False))
    monkeypatch.setattr(train, "YOLO", yolo_factory)

    result = train.train_safety_model()

    model = captured["instance"]
    assert model.model_name == "yolo26l.pt"
    assert model.train_called is True
    assert model.train_kwargs["device"] == "cpu"
    assert model.train_kwargs["data"] == str(Path("datasets/safety/data.yaml"))
    assert result["ok"] is True


def test_train_safety_model_uses_gpu_zero_when_cuda_available(monkeypatch, tmp_path):
    (tmp_path / "datasets" / "safety").mkdir(parents=True)
    (tmp_path / "datasets" / "safety" / "data.yaml").write_text("train: x\nval: y\n")
    monkeypatch.chdir(tmp_path)

    captured = {}

    def yolo_factory(model_name):
        instance = RecordingYOLO(model_name)
        captured["instance"] = instance
        return instance

    monkeypatch.setattr(train, "torch", FakeTorch(available=True))
    monkeypatch.setattr(train, "YOLO", yolo_factory)

    result = train.train_safety_model()

    model = captured["instance"]
    assert model.train_kwargs["device"] == 0
    assert result["ok"] is True


def test_train_safety_model_returns_early_when_model_load_fails(monkeypatch, tmp_path):
    (tmp_path / "datasets" / "safety").mkdir(parents=True)
    (tmp_path / "datasets" / "safety" / "data.yaml").write_text("train: x\nval: y\n")
    monkeypatch.chdir(tmp_path)

    train_called = {"value": False}

    class BrokenYOLO:
        def __init__(self, _):
            raise RuntimeError("corrupted model")

        def train(self, **kwargs):
            train_called["value"] = True
            return {}

    monkeypatch.setattr(train, "YOLO", BrokenYOLO)
    monkeypatch.setattr(train, "torch", FakeTorch(available=False))

    result = train.train_safety_model()

    assert result["ok"] is False
    assert result["error"] == "model_load_failed"
    assert train_called["value"] is False


def test_train_safety_model_returns_missing_data_yaml_when_config_absent(monkeypatch, tmp_path):
    """Exercises the missing_data_yaml branch (train.py L15-17)."""
    monkeypatch.chdir(tmp_path)  # no datasets/safety/data.yaml here

    monkeypatch.setattr(train, "torch", FakeTorch(available=False))
    monkeypatch.setattr(train, "YOLO", RecordingYOLO)

    result = train.train_safety_model()

    assert result["ok"] is False
    assert result["error"] == "missing_data_yaml"
