import numpy as np
from PIL import Image

import app


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path


class FakeBox:
    def __init__(self, conf, cls_id, xyxy):
        self.conf = np.array([conf], dtype=float)
        self.cls = np.array([cls_id], dtype=float)
        self.xyxy = np.array([xyxy], dtype=float)


class FakeResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


def test_model_loads_with_expected_path_for_non_custom(monkeypatch):
    monkeypatch.setattr(app, "YOLO", FakeYOLO)

    model, is_custom = app.load_model("YOLO26 Nano", custom_path=None)

    assert isinstance(model, FakeYOLO)
    assert model.model_path == "yolo26n.pt"
    assert is_custom is False


def test_prediction_output_type_and_shape_for_valid_result():
    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
    result = FakeResult(
        boxes=[FakeBox(conf=0.9, cls_id=0, xyxy=[1, 1, 10, 10])],
        names={0: "Hardhat"},
    )

    annotated, violations, safe = app.draw_safety_box(image, result, conf_threshold=0.4, is_custom=True)

    assert isinstance(annotated, np.ndarray)
    assert annotated.dtype == np.uint8
    assert annotated.shape == (24, 24, 3)
    assert isinstance(violations, list)
    assert isinstance(safe, list)


def test_missing_custom_model_falls_back_to_default(fake_streamlit, monkeypatch):
    monkeypatch.setattr(app, "st", fake_streamlit)
    monkeypatch.setattr(app, "YOLO", FakeYOLO)

    model, is_custom = app.load_model("Custom Trained Model", custom_path="not_found.pt")

    assert model.model_path == "yolov8n.pt"
    assert is_custom is False


def test_corrupted_model_load_returns_none_when_fallback_also_fails(fake_streamlit, monkeypatch):
    class BrokenYOLO:
        def __init__(self, _):
            raise RuntimeError("corrupted")

    monkeypatch.setattr(app, "st", fake_streamlit)
    monkeypatch.setattr(app, "YOLO", BrokenYOLO)

    model, is_custom = app.load_model("YOLO26 Nano", custom_path=None)
    assert model is None
    assert is_custom is False
    assert fake_streamlit.sidebar.errors


def test_empty_detection_list_returns_empty_reason_lists():
    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
    result = FakeResult(boxes=[], names={})

    _, violations, safe = app.draw_safety_box(image, result, conf_threshold=0.4, is_custom=True)

    assert violations == []
    assert safe == []


def test_invalid_detection_class_name_defaults_to_neutral():
    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8))
    result = FakeResult(
        boxes=[FakeBox(conf=0.9, cls_id=0, xyxy=[2, 2, 8, 8])],
        names={0: "UnknownClass"},
    )

    _, violations, safe = app.draw_safety_box(image, result, conf_threshold=0.4, is_custom=True)

    assert violations == []
    assert safe == []


def test_grayscale_image_auto_converts_to_rgb():
    gray = Image.fromarray(np.zeros((24, 24), dtype=np.uint8), mode='L')
    result = FakeResult(
        boxes=[FakeBox(conf=0.9, cls_id=0, xyxy=[1, 1, 8, 8])],
        names={0: "Hardhat"},
    )

    annotated, violations, safe = app.draw_safety_box(gray, result, conf_threshold=0.4, is_custom=True)

    assert annotated.shape == (24, 24, 3)
    assert safe == ["Hardhat"]


def test_rgba_image_auto_converts_to_rgb():
    rgba = Image.fromarray(np.zeros((24, 24, 4), dtype=np.uint8), mode='RGBA')
    result = FakeResult(
        boxes=[FakeBox(conf=0.9, cls_id=0, xyxy=[1, 1, 8, 8])],
        names={0: "NO-Mask"},
    )

    annotated, violations, safe = app.draw_safety_box(rgba, result, conf_threshold=0.4, is_custom=True)

    assert annotated.shape == (24, 24, 3)
    assert violations == ["NO-Mask"]
