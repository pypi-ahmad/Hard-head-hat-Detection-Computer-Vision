import numpy as np
import pytest
from PIL import Image
from pathlib import Path

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


def test_format_reasons_empty_list_returns_empty_string():
    assert app.format_reasons([]) == ""


def test_format_reasons_counts_duplicates():
    reasons = ["NO-Mask", "NO-Mask", "NO-Hardhat"]
    formatted = app.format_reasons(reasons)
    assert "NO-Mask (2)" in formatted
    assert "NO-Hardhat (1)" in formatted


def test_get_latest_model_path_returns_none_when_no_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert app.get_latest_model_path() is None


def test_get_latest_model_path_prefers_deep_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    deep_path = tmp_path / "datasets/safety/results_yolov8n_100e/kaggle/working/runs/detect/train/weights"
    deep_path.mkdir(parents=True)
    model_file = deep_path / "best.pt"
    model_file.write_text("model")

    selected = app.get_latest_model_path()

    assert selected == str(Path("datasets/safety/results_yolov8n_100e/kaggle/working/runs/detect/train/weights/best.pt"))


def test_get_latest_model_path_selects_newest_by_mtime(tmp_path, monkeypatch):
    import os, time
    monkeypatch.chdir(tmp_path)

    # Create two competing candidates under runs/detect
    old_dir = tmp_path / "runs" / "detect" / "safety_model" / "weights"
    old_dir.mkdir(parents=True)
    old_pt = old_dir / "best.pt"
    old_pt.write_text("old")
    # Force old mtime
    old_time = time.time() - 3600
    os.utime(old_pt, (old_time, old_time))

    new_dir = tmp_path / "runs" / "detect" / "safety_model2" / "weights"
    new_dir.mkdir(parents=True)
    new_pt = new_dir / "best.pt"
    new_pt.write_text("new")

    selected = app.get_latest_model_path()

    # Newest candidate must win
    assert selected == str(new_pt.relative_to(tmp_path))


def test_load_model_custom_path_exists_sets_custom_mode(tmp_path, monkeypatch):
    custom_model = tmp_path / "best.pt"
    custom_model.write_text("model")

    monkeypatch.setattr(app, "YOLO", FakeYOLO)

    model, is_custom = app.load_model("Custom Trained Model", str(custom_model))

    assert is_custom is True
    assert isinstance(model, FakeYOLO)
    assert model.model_path == str(custom_model)


def test_load_model_custom_missing_falls_back_and_reports(fake_streamlit, monkeypatch):
    monkeypatch.setattr(app, "st", fake_streamlit)
    monkeypatch.setattr(app, "YOLO", FakeYOLO)

    model, is_custom = app.load_model("Custom Trained Model", "missing.pt")

    assert is_custom is False
    assert model.model_path == "yolov8n.pt"
    assert fake_streamlit.sidebar.errors


def test_load_model_returns_none_when_primary_and_fallback_both_fail(fake_streamlit, monkeypatch):
    class BrokenYOLO:
        def __init__(self, _):
            raise RuntimeError("broken")

    monkeypatch.setattr(app, "st", fake_streamlit)
    monkeypatch.setattr(app, "YOLO", BrokenYOLO)

    model, is_custom = app.load_model("YOLO26 Nano", None)

    assert model is None
    assert is_custom is False
    assert len(fake_streamlit.sidebar.errors) >= 1


def test_draw_safety_box_non_custom_only_person_is_counted():
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    boxes = [
        FakeBox(conf=0.9, cls_id=0, xyxy=[1, 1, 10, 10]),
        FakeBox(conf=0.95, cls_id=2, xyxy=[2, 2, 8, 8]),
    ]
    names = {0: "person", 2: "car"}
    result = FakeResult(boxes=boxes, names=names)

    annotated, violations, safe = app.draw_safety_box(image, result, conf_threshold=0.4, is_custom=False)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (32, 32, 3)
    assert violations == []
    assert safe == []


def test_draw_safety_box_custom_tracks_safe_and_danger_classes():
    image = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    boxes = [
        FakeBox(conf=0.9, cls_id=0, xyxy=[1, 1, 4, 4]),
        FakeBox(conf=0.8, cls_id=1, xyxy=[5, 5, 9, 9]),
    ]
    names = {0: "Hardhat", 1: "NO-Mask"}
    result = FakeResult(boxes=boxes, names=names)

    annotated, violations, safe = app.draw_safety_box(image, result, conf_threshold=0.4, is_custom=True)

    assert isinstance(annotated, np.ndarray)
    assert safe == ["Hardhat"]
    assert violations == ["NO-Mask"]


def test_draw_safety_box_filters_low_confidence():
    image = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    boxes = [FakeBox(conf=0.1, cls_id=0, xyxy=[1, 1, 5, 5])]
    names = {0: "Hardhat"}
    result = FakeResult(boxes=boxes, names=names)

    _, violations, safe = app.draw_safety_box(image, result, conf_threshold=0.4, is_custom=True)

    assert violations == []
    assert safe == []


def test_draw_safety_box_handles_none_boxes():
    image = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8))
    result = FakeResult(boxes=None, names={})

    annotated, violations, safe = app.draw_safety_box(image, result, conf_threshold=0.4, is_custom=True)

    assert isinstance(annotated, np.ndarray)
    assert violations == []
    assert safe == []


def test_draw_safety_box_raises_for_none_image():
    result = FakeResult(boxes=[], names={})
    with pytest.raises(ValueError, match="Input image cannot be None"):
        app.draw_safety_box(None, result, conf_threshold=0.4, is_custom=True)


def test_draw_safety_box_handles_grayscale_image():
    gray = Image.fromarray(np.zeros((20, 20), dtype=np.uint8), mode='L')
    boxes = [FakeBox(conf=0.9, cls_id=0, xyxy=[1, 1, 5, 5])]
    names = {0: "Hardhat"}
    result = FakeResult(boxes=boxes, names=names)

    annotated, violations, safe = app.draw_safety_box(gray, result, conf_threshold=0.4, is_custom=True)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (20, 20, 3)
    assert safe == ["Hardhat"]


def test_draw_safety_box_handles_rgba_image():
    rgba = Image.fromarray(np.zeros((20, 20, 4), dtype=np.uint8), mode='RGBA')
    boxes = [FakeBox(conf=0.9, cls_id=0, xyxy=[1, 1, 5, 5])]
    names = {0: "NO-Hardhat"}
    result = FakeResult(boxes=boxes, names=names)

    annotated, violations, safe = app.draw_safety_box(rgba, result, conf_threshold=0.4, is_custom=True)

    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (20, 20, 3)
    assert violations == ["NO-Hardhat"]
