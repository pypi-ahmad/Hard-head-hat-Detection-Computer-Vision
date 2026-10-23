import numpy as np
import pytest
from PIL import Image

from safety import annotations as app


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
