import json
import zipfile
from collections import Counter

import pytest
from PIL import Image

from evaluate import threshold_scores
from safety import datasets, inference
from safety.annotations import draw_safety_box
from safety.profiles import checkpoint_profile


def test_training_jpeg_repair_preserves_source_and_is_stable(tmp_path):
    source, target, second = (tmp_path / name for name in ("raw.jpg", "train.jpg", "again.jpg"))
    Image.new("RGB", (32, 24), "red").save(source)
    source.write_bytes(source.read_bytes() + b"trailing bytes")
    original = source.read_bytes()
    datasets.copy_training_image(source, target)
    assert source.read_bytes() == original
    assert target.read_bytes().endswith(b"\xff\xd9")
    with Image.open(target) as image:
        image.verify()
    datasets.copy_training_image(target, second)
    assert datasets.sha256(target) == datasets.sha256(second)


def test_voc_heads_are_not_full_body_people(tmp_path):
    xml = tmp_path / "head.xml"
    xml.write_text('<annotation><object><name>person</name><bndbox><xmin>1</xmin>'
                   '<ymin>1</ymin><xmax>100</xmax><ymax>80</ymax></bndbox></object></annotation>')
    assert datasets.voc_boxes(xml, (100, 80), []) == [[1, .5, .5, 1., 1.]]
    assert checkpoint_profile({0: "helmet", 1: "bare_head"}) == "shwd"
    assert checkpoint_profile({0: "helmet", 1: "person"}) is None



def test_holdout_duplicate_is_removed_from_train_only(tmp_path):
    image = tmp_path / "train.jpg"
    copy = tmp_path / "test.jpg"
    image.write_bytes(b"same image")
    copy.write_bytes(image.read_bytes())
    annotation = tmp_path / "head.xml"
    annotation.write_text("<annotation/>")
    records = [("train", image, annotation), ("test", copy, annotation)]
    kept, excluded = datasets.filter_records(records, "shwd")
    assert kept == [("test", copy, annotation)]
    assert excluded[0]["split"] == "train"
    assert image.read_bytes() == copy.read_bytes()


def test_unknown_voc_class_quarantines_whole_image(tmp_path):
    annotation = tmp_path / "sample.xml"
    annotation.write_text('<annotation><object><name>dog</name></object></annotation>')
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"unchanged")
    kept, excluded = datasets.filter_records([("train", image, annotation)], "shwd")
    assert kept == []
    assert excluded[0]["classes"] == ["dog"]
    assert image.read_bytes() == b"unchanged"


def test_zip_traversal_does_not_write_outside_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        datasets.extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "escaped.txt").exists()


def test_download_failure_is_recorded_not_reported_as_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "dataset_root", lambda _: tmp_path)
    monkeypatch.setattr(datasets.gdown, "download", lambda **_: None)
    with pytest.raises(RuntimeError, match="Failed downloads"):
        datasets.download("shwd")
    manifest = json.loads((tmp_path / "raw/downloads.json").read_text())
    assert "error" in manifest[0]
    assert not (tmp_path / "data.yaml").exists()



def test_seeded_shwd_fallback_is_deterministic(tmp_path):
    folder = tmp_path / "Annotations"
    folder.mkdir()
    for index in range(20):
        (folder / f"{index:03}.xml").touch()
    first, policy = datasets.shwd_splits(tmp_path)
    assert first == datasets.shwd_splits(tmp_path)[0]
    assert [len(first[key]) for key in ("train", "val", "test")] == [16, 2, 2]
    assert len(set().union(*map(set, first.values()))) == 20
    assert "seed-42" in policy



def test_profile_metrics_use_their_own_targets():
    scores = threshold_scores([(1, .9, True)], Counter({0: 3, 1: 2}), .4, "shwd")
    assert scores["target_precision"] == 1
    assert scores["target_recall"] == .5


def test_selected_checkpoints_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "artifact_root", lambda profile: tmp_path / profile)
    weights = tmp_path / "shwd.pt"
    weights.write_bytes(b"model")
    (tmp_path / "shwd").mkdir()
    (tmp_path / "shwd/selected_model.json").write_text(json.dumps({"checkpoint": str(weights), "confidence": .6}))
    assert inference.selected_checkpoint("shwd") == (weights, .6)



def test_unknown_class_zero_is_not_mislabeled_as_person():
    from types import SimpleNamespace

    import numpy as np

    box = SimpleNamespace(conf=[.9], cls=[0], xyxy=[[0, 0, 20, 20]])
    result = SimpleNamespace(names={0: "helmet"}, boxes=[box])
    image = Image.new("RGB", (50, 50))
    annotated, violations, equipment = draw_safety_box(image, result, .4, False)
    assert np.array_equal(annotated, np.asarray(image))
    assert violations == equipment == []


def test_profile_selection_is_published_only_after_test_success(tmp_path, monkeypatch):
    import evaluate

    root = tmp_path / "shwd"
    root.mkdir()
    (root / "training_results.json").write_text(json.dumps([
        {"ok": True, "model": "yolo26s", "best_model": str(tmp_path / "model.pt")}]))
    selection = root / "selected_model.json"
    selection.write_text('"previous selection"')
    monkeypatch.setattr(evaluate, "artifact_root", lambda _: root)
    point = {"confidence": .4, "target_precision": .9, "target_recall": .7}

    def evaluate_fake(checkpoint, split="val", confidence=None, **kwargs):
        if split == "test":
            assert confidence == .4
            raise RuntimeError("test failed")
        return {"operating_point": point, "qualifying": point}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", evaluate_fake)
    with pytest.raises(RuntimeError, match="test failed"):
        evaluate.compare_profile("shwd", "cpu")
    assert selection.read_text() == '"previous selection"'


def test_full_video_verification_decodes_every_output_frame(tmp_path):
    from types import SimpleNamespace

    import cv2
    import numpy as np

    from verify_videos import verify_video

    source, output = tmp_path / "input.mp4", tmp_path / "output.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(6):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()
    detector = SimpleNamespace(annotate=lambda image, confidence: (np.asarray(image), [], ["helmet"]))
    result = verify_video(detector, source, output, .4)
    assert result["frames"] == result["output_decoded_frames"] == 6
    assert result["equipment_detections"] == {"helmet": 6}


@pytest.mark.parametrize("confidence", [-.1, 1.1, "nan", "inf"])
def test_selected_confidence_outside_supported_range_is_rejected(tmp_path, monkeypatch, confidence):
    selection = tmp_path / "selected_model.json"
    selection.write_text(json.dumps({"checkpoint": "model.pt", "confidence": confidence}))
    monkeypatch.setattr(inference, "SELECTION", selection)
    with pytest.raises(ValueError, match="between 0 and 1"):
        inference.selected_checkpoint("ppe")
