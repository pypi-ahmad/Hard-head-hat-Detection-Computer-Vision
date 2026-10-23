import pytest

from safety.events import Detection, EventMonitor
from safety.review import ReviewStore, validate_boxes


def test_events_require_zone_and_observed_evidence():
    head = Detection([0, 0, 20, 20], "bare_head", .8, 1)
    disabled = EventMonitor()
    assert disabled.update([head], 10, (100, 100)) == []
    monitor = EventMonitor((0, 0, 1, 1))
    for time in (0, .25, .5, .75):
        assert monitor.update([head], time, (100, 100)) == []
    assert len(monitor.update([head], 1, (100, 100))) == 1
    assert monitor.update([head], 1.25, (100, 100)) == []
    monitor.update([], 2.25, (100, 100))
    assert monitor.events[0]["end"] == 2.25
    assert monitor.events[0]["reason"] == "track lost"
    assert len(monitor.events) == 1


def test_gaps_low_scores_and_helmet_do_not_confirm():
    monitor = EventMonitor((0, 0, 1, 1))
    head = Detection([0, 0, 20, 20], "bare_head", .8, 1)
    monitor.update([head], 0, (100, 100))
    monitor.update([head], .4, (100, 100))
    monitor.update([head], 1., (100, 100))
    helmet = Detection(head.box, "helmet", .8, 1)
    monitor.update([helmet], 2., (100, 100))
    assert not monitor.events and not monitor.pending
    with pytest.raises(ValueError, match="monotonic"):
        monitor.update([], 1, (100, 100))


def test_review_export_blocks_drafts_and_invalid_boxes(tmp_path):
    store = ReviewStore(tmp_path / "review")
    with pytest.raises(ValueError, match="No approved"):
        store.export(tmp_path / "export")
    for box in ([0, 0, 0, 10], [0, 0, 101, 10], [float("nan"), 0, 10, 10]):
        with pytest.raises(ValueError):
            validate_boxes([{"class": "bare_head", "xyxy": box}], 100, 100)


def test_review_survives_reload_and_rejects_conflicting_edits(tmp_path):
    store = ReviewStore(tmp_path)
    with store.connect() as db:
        db.execute("INSERT INTO frames VALUES ('a','s',0,'image','hash',100,100,'[]','unreviewed',0)")
    frame = store.frames()[0]
    store.save(frame, [], "approved")
    assert ReviewStore(tmp_path).frames()[0]["status"] == "approved"
    with pytest.raises(ValueError, match="Another review"):
        store.save(frame, [], "excluded")


def test_export_approved_empty_frames_and_group_leakage(tmp_path):
    from PIL import Image

    from safety.datasets import sha256

    store = ReviewStore(tmp_path / "review")
    for index, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"{split}.jpg"
        Image.new("RGB", (10, 10), (index*50, 0, 0)).save(image)
        with store.connect() as db:
            db.execute("INSERT INTO sources VALUES (?, ?, ?, ?)", (split, str(image), split, split))
            db.execute("INSERT INTO frames VALUES (?, ?, 0, ?, ?, 10, 10, '[]', 'approved', 0)",
                       (split, split, str(image), sha256(image)))
    config = store.export(tmp_path / "v1")
    assert config.exists()
    assert (config.parent / "train/labels/train.txt").read_text() == ""
    store.assign("val", "train", "val")
    with pytest.raises(ValueError, match="cross splits"):
        store.export(tmp_path / "v2")
    assert not (tmp_path / "v2").exists()


def test_event_metrics_count_duplicate_alerts_as_false():
    from safety.event_evaluation import event_scores

    truth = [{"start": 0, "end": 5, "box": [0, 0, .2, .2]}]
    alert = {"confirmed": 1, "box": [0, 0, 20, 20]}
    result = event_scores([alert, alert], truth, 3600, (100, 100))
    assert result["event_recall"] == 1
    assert result["false_alerts_per_hour"] == 1
    assert result["mean_delay_seconds"] == 1


def test_finetuning_rejects_unreviewed_manifest_before_training(tmp_path):
    from finetune_camera import prepare_revision

    (tmp_path / "manifest.json").write_text('[{"status":"unreviewed"}]')
    with pytest.raises(ValueError, match="approved"):
        prepare_revision(tmp_path / "data.yaml", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_finetuning_checks_shwd_before_creating_run(tmp_path, monkeypatch):
    import json

    from PIL import Image

    import finetune_camera
    from safety.datasets import sha256

    store = ReviewStore(tmp_path / "review")
    for index, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"{split}.jpg"
        Image.new("RGB", (10, 10), (index * 50, 0, 0)).save(image)
        with store.connect() as db:
            db.execute("INSERT INTO sources VALUES (?, ?, ?, ?)", (split, str(image), split, split))
            db.execute("INSERT INTO frames VALUES (?, ?, 0, ?, ?, 10, 10, '[]', 'approved', 0)",
                       (split, split, str(image), sha256(image)))
    config = store.export(tmp_path / "v1")
    base = tmp_path / "shwd"
    (base / "train/images").mkdir(parents=True)
    (base / "train/labels").mkdir()
    image = base / "train/images/changed.jpg"
    image.write_bytes(b"changed")
    label = base / "train/labels/changed.txt"
    label.write_text("")
    (base / "manifest.json").write_text(json.dumps([{
        "image": "train/images/changed.jpg", "split": "train", "sha256": "original",
        "label_sha256": sha256(label),
    }]))
    monkeypatch.setattr(finetune_camera, "dataset_root", lambda _: base)
    with pytest.raises(ValueError, match="changed.jpg"):
        finetune_camera.prepare_revision(config, tmp_path / "run")
    assert not (tmp_path / "run").exists()
