from collections import Counter

import pytest

from audit_dataset import inspect_label
from evaluate import match_detections, threshold_scores


def test_duplicate_and_wrong_class_predictions_are_false_positives():
    labels = [[2, .5, .5, .4, .4]]
    predictions = [[30, 30, 70, 70, .9, 2], [30, 30, 70, 70, .8, 2], [30, 30, 70, 70, .95, 3]]
    assert match_detections(predictions, labels, (100, 100)) == [(3, .95, False), (2, .9, True), (2, .8, False)]


def test_threshold_macro_averages_all_three_violation_classes():
    matches = [(2, .9, True), (2, .3, False), (3, .8, True), (4, .7, True)]
    scores = threshold_scores(matches, Counter({2: 2, 3: 1, 4: 1}), .4)
    assert scores['violation_precision'] == 1
    assert scores['violation_recall'] == pytest.approx(5 / 6)
    assert threshold_scores(matches, Counter({2: 2, 3: 1, 4: 1}), .85)['violation_recall'] == pytest.approx(1 / 6)


@pytest.mark.parametrize('label', ['10 .5 .5 .1 .1', '0 nan .5 .1 .1', '0 .5 .5 0 .1', '0 .5 .5 .1', '0 .01 .5 .4 .4'])
def test_invalid_labels_reported(tmp_path, label):
    path = tmp_path / 'label.txt'
    path.write_text(label)
    assert inspect_label(path)[1]


def test_empty_label_is_valid_background(tmp_path):
    path = tmp_path / 'label.txt'
    path.write_text('')
    assert inspect_label(path) == ([], [])


def test_comparison_rejects_partial_training(tmp_path, monkeypatch):
    import evaluate

    (tmp_path / 'training_results.json').write_text('[{"model": "smoke", "ok": true}]')
    monkeypatch.setattr(evaluate, 'ARTIFACTS', tmp_path)
    with pytest.raises(RuntimeError, match='incomplete'):
        evaluate.compare()


def test_video_benchmark_aggregates_all_requested_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import evaluate

    folder = tmp_path / 'source_files/source_files'
    folder.mkdir(parents=True)
    for name in ('hardhat.mp4', 'indianworkers.mp4', 'JapanPPE.mp4'):
        (folder / name).touch()
    seen = []

    def measure(detector, confidence, path, frames):
        seen.append(path.name)
        return {'_samples': [.1, .3], 'frames': 2, 'peak_gpu_bytes': 123, 'video': path.name}

    monkeypatch.setattr(evaluate, 'DATASET', tmp_path)
    monkeypatch.setattr(evaluate, 'benchmark_video', measure)
    report = evaluate.benchmark(SimpleNamespace(device='cpu'), .4)
    assert set(seen) == {'custom Youtube video file by Mujtaba1_Part1Trim.mp4',
                         'hardhat.mp4', 'indianworkers.mp4', 'JapanPPE.mp4'}
    assert report['frames'] == 8
    assert report['mean_seconds'] == pytest.approx(.2)
    assert report['fps'] == pytest.approx(5)
    assert report['p95_seconds'] == pytest.approx(.3)
    assert len(report['videos']) == 4
    assert all('_samples' not in video for video in report['videos'])


def test_ppe_selection_is_preserved_when_test_evaluation_fails(tmp_path, monkeypatch):
    import json

    import evaluate

    journal = [{"ok": True, "model": f"yolo26{size}", "best_model": str(tmp_path / f"{size}.pt")}
               for size in "sml"]
    (tmp_path / "training_results.json").write_text(json.dumps(journal))
    selection = tmp_path / "selected_model.json"
    selection.write_text('"previous selection"')
    monkeypatch.setattr(evaluate, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(evaluate, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate, "BASELINE", tmp_path / "baseline.pt")
    monkeypatch.setattr(evaluate, "SELECTION", selection)
    point = {"confidence": .4, "violation_precision": .9, "violation_recall": .7}

    def measure(checkpoint, split="val", confidence=None, **kwargs):
        if split == "test":
            raise RuntimeError("test evaluation failed")
        return {"checkpoint": str(checkpoint), "qualifying": point,
                "operating_point": point, "benchmark": {"mean_seconds": .1}}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", measure)
    with pytest.raises(RuntimeError, match="test evaluation failed"):
        evaluate.compare(device="cpu")
    assert selection.read_text() == '\"previous selection\"'
