import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
from PIL import Image

from safety import inference
from safety.annotations import CLASSES


def test_prediction_confidence_rgb_and_serialization(monkeypatch, tmp_path):
    class Model:
        names = dict(enumerate(CLASSES))
        active = 0
        maximum = 0
        calls = []

        def predict(self, image, **kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            self.calls.append((image.mode, kwargs['conf']))
            time.sleep(0.01)
            self.active -= 1
            result = SimpleNamespace(boxes=[], names=self.names)
            result.cpu = lambda: result
            return [result]

    model = Model()
    monkeypatch.setattr(inference, 'YOLO', lambda _: model)
    detector = inference.Detector(tmp_path / 'fake.pt')
    image = Image.fromarray(np.zeros((16, 16), dtype=np.uint8))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: detector.annotate(image, 0.12), range(8)))
    assert model.maximum == 1
    assert model.calls == [('RGB', 0.12)] * 8
    assert all(result[0].shape == (16, 16, 3) for result in results)


def test_checkpoint_identity_changes_on_replacement(tmp_path):
    path = tmp_path / 'best.pt'
    path.write_bytes(b'old')
    old = inference.checkpoint_identity(path)
    path.write_bytes(b'replacement')
    assert inference.checkpoint_identity(path) != old


def test_selected_checkpoint_uses_explicit_selection(tmp_path, monkeypatch):
    import json
    path = tmp_path / 'best.pt'
    path.write_bytes(b'weights')
    selection = tmp_path / 'selected.json'
    selection.write_text(json.dumps({'checkpoint': str(path), 'confidence': 0.32}))
    monkeypatch.setattr(inference, 'SELECTION', selection)
    assert inference.selected_checkpoint("ppe") == (path, 0.32)
