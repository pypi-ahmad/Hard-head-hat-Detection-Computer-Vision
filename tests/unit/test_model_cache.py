import app
from safety.inference import checkpoint_identity


def test_model_cache_reuses_weights_and_invalidates_replacement(tmp_path, monkeypatch):
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'old')
    calls = []

    def load(checkpoint):
        calls.append(checkpoint)
        return object()

    monkeypatch.setattr(app, 'Detector', load)
    app.cached_detector.clear()
    try:
        first = app.cached_detector(checkpoint_identity(path))
        assert app.cached_detector(checkpoint_identity(path)) is first
        path.write_bytes(b'replacement')
        assert app.cached_detector(checkpoint_identity(path)) is not first
        assert calls == [str(path), str(path)]
    finally:
        app.cached_detector.clear()
