from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from safety.paths import ROOT


def test_input_switches(monkeypatch):
    import streamlit_webrtc

    from safety import inference

    # AppTest has no WebRTC/browser transport. Frame processing is checked separately.
    monkeypatch.setattr(streamlit_webrtc, 'webrtc_streamer', lambda **kwargs: None)
    monkeypatch.setattr(inference, 'Detector', lambda path: SimpleNamespace(path=ROOT / 'best.pt', is_custom=True))
    monkeypatch.setattr(inference, 'custom_checkpoints', lambda profile='ppe': [])
    original_selection = inference.selected_checkpoint
    monkeypatch.setattr(inference, 'selected_checkpoint',
                        lambda profile='ppe': original_selection("ppe") if profile == 'ppe' else (None, .4))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=15).run()
    assert not app.exception
    app.selectbox(key="detection_profile").set_value("ppe").run()
    for source in ('Video', 'Video link', 'Browser camera', 'Image'):
        app.radio[0].set_value(source).run()
        assert not app.exception
    assert 'YOLO26 Extra Large' in app.selectbox[1].options
    for profile in ('ppe', 'shwd'):
        app.selectbox(key='detection_profile').set_value(profile).run()
        assert not app.exception


def test_trained_profiles_support_all_inputs_and_close_previous_media(tmp_path, monkeypatch):
    import streamlit_webrtc

    from safety import inference, media

    weights = {}
    for profile in ('ppe', 'shwd'):
        weights[profile] = tmp_path / f'{profile}.pt'
        weights[profile].write_bytes(b'fake weights for UI contract')
    monkeypatch.setattr(inference, 'selected_checkpoint', lambda profile='ppe': (weights[profile], .4))
    monkeypatch.setattr(inference, 'custom_checkpoints', lambda profile='ppe': [])
    monkeypatch.setattr(inference, 'Detector', lambda path: SimpleNamespace(
        path=Path(path), profile=Path(path).stem, is_custom=True))
    monkeypatch.setattr(streamlit_webrtc, 'webrtc_streamer', lambda **kwargs: None)
    closed = []
    original_close = media.VideoSession.close

    def close(controller):
        closed.append(controller)
        return original_close(controller)

    monkeypatch.setattr(media.VideoSession, 'close', close)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=15).run()
    for profile in ('ppe', 'shwd'):
        previous_count = len(closed)
        app.selectbox(key='detection_profile').set_value(profile).run()
        assert not app.exception
        assert len(closed) > previous_count
        workspace = next(widget for widget in app.selectbox if widget.label == "Workspace")
        workspace.set_value("CCTV monitor").run()
        assert not app.exception
        next(widget for widget in app.selectbox if widget.label == "Workspace").set_value("Detection").run()
        for source in ('Video', 'Video link', 'Browser camera', 'Image'):
            app.radio[0].set_value(source).run()
            assert not app.exception
        assert not app.error


def test_invalid_selection_metadata_is_reported_without_crashing(tmp_path, monkeypatch):
    from safety import inference

    selection = tmp_path / "selected_model.json"
    selection.write_text('{"checkpoint":')
    monkeypatch.setattr(inference, "SELECTION", selection)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
    app.selectbox(key="detection_profile").set_value("ppe").run()
    assert not app.exception
    assert any("checkpoint" in item.value.lower() for item in app.error)
