import json
import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from safety import cctv
from scripts.configure_lan import gateway_config


def test_configuration_uses_environment_without_exposing_url(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CAMERA_URL", "rtsp://operator:private-test-value@192.168.1.2/live")
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras:\n  - id: entrance\n    url_env: TEST_CAMERA_URL\n"
                    "    helmet_required_zone: [0, 0, 1, 1]\n")
    settings = cctv.load_settings(path)
    assert settings.cameras[0].source.endswith("/live")
    assert "private-test-value" not in repr(settings)
    assert settings.cameras[0].zone == (0, 0, 1, 1)


@pytest.mark.parametrize("content", [
    "cameras: [{id: ../escape, url_env: TEST_CAMERA_URL}]",
    "cameras: [{id: one, url_env: TEST_CAMERA_URL}, {id: one, url_env: TEST_CAMERA_URL}]",
    "cameras: [{id: one, url_env: TEST_CAMERA_URL, helmet_required_zone: [0,0,2,1]}]",
    "inference_fps: .nan\ncameras: []",
])
def test_invalid_camera_configuration_is_rejected(tmp_path, monkeypatch, content):
    monkeypatch.setenv("TEST_CAMERA_URL", "rtsp://192.168.1.2/live")
    path = tmp_path / "cameras.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        cctv.load_settings(path)


def test_missing_camera_secret_names_only_the_variable(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras: [{id: one, url_env: CAMERA_NOT_CONFIGURED_FOR_TEST}]")
    with pytest.raises(ValueError, match="CAMERA_NOT_CONFIGURED_FOR_TEST"):
        cctv.load_settings(path)


def test_os_lock_prevents_duplicate_workers_and_releases(tmp_path):
    first = cctv.RuntimeLock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            cctv.RuntimeLock(tmp_path)
    finally:
        first.close()
    cctv.RuntimeLock(tmp_path).close()


def test_stale_worker_status_is_not_reported_online(tmp_path):
    cctv.atomic_json(tmp_path / "status.json", {
        "running": True, "heartbeat": time.time() - 20, "cameras": []})
    assert not cctv.read_status(tmp_path)["worker_online"]


def test_stalled_inference_is_stale_even_while_camera_receives_frames(tmp_path):
    worker = cctv.CameraWorker(cctv.Camera("one", "One", "unused"), object(),
                               cctv.Settings(()), tmp_path)
    worker.update(status="online", last_frame=time.time(), last_inference=time.time() - 20)
    assert worker.snapshot()["status"] == "stale"
    worker.update(last_inference=time.time(), last_processed_frame=time.time() - 20)
    assert worker.snapshot()["status"] == "stale"


def test_retention_deletes_only_finished_sessions(tmp_path):
    finished = tmp_path / "events/one/old/events.json"
    active = tmp_path / "events/one/current/events.json"
    cctv.atomic_json(finished, {"finished": True, "events": []})
    cctv.atomic_json(active, {"finished": False, "events": []})
    settings = replace(cctv.Settings(()), storage_mb=0)
    cctv.prune_events(tmp_path, settings)
    assert not finished.parent.exists()
    assert active.exists()


def test_crash_recovery_preserves_unknown_event_end_times(tmp_path):
    journal = tmp_path / "events/one/session/events.json"
    cctv.atomic_json(journal, {"finished": False, "events": [{"end": None}]})
    cctv.recover_journals(tmp_path)
    report = json.loads(journal.read_text())
    assert report["finished"] and report["interrupted"]
    assert report["events"][0]["end"] is None


def test_decoder_failure_reconnects_and_never_exposes_credentials(tmp_path, monkeypatch):
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(True)
        raise OSError("rtsp://operator:private-test-value@camera/live")

    monkeypatch.setattr(cctv.av, "open", fail)
    worker = cctv.CameraWorker(cctv.Camera("one", "One", "rtsp://camera/live"),
                               object(), cctv.Settings(()), tmp_path)
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while len(attempts) < 2 and time.monotonic() < deadline:
            time.sleep(.02)
        assert len(attempts) >= 2
        assert worker.snapshot()["reconnects"] >= 2
        assert "private-test-value" not in json.dumps(worker.snapshot())
    finally:
        worker.close()
    assert not worker.reader.is_alive() and not worker.processor.is_alive()


def test_processing_consumes_latest_frame_and_resets_after_reconnect(tmp_path, monkeypatch):
    processed, generations = [], []
    done = Event()

    class Processor:
        def __init__(self, *args, **kwargs):
            self.monitor = SimpleNamespace(events=[])
            self.directory = tmp_path / f"session-{len(generations)}"
            generations.append(self)

        def annotate(self, image, confidence, timestamp):
            processed.append(image.getpixel((0, 0))[0])
            done.set()
            return np.asarray(image), [], []

        def finish(self):
            pass

    monkeypatch.setattr(cctv, "StreamDetector", Processor)
    worker = cctv.CameraWorker(cctv.Camera("one", "One", "unused"), object(),
                               cctv.Settings(()), tmp_path)
    worker.generation = 1
    worker.latest = (100, 1, time.monotonic(), Image.new("RGB", (20, 20), (100, 0, 0)))
    worker.processor.start()
    try:
        assert done.wait(2)
        done.clear()
        with worker.condition:
            worker.generation = 2
            worker.latest = (200, 2, time.monotonic(), Image.new("RGB", (20, 20), (200, 0, 0)))
            worker.condition.notify()
        assert done.wait(2)
        assert processed == [100, 200]
        assert len(generations) == 2
    finally:
        worker.close()


def test_gateway_requires_valid_hash_and_has_no_public_upstream():
    password_hash = "$2a$14$" + "a" * 53
    config = gateway_config("cctv-pc", "operator", password_hash)
    assert "tls internal" in config
    assert "basic_auth" in config
    assert "reverse_proxy 127.0.0.1:8503" in config
    assert "skip_install_trust" in config
    with pytest.raises(ValueError):
        gateway_config("pc\nrespond unsafe", "operator", password_hash)
    with pytest.raises(ValueError):
        gateway_config("pc", "operator", "plaintext")


def test_gateway_setup_hashes_binary_lf_without_storing_password(tmp_path, monkeypatch):
    from scripts import configure_lan

    (tmp_path / "config").mkdir()
    monkeypatch.setattr(configure_lan, "ROOT", tmp_path)
    answers = iter(["camera-pc", "operator"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(configure_lan.getpass, "getpass", lambda _: "test-password-only")

    def hash_password(command, **kwargs):
        assert kwargs["input"] == b"test-password-only\n"
        assert "test-password-only" not in " ".join(command)
        return SimpleNamespace(stdout=("$2a$14$" + "a" * 53 + "\n").encode())

    monkeypatch.setattr(configure_lan.subprocess, "run", hash_password)
    configure_lan.main()
    assert "test-password-only" not in (tmp_path / "config/Caddyfile.local").read_text()
