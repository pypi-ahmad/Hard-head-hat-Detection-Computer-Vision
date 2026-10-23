import time

from streamlit.testing.v1 import AppTest

from safety.cctv import atomic_json


def test_dashboard_reports_offline_worker_without_crashing(tmp_path):
    app = AppTest.from_string(
        f"from cctv_app import main\nfrom pathlib import Path\nmain(Path({str(tmp_path)!r}))"
    ).run()
    assert not app.exception
    assert any("stopped" in error.value for error in app.error)


def test_dashboard_reads_live_camera_state_and_switches_views(tmp_path):
    atomic_json(tmp_path / "status.json", {"running": True, "heartbeat": time.time(),
        "cameras": [{"id": "one", "name": "Entrance", "status": "online",
            "last_frame": time.time(), "fps": 10, "latency_ms": 40,
            "last_inference": time.time(),
            "last_processed_frame": time.time(),
            "frames": 50, "reconnects": 0, "zone": None, "error": None,
            "bare_heads": 0, "helmets": 1}]})
    app = AppTest.from_string(
        f"from cctv_app import main\nfrom pathlib import Path\nmain(Path({str(tmp_path)!r}))"
    ).run()
    assert not app.exception and not app.error
    assert app.metric[0].value == "1 / 1"
    for workspace in ("Recorded events", "System", "Live cameras"):
        app.sidebar.radio[0].set_value(workspace).run()
        assert not app.exception
