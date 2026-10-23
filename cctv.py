"""Run continuous camera workers and the local dashboard independently of browsers."""

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from threading import Event, Thread

import av
import torch

from safety.cctv import (
    Camera,
    CameraWorker,
    RuntimeLock,
    Settings,
    atomic_json,
    load_settings,
    prune_events,
    recover_journals,
    runtime_path,
)
from safety.inference import Detector, selected_checkpoint, selected_resolution
from safety.paths import ROOT, project_path


def open_dashboard(port, lan, stop):
    while not stop.wait(.5):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open("https://localhost:8443" if lan else f"http://127.0.0.1:{port}")
                    return
        except (OSError, urllib.error.URLError):
            continue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/cameras.local.yaml")
    parser.add_argument("--runtime", help="Separate runtime directory for an isolated installation")
    parser.add_argument("--check", action="store_true", help="Validate configuration/model without connecting")
    parser.add_argument("--probe", action="store_true", help="Decode one frame per configured camera")
    parser.add_argument("--demo", help="Loop a local video in real time instead of connecting cameras")
    parser.add_argument("--headless", action="store_true", help="Workers only; no dashboard server")
    parser.add_argument("--lan", action="store_true", help="Run the configured authenticated HTTPS gateway")
    parser.add_argument("--open-browser", action="store_true", help="Open the dashboard after its server is ready")
    parser.add_argument("--port", type=int, default=8503)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds (verification)")
    parser.add_argument("--device", default=None, help="cpu or CUDA device, for example 0")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    if args.lan and (args.headless or args.port != 8503):
        parser.error("LAN gateway uses dashboard port 8503 and cannot run headless")
    if args.lan and not all((ROOT / name).is_file() for name in
                           ("vendor/caddy/caddy.exe", "config/Caddyfile.local")):
        parser.error("Configure the LAN gateway with Install.cmd first")
    try:
        if args.demo:
            video = project_path(args.demo)
            if not video.is_file():
                raise ValueError("Demo video does not exist")
            settings = Settings((Camera("demo", "Recorded video demo", str(video), file=True),))
        else:
            settings = load_settings(project_path(args.config))
        checkpoint, _ = selected_checkpoint("shwd")
        if checkpoint is None:
            raise ValueError("No SHWD model installed. Use the release app package or train a checkpoint.")
        detector = Detector(checkpoint, device=args.device, imgsz=selected_resolution("shwd"))
        if detector.profile != "shwd":
            raise ValueError("CCTV requires a helmet/bare-head checkpoint")
    except (OSError, ValueError, TypeError, KeyError):
        print("Setup incomplete. Check camera configuration, required environment variables, and SHWD model.",
              file=sys.stderr)
        return 2
    print(json.dumps({"cameras": len(settings.cameras), "cuda_available": torch.cuda.is_available(),
                      "model": checkpoint.name, "configuration": "valid"}))
    if args.probe:
        av.logging.set_level(None)
        for camera in settings.cameras:
            try:
                options = {"rtsp_transport": "tcp"} if camera.source.startswith("rtsp") else {}
                with av.open(camera.source, timeout=(5., 5.), options=options) as stream:
                    frame = next(stream.decode(video=0))
                    print(f"{camera.id}: decoded {frame.width}x{frame.height}")
            except Exception:
                print(f"{camera.id}: connection/decoding failed", file=sys.stderr)
                return 3
        return 0
    if args.check:
        return 0
    root = runtime_path(args.runtime)
    try:
        lock = RuntimeLock(root)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    workers, dashboard, gateway, stop = [], None, None, Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    started, pruned, storage = time.monotonic(), 0., 0
    try:
        recover_journals(root)
        workers = [CameraWorker(camera, detector, settings, root) for camera in settings.cameras]
        for worker in workers:
            worker.start()
        if not args.headless:
            dashboard = subprocess.Popen([sys.executable, "-m", "streamlit", "run",
                str(ROOT / "cctv_app.py"), "--server.address=127.0.0.1",
                f"--server.port={args.port}", "--server.headless=true", "--",
                "--runtime", str(root)], cwd=ROOT)
            print(f"CCTV dashboard: http://127.0.0.1:{args.port}", flush=True)
            if args.lan:
                gateway = subprocess.Popen([str(ROOT / "vendor/caddy/caddy.exe"), "run", "--config",
                    str(ROOT / "config/Caddyfile.local"), "--adapter", "caddyfile"], cwd=ROOT)
            if args.open_browser:
                Thread(target=open_dashboard, args=(args.port, args.lan, stop), daemon=True).start()
        while not stop.is_set():
            now = time.monotonic()
            if now - pruned >= 60:
                storage = prune_events(root, settings)
                pruned = now
            atomic_json(root / "status.json", {"running": True, "heartbeat": time.time(),
                "started_at": time.time() - (now - started), "storage_bytes": storage,
                "retention_days": settings.retention_days, "storage_mb": settings.storage_mb,
                "cameras": [worker.snapshot() for worker in workers]})
            if dashboard is not None and dashboard.poll() is not None:
                raise RuntimeError("Dashboard stopped; workers will stop for a clean restart")
            if gateway is not None and gateway.poll() is not None:
                raise RuntimeError("HTTPS gateway stopped; restarting the installation is required")
            if any(worker.snapshot()["status"] == "error" for worker in workers):
                raise RuntimeError("A camera worker failed; check model and disk before restarting")
            if args.duration is not None and now - started >= args.duration:
                break
            stop.wait(1)
    finally:
        stop.set()
        shutdown_errors = []
        for worker in workers:
            try:
                worker.close()
            except RuntimeError as exc:
                shutdown_errors.append(str(exc))
        for child in (gateway, dashboard):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        atomic_json(root / "status.json", {"running": False, "heartbeat": time.time(),
                    "cameras": [worker.snapshot() for worker in workers]})
        lock.close()
        if shutdown_errors:
            raise RuntimeError("; ".join(shutdown_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
