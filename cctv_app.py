"""Read-only local CCTV dashboard; workers live in the independent runner."""

import argparse
import json
import time
from datetime import UTC, datetime

import streamlit as st

from safety.cctv import RUNTIME, read_status, runtime_path
from safety.pc_camera import PcCameraSession, camera_controls


@st.cache_resource(scope="session", on_release=lambda session: session.close())
def pc_camera_session():
    return PcCameraSession()


@st.cache_resource(max_entries=2)
def pc_camera_detector(identity, resolution):
    from safety.inference import Detector

    return Detector(identity[0], imgsz=resolution)


def selected_pc_detector():
    from safety.inference import checkpoint_identity, selected_checkpoint, selected_resolution

    checkpoint, _ = selected_checkpoint("shwd")
    if checkpoint is None:
        raise ValueError("A selected SHWD checkpoint is required")
    return pc_camera_detector(checkpoint_identity(checkpoint), selected_resolution("shwd"))


@st.fragment(run_every=1)
def live_view(root):
    report = read_status(root)
    if not report["worker_online"]:
        st.error("Monitoring worker is stopped or not responding. Start CCTV with Start.cmd.")
    cameras = report["cameras"]
    if not cameras:
        st.info("No cameras configured. Follow docs/CCTV_INSTALL.md, then restart CCTV.")
        return
    now = time.time()
    online_ids = {c["id"] for c in cameras if report["worker_online"] and c["status"] == "online"
                  and all(now - (c.get(key) or 0) <= 5
                          for key in ("last_frame", "last_inference", "last_processed_frame"))}
    left, middle, right = st.columns(3)
    left.metric("Online cameras", f"{len(online_ids)} / {len(cameras)}")
    middle.metric("Bare heads in latest frames", sum(c["bare_heads"] for c in cameras
                if c["id"] in online_ids))
    right.metric("Event storage", f"{report.get('storage_bytes', 0) / 1048576:.1f} MB")
    columns = st.columns(2 if len(cameras) > 1 else 1)
    for index, camera in enumerate(cameras):
        with columns[index % len(columns)].container(border=True):
            st.subheader(camera["name"])
            fresh = camera["id"] in online_ids
            if fresh:
                st.success(f"Online · {camera['fps']:.1f} inference FPS · {camera['latency_ms']} ms")
            else:
                status = ("Stale" if camera["status"] == "online" else camera["status"].capitalize())
                st.warning(f"{status if report['worker_online'] else 'Worker stopped'} · preview may be stale")
            path = root / f"{camera['id']}.jpg"
            try:
                st.image(path.read_bytes(), alt=f"Latest annotated frame from {camera['name']}",
                         width="stretch")
            except OSError:
                st.caption("Waiting for the first processed frame.")
            st.caption(f"Frames processed: {camera['frames']} · Reconnects: {camera['reconnects']}")
            if camera["zone"] is None:
                st.caption("Detection only. Configure a helmet-required area to record events.")
            if camera["error"]:
                st.error(camera["error"])


def event_view(root):
    sessions = []
    for path in (root / "events").glob("*/*/events.json"):
        try:
            sessions.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            continue
    sessions.sort(reverse=True)
    journals = [path for _, path in sessions]
    labels = {path: f"{path.parent.parent.name} · " + datetime.fromtimestamp(
              modified, UTC).strftime("%Y-%m-%d %H:%M UTC") for modified, path in sessions}
    if not journals:
        st.info("No events recorded. Events need a configured helmet-required area.")
        return
    if len(journals) > 200:
        st.caption("Showing the 200 most recent recording sessions.")
    selected = st.selectbox("Recording session", journals[:200], format_func=labels.get)
    try:
        raw = selected.read_bytes()
        report = json.loads(raw)
    except (OSError, ValueError):
        st.info("Recording changed during refresh. Select the session again.")
        return
    rows = []
    for event in report["events"]:
        row = dict(event)
        reference = report.get("time_reference")
        if reference:
            confirmed = reference["unix_timestamp"] + event["confirmed"] - reference["stream_timestamp"]
            row["confirmed_utc"] = datetime.fromtimestamp(confirmed, UTC).isoformat()
        rows.append(row)
    if report.get("interrupted"):
        st.warning("This recording was interrupted. Open events have unknown end times.")
    st.dataframe(rows, alt="Recorded bare-head events")
    st.download_button("Download events JSON", raw, f"{report['source']}-events.json")
    if report["events"]:
        event = st.selectbox("Event", report["events"], format_func=lambda e: str(e["id"]))
        image = event.get("image")
        if image:
            from pathlib import Path
            path = Path(image).resolve()
            if path.is_relative_to((root / "events").resolve()):
                try:
                    st.image(path.read_bytes(), alt="Camera frame when the event was confirmed")
                except FileNotFoundError:
                    st.caption("Snapshot expired under the retention policy.")


def main(root=RUNTIME):
    st.set_page_config(page_title="Helmet CCTV", page_icon=":material/videocam:", layout="wide")
    st.title("Helmet CCTV")
    st.caption("Continuous helmet and bare-head monitoring. Counts describe detections in the latest frame.")
    page = st.sidebar.radio("Workspace", ["Live cameras", "PC camera", "Recorded events", "System"])
    session = pc_camera_session()
    if page != "PC camera":
        session.close()
    if page == "Live cameras":
        live_view(root)
    elif page == "PC camera":
        camera_controls(session, selected_pc_detector, 0.12)
        st.caption("Detection preview only. Configure network feeds for continuous event monitoring.")
    elif page == "Recorded events":
        event_view(root)
    else:
        report = read_status(root)
        st.json(report)
        st.caption("Local dashboard. Workers continue when you close this browser.")
        st.caption("Camera changes take effect after restarting Start.cmd. Installation guide: docs/CCTV_INSTALL.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime")
    args, _ = parser.parse_known_args()
    main(runtime_path(args.runtime))
