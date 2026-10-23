"""Enumerate attached cameras and run a user-started, session-owned preview."""

import sys
import time
from threading import Event, Lock, Thread

import cv2
from cv2_enumerate_cameras import enumerate_cameras
from PIL import Image


def attached_cameras():
    # One backend avoids listing the same Windows camera twice.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return enumerate_cameras(backend)


class PcCameraSession:
    def __init__(self):
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = None
        self.device = None
        self.latest = None
        self.error = None
        self.last_touch = time.monotonic()

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, camera, detector, confidence):
        self.close()
        from safety.events import StreamDetector

        if isinstance(detector, StreamDetector):
            detector.restart()
        self.device = (camera.index, camera.backend, camera.path)
        self.latest, self.error = None, None
        self.last_touch = time.monotonic()
        self.stop_event.clear()
        self.thread = Thread(target=self._run, args=(camera, detector, confidence), daemon=True)
        self.thread.start()

    def _run(self, camera, detector, confidence):
        capture = None
        try:
            capture = cv2.VideoCapture(camera.index, camera.backend)
            if not capture.isOpened():
                raise RuntimeError("Camera unavailable. Close other apps using it and refresh the list.")
            while not self.stop_event.is_set() and time.monotonic() - self.last_touch < 60:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera disconnected or stopped delivering frames. Refresh and retry.")
                captured = time.monotonic()
                result = detector.annotate(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                           confidence)
                with self.lock:
                    self.latest = (captured, result)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            from safety.events import StreamDetector

            if isinstance(detector, StreamDetector):
                detector.finish()

    def close(self):
        self.stop_event.set()
        if self.running:
            self.thread.join(timeout=15)
            if self.running:
                raise RuntimeError("Camera is still stopping; wait for the current capture/inference.")
        self.thread = None
        self.latest = None


def camera_controls(session, detector_factory, confidence):
    import streamlit as st

    st.caption("Cameras attached to the PC running this app, including virtual cameras.")
    st.caption("This preview runs while this workspace is open; configured CCTV feeds remain independent.")
    if st.button("Refresh cameras"):
        session.close()
        st.session_state.pop("attached_pc_cameras", None)
    if "attached_pc_cameras" not in st.session_state:
        try:
            st.session_state.attached_pc_cameras = attached_cameras()
        except Exception as exc:
            st.error(f"Unable to list PC cameras: {exc}")
            return
    cameras = st.session_state.attached_pc_cameras
    if not cameras:
        session.close()
        st.info("No PC cameras found. Connect a camera, allow Windows camera access, then refresh.")
        return
    camera = st.selectbox("PC camera", cameras,
                          format_func=lambda c: f"{c.name} (device {c.index})")
    device = (camera.index, camera.backend, camera.path)
    if session.device != device:
        session.close()
        session.device = device
    start, stop = st.columns(2)
    if start.button("Start PC camera", disabled=session.running):
        try:
            session.start(camera, detector_factory(), confidence)
        except Exception as exc:
            st.error(f"Unable to start camera: {exc}")
    if stop.button("Stop PC camera"):
        session.close()
    _preview(session)


def _preview(session):
    import streamlit as st

    @st.fragment(run_every=0.2)
    def render():
        session.last_touch = time.monotonic()
        with session.lock:
            latest = session.latest
        if latest:
            captured, result = latest
            if not session.running or time.monotonic() - captured > 5:
                st.warning("Preview stopped or stale.")
            st.image(result[0], alt="PC camera helmet and bare-head detections", width="stretch")
            left, right = st.columns(2)
            left.metric("Violation detections", len(result[1]))
            right.metric("Equipment detections", len(result[2]))
        elif session.running:
            st.info("Waiting for the first processed camera frame.")
        if session.error:
            st.error(session.error)

    render()
