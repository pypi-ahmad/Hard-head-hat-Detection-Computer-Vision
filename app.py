"""Streamlit interface for image, video, browser camera, and public video URLs."""

from threading import Lock

import av
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_webrtc import webrtc_streamer

from safety.annotations import format_reasons
from safety.events import StreamDetector
from safety.inference import (
    PRESETS,
    Detector,
    checkpoint_identity,
    custom_checkpoints,
    selected_checkpoint,
    selected_resolution,
)
from safety.media import DownloadJob, VideoSession, save_upload
from safety.profiles import PROFILES


@st.cache_resource(max_entries=2, show_spinner="Loading detector...")
def cached_detector(identity):
    return Detector(identity[0], imgsz=identity[3]) if len(identity) > 3 else Detector(identity[0])


class CameraProcessor:
    def __init__(self, detector, confidence):
        self.detector = detector
        self.confidence = confidence
        self.lock = Lock()
        self.processing = Lock()
        self.latest = None
        self.error = None
        self.active = True

    def process(self, frame):
        with self.processing:
            return self._process(frame)

    def _process(self, frame):
        with self.lock:
            confidence, active = self.confidence, self.active
        if not active:
            return frame
        try:
            picture = Image.fromarray(frame.to_ndarray(format="rgb24"))
            result = self.detector.annotate(picture, confidence)
            with self.lock:
                self.latest = result
            return av.VideoFrame.from_ndarray(result[0], format="rgb24")
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.active = False
            return frame

    def close(self):
        with self.lock:
            self.active = False
        with self.processing:
            if isinstance(self.detector, StreamDetector):
                self.detector.finish()


class MediaResources:
    def __init__(self):
        self.video = VideoSession()
        self.download = None
        self.camera = None
        self.key = None

    def close(self):
        self.video.close()
        if self.download:
            self.download.close()
            self.download = None
        if self.camera:
            self.camera.close()
            self.camera = None


@st.cache_resource(scope="session", on_release=lambda resource: resource.close())
def session_media():
    return MediaResources()


def metrics(violations, equipment):
    left, right = st.columns(2)
    shwd = st.session_state.get("detection_profile", "shwd") == "shwd"
    left.metric("Bare heads" if shwd else "Violation detections", len(violations))
    right.metric("Helmets" if shwd else "PPE items detected", len(equipment))
    if violations:
        left.error(format_reasons(violations))
    if equipment:
        right.success(format_reasons(equipment))


@st.fragment(run_every=0.2)
def video_status(controller):
    controller.touch()
    with controller.lock:
        latest, position, total = controller.latest, controller.position, controller.total
    if latest is not None:
        st.image(latest[0], alt="Annotated video frame")
        metrics(latest[1], latest[2])
    if total:
        st.progress(min(1.0, position / total), text=f"Frame {position} of {total}")
    if controller.error:
        st.error(controller.error)
    elif controller.finished:
        st.caption("Processing finished. Start again to replay.")
    processor = getattr(controller, "processor", None)
    if isinstance(processor, StreamDetector):
        event_status(processor, "video-events")


@st.fragment(run_every=0.5)
def camera_status(processor):
    with processor.lock:
        latest, error = processor.latest, processor.error
    if latest:
        metrics(latest[1], latest[2])
    if error:
        st.error(error)
    if isinstance(processor.detector, StreamDetector):
        event_status(processor.detector, "camera-events")


def event_status(processor, key):
    import json

    events = json.loads(processor.json())["events"]
    st.caption(f"{len(events)} confirmed events in this stream")
    st.download_button("Export events JSON", processor.json(), "events.json", key=f"{key}-json")
    st.download_button("Export events CSV", processor.csv(), "events.csv", key=f"{key}-csv")


@st.fragment(run_every=0.5)
def download_status(resources):
    job = resources.download
    if job is None:
        return
    if job.done.is_set():
        if job.error:
            st.error(f"Download failed: {job.error}")
        else:
            st.success("Video ready. Select Start to analyze it.")
        return
    with job.lock:
        size, total = job.bytes, job.total
    if total:
        st.progress(min(size / total, 1.0), text=f"Downloaded {size / 1048576:.1f} MB")
    else:
        st.caption(f"Resolving or downloading video: {size / 1048576:.1f} MB")
    if st.button("Cancel download"):
        job.close()
        resources.download = None
        st.rerun()


def video_controls(resources, detector, confidence, upload=None):
    controller = resources.video
    speed = st.select_slider("Playback speed", options=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], value=1.0)
    controller.touch(confidence, speed)
    start, pause, stop = st.columns(3)
    if start.button("Start", key="start_video"):
        try:
            if upload is not None:
                path, owner = save_upload(upload)
                controller.start(path, detector, owner)
            elif resources.download and resources.download.done.is_set() and not resources.download.error:
                controller.start(resources.download.path, detector)
            else:
                st.info("Upload or download a video first.")
        except Exception as exc:
            st.error(str(exc))
    if pause.button("Pause / resume", key="pause_video"):
        if controller.playing.is_set():
            controller.playing.clear()
        elif controller.thread and not controller.finished:
            controller.playing.set()
    if stop.button("Stop", key="stop_video"):
        controller.close()
    video_status(controller)


def main():
    st.set_page_config(page_title="Helmet detection", page_icon=":material/health_and_safety:", layout="wide")
    st.title("Helmet detection")
    st.caption("Detect helmets and bare heads. Counts describe detections, not unique workers.")
    resources = session_media()
    profile = st.sidebar.selectbox("Detection mode", ["shwd", "ppe"],
                                   format_func=lambda value: PROFILES[value]["title"],
                                   key="detection_profile")
    if st.session_state.get("active_profile") != profile:
        resources.close()
        st.session_state.active_profile = profile
    try:
        default, default_confidence = selected_checkpoint(profile)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Unable to read selected checkpoint metadata: {exc}")
        return
    options = {"Selected checkpoint": default} if default else {}
    if profile == "ppe":
        options.update(PRESETS)
    options.update({f"Custom: {path.parent.parent.name}": path for path in custom_checkpoints(profile)})
    if not options:
        st.info("This detection mode has no trained checkpoint yet.")
        return
    if profile != "ppe" and default is None:
        st.warning("Experimental checkpoint: validation requirements have not been met yet.")
    label = st.sidebar.selectbox("Model", list(options))
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, default_confidence, step=0.01,
                                   key=f"confidence-{profile}-{label}")
    page = st.sidebar.selectbox("Workspace", ["Detection", "CCTV monitor", "Annotation review", "Event review"])
    if page == "CCTV monitor":
        from cctv_app import live_view
        from safety.cctv import RUNTIME

        resources.close()
        st.subheader("CCTV monitor")
        live_view(RUNTIME)
        return
    source = st.radio("Input", ["Image", "Video", "Browser camera", "Video link"], horizontal=True)
    zone = None
    if profile == "shwd" and source != "Image" and page == "Detection":
        required = st.sidebar.checkbox("Helmet-required area")
        if required:
            rectangle = st.sidebar.checkbox("Use rectangular area")
            if rectangle:
                from safety.review_ui import zone_editor

                with st.sidebar:
                    st.caption("Drag the area in frame space. The yellow outline appears on video.")
                    zone = zone_editor()
            else:
                zone = (0., 0., 1., 1.)
        st.sidebar.caption("Events require 1 second of bare-head evidence. IDs belong to this stream only.")
    identity = checkpoint_identity(options[label])
    try:
        if label == "Selected checkpoint" and selected_resolution(profile) != 640:
            identity += (selected_resolution(profile),)
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"Unable to read inference resolution: {exc}")
        return
    key = (profile, identity, source, confidence, zone, page)
    if resources.key != key:
        resources.close()
        resources.key = key
    try:
        detector = cached_detector(identity)
        actual_profile = getattr(detector, "profile", "ppe" if detector.is_custom else None)
        generic_preset = profile == "ppe" and options[label] in PRESETS.values()
        if not generic_preset and actual_profile != profile:
            raise ValueError("Checkpoint classes do not match this detection mode")
    except Exception as exc:
        st.error(f"Unable to load selected model: {exc}")
        return
    st.sidebar.caption(str(detector.path.name))
    if page == "Annotation review":
        if profile != "shwd":
            st.info("Select the helmet detection mode to review annotations.")
            return
        from safety.review_ui import review_page
        review_page(detector)
        return
    if page == "Event review":
        import json

        from safety.paths import ARTIFACTS

        journals = sorted((ARTIFACTS / "events").glob("*/events.json"),
                          key=lambda path: path.stat().st_mtime, reverse=True)
        if not journals:
            st.info("No confirmed events recorded yet. Mark a helmet-required area during detection.")
            return
        journal = st.selectbox("Recorded stream", journals, format_func=lambda p: p.parent.name)
        report = json.loads(journal.read_text(encoding="utf-8"))
        st.write(report["source"])
        st.dataframe(report["events"])
        event = st.selectbox("Event image", report["events"], format_func=lambda e: str(e["id"]))
        if event.get("image"):
            st.image(event["image"], alt="Recorded event image")
        st.download_button("Download journal", journal.read_bytes(), "events.json")
        return
    if profile == "shwd" and source != "Image":
        if not hasattr(resources, "stream") or resources.stream_key != key:
            resources.stream = StreamDetector(detector, confidence, zone, source)
            resources.stream_key = key
        detector = resources.stream
    if not getattr(detector, "is_custom", True):
        st.warning("This pretrained model shows person detections only. Select a PPE checkpoint for equipment detection.")
    if source == "Image":
        uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, alt="Uploaded image", width="stretch")
                if st.button("Analyze image"):
                    annotated, violations, equipment = detector.annotate(image, confidence)
                    st.image(annotated, alt="PPE detections", width="stretch")
                    metrics(violations, equipment)
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
                st.error(f"Image analysis failed: {exc}")
    elif source == "Video":
        uploaded = st.file_uploader("Video", type=["mp4", "avi", "mov"])
        upload_key = (uploaded.file_id if uploaded else None)
        if st.session_state.get("upload_key") != upload_key:
            resources.video.close()
            st.session_state.upload_key = upload_key
        if uploaded:
            video_controls(resources, detector, confidence, uploaded)
    elif source == "Video link":
        url = st.text_input("Public YouTube or direct video URL", placeholder="https://...")
        st.caption("Prerecorded videos only, up to 1,000 MB. YouTube downloads use up to 720p.")
        if st.button("Download video", disabled=not url):
            resources.close()
            resources.download = DownloadJob(url)
        download_status(resources)
        video_controls(resources, detector, confidence)
    else:
        if resources.camera is None:
            resources.camera = CameraProcessor(detector, confidence)
        with resources.camera.lock:
            resources.camera.confidence = confidence
        webrtc_streamer(
            key=f"camera-{identity}", video_frame_callback=resources.camera.process,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        st.caption("Allow browser camera access. Remote connections require HTTPS; some networks also require TURN.")
        camera_status(resources.camera)


if __name__ == "__main__":
    main()
