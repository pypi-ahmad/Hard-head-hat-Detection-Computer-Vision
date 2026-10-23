"""Bounded public video downloads and cancellable video processing."""

import http.client
import ipaddress
import socket
import ssl
import tempfile
import time
from pathlib import Path
from threading import Event, Lock, Thread
from urllib.parse import parse_qs, urljoin, urlsplit

import cv2
from PIL import Image

MAX_BYTES = 1000 * 1024 * 1024


def validate_video_file(path):
    """Reject text playlists before a decoder could follow embedded network URLs."""
    with Path(path).open("rb") as handle:
        header = handle.read(12)
    mp4_mov = len(header) >= 8 and header[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free")
    avi = header[:4] == b"RIFF" and header[8:12] == b"AVI "
    if not (mp4_mov or avi):
        raise ValueError("Use a prerecorded MP4, MOV, or AVI file; playlists and streams are not supported")


def public_destination(url):
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("Use a public HTTP(S) video URL without credentials")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in (80, 443):
        raise ValueError("Only public HTTP(S) ports are supported")
    addresses = {item[4][0] for item in socket.getaddrinfo(parts.hostname, port, type=socket.SOCK_STREAM)}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Local and private network destinations are not supported")
    return parts, sorted(addresses)[0], port


def download_public(url, path, cancel, progress, max_bytes=MAX_BYTES):
    """Pin each connection to its validated IP, retaining HTTPS hostname verification."""
    for _ in range(6):
        parts, address, port = public_destination(url)
        if cancel.is_set():
            raise InterruptedError("Download cancelled")
        connection = http.client.HTTPConnection(parts.hostname, port, timeout=10)
        raw = socket.create_connection((address, port), timeout=10)
        try:
            connection.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=parts.hostname) \
                if parts.scheme == "https" else raw
            target = parts.path or "/"
            if parts.query:
                target += "?" + parts.query
            connection.request("GET", target, headers={"User-Agent": "Mozilla/5.0", "Accept": "video/*,*/*;q=0.8"})
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise ValueError("Redirect has no destination")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"Video download returned HTTP {response.status}")
            total = int(response.getheader("Content-Length", "0"))
            if total > max_bytes:
                raise ValueError("Video exceeds the 1,000 MB limit")
            size = 0
            with Path(path).open("wb") as handle:
                while chunk := response.read(256 * 1024):
                    if cancel.is_set():
                        raise InterruptedError("Download cancelled")
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("Video exceeds the 1,000 MB limit")
                    handle.write(chunk)
                    progress(size, total)
            if not size:
                raise ValueError("The video download was empty")
            if total and size != total:
                raise ValueError("Video download ended before all bytes arrived")
            return Path(path)
        finally:
            connection.close()
            raw.close()
    raise ValueError("Too many redirects")


def resolve_video_url(url):
    parts, _, _ = public_destination(url)
    host = parts.hostname.lower()
    if host not in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"):
        return url
    query = parse_qs(parts.query)
    if "list" in query or parts.path == "/playlist":
        raise ValueError("Playlists are not supported; paste a single video link")
    if parts.path.startswith("/live"):
        raise ValueError("Live streams are not supported")
    from yt_dlp import YoutubeDL

    options = {"noplaylist": True, "quiet": True, "no_warnings": True, "socket_timeout": 10,
               "retries": 1, "extractor_retries": 1, "js_runtimes": {"node": {}},
               "format": "bestvideo[height<=720][ext=mp4][protocol=https]/best[height<=720][protocol=https]"}
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming") or info.get("entries") is not None:
        raise ValueError("Only single prerecorded videos are supported")
    if info.get("filesize", 0) and info["filesize"] > MAX_BYTES:
        raise ValueError("Video exceeds the 1,000 MB limit")
    return info["url"]


class DownloadJob:
    def __init__(self, url):
        self.temp = tempfile.TemporaryDirectory(prefix="ppe-download-")
        self.path = Path(self.temp.name) / "video.mp4"
        self.cancel = Event()
        self.lock = Lock()
        self.done = Event()
        self.error = None
        self.bytes = self.total = 0
        # Let pending network I/O close its file before Python runs temp-dir finalizers.
        self.thread = Thread(target=self._run, args=(url,), daemon=False)
        self.thread.start()

    def _progress(self, size, total):
        with self.lock:
            self.bytes, self.total = size, total

    def _run(self, url):
        try:
            resolved = resolve_video_url(url)
            download_public(resolved, self.path, self.cancel, self._progress)
            validate_video_file(self.path)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.cancel.is_set() or self.error:
                self.temp.cleanup()
            self.done.set()

    def close(self):
        self.cancel.set()
        self.thread.join(timeout=0.1)
        if self.done.is_set():
            self.temp.cleanup()


class VideoSession:
    """One decoder worker, one latest frame, and a UI heartbeat for disconnect cleanup."""

    def __init__(self):
        self.lock = Lock()
        self.stop_event = Event()
        self.playing = Event()
        self.thread = None
        self.confidence = 0.4
        self.speed = 1.0
        self.last_touch = time.monotonic()
        self.latest = None
        self.position = self.total = 0
        self.error = None
        self.finished = False
        self.temp = None

    def touch(self, confidence=None, speed=None):
        with self.lock:
            self.last_touch = time.monotonic()
            if confidence is not None:
                self.confidence = confidence
            if speed is not None:
                self.speed = speed

    def start(self, path, detector, owner=None):
        self.close()
        from safety.events import StreamDetector

        if isinstance(detector, StreamDetector):
            detector.restart()
        self.processor = detector
        self.stop_event.clear()
        self.playing.set()
        self.latest, self.error, self.finished = None, None, False
        self.position = self.total = 0
        self.temp = owner
        self.touch()
        self.thread = Thread(target=self._run, args=(str(path), detector), daemon=True)
        self.thread.start()

    def _run(self, path, detector):
        capture = None
        try:
            validate_video_file(path)
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                raise ValueError("Unable to decode video")
            self.total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS) or 25
            while not self.stop_event.is_set():
                if time.monotonic() - self.last_touch > 60:
                    break
                if not self.playing.wait(timeout=0.1):
                    continue
                started = time.monotonic()
                success, frame = capture.read()
                if not success:
                    if not self.position:
                        raise ValueError("Video contains no decodable frames")
                    break
                with self.lock:
                    confidence, speed = self.confidence, self.speed
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                from safety.events import StreamDetector

                source_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if source_time <= 0:
                    source_time = self.position / fps
                result = (detector.annotate(image, confidence, source_time)
                          if isinstance(detector, StreamDetector) else detector.annotate(image, confidence))
                with self.lock:
                    self.latest = result
                    self.position += 1
                self.stop_event.wait(max(0, 1 / (fps * speed) - (time.monotonic() - started)))
        except Exception as exc:
            self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()
            from safety.events import StreamDetector

            if isinstance(detector, StreamDetector):
                detector.finish()
            self.finished = True
            self.playing.clear()
            if self.temp:
                self.temp.cleanup()

    def close(self):
        self.stop_event.set()
        self.playing.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                raise RuntimeError("Video processor is still stopping; retry after current inference")
        self.thread = None


def save_upload(upload):
    if upload.size > MAX_BYTES:
        raise ValueError("Video exceeds the 1,000 MB limit")
    owner = tempfile.TemporaryDirectory(prefix="ppe-upload-")
    path = Path(owner.name) / ("video" + Path(upload.name).suffix)
    try:
        with path.open("wb") as handle:
            handle.write(upload.getbuffer())
    except Exception:
        owner.cleanup()
        raise
    return path, owner
