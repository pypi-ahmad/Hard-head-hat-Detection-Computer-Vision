from pathlib import Path
from threading import Event
from types import SimpleNamespace

import av
import numpy as np
import pytest

from app import CameraProcessor
from safety import media


@pytest.mark.parametrize('url', [
    'file:///tmp/video.mp4', 'ftp://example.com/x', 'http://user:password@example.com/x',
    'http://example.com:8080/x',
])
def test_unsupported_urls_rejected(url):
    with pytest.raises(ValueError):
        media.public_destination(url)


@pytest.mark.parametrize('address', ['127.0.0.1', '10.0.0.2', '169.254.169.254', '::1', '192.168.1.2'])
def test_private_destinations_rejected(monkeypatch, address):
    monkeypatch.setattr(media.socket, 'getaddrinfo', lambda *a, **kw: [(None, None, None, None, (address, 80))])
    with pytest.raises(ValueError, match='private'):
        media.public_destination('http://example.com/video')


def test_redirect_revalidates_destination(monkeypatch, tmp_path):
    monkeypatch.setattr(media.socket, 'getaddrinfo', lambda host, *a, **kw: [
        (None, None, None, None, ('127.0.0.1' if host == 'localhost' else '8.8.8.8', 80))])
    sock = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(media.socket, 'create_connection', lambda *a, **kw: sock)
    response = SimpleNamespace(status=302, getheader=lambda _: 'http://localhost/video')
    connection = SimpleNamespace(request=lambda *a, **kw: None, getresponse=lambda: response, close=lambda: None)
    monkeypatch.setattr(media.http.client, 'HTTPConnection', lambda *a, **kw: connection)
    with pytest.raises(ValueError, match='private'):
        media.download_public('http://example.com/video', tmp_path / 'video', Event(), lambda *a: None)


@pytest.mark.parametrize('declared_size', ['100', '0'])
def test_download_limit_enforced_with_or_without_content_length(monkeypatch, tmp_path, declared_size):
    monkeypatch.setattr(media.socket, 'getaddrinfo', lambda *a, **kw: [(None, None, None, None, ('8.8.8.8', 80))])
    monkeypatch.setattr(media.socket, 'create_connection', lambda *a, **kw: SimpleNamespace(close=lambda: None))
    response = SimpleNamespace(status=200, getheader=lambda *a: declared_size, read=lambda _: b'x' * 100)
    connection = SimpleNamespace(request=lambda *a, **kw: None, getresponse=lambda: response, close=lambda: None)
    monkeypatch.setattr(media.http.client, 'HTTPConnection', lambda *a, **kw: connection)
    with pytest.raises(ValueError, match='limit'):
        media.download_public('http://example.com/video', tmp_path / 'video', Event(), lambda *a: None, max_bytes=10)


def test_failed_download_cleans_temporary_directory(monkeypatch):
    def fail(_):
        raise ValueError('bad URL')
    monkeypatch.setattr(media, 'resolve_video_url', fail)
    job = media.DownloadJob('bad')
    job.thread.join(timeout=2)
    assert job.done.is_set()
    assert job.error == 'bad URL'
    assert not Path(job.temp.name).exists()


def test_cancelled_download_cleans_temporary_directory(monkeypatch):
    entered = Event()
    monkeypatch.setattr(media, 'resolve_video_url', lambda url: url)
    def download(url, path, cancel, progress):
        entered.set()
        assert cancel.wait(timeout=2)
        raise InterruptedError('cancelled')
    monkeypatch.setattr(media, 'download_public', download)
    job = media.DownloadJob('url')
    assert entered.wait(timeout=2)
    job.close()
    job.thread.join(timeout=2)
    assert not job.thread.is_alive()
    assert not Path(job.temp.name).exists()


def test_invalid_video_releases_decoder_and_upload(monkeypatch):
    released, cleaned = [], []
    monkeypatch.setattr(media, 'validate_video_file', lambda _: None)
    monkeypatch.setattr(media.cv2, 'VideoCapture', lambda _: SimpleNamespace(
        isOpened=lambda: False, release=lambda: released.append(True)))
    controller = media.VideoSession()
    controller.start('invalid', None, SimpleNamespace(cleanup=lambda: cleaned.append(True)))
    controller.thread.join(timeout=2)
    assert controller.error == 'Unable to decode video'
    assert controller.finished and released and cleaned
    controller.close()


def test_camera_frame_callback_and_stop():
    calls = []
    def annotate(image, confidence):
        calls.append(confidence)
        return np.asarray(image), ['NO-Mask'], []
    camera = CameraProcessor(SimpleNamespace(annotate=annotate), 0.17)
    frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format='rgb24')
    output = camera.process(frame)
    assert output.width == 16
    assert calls == [0.17]
    assert camera.latest[1] == ['NO-Mask']
    camera.close()
    assert camera.process(frame) is frame
    assert calls == [0.17]


def test_text_playlist_rejected_before_decoder(tmp_path):
    path = tmp_path / 'fake.mp4'
    path.write_text('#EXTM3U\nhttp://127.0.0.1/private\n')
    with pytest.raises(ValueError, match='playlists'):
        media.validate_video_file(path)
