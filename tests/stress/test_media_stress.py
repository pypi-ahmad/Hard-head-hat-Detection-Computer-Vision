import tempfile
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np

from safety.media import VideoSession


def test_repeated_real_video_decoder_cleanup(tmp_path):
    source = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 32))
    assert writer.isOpened()
    try:
        for _ in range(8):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    detector = SimpleNamespace(annotate=lambda image, confidence: (np.asarray(image), [], []))
    controller = VideoSession()
    for _ in range(20):
        owner = tempfile.TemporaryDirectory(prefix='ppe-test-')
        path = Path(owner.name) / 'video.avi'
        path.write_bytes(source.read_bytes())
        controller.touch(speed=2.0)
        controller.start(path, detector, owner)
        controller.thread.join(timeout=5)
        assert not controller.thread.is_alive()
        assert controller.position == 8
        assert controller.error is None
        assert not path.parent.exists()
        controller.close()


def test_video_pause_resume_confidence_and_stop(tmp_path):
    source = tmp_path / 'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 30, (32, 32))
    assert writer.isOpened()
    for _ in range(60):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    entered, release, second = Event(), Event(), Event()
    confidences = []

    def annotate(image, confidence):
        confidences.append(confidence)
        if len(confidences) == 1:
            entered.set()
            assert release.wait(timeout=2)
        else:
            second.set()
        return np.asarray(image), [], []

    controller = VideoSession()
    controller.touch(confidence=.2)
    controller.start(source, SimpleNamespace(annotate=annotate))
    try:
        assert entered.wait(timeout=2)
        controller.playing.clear()
        release.set()
        assert not second.wait(timeout=.15)
        controller.touch(confidence=.7)
        controller.playing.set()
        assert second.wait(timeout=2)
        assert confidences[:2] == [.2, .7]
    finally:
        release.set()
        controller.close()
    position = controller.position
    time.sleep(.05)
    assert controller.position == position
    source.unlink()  # Windows fails this if the decoder still owns the file.
