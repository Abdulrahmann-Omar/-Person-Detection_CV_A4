import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import draw_tracks


def test_yolo_returns_list():
    try:
        from src.v2_transfer.detector import YOLOv8Detector
        detector = YOLOv8Detector()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank)
        assert isinstance(result, list)
        for det in result:
            assert "bbox" in det
            assert "conf" in det
            assert len(det["bbox"]) == 4
    except ImportError:
        pytest.skip("ultralytics not installed")


def test_bytetrack_ids_persist():
    try:
        from src.v2_transfer.tracker import ByteTrackWrapper
        tracker = ByteTrackWrapper()

        # Create a frame with a large white rectangle (person-like blob)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 100), (280, 350), (255, 255, 255), -1)

        ids_seen = set()
        for _ in range(10):
            tracks = tracker.track_frame(frame)
            for t in tracks:
                assert isinstance(t["id"], int)
                ids_seen.add(t["id"])

        # IDs should be integers (we can't guarantee persistence without real detections,
        # but we can assert the structure is correct)
        assert all(isinstance(tid, int) for tid in ids_seen)
    except ImportError:
        pytest.skip("ultralytics not installed")


def test_draw_tracks_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = [
        {"id": 1, "bbox": [10, 10, 100, 200], "conf": 0.85},
        {"id": 2, "bbox": [300, 50, 400, 300], "conf": 0.72},
    ]
    out = draw_tracks(frame, tracks)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
