import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.v1_scratch.detector import MOG2Detector
from src.v1_scratch.tracker import CentroidTracker
from src.utils import compute_iou, draw_tracks


def _make_synthetic_frames(n=30, width=320, height=240):
    """Create synthetic frames with a moving rectangle."""
    frames = []
    for i in range(n):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)
        x = 50 + i * 3
        cv2.rectangle(frame, (x, 60), (x + 40, 160), (0, 0, 255), -1)
        frames.append(frame)
    return frames


def test_detector_returns_list():
    frames = _make_synthetic_frames(30)
    detector = MOG2Detector(min_area=500)
    detector.warmup(frames[:10])
    all_detections = []
    for frame in frames[10:]:
        dets = detector.detect(frame)
        all_detections.append(dets)
    assert isinstance(all_detections, list)
    for dets in all_detections:
        assert isinstance(dets, list)
        for d in dets:
            assert "bbox" in d
            assert "conf" in d
            assert len(d["bbox"]) == 4


def test_tracker_assigns_ids():
    tracker = CentroidTracker()
    fake_detections = [
        {"bbox": [10, 10, 50, 50], "conf": 0.9},
        {"bbox": [200, 200, 250, 250], "conf": 0.8},
    ]
    tracks = tracker.update(fake_detections)
    assert isinstance(tracks, list)
    for t in tracks:
        assert "id" in t
        assert isinstance(t["id"], int)


def test_iou_zero_nonoverlap():
    box_a = [0, 0, 10, 10]
    box_b = [20, 20, 30, 30]
    assert compute_iou(box_a, box_b) == 0.0


def test_iou_one_identical():
    box = [10, 10, 50, 50]
    assert abs(compute_iou(box, box) - 1.0) < 1e-6


def test_tracker_disappear_removal():
    tracker = CentroidTracker(max_disappeared=3)
    det = [{"bbox": [10, 10, 50, 50], "conf": 0.9}]
    tracker.update(det)

    # Stop sending detections
    for i in range(4):
        tracks = tracker.update([])

    # After max_disappeared + 1 updates with no detections, track should be removed
    assert len(tracker.tracks) == 0


def test_draw_tracks_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracks = [{"id": 1, "bbox": [10, 10, 100, 200], "conf": 0.85}]
    out = draw_tracks(frame, tracks)
    assert out.shape == frame.shape
