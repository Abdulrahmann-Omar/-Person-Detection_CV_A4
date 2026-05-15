"""
V1: Background Subtraction + Centroid Tracker on Modal (CPU).
Run with: modal run modal_app/modal_v1_scratch.py
"""
import sys
import os
import time
import numpy as np
import cv2

# ── local imports for the entrypoint only ───────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modal_app.common import app, image, volume, VOLUME_PATH, src_mount


# ── helpers (also used inside Modal containers) ──────────────────────────────
def _generate_synthetic_video(path: str, n_frames: int = 150, fps: float = 30.0,
                               width: int = 640, height: int = 480):
    """Create a synthetic video with three moving coloured rectangles."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    objects = [
        {"x": 50,  "y": 140, "w": 40, "h": 110, "vx": 4,  "vy": 0, "color": (50, 50, 220)},
        {"x": 500, "y": 190, "w": 38, "h": 100, "vx": -3, "vy": 1, "color": (50, 200, 50)},
        {"x": 280, "y": 80,  "w": 44, "h": 120, "vx": 2,  "vy": 2, "color": (220, 50, 50)},
    ]
    for _ in range(n_frames):
        frame = np.full((height, width, 3), 210, dtype=np.uint8)
        cv2.rectangle(frame, (0, height - 60), (width, height), (120, 120, 120), -1)
        for obj in objects:
            obj["x"] = int(obj["x"] + obj["vx"])
            obj["y"] = int(obj["y"] + obj["vy"])
            if obj["x"] <= 0 or obj["x"] + obj["w"] >= width:
                obj["vx"] *= -1
            if obj["y"] <= 0 or obj["y"] + obj["h"] >= height - 60:
                obj["vy"] *= -1
            obj["x"] = max(0, min(obj["x"], width - obj["w"]))
            obj["y"] = max(0, min(obj["y"], height - obj["h"]))
            cv2.rectangle(frame,
                          (obj["x"], obj["y"]),
                          (obj["x"] + obj["w"], obj["y"] + obj["h"]),
                          obj["color"], -1)
        writer.write(frame)
    writer.release()


# ── Modal function ────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    mounts=[src_mount],
    timeout=600,
    cpu=2,
)
def run_v1(video_filename: str = "synthetic_test.mp4", warmup_frames: int = 30) -> dict:
    """Run the full V1 (MOG2 + Centroid Tracker) pipeline on Modal."""
    import sys
    sys.path.insert(0, "/root")

    from src.v1_scratch.detector import MOG2Detector
    from src.v1_scratch.tracker import CentroidTracker
    from src.utils import draw_tracks, write_output_video, benchmark_speed

    input_path = os.path.join(VOLUME_PATH, video_filename)

    # Generate synthetic video if not present in the volume
    if not os.path.exists(input_path):
        print(f"[V1] Generating synthetic video → {input_path}")
        _generate_synthetic_video(input_path)
        volume.commit()

    # Read all frames
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[V1] Loaded {len(frames)} frames at {fps:.1f} FPS")

    detector = MOG2Detector(min_area=500)
    tracker  = CentroidTracker()

    # Warmup
    wup = min(warmup_frames, len(frames) // 4)
    if wup > 0:
        detector.warmup(frames[:wup])
        print(f"[V1] Warmed up on {wup} frames")

    annotated_frames = []
    total_detections = 0
    t_start = time.time()

    for i, frame in enumerate(frames):
        dets   = detector.detect(frame)
        tracks = tracker.update(dets)
        total_detections += len(dets)
        annotated = draw_tracks(frame, tracks)
        annotated_frames.append(annotated)
        if (i + 1) % 50 == 0:
            print(f"[V1] Frame {i+1}/{len(frames)}")

    elapsed = time.time() - t_start
    proc_fps = len(frames) / elapsed if elapsed > 0 else 0

    # Speed benchmark (single frame)
    if frames:
        def _step(f):
            return tracker.update(detector.detect(f))
        speed = benchmark_speed(_step, frames[len(frames)//2], n_runs=5)
    else:
        speed = {"mean_ms": 0, "std_ms": 0, "fps": 0}

    # Write output video
    output_path = os.path.join(VOLUME_PATH, f"output_v1_{video_filename}")
    write_output_video(annotated_frames, output_path, fps)
    volume.commit()

    result = {
        "total_frames": len(frames),
        "total_detections": total_detections,
        "elapsed_seconds": round(elapsed, 2),
        "processing_fps": round(proc_fps, 1),
        "single_frame_benchmark": speed,
        "output_path": output_path,
    }
    print(f"[V1] Done → {result}")
    return result


# ── local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import json
    print("=== Running V1 on Modal (CPU) ===")
    result = run_v1.remote("synthetic_test.mp4")
    print("\n=== V1 Results ===")
    print(json.dumps(result, indent=2))
