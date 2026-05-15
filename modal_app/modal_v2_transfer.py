"""
V2: YOLOv8n + ByteTrack on Modal (GPU T4).
Run with: modal run modal_app/modal_v2_transfer.py
"""
import sys
import os
import time
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from modal_app.common import app, image, volume, VOLUME_PATH, src_mount
from modal_app.modal_v1_scratch import _generate_synthetic_video


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    mounts=[src_mount],
    gpu="T4",
    timeout=600,
)
def run_v2(video_filename: str = "synthetic_test.mp4") -> dict:
    """Run the full V2 (YOLOv8n + ByteTrack) pipeline on Modal with GPU."""
    import sys
    sys.path.insert(0, "/root")

    from src.v2_transfer.tracker import ByteTrackWrapper
    from src.utils import draw_tracks, write_output_video, benchmark_speed

    input_path = os.path.join(VOLUME_PATH, video_filename)

    if not os.path.exists(input_path):
        print(f"[V2] Generating synthetic video → {input_path}")
        _generate_synthetic_video(input_path)
        volume.commit()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"[V2] Loaded {len(frames)} frames at {fps:.1f} FPS")

    tracker = ByteTrackWrapper(device="cuda")
    print("[V2] Model loaded, starting tracking...")

    annotated_frames = []
    total_tracks = 0
    t_start = time.time()

    for i, frame in enumerate(frames):
        tracks = tracker.track_frame(frame)
        total_tracks += len(tracks)
        annotated = draw_tracks(frame, tracks)
        annotated_frames.append(annotated)
        if (i + 1) % 50 == 0:
            print(f"[V2] Frame {i+1}/{len(frames)}")

    elapsed = time.time() - t_start
    proc_fps = len(frames) / elapsed if elapsed > 0 else 0

    # Speed benchmark on single frame
    if frames:
        speed = benchmark_speed(tracker.track_frame, frames[len(frames)//2], n_runs=5)
    else:
        speed = {"mean_ms": 0, "std_ms": 0, "fps": 0}

    output_path = os.path.join(VOLUME_PATH, f"output_v2_{video_filename}")
    write_output_video(annotated_frames, output_path, fps)
    volume.commit()

    result = {
        "total_frames": len(frames),
        "total_tracks_detected": total_tracks,
        "elapsed_seconds": round(elapsed, 2),
        "processing_fps": round(proc_fps, 1),
        "single_frame_benchmark": speed,
        "output_path": output_path,
    }
    print(f"[V2] Done → {result}")
    return result


@app.local_entrypoint()
def main():
    print("=== Running V2 on Modal (GPU T4) ===")
    result = run_v2.remote("synthetic_test.mp4")
    print("\n=== V2 Results ===")
    print(json.dumps(result, indent=2))
