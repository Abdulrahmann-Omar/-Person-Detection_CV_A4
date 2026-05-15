# Person Detection & Tracking — CLAUDE.md

## Project Overview

Two-version person detection and tracking system for Zewail City CV Assignment 4.

| Version | Approach | Detection | Tracking |
|---------|----------|-----------|----------|
| V1 | From Scratch | MOG2 background subtraction + contour filter | CentroidTracker (IoU + centroid distance, Hungarian assignment) |
| V2 | Transfer Learning | YOLOv8n (ultralytics, COCO-pretrained, class=0) | ByteTrack (built into ultralytics) |

Both versions run on **Modal.com** — V1 on CPU, V2 on T4 GPU.

## Running the Code

### Local (no pip install needed for V1)
```bash
# Run tests (V1 tests pass without ultralytics)
pytest tests/ -v

# Generate report figures (needs matplotlib + opencv)
python report/generate_figures.py

# Compile LaTeX report
bash report/compile.sh
```

### Modal Deployment
```bash
# Authenticate once
modal setup

# Run V1 on CPU
modal run modal_app/modal_v1_scratch.py

# Run V2 on T4 GPU
modal run modal_app/modal_v2_transfer.py

# Run everything at once
modal run modal_app/modal_run_all.py
```

## Project Layout
```
src/
  utils.py           — shared: video I/O, draw_tracks, compute_iou, evaluate_tracking, benchmark_speed
  v1_scratch/
    detector.py      — MOG2Detector (cv2.BackgroundSubtractorMOG2 + morphological cleanup)
    tracker.py       — CentroidTracker (greedy/Hungarian matching on IoU + centroid distance)
  v2_transfer/
    detector.py      — YOLOv8Detector (ultralytics lazy-loaded, person class only)
    tracker.py       — ByteTrackWrapper (ultralytics .track() with bytetrack.yaml)
modal_app/
  common.py          — Modal App, Image (debian-slim + pip packages), Volume, Mount
  modal_v1_scratch.py— @app.function(cpu) + @app.local_entrypoint
  modal_v2_transfer.py— @app.function(gpu="T4") + @app.local_entrypoint
  modal_run_all.py   — Parallel dispatch of V1+V2, figure generation, video download
notebooks/           — Jupyter walkthroughs for V1 and V2
report/              — LaTeX report + auto-generated figures
tests/               — pytest suite (V2 tests skip when ultralytics unavailable)
outputs/             — output videos (gitignored)
```

## Key Design Decisions

- **V1 imports ultralytics lazily** (`_load_model` called on first `detect()`), so importing `src.v1_scratch` never fails even without ultralytics installed.
- **V2 imports ultralytics lazily** similarly — `test_v2.py` catches `ImportError` and skips.
- **Modal Volume** `person-tracking-data` stores input/output videos at `/data`.
- **`src/` is mounted** into Modal containers via `modal.Mount.from_local_dir` so the same code runs locally and in the cloud.
- **Synthetic video generator** (`_generate_synthetic_video`) inside `modal_v1_scratch.py` creates a self-contained test video with 3 moving coloured rectangles — no external dataset needed.

## Python Version
Python 3.10+ required (uses `list[dict]`, `tuple[...]`, `match` syntax not used but 3.10 type hints are).
