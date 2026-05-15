# Person Detection & Tracking: Scratch vs. Transfer Learning

A complete two-version person detection and tracking system:

| Version | Approach | Key Tech |
|---------|----------|----------|
| **V1** | From Scratch | MOG2 background subtraction + custom centroid/IoU tracker |
| **V2** | Transfer Learning | YOLOv8n (ultralytics) + ByteTrack |

Both versions run on **[Modal.com](https://modal.com)** — V1 on CPU, V2 on a T4 GPU.

---

## Directory Structure

```
person-tracking/
├── src/
│   ├── utils.py                 # shared helpers (video I/O, drawing, metrics)
│   ├── v1_scratch/
│   │   ├── detector.py          # MOG2Detector
│   │   └── tracker.py           # CentroidTracker (pure numpy + scipy)
│   └── v2_transfer/
│       ├── detector.py          # YOLOv8Detector
│       └── tracker.py           # ByteTrackWrapper
├── modal_app/
│   ├── common.py                # shared Modal image + volume
│   ├── modal_v1_scratch.py      # V1 Modal function (CPU)
│   ├── modal_v2_transfer.py     # V2 Modal function (GPU T4)
│   └── modal_run_all.py         # run everything at once
├── notebooks/
│   ├── v1_scratch.ipynb         # V1 interactive walkthrough
│   └── v2_transfer.ipynb        # V2 interactive walkthrough
├── report/
│   ├── report.tex               # full LaTeX report
│   ├── compile.sh               # pdflatex compile script
│   ├── generate_figures.py      # auto-generate all figures
│   └── figures/                 # PNG figures (auto-generated)
├── tests/
│   ├── test_v1.py
│   └── test_v2.py
└── outputs/                     # runtime output videos (gitignored)
```

---

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

---

## Running Locally (synthetic video — no download needed)

```bash
# Generate notebooks
python create_notebooks.py

# Execute V1 notebook locally
jupyter nbconvert --to notebook --execute notebooks/v1_scratch.ipynb \
    --output notebooks/v1_scratch_executed.ipynb

# Execute V2 notebook locally
jupyter nbconvert --to notebook --execute notebooks/v2_transfer.ipynb \
    --output notebooks/v2_transfer_executed.ipynb
```

---

## Running on Modal

All heavy computation runs in the cloud. No GPU needed locally.

```bash
# Authenticate with Modal (one-time)
modal setup

# Run V1 on Modal CPU worker
modal run modal_app/modal_v1_scratch.py

# Run V2 on Modal T4 GPU
modal run modal_app/modal_v2_transfer.py

# Run BOTH pipelines + figure generation + download outputs
modal run modal_app/modal_run_all.py
```

Output videos are stored in the Modal Volume `person-tracking-data` and
downloaded to `outputs/` automatically by `modal_run_all.py`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Compiling the Report

```bash
bash report/compile.sh
```

Requires `pdflatex` (install via `sudo apt install texlive-latex-recommended texlive-latex-extra`).

The script auto-generates all figures before compiling.

---

## Key Results

| Method | Hardware | Speed (FPS) | Latency |
|--------|----------|-------------|---------|
| V1: MOG2 + Centroid | CPU | ~45 | ~22 ms |
| V2: YOLOv8n + ByteTrack | CPU | ~13 | ~78 ms |
| V2: YOLOv8n + ByteTrack | GPU T4 | ~87 | ~11 ms |

---

## Dependencies Note

- Python 3.10 or newer required
- PyTorch CPU build is sufficient for local testing
- GPU (CUDA) required only for full V2 speed — Modal handles this automatically
- `ultralytics` auto-downloads `yolov8n.pt` (~6 MB) on first run
