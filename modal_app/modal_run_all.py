"""
Run everything on Modal: V1 (CPU) + V2 (GPU T4) + figure generation.
Then download output videos from the Modal Volume to local outputs/.

Usage: modal run modal_app/modal_run_all.py
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import modal
from modal_app.common import app, image, volume, VOLUME_PATH, src_mount

# ── Figure generation on Modal ────────────────────────────────────────────────
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "report")

report_mount = modal.Mount.from_local_dir(
    os.path.join(os.path.dirname(__file__), ".."),
    remote_path="/root/project",
)

@app.function(
    image=image,
    mounts=[src_mount, report_mount],
    timeout=300,
)
def generate_figures_remote() -> dict:
    """Generate all report figures inside Modal and return them as bytes."""
    import sys, os
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/project")

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_figures",
        "/root/project/report/generate_figures.py"
    )
    mod = importlib.util.module_from_spec(spec)

    # Override FIGURES_DIR to a writable location inside Modal
    import tempfile
    tmpdir = tempfile.mkdtemp()

    # Patch the module to use tmpdir
    import builtins
    orig_open = builtins.open

    # Execute the script in a subprocess-like way
    exec(open("/root/project/report/generate_figures.py").read(), {
        "__name__": "__main__",
        "__file__": "/root/project/report/generate_figures.py",
    })

    # Read generated files
    figures_path = "/root/project/report/figures"
    result = {}
    if os.path.exists(figures_path):
        for fname in os.listdir(figures_path):
            if fname.endswith(".png"):
                with open(os.path.join(figures_path, fname), "rb") as f:
                    result[fname] = f.read()
    return result


# ── V1 function (imported from modal_v1_scratch) ─────────────────────────────
from modal_app.modal_v1_scratch import run_v1, _generate_synthetic_video
from modal_app.modal_v2_transfer import run_v2


# ── local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import concurrent.futures

    print("=" * 60)
    print("  Person Detection & Tracking — Full Modal Run")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)

    # ── Step 1: Generate figures ─────────────────────────────────────────────
    print("\n[1/3] Generating report figures on Modal...")
    try:
        figures = generate_figures_remote.remote()
        for fname, data in figures.items():
            out_path = os.path.join("report", "figures", fname)
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"  Saved figure: {out_path}")
    except Exception as e:
        print(f"  Figure generation warning: {e}")
        print("  Running locally as fallback...")
        os.chdir(os.path.join(os.path.dirname(__file__), ".."))
        import subprocess
        subprocess.run([sys.executable, "report/generate_figures.py"], check=False)

    # ── Step 2: Run V1 and V2 in parallel on Modal ───────────────────────────
    print("\n[2/3] Launching V1 (CPU) and V2 (GPU T4) on Modal in parallel...")
    t_all = time.time()

    v1_handle = run_v1.spawn("synthetic_test.mp4")
    v2_handle = run_v2.spawn("synthetic_test.mp4")

    print("  Both jobs submitted. Waiting for results...")

    v1_result = v1_handle.get()
    print(f"\n  V1 finished: {json.dumps(v1_result, indent=4)}")

    v2_result = v2_handle.get()
    print(f"\n  V2 finished: {json.dumps(v2_result, indent=4)}")

    total_elapsed = time.time() - t_all
    print(f"\n  Total wall time: {total_elapsed:.1f}s")

    # ── Step 3: Download output videos from the Modal Volume ─────────────────
    print("\n[3/3] Downloading output videos from Modal Volume...")
    try:
        vol = modal.Volume.from_name("person-tracking-data")
        for remote_name, local_name in [
            ("output_v1_synthetic_test.mp4", "outputs/v1_output.mp4"),
            ("output_v2_synthetic_test.mp4", "outputs/v2_output.mp4"),
        ]:
            try:
                data = b"".join(vol.read_file(remote_name))
                with open(local_name, "wb") as f:
                    f.write(data)
                print(f"  Downloaded: {local_name}  ({len(data)//1024} KB)")
            except Exception as e:
                print(f"  Could not download {remote_name}: {e}")
    except Exception as e:
        print(f"  Volume download error: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  V1 (CPU)  — {v1_result.get('processing_fps', 'N/A')} FPS "
          f"| {v1_result.get('total_frames', 0)} frames "
          f"| {v1_result.get('elapsed_seconds', 0)}s")
    print(f"  V2 (GPU)  — {v2_result.get('processing_fps', 'N/A')} FPS "
          f"| {v2_result.get('total_frames', 0)} frames "
          f"| {v2_result.get('elapsed_seconds', 0)}s")
    print("\n  Outputs:")
    print("    outputs/v1_output.mp4")
    print("    outputs/v2_output.mp4")
    print("    report/figures/  (4 PNG files)")
    print("=" * 60)
