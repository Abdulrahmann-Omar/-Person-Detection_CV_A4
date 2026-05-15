"""
Modal app that:
  1. Executes v1_scratch.ipynb  (CPU)
  2. Executes v2_transfer.ipynb (T4 GPU)
  3. Compiles report/report.tex → report.pdf
  4. Returns all artefacts to the local machine
"""
import modal
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Image — bake the project src + notebooks + report into the image ──────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "texlive-latex-base", "texlive-latex-extra",
        "texlive-fonts-recommended", "texlive-fonts-extra",
        "texlive-science", "cm-super", "dvipng",
        "libgl1-mesa-glx", "libglib2.0-0",
        "ffmpeg",
    ])
    .pip_install([
        "numpy", "opencv-python-headless", "Pillow",
        "matplotlib", "seaborn", "pandas", "scipy",
        "ultralytics", "torch", "torchvision",
        "jupyter", "nbconvert", "ipykernel", "nbformat",
    ])
    # Add the project tree into the image at /project
    .add_local_dir(PROJECT_DIR, remote_path="/project")
)

app = modal.App("person-tracking-execute")


# ─────────────────────────────────────────────────────────────────────────────
# Helper that runs inside Modal
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    timeout=2400,
    cpu=4,
)
def execute_all() -> dict:
    import subprocess, os, shutil, sys

    # Copy project to a writable workspace (mount is read-only)
    src = "/project"
    dst = "/workspace"
    shutil.copytree(src, dst)
    os.chdir(dst)
    sys.path.insert(0, dst)

    # Install ipykernel so nbconvert can find a kernel
    subprocess.run(
        ["python", "-m", "ipykernel", "install", "--user", "--name", "python3"],
        capture_output=True,
    )

    results: dict = {}

    # ── 1. Execute V1 notebook (CPU path) ────────────────────────────────────
    print("=" * 60)
    print("Executing V1 notebook …")
    r1 = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output", f"{dst}/notebooks/v1_scratch.ipynb",
            f"{dst}/notebooks/v1_scratch.ipynb",
        ],
        capture_output=True, text=True, cwd=dst,
    )
    print("V1 stdout:", r1.stdout[-800:])
    if r1.returncode != 0:
        print("V1 stderr:", r1.stderr[-1500:])
    results["v1_ok"] = r1.returncode == 0

    # ── 2. Execute V2 notebook (GPU path) ────────────────────────────────────
    print("=" * 60)
    print("Executing V2 notebook …")
    r2 = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=900",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output", f"{dst}/notebooks/v2_transfer.ipynb",
            f"{dst}/notebooks/v2_transfer.ipynb",
        ],
        capture_output=True, text=True, cwd=dst,
    )
    print("V2 stdout:", r2.stdout[-800:])
    if r2.returncode != 0:
        print("V2 stderr:", r2.stderr[-1500:])
    results["v2_ok"] = r2.returncode == 0

    # ── 3. Generate figures + compile LaTeX ──────────────────────────────────
    print("=" * 60)
    print("Generating figures …")
    subprocess.run(["python", "report/generate_figures.py"], cwd=dst, capture_output=True)

    print("Compiling LaTeX (pass 1) …")
    def pdflatex():
        return subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", f"{dst}/report", f"{dst}/report/report.tex"],
            capture_output=True, text=True, cwd=f"{dst}/report",
        )
    r_tex1 = pdflatex()
    r_tex2 = pdflatex()   # second pass for cross-refs / TOC
    print("LaTeX stdout:", r_tex2.stdout[-500:])
    if r_tex2.returncode != 0:
        print("LaTeX stderr:", r_tex2.stderr[-800:])
    results["tex_ok"] = r_tex2.returncode == 0

    # ── 4. Collect artefacts ─────────────────────────────────────────────────
    def read_bytes(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    for name in ["v1_scratch", "v2_transfer"]:
        data = read_bytes(f"{dst}/notebooks/{name}.ipynb")
        results[f"nb_{name}"] = data
        if data:
            print(f"✓  {name}.ipynb  ({len(data)//1024} KB)")

    pdf = read_bytes(f"{dst}/report/report.pdf")
    results["pdf"] = pdf
    if pdf:
        print(f"✓  report.pdf  ({len(pdf)//1024} KB)")

    for vname in ["v1_output.mp4", "v2_output.mp4"]:
        data = read_bytes(f"{dst}/outputs/{vname}")
        results[f"video_{vname}"] = data
        if data:
            print(f"✓  {vname}  ({len(data)//1024} KB)")

    for tag in ["v1", "v2"]:
        for suffix in ["before.gif", "after.gif"]:
            key = f"{tag}_{suffix}"
            data = read_bytes(f"{dst}/outputs/{key}")
            results[f"gif_{key}"] = data
            if data:
                print(f"✓  {key}  ({len(data)//1024} KB)")

    return results


# ── Local entrypoint ─────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import os

    base = PROJECT_DIR

    print("Dispatching to Modal …")
    results = execute_all.remote()

    # ── Save notebooks ────────────────────────────────────────────────────────
    for name in ["v1_scratch", "v2_transfer"]:
        data = results.get(f"nb_{name}")
        if data:
            path = os.path.join(base, "notebooks", f"{name}.ipynb")
            with open(path, "wb") as f:
                f.write(data)
            print(f"✓  Saved {name}.ipynb")
        else:
            print(f"✗  {name}.ipynb — execution failed (check Modal logs)")

    # ── Save PDF ──────────────────────────────────────────────────────────────
    pdf = results.get("pdf")
    if pdf:
        path = os.path.join(base, "report", "report.pdf")
        with open(path, "wb") as f:
            f.write(pdf)
        print(f"✓  Saved report.pdf  ({len(pdf)//1024} KB)")
    else:
        print("✗  report.pdf — LaTeX failed (check Modal logs)")

    # ── Save videos ───────────────────────────────────────────────────────────
    os.makedirs(os.path.join(base, "outputs"), exist_ok=True)
    for vname in ["v1_output.mp4", "v2_output.mp4"]:
        data = results.get(f"video_{vname}")
        if data:
            path = os.path.join(base, "outputs", vname)
            with open(path, "wb") as f:
                f.write(data)
            print(f"✓  Saved {vname}  ({len(data)//1024} KB)")

    # ── Save GIFs ─────────────────────────────────────────────────────────────
    for tag in ["v1", "v2"]:
        for suffix in ["before.gif", "after.gif"]:
            key = f"{tag}_{suffix}"
            data = results.get(f"gif_{key}")
            if data:
                path = os.path.join(base, "outputs", key)
                with open(path, "wb") as f:
                    f.write(data)
                print(f"✓  Saved {key}  ({len(data)//1024} KB)")

    print("\nAll done.")
    ok = results.get("v1_ok"), results.get("v2_ok"), results.get("tex_ok")
    print(f"  V1 notebook : {'OK' if ok[0] else 'FAILED'}")
    print(f"  V2 notebook : {'OK' if ok[1] else 'FAILED'}")
    print(f"  LaTeX PDF   : {'OK' if ok[2] else 'FAILED'}")
