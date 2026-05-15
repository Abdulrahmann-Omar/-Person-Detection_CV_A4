import modal
import os

# Path to local src/ directory so Modal can copy it into the container
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "numpy",
        "opencv-python-headless",
        "ultralytics",
        "torch",
        "torchvision",
        "matplotlib",
        "pandas",
        "seaborn",
        "scipy",
        "pillow",
    ])
    .copy_local_dir(SRC_DIR, "/root/src")
    .run_commands("echo 'Image ready'")
)

# Mount src code into /root/src inside the container
src_mount = modal.Mount.from_local_dir(
    SRC_DIR,
    remote_path="/root/src",
)

volume = modal.Volume.from_name("person-tracking-data", create_if_missing=True)
VOLUME_PATH = "/data"

app = modal.App("person-tracking")
