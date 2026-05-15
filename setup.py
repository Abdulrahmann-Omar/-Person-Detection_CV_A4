from setuptools import setup, find_packages

setup(
    name="person-tracking",
    version="1.0.0",
    description="Person detection and tracking: scratch vs transfer learning",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "ultralytics>=8.2",
        "torch>=2.1",
        "torchvision>=0.16",
        "modal>=0.62",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "pandas>=2.0",
        "scipy>=1.11",
    ],
)
