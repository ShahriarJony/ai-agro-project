"""Report availability and versions of ML dependencies."""

import importlib

MODULES = [
    "torch",
    "torchvision",
    "albumentations",
    "cv2",
    "PIL",
    "sklearn",
    "matplotlib",
    "seaborn",
    "numpy",
    "pandas",
    "skimage",
    "kaggle",
]

for name in MODULES:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "?")
        print(f"{name:15s} OK       {version}")
    except Exception as exc:
        print(f"{name:15s} MISSING  {type(exc).__name__}: {exc}")
