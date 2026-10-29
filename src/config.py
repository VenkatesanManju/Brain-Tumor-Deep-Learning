from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "logs"

PRIMARY_CLASS_NAMES = ["notumor", "glioma", "meningioma", "pituitary"]
EXTERNAL_CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
CLASS_TO_ID = {"notumor": 0, "glioma": 1, "meningioma": 2, "pituitary": 3}
CLASS_TO_LABEL = CLASS_TO_ID.copy()
LABEL_TO_CLASS = {label: class_name for class_name, label in CLASS_TO_ID.items()}
MODEL_NAMES = ["ResNet50", "EfficientNetB0", "DenseNet121"]


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Project root directory")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root folder containing primary/, external/, external_raw/")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for metrics, plots, checkpoints, and logs")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for text logs")


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu", help="Torch device")


def add_eas_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--alpha", type=float, default=0.5, help="Agreement weight for confidence fusion")
