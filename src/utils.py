from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name: str, log_dir: str | Path) -> logging.Logger:
    log_dir = ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def normalize_image_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3 and img.shape[-1] in (1, 3):
        img = img[..., 0] if img.shape[-1] == 1 else img
    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if max_val > min_val:
        img = 255.0 * (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img, dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def save_png(array: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = normalize_image_uint8(array)
    Image.fromarray(img).save(output_path, format="PNG")


def iter_mat_files(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.mat") if p.is_file())


def load_mat_cjdata(mat_path: str | Path) -> dict[str, Any]:
    """Load MATLAB files in either v5 or v7.3 format and return the cjdata dictionary."""
    mat_path = Path(mat_path)
    try:
        import scipy.io as sio

        data = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        if "cjdata" in data:
            cjdata = data["cjdata"]
            if hasattr(cjdata, "__dict__"):
                return cjdata.__dict__
            return {k: getattr(cjdata, k) for k in dir(cjdata) if not k.startswith("__") and not callable(getattr(cjdata, k))}
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not k.startswith("__")}
    except Exception:
        pass

    try:
        with h5py.File(str(mat_path), "r") as handle:
            key = "cjdata" if "cjdata" in handle else next(iter(handle.keys()))
            group = handle[key]
            return {name: group[name][()] for name in group.keys()}
    except Exception as exc:  # pragma: no cover - debugging aid
        raise RuntimeError(f"Unable to read MAT file: {mat_path}") from exc


def map_label_to_class(label: int | np.integer) -> str:
    label = int(label)
    if label == 1:
        return "meningioma"
    if label == 2:
        return "glioma"
    if label == 3:
        return "pituitary"
    raise ValueError(f"Unsupported MRI class label {label}; expected 1, 2, or 3.")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
