#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from config import CLASS_TO_LABEL, LABEL_TO_CLASS  # type: ignore
from utils import ensure_dir, iter_mat_files, load_mat_cjdata, map_label_to_class, save_png, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert external MATLAB MRI slices to PNG files.")
    parser.add_argument("--input-root", type=Path, default=Path("data/external_raw"), help="Folder containing the external_raw/*.mat files")
    parser.add_argument("--output-root", type=Path, default=Path("data/external"), help="Destination folder for converted images")
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"), help="Directory used for stage logs")
    return parser.parse_args()


def extract_image_and_label(mat_path: Path) -> tuple[np.ndarray, int]:
    cjdata = load_mat_cjdata(mat_path)
    if "label" not in cjdata:
        raise KeyError(f"Missing 'label' field in {mat_path}")
    if "image" not in cjdata:
        raise KeyError(f"Missing 'image' field in {mat_path}")

    label_value = np.asarray(cjdata["label"]).reshape(-1)[0]
    label = int(label_value)
    if label not in {1, 2, 3}:
        raise ValueError(f"Unsupported label {label} in {mat_path}; expected 1, 2, or 3.")
    image = np.asarray(cjdata["image"])
    return image, label


def main() -> None:
    args = parse_args()
    logger = setup_logger("convert_external", args.log_dir)
    logger.info("Step 0: begin external MAT conversion")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    destination_dirs = {1: output_root / "meningioma", 2: output_root / "glioma", 3: output_root / "pituitary"}
    for directory in destination_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    mat_files = [p for p in iter_mat_files(input_root) if p.name != "cvind.mat"]
    logger.info("Found %d MAT files under %s", len(mat_files), input_root)

    counts = Counter()
    total_saved = 0
    invalid_files = []

    for mat_path in mat_files:
        try:
            image, label = extract_image_and_label(mat_path)
            class_name = map_label_to_class(label)
            output_dir = destination_dirs[label]
            output_path = output_dir / f"{mat_path.stem}.png"
            save_png(image, output_path)
            counts[class_name] += 1
            total_saved += 1
        except Exception as exc:
            invalid_files.append((mat_path.name, str(exc)))
            logger.warning("Skipped %s (%s)", mat_path.name, exc)

    logger.info("Saved %d images across external dataset", total_saved)
    logger.info("Per-class counts: %s", dict(sorted(counts.items())))

    if invalid_files:
        logger.warning("Invalid MAT files: %s", invalid_files)

    print("\nConverted external dataset summary:")
    for class_name in ["glioma", "meningioma", "pituitary"]:
        print(f"  {class_name}: {counts.get(class_name, 0)}")
    print(f"Total saved: {total_saved}")

    logger.info("Step 0: complete")


if __name__ == "__main__":
    main()
