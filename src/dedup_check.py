#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from imagehash import phash
from PIL import Image

from utils import ensure_dir, setup_logger


def list_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in valid_exts)


def image_hash(path: Path):
    with Image.open(path) as image:
        return phash(image, hash_size=8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag near-duplicate MRI images between external and primary datasets.")
    parser.add_argument("--external-root", type=Path, default=Path("data/external"), help="Folder with external converted images")
    parser.add_argument("--primary-root", type=Path, action="append", help="Primary image root; repeat to include additional training roots")
    parser.add_argument("--threshold", type=int, default=5, help="Maximum hash distance to classify as a near-duplicate")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory to save duplicate reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("dedup_check", args.output_dir / "logs")
    logger.info("Step 1: begin duplicate detection")

    external_files = list_image_files(args.external_root)
    primary_roots = args.primary_root or [Path("data/primary")]
    primary_files = [path for root in primary_roots for path in list_image_files(root)]
    logger.info("Found %d external images and %d primary images.", len(external_files), len(primary_files))

    if not external_files or not primary_files:
        logger.warning("No images were found; writing an empty duplicate report.")
        report = {"threshold": args.threshold, "duplicates": []}
        out_dir = ensure_dir(args.output_dir)
        with (out_dir / "duplicates.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["external_image", "primary_image", "distance"])
        with (out_dir / "duplicates.json").open("w", encoding="utf-8") as json_file:
            json.dump(report, json_file, indent=2)
        logger.info("Step 1: complete (empty report)")
        return

    ext_hashes = {path: image_hash(path) for path in external_files}
    primary_hashes = {path: image_hash(path) for path in primary_files}

    duplicates: list[dict[str, str | int]] = []
    for ext_path, ext_hash in ext_hashes.items():
        for primary_path, primary_hash in primary_hashes.items():
            distance = int(ext_hash - primary_hash)
            if distance < args.threshold:
                duplicates.append(
                    {
                        "external_image": str(ext_path),
                        "primary_image": str(primary_path),
                        "distance": distance,
                    }
                )

    duplicates_sorted = sorted(duplicates, key=lambda item: (item["distance"], str(item["external_image"]), str(item["primary_image"])))
    out_dir = ensure_dir(args.output_dir)

    with (out_dir / "duplicates.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["external_image", "primary_image", "distance"])
        for item in duplicates_sorted:
            writer.writerow([item["external_image"], item["primary_image"], item["distance"]])

    with (out_dir / "duplicates.json").open("w", encoding="utf-8") as json_file:
        json.dump({"threshold": args.threshold, "duplicate_count": len(duplicates_sorted), "duplicates": duplicates_sorted}, json_file, indent=2)

    logger.info("Found %d near-duplicate pairs under threshold %d.", len(duplicates_sorted), args.threshold)
    if duplicates_sorted:
        for row in duplicates_sorted[:10]:
            logger.warning("Duplicate candidate: %s vs %s (distance=%s)", row["external_image"], row["primary_image"], row["distance"])
    logger.info("Step 1: complete")

    print(f"Near duplicates under threshold {args.threshold}: {len(duplicates_sorted)}")
    if duplicates_sorted:
        print("First few matches:")
        for row in duplicates_sorted[:10]:
            print(f"  {row['external_image']} <-> {row['primary_image']}  distance={row['distance']}")


if __name__ == "__main__":
    main()
