from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in [str(PROJECT_ROOT), str(SRC_ROOT)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)

from config import CLASS_TO_ID
from data import ImageFolderDataset, build_image_transform


def _write_rgb(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color=(12, 34, 56))
    image.save(path)


def test_canonical_class_mapping_is_used_for_primary_and_external_datasets() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for class_name in ["notumor", "glioma", "meningioma", "pituitary"]:
            class_dir = root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            _write_rgb(class_dir / "sample.png")

        primary_dataset = ImageFolderDataset(
            root,
            ["notumor", "glioma", "meningioma", "pituitary"],
            transform=build_image_transform(False),
        )
        primary_seen = sorted({int(label) for _, label in primary_dataset.samples})
        assert primary_dataset.class_to_idx == CLASS_TO_ID
        assert primary_seen == sorted(CLASS_TO_ID.values()), (
            f"Primary dataset labels drifted: expected {sorted(CLASS_TO_ID.values())}, got {primary_seen}"
        )

        external_root = root / "external"
        for class_name in ["glioma", "meningioma", "pituitary"]:
            class_dir = external_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            _write_rgb(class_dir / "sample.png")

        external_dataset = ImageFolderDataset(
            external_root,
            ["glioma", "meningioma", "pituitary"],
            transform=build_image_transform(False),
        )
        external_seen = sorted({int(label) for _, label in external_dataset.samples})
        assert external_dataset.class_to_idx == CLASS_TO_ID
        assert external_seen == [CLASS_TO_ID["glioma"], CLASS_TO_ID["meningioma"], CLASS_TO_ID["pituitary"]], (
            f"External dataset labels must map to the canonical four-class space: {CLASS_TO_ID}"
        )


if __name__ == "__main__":
    test_canonical_class_mapping_is_used_for_primary_and_external_datasets()
    print("Dataset mapping test passed.")
