#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CLASS_TO_ID, PRIMARY_CLASS_NAMES, add_common_paths, add_common_training_args
from utils import ensure_dir, setup_logger


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str], transform: transforms.Compose | None = None):
        self.root = Path(root)
        self.class_names = list(class_names)
        self.class_to_idx = dict(CLASS_TO_ID)
        self.transform = transform or transforms.Compose([])
        self.samples = self._collect_samples()

    def _collect_samples(self):
        samples = []
        for class_name in self.class_names:
            if class_name not in CLASS_TO_ID:
                raise KeyError(f"Unsupported class name '{class_name}' for canonical mapping {CLASS_TO_ID}")
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    samples.append((image_path, CLASS_TO_ID[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def assert_dataset_class_mapping(dataset_name: str, dataset: ImageFolderDataset, expected_classes: list[str]):
    if dataset.class_to_idx != CLASS_TO_ID:
        raise AssertionError(f"{dataset_name} class_to_idx drifted: {dataset.class_to_idx} != {CLASS_TO_ID}")
    expected_ids = [CLASS_TO_ID[class_name] for class_name in expected_classes if class_name in CLASS_TO_ID]
    seen_ids = sorted({int(label) for _, label in dataset.samples})
    print(f"{dataset_name} mapping: expected={expected_classes} -> {expected_ids}; seen={seen_ids}")
    if seen_ids != sorted(set(expected_ids)):
        raise AssertionError(
            f"{dataset_name} label drifted: expected ids {sorted(set(expected_ids))}, but found {seen_ids}. "
            f"Canonical mapping: {CLASS_TO_ID}"
        )
    return True


def build_image_transform(train: bool = False) -> transforms.Compose:
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_mean, std=image_std),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])


def collect_samples(dataset_roots: list[Path] | Path, class_names: list[str]) -> list[tuple[str, str]]:
    if isinstance(dataset_roots, Path):
        dataset_roots = [dataset_roots]

    samples = []
    for dataset_root in dataset_roots:
        for class_name in class_names:
            class_dir = dataset_root / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    samples.append((str(image_path), class_name))
    return samples


def build_primary_loaders(data_root: Path, batch_size: int = 32, num_workers: int = 0):
    train_root = data_root / "primary" / "Training"
    archive_root = data_root / "archive" / "Training"
    test_root = data_root / "primary" / "Testing"

    training_roots = [train_root, archive_root]
    train_samples = collect_samples(training_roots, PRIMARY_CLASS_NAMES)
    labels = [CLASS_TO_ID[class_name] for _, class_name in train_samples]
    train_samples, val_samples = train_test_split(train_samples, test_size=0.15, stratify=labels, random_state=42)

    train_dataset = ImageFolderDataset(train_root, PRIMARY_CLASS_NAMES, transform=build_image_transform(train=True))
    val_dataset = ImageFolderDataset(train_root, PRIMARY_CLASS_NAMES, transform=build_image_transform(train=False))
    test_dataset = ImageFolderDataset(test_root, PRIMARY_CLASS_NAMES, transform=build_image_transform(train=False))

    train_dataset.samples = [(Path(image_path), CLASS_TO_ID[class_name]) for image_path, class_name in train_samples]
    val_dataset.samples = [(Path(image_path), CLASS_TO_ID[class_name]) for image_path, class_name in val_samples]
    assert_dataset_class_mapping("train", train_dataset, PRIMARY_CLASS_NAMES)
    assert_dataset_class_mapping("val", val_dataset, PRIMARY_CLASS_NAMES)
    assert_dataset_class_mapping("internal_test", test_dataset, PRIMARY_CLASS_NAMES)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, val_loader, test_loader


def build_external_loader(data_root: Path, batch_size: int = 32, num_workers: int = 0):
    external_root = data_root / "external"
    dataset = ImageFolderDataset(external_root, ["glioma", "meningioma", "pituitary"], transform=build_image_transform(train=False))
    assert_dataset_class_mapping("external_test", dataset, ["glioma", "meningioma", "pituitary"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return loader


def validate_all_dataset_mappings(data_root: Path, batch_size: int = 32, num_workers: int = 0):
    train_loader, val_loader, test_loader = build_primary_loaders(data_root, batch_size, num_workers)
    external_loader = build_external_loader(data_root, batch_size, num_workers)
    datasets = {
        "train": train_loader.dataset,
        "val": val_loader.dataset,
        "internal_test": test_loader.dataset,
        "external_test": external_loader.dataset,
    }
    for dataset_name, dataset in datasets.items():
        if dataset.class_to_idx != CLASS_TO_ID:
            raise AssertionError(f"{dataset_name} class_to_idx differs from canonical mapping: {dataset.class_to_idx}")
    print(f"Verified identical class_to_idx across train/val/internal_test/external_test: {CLASS_TO_ID}")
    return train_loader, val_loader, test_loader, external_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PyTorch dataloaders for the primary and external MRI datasets.")
    add_common_paths(parser)
    add_common_training_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("data", args.log_dir)
    logger.info("Step 2: create dataset loaders")
    train_loader, val_loader, test_loader = build_primary_loaders(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    external_loader = build_external_loader(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    logger.info("Primary train/val/test sizes: %d / %d / %d", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    logger.info("External dataset size: %d", len(external_loader.dataset))
    print("Primary train/val/test sizes:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print("External dataset size:", len(external_loader.dataset))


if __name__ == "__main__":
    main()
