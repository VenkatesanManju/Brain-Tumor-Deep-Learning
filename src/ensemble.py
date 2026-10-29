#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import MODEL_NAMES, add_common_paths
from data import validate_all_dataset_mappings
from train import build_model
from utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average model probabilities across the trained CNN ensemble.")
    add_common_paths(parser)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"), help="Directory containing checkpoint files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("ensemble", args.log_dir)
    logger.info("Step 4: soft-voting ensemble")

    checkpoint_dir = args.checkpoint_dir
    missing = [model for model in MODEL_NAMES if not (checkpoint_dir / f"{model}.pt").exists()]
    if missing:
        logger.warning("Missing checkpoints: %s", missing)
        print(f"Missing checkpoints: {missing}")
        return

    _, _, loader, _ = validate_all_dataset_mappings(args.data_root, batch_size=16, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logits_by_model = {}
    for name in MODEL_NAMES:
        checkpoint = checkpoint_dir / f"{name}.pt"
        model = build_model(name, num_classes=4).to(device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        model.eval()
        logits = []
        with torch.no_grad():
            for images, _ in loader:
                logits.append(model(images.to(device)).cpu())
        logits_by_model[name] = torch.cat(logits, dim=0)

    probs = [torch.softmax(logits_by_model[name], dim=1) for name in MODEL_NAMES]
    mean_probs = torch.stack(probs, dim=0).mean(dim=0)
    preds = mean_probs.argmax(dim=1)
    print(f"Ensemble predictions generated: {len(preds)}")
    print("Sample predictions:", preds[:10].tolist())
    logger.info("Ensemble inference completed for %d samples", len(preds))


if __name__ == "__main__":
    main()
