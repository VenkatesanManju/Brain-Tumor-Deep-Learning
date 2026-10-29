#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score

from config import MODEL_NAMES, add_common_paths, add_common_training_args
from data import validate_all_dataset_mappings
from train import build_model
from utils import setup_logger


def evaluate_loader(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    true_labels = []
    pred_labels = []
    logits_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())
            logits_all.append(logits.cpu())
    if logits_all:
        ensemble_logits = torch.cat(logits_all, dim=0)
    else:
        ensemble_logits = torch.empty(0, 4)
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(f1_score(true_labels, pred_labels, average="macro")),
        "logits": ensemble_logits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the primary-model predictions on the internal test set.")
    add_common_paths(parser)
    add_common_training_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"), help="Directory containing saved checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("evaluate_internal", args.log_dir)
    logger.info("Step 5: internal evaluation")

    _, _, test_loader, _ = validate_all_dataset_mappings(args.data_root, args.batch_size, args.num_workers)
    device = torch.device(args.device)

    results = {}
    logits_by_model = {}
    for model_name in MODEL_NAMES:
        checkpoint_path = args.checkpoint_dir / f"{model_name}.pt"
        if not checkpoint_path.exists():
            logger.warning("Missing checkpoint for %s at %s", model_name, checkpoint_path)
            continue
        model = build_model(model_name, num_classes=4).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        metrics = evaluate_loader(model, test_loader, device)
        results[model_name] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        logits_by_model[model_name] = metrics["logits"]

    if logits_by_model:
        ensemble_probs = torch.stack([
            torch.softmax(logits_by_model[name], dim=1) for name in MODEL_NAMES if name in logits_by_model
        ], dim=0).mean(dim=0)
        ensemble_preds = ensemble_probs.argmax(dim=1)
        true_labels = []
        for _, labels in test_loader:
            true_labels.extend(labels.cpu().tolist())
        results["Ensemble"] = {
            "accuracy": float(accuracy_score(true_labels, ensemble_preds.cpu().tolist())),
            "macro_f1": float(f1_score(true_labels, ensemble_preds.cpu().tolist(), average="macro")),
        }

    output_path = args.output_dir / "internal_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    logger.info("Internal metric summary saved to %s", output_path)


if __name__ == "__main__":
    main()
