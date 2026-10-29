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
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "macro_f1": float(f1_score(true_labels, pred_labels, labels=[1, 2, 3], average="macro", zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model performance on the external dataset after the internal training phase.")
    add_common_paths(parser)
    add_common_training_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory containing saved checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("evaluate_external", args.log_dir)
    logger.info("Step 6: external evaluation")

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else args.output_dir / "checkpoints"
    _, _, _, loader = validate_all_dataset_mappings(args.data_root, args.batch_size, args.num_workers)
    device = torch.device(args.device)

    results = {}
    logits_by_model = {}
    for model_name in MODEL_NAMES:
        checkpoint_path = checkpoint_dir / f"{model_name}.pt"
        if not checkpoint_path.exists():
            logger.warning("Missing checkpoint for %s at %s", model_name, checkpoint_path)
            continue
        model = build_model(model_name, num_classes=4).to(device)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        metrics = evaluate_loader(model, loader, device)
        results[model_name] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}

    if not results:
        logger.warning("No model checkpoints were available for external evaluation.")
    else:
        logits_by_model = {}
        for model_name in MODEL_NAMES:
            checkpoint_path = checkpoint_dir / f"{model_name}.pt"
            if not checkpoint_path.exists():
                continue
            model = build_model(model_name, num_classes=4).to(device)
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            logits = []
            labels = []
            with torch.no_grad():
                for images, batch_labels in loader:
                    images = images.to(device)
                    logits.append(model(images).cpu())
                    labels.extend(batch_labels.tolist())
            logits_by_model[model_name] = torch.cat(logits, dim=0)
        if logits_by_model:
            ensemble_probs = torch.stack([
                torch.softmax(logits_by_model[name], dim=1) for name in MODEL_NAMES if name in logits_by_model
            ], dim=0).mean(dim=0)
            ensemble_preds = ensemble_probs.argmax(dim=1)
            ensemble_labels = torch.tensor(labels)
            results["Ensemble"] = {
                "accuracy": float(accuracy_score(ensemble_labels.tolist(), ensemble_preds.tolist())),
                "macro_f1": float(f1_score(ensemble_labels.tolist(), ensemble_preds.tolist(), labels=[1, 2, 3], average="macro", zero_division=0)),
            }

    output_path = args.output_dir / "external_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    logger.info("External metric summary saved to %s", output_path)


if __name__ == "__main__":
    main()
