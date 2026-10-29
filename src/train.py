#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torchvision import models

from config import MODEL_NAMES, add_common_paths, add_common_training_args
from data import build_primary_loaders
from utils import ensure_dir, setup_logger


def build_model(model_name: str, num_classes: int = 4):
    if model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    if model_name == "DenseNet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_model(model: nn.Module, loader, device: torch.device):
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
    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune pretrained CNNs on the primary brain tumor dataset.")
    add_common_paths(parser)
    add_common_training_args(parser)
    parser.add_argument("--model", choices=MODEL_NAMES, default="ResNet50")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory where model checkpoints are saved")
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Minimum validation macro-F1 improvement required to reset early stopping")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("train", args.log_dir)
    logger.info("Step 3: training stage for %s", args.model)
    start = time.time()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else args.output_dir / "checkpoints"
    checkpoint_dir = ensure_dir(checkpoint_dir)

    train_loader, val_loader, _ = build_primary_loaders(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = build_model(args.model, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_state = None
    patience_counter = 0
    metrics = {"model": args.model, "epochs": [], "best_val_accuracy": 0.0, "best_val_macro_f1": 0.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, device)
        logger.info("Epoch %d/%d | %s | train_loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f",
                    epoch, args.epochs, args.model, train_loss, val_metrics["accuracy"], val_metrics["macro_f1"])
        metrics["epochs"].append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        })

        improved = val_metrics["macro_f1"] > best_val_macro_f1 + args.min_delta
        if improved:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_val_acc = float(val_metrics["accuracy"])
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered for %s after epoch %d with patience %d", args.model, epoch, args.patience)
            break

    if args.epochs == 0:
        logger.info("Epochs=0 specified. Skipping training and leaving existing checkpoint intact.")
        return

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save(best_state, checkpoint_dir / f"{args.model}.pt")
    metrics["best_val_accuracy"] = float(best_val_acc)
    metrics["best_val_macro_f1"] = float(best_val_macro_f1)
    metrics_path = args.output_dir / f"{args.model}_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Best checkpoint saved to %s", checkpoint_dir / f"{args.model}.pt")
    logger.info("Final validation accuracy for %s: %.4f", args.model, best_val_acc)
    logger.info("Step 3 complete in %.2f seconds", time.time() - start)

    print(f"{args.model} training finished. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
