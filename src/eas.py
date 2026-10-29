#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import MODEL_NAMES, add_common_paths, add_common_training_args
from data import build_external_loader, build_primary_loaders
from gradcam import GradCAM, get_target_layer
from train import build_model
from utils import ensure_dir, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Explainability-Agreement Score (EAS) and agreement-weighted confidence.")
    add_common_paths(parser)
    add_common_training_args(parser)
    parser.add_argument("--alpha", type=float, default=0.5, help="Mixture weight used in Confidence = p_ensemble_max * (alpha * P_agree + (1-alpha) * EAS)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory containing saved checkpoints")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of test samples to analyze for EAS (default: all samples)")
    parser.add_argument("--dry-run", action="store_true", help="Skip metric computation and only validate script wiring")
    return parser.parse_args()


def pairwise_iou(map_a: np.ndarray, map_b: np.ndarray) -> float:
    threshold_a = map_a >= map_a.mean()
    threshold_b = map_b >= map_b.mean()
    union = np.logical_or(threshold_a, threshold_b)
    intersection = np.logical_and(threshold_a, threshold_b)
    if union.sum() == 0:
        return 0.0
    return float(intersection.sum() / union.sum())


def cosine_similarity(map_a: np.ndarray, map_b: np.ndarray) -> float:
    a = map_a.reshape(-1)
    b = map_b.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def agreement_weighted_confidence(probabilities: np.ndarray, eas_score: float, alpha: float = 0.5) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    p_ensemble_max = float(np.max(probs))
    # Uniform variance for max disagreement among 4 classes
    uniform_var = np.var(np.ones(len(probs)) * (1.0 / len(probs)))
    p_agree = max(0.0, 1.0 - np.var(probs) / max(uniform_var, 1e-12))
    return float(p_ensemble_max * (alpha * p_agree + (1.0 - alpha) * eas_score))


def main() -> None:
    args = parse_args()
    logger = setup_logger("eas", args.log_dir)
    logger.info("Step 9: EAS and confidence aggregation")
    if args.dry_run:
        logger.info("Dry-run mode enabled: EAS logic scaffold ready.")
        print("[dry-run] EAS scaffolding is ready.")
        return

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else args.output_dir / "checkpoints"
    device = torch.device(args.device)

    _, _, test_loader = build_primary_loaders(args.data_root, batch_size=1, num_workers=0)
    ext_loader = build_external_loader(args.data_root, batch_size=1, num_workers=0)

    models_dict = {}
    cam_extractors = {}
    for model_name in MODEL_NAMES:
        ckpt = checkpoint_dir / f"{model_name}.pt"
        if not ckpt.exists():
            continue
        model = build_model(model_name, num_classes=4).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        target_layer = get_target_layer(model_name, model)
        models_dict[model_name] = model
        cam_extractors[model_name] = GradCAM(model, target_layer)

    if not models_dict:
        logger.warning("No model checkpoints available for EAS computation.")
        return

    records = []
    datasets_to_process = [("internal", test_loader), ("external", ext_loader)]

    for ds_type, loader in datasets_to_process:
        count = 0
        for i, (images, labels) in enumerate(loader):
            if count >= args.num_samples:
                break
            images = images.to(device)
            target_class = labels.item()

            probs_list = []
            maps = []
            for name in MODEL_NAMES:
                if name not in models_dict:
                    continue
                model = models_dict[name]
                extractor = cam_extractors[name]

                with torch.no_grad():
                    logits = model(images)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    probs_list.append(probs)

                cam = extractor.generate(images, target_class=target_class)
                maps.append(cam)

            if len(maps) >= 2:
                ious = []
                cossims = []
                for idx1 in range(len(maps)):
                    for idx2 in range(idx1 + 1, len(maps)):
                        ious.append(pairwise_iou(maps[idx1], maps[idx2]))
                        cossims.append(cosine_similarity(maps[idx1], maps[idx2]))

                mean_iou = float(np.mean(ious))
                mean_cossim = float(np.mean(cossims))
                eas_score = 0.5 * mean_iou + 0.5 * mean_cossim

                ensemble_probs = np.mean(probs_list, axis=0)
                conf = agreement_weighted_confidence(ensemble_probs, eas_score, alpha=args.alpha)
                pred_label = int(np.argmax(ensemble_probs))

                records.append({
                    "dataset_type": ds_type,
                    "sample_index": i,
                    "true_label": target_class,
                    "pred_label": pred_label,
                    "is_correct": int(pred_label == target_class),
                    "mean_iou": mean_iou,
                    "mean_cossim": mean_cossim,
                    "eas_score": eas_score,
                    "agreement_confidence": conf,
                })
            count += 1

    for extractor in cam_extractors.values():
        extractor.remove_hooks()

    df = pd.DataFrame(records)
    output_path = args.output_dir / "eas_scores.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Computed EAS and confidence scores for %d samples saved to %s", len(df), output_path)
    print(f"EAS evaluation completed. Output saved to {output_path}")
    if not df.empty:
        print(df.groupby("dataset_type")[["eas_score", "agreement_confidence", "is_correct"]].mean())


if __name__ == "__main__":
    main()
