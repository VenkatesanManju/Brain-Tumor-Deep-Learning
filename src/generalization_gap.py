#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import add_common_paths
from utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute the internal-to-external generalization gap across models and the ensemble.")
    add_common_paths(parser)
    parser.add_argument("--internal-json", type=Path, default=Path("outputs/internal_metrics.json"), help="Path to the internal evaluation summary JSON")
    parser.add_argument("--external-json", type=Path, default=Path("outputs/external_metrics.json"), help="Path to the external evaluation summary JSON")
    return parser.parse_args()


def compute_gap(internal: float, external: float) -> float:
    return internal - external


def main() -> None:
    args = parse_args()
    logger = setup_logger("generalization_gap", args.log_dir)
    logger.info("Step 7: generalization gap")

    rows = []

    if args.internal_json.exists() and args.external_json.exists():
        with args.internal_json.open("r", encoding="utf-8") as handle:
            internal = json.load(handle)
        with args.external_json.open("r", encoding="utf-8") as handle:
            external = json.load(handle)

        for model_name in sorted(set(internal) | set(external)):
            if model_name not in internal or model_name not in external:
                continue
            acc_gap = compute_gap(internal[model_name].get("accuracy", 0.0), external[model_name].get("accuracy", 0.0))
            f1_gap = compute_gap(internal[model_name].get("macro_f1", 0.0), external[model_name].get("macro_f1", 0.0))
            rows.append({
                "model": model_name,
                "accuracy_gap": acc_gap,
                "macro_f1_gap": f1_gap,
            })
    else:
        rows.append({"model": "ResNet50", "accuracy_gap": 0.0, "macro_f1_gap": 0.0})
        rows.append({"model": "EfficientNetB0", "accuracy_gap": 0.0, "macro_f1_gap": 0.0})
        rows.append({"model": "DenseNet121", "accuracy_gap": 0.0, "macro_f1_gap": 0.0})
        rows.append({"model": "Ensemble", "accuracy_gap": 0.0, "macro_f1_gap": 0.0})
        logger.warning("Expected metric JSON files were not found; writing placeholder gap values.")

    output = pd.DataFrame(rows)
    output_path = args.output_dir / "generalization_gap.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    logger.info("Generalization gap summary saved to %s", output_path)


if __name__ == "__main__":
    main()
