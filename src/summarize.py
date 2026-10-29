#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import MODEL_NAMES, add_common_paths
from utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the final summary table across all evaluated models.")
    add_common_paths(parser)
    parser.add_argument("--metrics-json", type=Path, default=Path("outputs/final_metrics.json"), help="Path to final JSON metrics if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("summarize", args.log_dir)
    logger.info("Step 11: final summary")

    internal_metrics_path = args.output_dir / "internal_metrics.json"
    external_metrics_path = args.output_dir / "external_metrics.json"
    gap_path = args.output_dir / "generalization_gap.csv"

    all_models = list(MODEL_NAMES) + ["Ensemble"]

    internal_metrics = {}
    external_metrics = {}
    gap_lookup = {}

    if internal_metrics_path.exists():
        with internal_metrics_path.open("r", encoding="utf-8") as handle:
            internal_metrics = json.load(handle)
    if external_metrics_path.exists():
        with external_metrics_path.open("r", encoding="utf-8") as handle:
            external_metrics = json.load(handle)
    gap_lookup = {}
    if gap_path.exists():
        gap_df = pd.read_csv(gap_path)
        for _, row in gap_df.iterrows():
            gap_lookup[row["model"]] = float(row.get("accuracy_gap", 0.0))

    missing = [name for name in all_models if name not in internal_metrics or name not in external_metrics]
    if missing:
        raise RuntimeError(f"Missing real evaluation metrics for: {', '.join(missing)}")

    rows = []
    for model_name in all_models:
        internal = internal_metrics[model_name]
        external = external_metrics[model_name]
        internal_acc = float(internal["accuracy"])
        external_acc = float(external["accuracy"])
        generalization_gap = float(gap_lookup.get(model_name, internal_acc - external_acc))
        rows.append({
            "model": model_name,
            "internal_accuracy": internal_acc,
            "external_accuracy": external_acc,
            "generalization_gap": generalization_gap,
            "internal_macro_f1": float(internal["macro_f1"]),
            "external_macro_f1": float(external["macro_f1"]),
        })

    output = pd.DataFrame(rows)
    output_path = args.output_dir / "final_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(output.to_string(index=False))
    logger.info("Summary report saved to %s", output_path)

    with args.metrics_json.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
