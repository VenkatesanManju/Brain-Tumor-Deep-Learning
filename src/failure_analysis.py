#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import add_common_paths
from utils import ensure_dir, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect representative success and failure cases for qualitative review.")
    add_common_paths(parser)
    parser.add_argument("--eas-csv", type=Path, default=None, help="Path to eas_scores.csv")
    parser.add_argument("--dry-run", action="store_true", help="Skip visualization workflow and display planned report structure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("failure_analysis", args.log_dir)
    logger.info("Step 10: failure-case analysis")

    if args.dry_run:
        logger.info("Dry-run mode enabled: failure-case pipeline ready.")
        print("[dry-run] Failure-case analysis scaffold ready.")
        return

    eas_csv = args.eas_csv if args.eas_csv is not None else args.output_dir / "eas_scores.csv"
    output_dir = ensure_dir(args.output_dir / "failure_cases")

    if not eas_csv.exists():
        logger.warning("EAS score CSV not found at %s. Writing placeholder report.", eas_csv)
        summary_path = output_dir / "failure_summary.csv"
        df_empty = pd.DataFrame(columns=["dataset_type", "total_evaluated", "correct_count", "failure_count", "accuracy"])
        df_empty.to_csv(summary_path, index=False)
        return

    df = pd.read_csv(eas_csv)
    summary_rows = []
    for ds_type, group in df.groupby("dataset_type"):
        total = len(group)
        correct = group["is_correct"].sum()
        failures = total - correct
        accuracy = float(correct / total) if total > 0 else 0.0
        summary_rows.append({
            "dataset_type": ds_type,
            "total_evaluated": total,
            "correct_count": correct,
            "failure_count": failures,
            "accuracy": accuracy,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "failure_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    failures_df = df[df["is_correct"] == 0]
    failures_path = output_dir / "failure_cases_details.csv"
    failures_df.to_csv(failures_path, index=False)

    logger.info("Failure analysis completed. Saved summary to %s and details to %s", summary_path, failures_path)
    print("Failure-case summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
