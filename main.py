#!/usr/bin/env python3
"""Top-level entry point for the brain tumor MRI classification pipeline."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import MODEL_NAMES

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"


def run_step(script_name: str, *extra_args: str) -> None:
    script_path = SRC_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    command = [sys.executable, str(script_path), *extra_args]
    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full brain tumor MRI pipeline.")
    parser.add_argument("--skip-early-stages", action="store_true", help="Skip step 0 and 1 and run only later stages.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Project root: {ROOT}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")

    if not args.skip_early_stages:
        run_step("convert_external.py", "--input-root", str(args.data_root / "external_raw"), "--output-root", str(args.data_root / "external"), "--log-dir", str(args.output_dir / "logs"))
        run_step("dedup_check.py", "--external-root", str(args.data_root / "external"), "--primary-root", str(args.data_root / "primary"), "--output-dir", str(args.output_dir))

    run_step("data.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--batch-size", str(args.batch_size))
    for model_name in MODEL_NAMES:
        run_step(
            "train.py",
            "--data-root",
            str(args.data_root),
            "--output-dir",
            str(args.output_dir),
            "--checkpoint-dir",
            str(args.output_dir / "checkpoints"),
            "--log-dir",
            str(args.output_dir / "logs"),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--model",
            model_name,
        )
    run_step("ensemble.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--log-dir", str(args.output_dir / "logs"), "--checkpoint-dir", str(args.output_dir / "checkpoints"))
    run_step("evaluate_internal.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--log-dir", str(args.output_dir / "logs"), "--batch-size", str(args.batch_size), "--checkpoint-dir", str(args.output_dir / "checkpoints"))
    run_step("evaluate_external.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--log-dir", str(args.output_dir / "logs"), "--batch-size", str(args.batch_size), "--checkpoint-dir", str(args.output_dir / "checkpoints"))
    run_step("generalization_gap.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--log-dir", str(args.output_dir / "logs"), "--internal-json", str(args.output_dir / "internal_metrics.json"), "--external-json", str(args.output_dir / "external_metrics.json"))
    run_step("gradcam.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--checkpoint-dir", str(args.output_dir / "checkpoints"), "--log-dir", str(args.output_dir / "logs"), "--batch-size", str(args.batch_size))
    run_step("eas.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--checkpoint-dir", str(args.output_dir / "checkpoints"), "--log-dir", str(args.output_dir / "logs"), "--alpha", str(args.alpha))
    run_step("failure_analysis.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--log-dir", str(args.output_dir / "logs"))
    run_step("summarize.py", "--data-root", str(args.data_root), "--output-dir", str(args.output_dir), "--log-dir", str(args.output_dir / "logs"), "--metrics-json", str(args.output_dir / "final_metrics.json"))

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
