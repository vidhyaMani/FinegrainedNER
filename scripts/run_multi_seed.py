#!/usr/bin/env python
"""Run multi-seed experiments for statistical significance.

Trains BERT, DAPT-BERT, and BiLSTM-CRF with multiple seeds
and reports mean ± std for all metrics.

Usage:
    python scripts/run_multi_seed.py --seeds 42 123 456
    python scripts/run_multi_seed.py --models bert_ner bilstm_crf --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.utils.helpers import ensure_dir, get_logger

logger = get_logger(__name__)

DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_MODELS = ["bert_ner", "bilstm_crf"]  # Add "dapt_bert" when trained


def run_training(
    model_type: str,
    seed: int,
    output_dir: Path,
    data_dir: str = "data/annotations",
    config_path: str = "configs/train_config.yaml",
) -> dict:
    """Run training for a single model with a specific seed.

    Args:
        model_type: Type of model ('bert_ner', 'bilstm_crf', 'dapt_bert').
        seed: Random seed.
        output_dir: Directory to save model and results.
        data_dir: Path to data directory.
        config_path: Path to training config.

    Returns:
        Dict with test metrics.
    """
    model_output_dir = output_dir / f"{model_type}_seed{seed}"
    ensure_dir(model_output_dir)

    if model_type in ["bert_ner", "roberta_ner", "dapt_bert"]:
        script = "scripts/train_transformer.py"
        cmd = [
            sys.executable, script,
            "--model", model_type.replace("_ner", "").replace("dapt_", ""),
            "--data-dir", data_dir,
            "--output-dir", str(model_output_dir),
            "--seed", str(seed),
        ]
        if model_type == "dapt_bert":
            cmd.extend(["--checkpoint", "outputs/esci_bert_dapt/final"])
    else:
        # Classical models: bilstm_crf, cnn_bilstm - pass model name as-is
        script = "scripts/train_classical.py"
        cmd = [
            sys.executable, script,
            "--model", model_type,
            "--data-dir", data_dir,
            "--output-dir", str(model_output_dir),
            "--seed", str(seed),
        ]

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        if result.returncode != 0:
            logger.error(f"Training failed for {model_type} seed {seed}:")
            logger.error(result.stderr)
            return {}
    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for {model_type} seed {seed}")
        return {}

    # Load test metrics
    metrics_path = model_output_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)

    return {}


def aggregate_results(
    results: dict,
    output_path: Path,
) -> pd.DataFrame:
    """Aggregate multi-seed results and compute mean ± std.

    Args:
        results: Dict mapping (model, seed) -> metrics.
        output_path: Path to save aggregated CSV.

    Returns:
        DataFrame with aggregated metrics.
    """
    # Organize by model
    model_results = {}
    for (model, seed), metrics in results.items():
        if model not in model_results:
            model_results[model] = []
        if metrics:
            model_results[model].append(metrics)

    # Compute aggregates
    rows = []
    for model, metrics_list in model_results.items():
        if not metrics_list:
            continue

        # Extract numeric metrics
        metric_keys = ["precision", "recall", "f1"]
        for key in metric_keys:
            values = [m.get(key, m.get(f"test_{key}", 0)) for m in metrics_list if key in m or f"test_{key}" in m]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                rows.append({
                    "model": model,
                    "metric": key,
                    "mean": mean_val,
                    "std": std_val,
                    "mean_std": f"{mean_val:.4f} ± {std_val:.4f}",
                    "n_seeds": len(values),
                    "values": values,
                })

    df = pd.DataFrame(rows)

    # Create pivot table for easier reading
    if not df.empty:
        pivot_df = df.pivot(index="model", columns="metric", values="mean_std")
        pivot_df.to_csv(output_path.with_suffix(".pivot.csv"))
        logger.info(f"Saved pivot table to {output_path.with_suffix('.pivot.csv')}")

    df.to_csv(output_path, index=False)
    logger.info(f"Saved aggregated results to {output_path}")

    return df


def load_existing_results(output_dir: Path) -> dict:
    """Load existing results from previous runs.

    Args:
        output_dir: Directory containing seed-specific model outputs.

    Returns:
        Dict mapping (model, seed) -> metrics.
    """
    results = {}

    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Parse model name and seed from directory name
        name = model_dir.name
        if "_seed" not in name:
            continue

        parts = name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue

        model = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            continue

        metrics_path = model_dir / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                results[(model, seed)] = json.load(f)
            logger.info(f"Loaded existing results for {model} seed {seed}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed experiments")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Models to train (default: bert_ner bilstm_crf)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Random seeds to use (default: 42 123 456)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/annotations",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/multi_seed",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip training if results already exist",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Only aggregate existing results, don't train",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Load existing results
    results = load_existing_results(output_dir)
    logger.info(f"Found {len(results)} existing results")

    if not args.aggregate_only:
        # Run training for each model and seed
        for model in args.models:
            for seed in args.seeds:
                key = (model, seed)

                if args.skip_existing and key in results:
                    logger.info(f"Skipping {model} seed {seed} (already exists)")
                    continue

                logger.info(f"=== Training {model} with seed {seed} ===")
                metrics = run_training(
                    model_type=model,
                    seed=seed,
                    output_dir=output_dir,
                    data_dir=args.data_dir,
                )
                results[key] = metrics

    # Aggregate results
    if results:
        aggregate_df = aggregate_results(
            results,
            output_dir / "multi_seed_results.csv",
        )

        # Print summary
        print("\n" + "=" * 60)
        print("Multi-Seed Experiment Results (mean ± std)")
        print("=" * 60)

        for model in args.models:
            model_rows = [r for r in aggregate_df.to_dict("records") if r["model"] == model]
            if model_rows:
                print(f"\n{model}:")
                for row in model_rows:
                    print(f"  {row['metric']}: {row['mean_std']}")
    else:
        logger.warning("No results to aggregate")


if __name__ == "__main__":
    main()

