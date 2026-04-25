#!/usr/bin/env python
"""Run learning curve experiments.

Trains models on different dataset sizes (1K, 2.5K, 5K, full)
and plots learning curves.

Usage:
    python scripts/run_learning_curve.py
    python scripts/run_learning_curve.py --models bert_ner bilstm_crf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.helpers import ensure_dir, get_logger

logger = get_logger(__name__)

# Dataset sizes to evaluate
DATASET_SIZES = ["1k", "2.5k", "5k", "full"]
SIZE_VALUES = {"1k": 1000, "2.5k": 2500, "5k": 5000, "full": 52500}

DEFAULT_MODELS = ["bert_ner", "bilstm_crf"]


def create_2_5k_subset(data_dir: Path, seed: int = 42) -> Path:
    """Create 2.5K subset if it doesn't exist.

    Args:
        data_dir: Path to data/annotations directory.
        seed: Random seed for sampling.

    Returns:
        Path to 2.5K parquet file.
    """
    output_path = data_dir / "2.5k.parquet"

    if output_path.exists():
        logger.info("2.5K subset already exists")
        return output_path

    logger.info("Creating 2.5K subset...")

    # Load training data
    train_path = data_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pd.read_parquet(train_path)

    # Sample 2.5K
    sampled = df.sample(n=2500, random_state=seed)
    sampled.to_parquet(output_path)

    # Also create CoNLL version
    conll_path = data_dir / "2.5k.conll"
    with open(conll_path, "w") as f:
        tokens_col = "query_tokens" if "query_tokens" in sampled.columns else "tokens"
        for _, row in sampled.iterrows():
            tokens = row[tokens_col]
            tags = row["ner_tags"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            if hasattr(tags, "tolist"):
                tags = tags.tolist()
            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

    logger.info(f"Created 2.5K subset: {output_path}")
    return output_path


def run_training_for_size(
    model_type: str,
    data_size: str,
    output_dir: Path,
    data_dir: str = "data/annotations",
    seed: int = 42,
) -> dict:
    """Train a model on a specific dataset size.

    Args:
        model_type: Type of model ('bert_ner', 'bilstm_crf', etc.).
        data_size: Dataset size ('1k', '2.5k', '5k', 'full').
        output_dir: Directory to save model and results.
        data_dir: Path to data directory.
        seed: Random seed.

    Returns:
        Dict with test metrics.
    """
    model_output_dir = output_dir / f"{model_type}_{data_size}"
    ensure_dir(model_output_dir)

    # Determine training data path
    if data_size == "full":
        train_file = "train.parquet"
    else:
        train_file = f"{data_size}.parquet"

    if model_type in ["bert_ner", "roberta_ner", "dapt_bert"]:
        script = "scripts/train_transformer.py"
        cmd = [
            sys.executable, script,
            "--model", model_type.replace("_ner", "").replace("dapt_", ""),
            "--data-dir", data_dir,
            "--train-file", train_file,
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
            "--train-file", train_file,
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
            logger.error(f"Training failed for {model_type} on {data_size}:")
            logger.error(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            return {}
    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for {model_type} on {data_size}")
        return {}

    # Load test metrics
    metrics_path = model_output_dir / "test_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
            metrics["data_size"] = data_size
            metrics["data_size_value"] = SIZE_VALUES.get(data_size, 0)
            return metrics

    return {}


def load_existing_results(output_dir: Path) -> dict:
    """Load existing results from previous runs."""
    results = {}

    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue

        name = model_dir.name
        # Parse model name and data size
        for size in DATASET_SIZES:
            if name.endswith(f"_{size}"):
                model = name.rsplit(f"_{size}", 1)[0]
                metrics_path = model_dir / "test_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                        metrics["data_size"] = size
                        metrics["data_size_value"] = SIZE_VALUES.get(size, 0)
                        results[(model, size)] = metrics
                    logger.info(f"Loaded existing results for {model} on {size}")
                break

    return results


def plot_learning_curves(
    results: dict,
    output_path: Path,
    metric: str = "f1",
):
    """Plot learning curves for all models.

    Args:
        results: Dict mapping (model, size) -> metrics.
        output_path: Path to save figure.
        metric: Metric to plot ('f1', 'precision', 'recall').
    """
    # Organize data
    plot_data = []
    for (model, size), metrics in results.items():
        if not metrics:
            continue
        value = metrics.get(metric, metrics.get(f"test_{metric}", 0))
        plot_data.append({
            "model": model,
            "size": size,
            "size_value": SIZE_VALUES.get(size, 0),
            metric: value,
        })

    df = pd.DataFrame(plot_data)
    if df.empty:
        logger.warning("No data to plot")
        return

    # Sort by size
    df = df.sort_values("size_value")

    # Plot
    plt.figure(figsize=(10, 6))

    for model in df["model"].unique():
        model_df = df[df["model"] == model].sort_values("size_value")
        plt.plot(
            model_df["size_value"],
            model_df[metric],
            marker="o",
            linewidth=2,
            markersize=8,
            label=model,
        )

    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel(f"{metric.upper()} Score", fontsize=12)
    plt.title(f"Learning Curves: {metric.upper()} vs Training Size", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks
    plt.xticks(
        list(SIZE_VALUES.values()),
        [f"{k}\n({v:,})" for k, v in SIZE_VALUES.items()],
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved learning curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run learning curve experiments")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Models to train",
    )
    parser.add_argument(
        "--sizes", nargs="+", default=DATASET_SIZES,
        help="Dataset sizes to use",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/annotations",
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/learning_curve",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip training if results already exist",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Only plot from existing results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    ensure_dir(output_dir)
    ensure_dir(results_dir)

    # Create 2.5K subset if needed
    if "2.5k" in args.sizes:
        create_2_5k_subset(data_dir, args.seed)

    # Load existing results
    results = load_existing_results(output_dir)
    logger.info(f"Found {len(results)} existing results")

    if not args.plot_only:
        # Run training for each model and size
        for model in args.models:
            for size in args.sizes:
                key = (model, size)

                if args.skip_existing and key in results:
                    logger.info(f"Skipping {model} on {size} (already exists)")
                    continue

                logger.info(f"=== Training {model} on {size} ===")
                metrics = run_training_for_size(
                    model_type=model,
                    data_size=size,
                    output_dir=output_dir,
                    data_dir=str(data_dir),
                    seed=args.seed,
                )
                results[key] = metrics

    # Export results
    if results:
        # Save raw results
        rows = []
        for (model, size), metrics in results.items():
            if metrics:
                row = {"model": model, "size": size, "size_value": SIZE_VALUES.get(size, 0)}
                row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(results_dir / "learning_curve.csv", index=False)
        logger.info(f"Saved results to {results_dir / 'learning_curve.csv'}")

        # Plot learning curves
        ensure_dir(results_dir / "figures_export")
        plot_learning_curves(results, results_dir / "figures_export" / "learning_curve_f1.png", "f1")
        plot_learning_curves(results, results_dir / "figures_export" / "learning_curve_precision.png", "precision")
        plot_learning_curves(results, results_dir / "figures_export" / "learning_curve_recall.png", "recall")

        # Print summary
        print("\n" + "=" * 60)
        print("Learning Curve Results (F1 Score)")
        print("=" * 60)
        print(df.pivot(index="model", columns="size", values="f1").to_string())
    else:
        logger.warning("No results to export")


if __name__ == "__main__":
    main()

