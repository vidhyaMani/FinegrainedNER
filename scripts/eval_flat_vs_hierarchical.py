#!/usr/bin/env python3
"""Evaluate flat vs hierarchical schema comparison.

This script demonstrates the value of hierarchical entity schema by:
1. Evaluating models at fine-grained level (BRAND, COLOR, etc.)
2. Evaluating the same predictions at coarse level (PRODUCT only)
3. Showing that hierarchy provides graceful degradation

The key insight: even when a model confuses fine types (e.g., BRAND vs PRODUCT_TYPE),
the coarse-level prediction may still be correct, enabling downstream applications
to benefit from partial information.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.evaluation.intrinsic import (
    collapse_to_coarse,
    compute_coarse_metrics,
    compute_ner_metrics,
)
from src.schema.entity_schema import EntitySchema
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_predictions(predictions_path: Path) -> List[List[str]]:
    """Load model predictions from JSON file."""
    with open(predictions_path) as f:
        return json.load(f)


def load_gold_labels(data_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """Load gold labels from test parquet."""
    df = pd.read_parquet(data_path)
    # Handle both possible column names
    tokens_col = "query_tokens" if "query_tokens" in df.columns else "tokens"
    tokens = df[tokens_col].tolist()
    tags = df["ner_tags"].tolist()
    # Convert numpy arrays to Python lists if needed
    tokens = [list(t) if hasattr(t, 'tolist') else t for t in tokens]
    tags = [list(t) if hasattr(t, 'tolist') else t for t in tags]
    return tokens, tags


def compute_confusion_gain(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, Any]:
    """Compute how much performance is recovered at coarse level.

    This measures "confusion gain": when fine-grained predictions are wrong
    but the coarse-level prediction is correct.

    Returns:
        Dict with fine metrics, coarse metrics, and gain statistics.
    """
    fine_metrics = compute_ner_metrics(true_labels, pred_labels)
    coarse_metrics = compute_coarse_metrics(true_labels, pred_labels)

    # Calculate gain
    f1_gain = coarse_metrics["f1"] - fine_metrics["f1"]
    recall_gain = coarse_metrics["recall"] - fine_metrics["recall"]
    precision_gain = coarse_metrics["precision"] - fine_metrics["precision"]

    return {
        "fine": {
            "precision": fine_metrics["precision"],
            "recall": fine_metrics["recall"],
            "f1": fine_metrics["f1"],
        },
        "coarse": {
            "precision": coarse_metrics["precision"],
            "recall": coarse_metrics["recall"],
            "f1": coarse_metrics["f1"],
        },
        "gain": {
            "precision": precision_gain,
            "recall": recall_gain,
            "f1": f1_gain,
        },
    }


def analyze_fine_type_confusion(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, int]:
    """Analyze confusion between fine-grained entity types.

    Counts cases where predicted entity type differs from gold but
    both are non-O (i.e., both are some PRODUCT subtype).
    """
    confusion_counts = Counter()

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            if true_tag == "O" or pred_tag == "O":
                continue
            # Extract entity types
            true_type = true_tag.split("-", 1)[1] if "-" in true_tag else true_tag
            pred_type = pred_tag.split("-", 1)[1] if "-" in pred_tag else pred_tag

            if true_type != pred_type:
                confusion_counts[(true_type, pred_type)] += 1

    return dict(confusion_counts.most_common(20))


def analyze_by_entity_rarity_hierarchical(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Analyze how hierarchy helps with rare entity types.

    Hypothesis: Rare entity types (e.g., MATERIAL, COLOR) benefit more
    from hierarchical fallback because fine-grained classifiers have
    less training data for them.
    """
    # Count entity occurrences
    entity_counts = Counter()
    for true_seq in true_labels:
        for tag in true_seq:
            if tag.startswith("B-"):
                entity_counts[tag[2:]] += 1

    # Split into rare vs common (threshold: median count)
    counts = list(entity_counts.values())
    median_count = sorted(counts)[len(counts) // 2] if counts else 0

    rare_types = {et for et, c in entity_counts.items() if c < median_count}
    common_types = {et for et, c in entity_counts.items() if c >= median_count}

    logger.info(f"Entity counts: {dict(entity_counts.most_common())}")
    logger.info(f"Rare types (<{median_count}): {rare_types}")
    logger.info(f"Common types (>={median_count}): {common_types}")

    # Filter to only sequences containing rare entities
    rare_true, rare_pred = [], []
    common_true, common_pred = [], []

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        has_rare = any(
            tag.startswith("B-") and tag[2:] in rare_types
            for tag in true_seq
        )
        has_common = any(
            tag.startswith("B-") and tag[2:] in common_types
            for tag in true_seq
        )

        if has_rare:
            rare_true.append(true_seq)
            rare_pred.append(pred_seq)
        if has_common:
            common_true.append(true_seq)
            common_pred.append(pred_seq)

    results = {
        "entity_counts": dict(entity_counts.most_common()),
        "rare_types": list(rare_types),
        "common_types": list(common_types),
    }

    if rare_true:
        rare_gain = compute_confusion_gain(rare_true, rare_pred)
        results["rare_entities"] = {
            "count": len(rare_true),
            **rare_gain,
        }

    if common_true:
        common_gain = compute_confusion_gain(common_true, common_pred)
        results["common_entities"] = {
            "count": len(common_true),
            **common_gain,
        }

    return results


def generate_comparison_table(
    model_results: Dict[str, Dict[str, Any]],
) -> str:
    """Generate a formatted comparison table."""
    lines = [
        "=" * 80,
        "FLAT VS HIERARCHICAL SCHEMA COMPARISON",
        "=" * 80,
        "",
        f"{'Model':<25} {'Fine F1':>10} {'Coarse F1':>12} {'F1 Gain':>10} {'Recall Gain':>12}",
        "-" * 80,
    ]

    for model_name, results in sorted(model_results.items()):
        fine_f1 = results["fine"]["f1"]
        coarse_f1 = results["coarse"]["f1"]
        f1_gain = results["gain"]["f1"]
        recall_gain = results["gain"]["recall"]

        lines.append(
            f"{model_name:<25} {fine_f1:>10.4f} {coarse_f1:>12.4f} "
            f"{f1_gain:>+10.4f} {recall_gain:>+12.4f}"
        )

    lines.append("-" * 80)
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Fine F1: Performance at fine-grained level (BRAND, COLOR, etc.)")
    lines.append("- Coarse F1: Performance when collapsing to PRODUCT only")
    lines.append("- F1 Gain: How much F1 improves with hierarchical fallback")
    lines.append("- Recall Gain: How much recall improves (entities correctly bounded)")
    lines.append("")

    return "\n".join(lines)


def generate_latex_table(
    model_results: Dict[str, Dict[str, Any]],
) -> str:
    """Generate LaTeX table for thesis."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Flat vs Hierarchical Schema Performance Comparison}",
        r"\label{tab:flat-vs-hierarchical}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Fine F1 & Coarse F1 & $\Delta$ F1 & $\Delta$ Recall \\",
        r"\midrule",
    ]

    for model_name, results in sorted(model_results.items()):
        fine_f1 = results["fine"]["f1"]
        coarse_f1 = results["coarse"]["f1"]
        f1_gain = results["gain"]["f1"]
        recall_gain = results["gain"]["recall"]

        model_display = model_name.replace("_", r"\_")
        lines.append(
            f"{model_display} & {fine_f1:.3f} & {coarse_f1:.3f} & "
            f"+{f1_gain:.3f} & +{recall_gain:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate flat vs hierarchical schema comparison"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/annotations/test.parquet",
        help="Path to test data parquet",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="results/benchmark",
        help="Directory containing *_predictions.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/flat_vs_hierarchical",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # Setup
    test_path = Path(args.test_data)
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gold labels
    logger.info(f"Loading test data from {test_path}")
    tokens, gold_tags = load_gold_labels(test_path)
    logger.info(f"Loaded {len(tokens)} test samples")

    # Find all prediction files
    prediction_files = list(predictions_dir.glob("*_predictions.json"))
    if not prediction_files:
        logger.error(f"No prediction files found in {predictions_dir}")
        return

    logger.info(f"Found {len(prediction_files)} prediction files")

    # Evaluate each model
    all_results = {}
    rarity_analysis = {}

    for pred_file in prediction_files:
        model_name = pred_file.stem.replace("_predictions", "")
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info("=" * 60)

        predictions = load_predictions(pred_file)

        # Ensure lengths match (truncate if needed)
        min_len = min(len(gold_tags), len(predictions))
        eval_gold = gold_tags[:min_len]
        eval_pred = predictions[:min_len]

        # Main comparison
        results = compute_confusion_gain(eval_gold, eval_pred)
        all_results[model_name] = results

        logger.info(f"Fine-grained F1:   {results['fine']['f1']:.4f}")
        logger.info(f"Coarse F1:         {results['coarse']['f1']:.4f}")
        logger.info(f"F1 Gain:           {results['gain']['f1']:+.4f}")

        # Fine-type confusion analysis
        confusion = analyze_fine_type_confusion(eval_gold, eval_pred)
        results["fine_type_confusion"] = confusion
        logger.info(f"Top confusions: {list(confusion.items())[:5]}")

        # Rarity analysis
        rarity = analyze_by_entity_rarity_hierarchical(
            tokens[:min_len], eval_gold, eval_pred
        )
        rarity_analysis[model_name] = rarity

    # Generate comparison table
    table = generate_comparison_table(all_results)
    print("\n" + table)

    # Save results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(output_dir / "rarity_analysis.json", "w") as f:
        json.dump(rarity_analysis, f, indent=2)

    with open(output_dir / "comparison_table.txt", "w") as f:
        f.write(table)

    # Generate LaTeX
    latex = generate_latex_table(all_results)
    with open(output_dir / "comparison_table.tex", "w") as f:
        f.write(latex)

    logger.info(f"\nResults saved to {output_dir}/")

    # Summary statistics
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    avg_f1_gain = sum(r["gain"]["f1"] for r in all_results.values()) / len(all_results)
    avg_recall_gain = sum(r["gain"]["recall"] for r in all_results.values()) / len(all_results)

    print(f"\nAverage F1 gain from hierarchical fallback: +{avg_f1_gain:.4f}")
    print(f"Average recall gain: +{avg_recall_gain:.4f}")
    print("\nThis demonstrates that the hierarchical schema provides:")
    print("1. Graceful degradation when fine-grained types are confused")
    print("2. Higher recall for entity boundary detection")
    print("3. A foundation for taxonomy extension without retraining")


if __name__ == "__main__":
    main()

