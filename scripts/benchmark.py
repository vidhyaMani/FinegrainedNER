"""Benchmarking script: run all models on test set and compare results.

Usage:
    python scripts/benchmark.py --models-dir outputs/ --data-dir data/annotations
"""

from __future__ import annotations

# Fix SSL certificate issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import src.utils.ssl_fix  # noqa: F401

import argparse
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.evaluation.error_analysis import (
    generate_error_report,
    extract_qualitative_examples,
    export_qualitative_examples_markdown,
)
from src.evaluation.extrinsic import (
    attribute_coverage,
    query_understanding_accuracy,
    slot_precision_recall_f1,
)
from src.evaluation.intrinsic import compare_models, compute_ner_metrics
from src.evaluation.efficiency import collect_efficiency_metrics, export_efficiency_csv
from src.models.classical.bilstm_crf import BiLSTMCRF
from src.models.classical.cnn_bilstm import CNNBiLSTMCRF
from src.models.transformer.bert_ner import BertNER
from src.models.transformer.roberta_ner import RobertaNER
from src.schema.entity_schema import EntitySchema
from src.utils.helpers import ensure_dir, get_logger, set_seed

logger = get_logger(__name__)


MODEL_CLASSES = {
    "bilstm_crf": BiLSTMCRF,
    "cnn_bilstm": CNNBiLSTMCRF,
    "bert_ner": BertNER,
    "roberta_ner": RobertaNER,
}


def load_split(path: Path):
    """Load a dataset split and extract tokens + tags."""
    df = pd.read_parquet(path)
    # Handle both possible column names
    tokens_col = "query_tokens" if "query_tokens" in df.columns else "tokens"
    tokens_list = df[tokens_col].tolist()
    tags_list = df["ner_tags"].tolist()
    # Convert numpy arrays to Python lists if needed
    tokens_list = [list(t) if hasattr(t, 'tolist') else t for t in tokens_list]
    tags_list = [list(t) if hasattr(t, 'tolist') else t for t in tags_list]
    # Get query IDs if available
    if "query_id" in df.columns:
        query_ids = df["query_id"].tolist()
    elif "query" in df.columns:
        query_ids = df["query"].tolist()
    else:
        query_ids = list(range(len(tokens_list)))
    return tokens_list, tags_list, query_ids


def plot_comparison_bar(
    results: Dict[str, Dict],
    output_path: str,
    metric: str = "f1",
):
    """Bar chart comparing models by a single metric."""
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, values, color=sns.color_palette("viridis", len(models)))
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    ax.set_xlim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved comparison chart to {output_path}")


def plot_per_entity_radar(
    results: Dict[str, Dict],
    output_path: str,
):
    """Radar chart comparing models across entity types."""
    # Collect entity types from the first model's per-entity breakdown
    first_model = list(results.values())[0]
    entity_report = first_model.get("per_entity_type", {})
    entity_types = [
        k for k in entity_report.keys()
        if k not in ("micro avg", "macro avg", "weighted avg")
    ]

    if not entity_types:
        logger.warning("No per-entity metrics available for radar chart.")
        return

    N = len(entity_types)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model_name, metrics in results.items():
        entity_data = metrics.get("per_entity_type", {})
        values = [entity_data.get(et, {}).get("f1-score", 0) for et in entity_types]
        values += values[:1]
        ax.plot(angles, values, label=model_name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(entity_types, size=9)
    ax.set_ylim(0, 1)
    ax.set_title("Per-Entity F1 Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved radar chart to {output_path}")


def export_overall_metrics_csv(
    results: Dict[str, Dict],
    output_path: str,
):
    """Export overall metrics to CSV: one row per model with P/R/F1."""
    rows = []
    for model_name, metrics in results.items():
        per_entity = metrics.get("per_entity_type", {})
        row = {
            "model": model_name,
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1", 0),
            "micro_precision": per_entity.get("micro avg", {}).get("precision", metrics.get("precision", 0)),
            "micro_recall": per_entity.get("micro avg", {}).get("recall", metrics.get("recall", 0)),
            "micro_f1": per_entity.get("micro avg", {}).get("f1-score", metrics.get("f1", 0)),
            "macro_precision": per_entity.get("macro avg", {}).get("precision", 0),
            "macro_recall": per_entity.get("macro avg", {}).get("recall", 0),
            "macro_f1": per_entity.get("macro avg", {}).get("f1-score", 0),
            "weighted_precision": per_entity.get("weighted avg", {}).get("precision", 0),
            "weighted_recall": per_entity.get("weighted avg", {}).get("recall", 0),
            "weighted_f1": per_entity.get("weighted avg", {}).get("f1-score", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved overall metrics to {output_path}")


def export_per_entity_metrics_csv(
    results: Dict[str, Dict],
    output_path: str,
):
    """Export per-entity metrics to CSV: one row per (model, entity_type)."""
    rows = []
    for model_name, metrics in results.items():
        per_entity = metrics.get("per_entity_type", {})
        for entity_type, entity_metrics in per_entity.items():
            # Skip aggregate rows
            if entity_type in ("micro avg", "macro avg", "weighted avg"):
                continue
            row = {
                "model": model_name,
                "entity_type": entity_type,
                "precision": entity_metrics.get("precision", 0),
                "recall": entity_metrics.get("recall", 0),
                "f1": entity_metrics.get("f1-score", 0),
                "support": entity_metrics.get("support", 0),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved per-entity metrics to {output_path}")


def export_query_level_metrics_csv(
    results: Dict[str, Dict],
    output_path: str,
):
    """Export query-level metrics to CSV: one row per model."""
    rows = []
    for model_name, metrics in results.items():
        extrinsic = metrics.get("extrinsic", {})
        qu = extrinsic.get("query_understanding", {})
        ac = extrinsic.get("attribute_coverage", {})
        slot = extrinsic.get("slot_metrics", {})

        row = {
            "model": model_name,
            "exact_match_accuracy": qu.get("exact_match_accuracy", 0),
            "partial_match_accuracy": qu.get("partial_match_accuracy", 0),
            "attribute_coverage": ac.get("overall_coverage", 0),
            "slot_precision": slot.get("slot_precision", 0),
            "slot_recall": slot.get("slot_recall", 0),
            "slot_f1": slot.get("slot_f1", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved query-level metrics to {output_path}")


def export_predictions_jsonl(
    test_tokens: list,
    test_tags: list,
    query_ids: list,
    results: Dict[str, Dict],
    output_path: str,
):
    """Export predictions to JSONL: one JSON object per (query, model)."""
    with open(output_path, "w") as f:
        for i, (tokens, gold_tags, qid) in enumerate(zip(test_tokens, test_tags, query_ids)):
            for model_name, metrics in results.items():
                preds = metrics.get("predictions", [])
                if i < len(preds):
                    pred_tags = preds[i]
                else:
                    pred_tags = []

                record = {
                    "query_id": str(qid),
                    "query_index": i,
                    "model": model_name,
                    "tokens": tokens,
                    "gold_tags": gold_tags,
                    "pred_tags": pred_tags,
                }
                f.write(json.dumps(record) + "\n")

    logger.info(f"Saved predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all NER models")
    parser.add_argument(
        "--models-dir", type=str, default="outputs",
        help="Directory containing trained model subdirectories",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/annotations",
        help="Directory containing test split",
    )
    parser.add_argument(
        "--test-file", type=str, default="test.parquet",
        help="Test data file name",
    )
    parser.add_argument(
        "--model-prefix", type=str, default="",
        help="Prefix for model directories (e.g., 'gold_' for gold_bert_ner)",
    )
    parser.add_argument(
        "--schema", type=str, default="configs/entity_schema.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/benchmark",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    schema = EntitySchema.from_yaml(args.schema)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # Load test data
    test_tokens, test_tags, query_ids = load_split(Path(args.data_dir) / args.test_file)
    logger.info(f"Test file: {args.test_file}")
    logger.info(f"Test set: {len(test_tokens):,} samples")

    all_results: Dict[str, Dict] = {}

    # 1. Run spaCy baseline (optional - may fail due to SSL issues)
    logger.info("=== Running spaCy Baseline ===")
    try:
        from src.evaluation.baseline import run_spacy_baseline
        spacy_metrics = run_spacy_baseline(
            test_tokens, test_tags, schema.label2id, schema.id2label,
        )
        all_results["spaCy Baseline"] = spacy_metrics
    except Exception as e:
        logger.warning(f"spaCy baseline failed: {e}. Skipping.")

    # 2. Load and evaluate each trained model
    models_dir = Path(args.models_dir)
    model_prefix = args.model_prefix
    for model_name, model_cls in MODEL_CLASSES.items():
        model_path = models_dir / f"{model_prefix}{model_name}"
        if not model_path.exists():
            logger.warning(f"Model directory '{model_path}' not found. Skipping.")
            continue

        logger.info(f"=== Evaluating {model_prefix}{model_name} ===")
        try:
            model = model_cls(
                label2id=schema.label2id,
                id2label=schema.id2label,
            )
            model.load(model_path)

            predictions = model.predict(test_tokens)
            metrics = compute_ner_metrics(test_tags, predictions)
            metrics["predictions"] = predictions  # Store for error analysis
            result_name = f"{model_prefix}{model_name}" if model_prefix else model_name
            all_results[result_name] = metrics

            # Save predictions to file for error analysis notebook
            with open(output_dir / f"{model_prefix}{model_name}_predictions.json", "w") as f:
                json.dump(predictions, f)

            logger.info(f"{model_prefix}{model_name} — F1: {metrics['f1']:.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")

    # 3. Also check for hierarchical model
    hierarchical_path = models_dir / "hierarchical_ner"
    if hierarchical_path.exists():
        logger.info("=== Evaluating Hierarchical NER ===")
        try:
            from src.models.hierarchical.hierarchical_ner import HierarchicalNER
            model = HierarchicalNER(
                label2id=schema.label2id,
                id2label=schema.id2label,
                schema=schema,
            )
            model.load(hierarchical_path)
            predictions = model.predict(test_tokens)
            metrics = compute_ner_metrics(test_tags, predictions)
            metrics["predictions"] = predictions
            all_results["Hierarchical NER"] = metrics

            # Save predictions
            with open(output_dir / "hierarchical_ner_predictions.json", "w") as f:
                json.dump(predictions, f)
        except Exception as e:
            logger.error(f"Failed to evaluate Hierarchical NER: {e}")

    # 4. Generate comparison table
    comparison_table = compare_models(all_results)

    # 5. Extrinsic evaluation
    logger.info("=== Extrinsic Evaluation ===")
    for model_name, metrics in all_results.items():
        if "predictions" in metrics:
            preds = metrics["predictions"]
        else:
            continue
        qu_metrics = query_understanding_accuracy(test_tokens, preds, test_tags)
        ac_metrics = attribute_coverage(test_tokens, preds, test_tags)
        slot_metrics = slot_precision_recall_f1(test_tokens, preds, test_tags)
        all_results[model_name]["extrinsic"] = {
            "query_understanding": qu_metrics,
            "attribute_coverage": ac_metrics,
            "slot_metrics": slot_metrics,
        }

    # 6. Save results
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "report_str"}
             for k, v in all_results.items()},
            f, indent=2, default=str,
        )

    with open(output_dir / "comparison_table.txt", "w") as f:
        f.write(comparison_table)

    # 6b. Collect efficiency metrics
    logger.info("=== Collecting Efficiency Metrics ===")
    efficiency_results = []
    for model_name, model_cls in MODEL_CLASSES.items():
        model_path = models_dir / model_name
        if not model_path.exists():
            continue
        try:
            model = model_cls(
                label2id=schema.label2id,
                id2label=schema.id2label,
            )
            model.load(model_path)
            eff_metrics = collect_efficiency_metrics(
                model, model_name, test_tokens,
                log_dir=models_dir.parent / "outputs",
                num_latency_runs=50,
            )
            efficiency_results.append(eff_metrics)
            logger.info(f"{model_name}: {eff_metrics.get('total_params_millions', -1):.2f}M params, "
                       f"{eff_metrics.get('avg_latency_ms', -1):.2f}ms latency")
        except Exception as e:
            logger.warning(f"Failed to collect efficiency metrics for {model_name}: {e}")

    if efficiency_results:
        export_efficiency_csv(efficiency_results, str(output_dir / "efficiency_metrics.csv"))

    # 7. Generate plots
    plot_comparison_bar(all_results, str(output_dir / "model_comparison_f1.png"), "f1")
    plot_comparison_bar(all_results, str(output_dir / "model_comparison_precision.png"), "precision")
    plot_comparison_bar(all_results, str(output_dir / "model_comparison_recall.png"), "recall")
    plot_per_entity_radar(all_results, str(output_dir / "per_entity_radar.png"))

    # 8. Error analysis for best model
    if all_results:
        best_model = max(all_results.items(), key=lambda x: x[1].get("f1", 0))
        best_model_name, best_metrics = best_model
        logger.info(f"Best model: {best_model_name} (F1={best_metrics['f1']:.4f})")

        # Generate comprehensive error report
        if "predictions" in best_metrics:
            logger.info("=== Generating Error Analysis ===")
            error_report = generate_error_report(
                test_tokens, test_tags, best_metrics["predictions"],
                output_dir=str(output_dir),
            )
            with open(output_dir / "error_analysis.json", "w") as f:
                json.dump(error_report, f, indent=2, default=str)

            # Export qualitative examples
            examples = extract_qualitative_examples(
                test_tokens, test_tags, best_metrics["predictions"],
                num_examples=30,
            )
            export_qualitative_examples_markdown(
                examples,
                str(output_dir / "qualitative_examples.md"),
                model_name=best_model_name,
            )

    # 9. Export CSV and JSONL files
    logger.info("=== Exporting CSV/JSONL Files ===")
    export_overall_metrics_csv(all_results, str(output_dir / "overall_metrics.csv"))
    export_per_entity_metrics_csv(all_results, str(output_dir / "per_entity_metrics.csv"))
    export_query_level_metrics_csv(all_results, str(output_dir / "query_level_metrics.csv"))
    export_predictions_jsonl(
        test_tokens, test_tags, query_ids, all_results,
        str(output_dir / "predictions_test.jsonl")
    )

    logger.info(f"Benchmark complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

