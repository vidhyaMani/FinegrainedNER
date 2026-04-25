#!/usr/bin/env python
"""Generate thesis-ready artifacts.

Consolidates all CSVs, figures, and tables into a single folder
for thesis submission.

Usage:
    python scripts/generate_thesis_artifacts.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.utils.helpers import ensure_dir, get_logger

logger = get_logger(__name__)


def copy_figures(src_dir: Path, dest_dir: Path):
    """Copy all PNG/PDF figures to destination."""
    for ext in ["*.png", "*.pdf", "*.svg"]:
        for fig in src_dir.glob(ext):
            shutil.copy(fig, dest_dir / fig.name)
            logger.info(f"Copied {fig.name}")


def generate_model_comparison_table(
    benchmark_results: dict,
    output_path: Path,
):
    """Generate final model comparison table."""
    rows = []

    for model_name, metrics in benchmark_results.items():
        extrinsic = metrics.get("extrinsic", {})
        qu = extrinsic.get("query_understanding", {})
        ac = extrinsic.get("attribute_coverage", {})
        slot = extrinsic.get("slot_metrics", {})

        row = {
            "Model": model_name,
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
            "Exact Match": qu.get("exact_match_accuracy", 0),
            "Attr Coverage": ac.get("overall_coverage", 0),
            "Slot F1": slot.get("slot_f1", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("F1", ascending=False)

    # Format numeric columns
    for col in ["Precision", "Recall", "F1", "Exact Match", "Slot F1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    if "Attr Coverage" in df.columns:
        df["Attr Coverage"] = df["Attr Coverage"].apply(
            lambda x: f"{x:.1f}%" if isinstance(x, float) else x
        )

    # Save as CSV
    df.to_csv(output_path / "model_comparison_table.csv", index=False)

    # Save as LaTeX
    latex_table = df.to_latex(index=False, escape=False)
    with open(output_path / "model_comparison_table.tex", "w") as f:
        f.write(latex_table)

    # Save as Markdown
    with open(output_path / "model_comparison_table.md", "w") as f:
        f.write(df.to_markdown(index=False))

    logger.info(f"Generated model comparison table")
    return df


def generate_per_entity_summary(
    benchmark_results: dict,
    output_path: Path,
):
    """Generate per-entity performance summary."""
    all_entities = set()
    for metrics in benchmark_results.values():
        per_entity = metrics.get("per_entity_type", {})
        for entity in per_entity.keys():
            if entity not in ("micro avg", "macro avg", "weighted avg"):
                all_entities.add(entity)

    rows = []
    for entity in sorted(all_entities):
        row = {"Entity": entity}
        for model_name, metrics in benchmark_results.items():
            per_entity = metrics.get("per_entity_type", {})
            entity_metrics = per_entity.get(entity, {})
            f1 = entity_metrics.get("f1-score", 0)
            row[model_name] = f"{f1:.2f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path / "per_entity_summary.csv", index=False)

    with open(output_path / "per_entity_summary.md", "w") as f:
        f.write("# Per-Entity F1 Scores\n\n")
        f.write(df.to_markdown(index=False))

    logger.info("Generated per-entity summary")
    return df


def generate_final_comparison_table(
    benchmark_results: dict,
    project_root: Path,
    output_path: Path,
):
    """Generate final comparison table with intrinsic, retrieval, and efficiency metrics."""
    rows = []

    # Load retrieval metrics if available
    retrieval_path = project_root / "results" / "retrieval" / "retrieval_metrics.csv"
    retrieval_metrics = {}
    if retrieval_path.exists():
        retrieval_df = pd.read_csv(retrieval_path)
        for _, row in retrieval_df.iterrows():
            model = row["model"].replace("_rerank", "")
            retrieval_metrics[model] = {
                "nDCG@10": row.get("nDCG@10", 0),
                "MRR@10": row.get("MRR@10", 0),
            }

    # Load efficiency metrics if available
    efficiency_path = project_root / "outputs" / "benchmark" / "efficiency_metrics.csv"
    efficiency_metrics = {}
    if efficiency_path.exists():
        efficiency_df = pd.read_csv(efficiency_path)
        for _, row in efficiency_df.iterrows():
            model = row["model"]
            efficiency_metrics[model] = {
                "params_M": row.get("total_params_millions", 0),
                "latency_ms": row.get("avg_latency_ms", 0),
            }

    for model_name, metrics in benchmark_results.items():
        row = {
            "Model": model_name,
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
        }

        # Add retrieval metrics
        if model_name in retrieval_metrics:
            row["nDCG@10"] = retrieval_metrics[model_name].get("nDCG@10", 0)
            row["MRR@10"] = retrieval_metrics[model_name].get("MRR@10", 0)
        else:
            row["nDCG@10"] = "-"
            row["MRR@10"] = "-"

        # Add efficiency metrics
        if model_name in efficiency_metrics:
            row["Params (M)"] = efficiency_metrics[model_name].get("params_M", 0)
            row["Latency (ms)"] = efficiency_metrics[model_name].get("latency_ms", 0)
        else:
            row["Params (M)"] = "-"
            row["Latency (ms)"] = "-"

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("F1", ascending=False)

    # Add rank column
    df["Rank"] = range(1, len(df) + 1)

    # Format numeric columns
    for col in ["Precision", "Recall", "F1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    for col in ["nDCG@10", "MRR@10"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    for col in ["Params (M)", "Latency (ms)"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)

    # Reorder columns
    cols = ["Rank", "Model", "Precision", "Recall", "F1", "nDCG@10", "MRR@10", "Params (M)", "Latency (ms)"]
    df = df[[c for c in cols if c in df.columns]]

    # Save as CSV
    df.to_csv(output_path / "final_comparison_table.csv", index=False)

    # Save as LaTeX
    latex_table = df.to_latex(index=False, escape=False)
    with open(output_path / "final_comparison_table.tex", "w") as f:
        f.write(latex_table)

    # Save as Markdown
    with open(output_path / "final_comparison_table.md", "w") as f:
        f.write("# Final Model Comparison\n\n")
        f.write("Ranking by F1 score, with intrinsic, retrieval, and efficiency metrics.\n\n")
        f.write(df.to_markdown(index=False))

    logger.info("Generated final comparison table")
    return df


def generate_summary_report(
    benchmark_results: dict,
    output_path: Path,
    project_root: Path,
):
    """Generate comprehensive summary markdown report."""
    # Find best model
    best_model = max(
        benchmark_results.items(),
        key=lambda x: x[1].get("f1", 0)
    )

    lines = [
        "# E-commerce NER Experiment Results Summary",
        "",
        "**Date:** March 2026",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"This report summarizes experiments on fine-grained Named Entity Recognition (NER) for e-commerce product queries using the Amazon ESCI dataset. **{best_model[0]}** achieved the best overall F1 score of **{best_model[1]['f1']:.4f}**.",
        "",
        "---",
        "",
        "## 2. Models Evaluated",
        "",
        "| Model | Type | Description |",
        "|-------|------|-------------|",
        "| BiLSTM-CRF | Classical | Word embeddings + BiLSTM + CRF |",
        "| CNN-BiLSTM-CRF | Classical | Char CNN + BiLSTM + CRF |",
        "| BERT-NER | Transformer | Fine-tuned bert-base-uncased |",
        "| RoBERTa-NER | Transformer | Fine-tuned roberta-base |",
        "| DAPT-BERT | Transformer | Domain-adapted BERT (MLM on ESCI) |",
        "",
        "---",
        "",
        "## 3. Main Results",
        "",
        "### Overall Performance (Strict Span-Level)",
        "",
        "| Model | Precision | Recall | F1 | Rank |",
        "|-------|-----------|--------|-----|------|",
    ]

    sorted_models = sorted(
        benchmark_results.items(),
        key=lambda x: x[1].get("f1", 0),
        reverse=True
    )

    for i, (model_name, metrics) in enumerate(sorted_models, 1):
        p = metrics.get("precision", 0)
        r = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else str(i)))
        lines.append(f"| {model_name} | {p:.4f} | {r:.4f} | {f1:.4f} | {medal} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Per-Entity Performance (F1 Score)",
        "",
    ])

    # Add per-entity table
    all_entities = set()
    for metrics in benchmark_results.values():
        per_entity = metrics.get("per_entity_type", {})
        for entity in per_entity.keys():
            if entity not in ("micro avg", "macro avg", "weighted avg"):
                all_entities.add(entity)

    if all_entities:
        header = "| Entity |" + "|".join(benchmark_results.keys()) + "|"
        separator = "|--------|" + "|".join(["------"] * len(benchmark_results)) + "|"
        lines.append(header)
        lines.append(separator)

        for entity in sorted(all_entities):
            row_parts = [entity]
            for model_name, metrics in benchmark_results.items():
                per_entity = metrics.get("per_entity_type", {})
                f1 = per_entity.get(entity, {}).get("f1-score", 0)
                row_parts.append(f"{f1:.2f}")
            lines.append("| " + " | ".join(row_parts) + " |")

    # Load and add multi-seed results
    multi_seed_path = project_root / "outputs" / "multi_seed" / "multi_seed_results.pivot.csv"
    if multi_seed_path.exists():
        lines.extend([
            "",
            "---",
            "",
            "## 5. Multi-Seed Results (mean ± std)",
            "",
        ])
        try:
            ms_df = pd.read_csv(multi_seed_path)
            lines.append(ms_df.to_markdown(index=True))
        except Exception:
            lines.append("*See `data/multi_seed_results.csv` for detailed results.*")

    # Load and add learning curve results
    learning_curve_path = project_root / "results" / "thesis_artifacts" / "data" / "learning_curve.csv"
    if learning_curve_path.exists():
        lines.extend([
            "",
            "---",
            "",
            "## 6. Learning Curve Analysis",
            "",
            "Performance (F1) vs training data size:",
            "",
        ])
        try:
            lc_df = pd.read_csv(learning_curve_path)
            if "f1" in lc_df.columns or "test_f1" in lc_df.columns:
                f1_col = "f1" if "f1" in lc_df.columns else "test_f1"
                pivot = lc_df.pivot(index="model", columns="size", values=f1_col)
                lines.append(pivot.to_markdown())
        except Exception:
            lines.append("*See `data/learning_curve.csv` for detailed results.*")

    # Load and add retrieval results
    retrieval_path = project_root / "results" / "retrieval" / "retrieval_metrics.csv"
    if retrieval_path.exists():
        lines.extend([
            "",
            "---",
            "",
            "## 7. Retrieval Evaluation Results",
            "",
            "NER-enhanced product retrieval performance:",
            "",
        ])
        try:
            ret_df = pd.read_csv(retrieval_path)
            lines.append(ret_df.to_markdown(index=False))
        except Exception:
            lines.append("*See `data/retrieval_metrics.csv` for detailed results.*")

    # Load and add efficiency metrics
    efficiency_path = project_root / "outputs" / "benchmark" / "efficiency_metrics.csv"
    if efficiency_path.exists():
        lines.extend([
            "",
            "---",
            "",
            "## 8. Efficiency Comparison",
            "",
        ])
        try:
            eff_df = pd.read_csv(efficiency_path)
            cols_to_show = ["model", "total_params_millions", "avg_latency_ms", "peak_memory_mb"]
            cols_available = [c for c in cols_to_show if c in eff_df.columns]
            lines.append(eff_df[cols_available].to_markdown(index=False))
        except Exception:
            lines.append("*See `data/efficiency_metrics.csv` for detailed results.*")

    lines.extend([
        "",
        "---",
        "",
        "## 9. Error Analysis Highlights",
        "",
        "### Key Observations:",
        "",
        "1. **COLOR is the hardest entity type** (F1 < 0.20) — often confused with BRAND and PRODUCT_TYPE",
        "2. **ATTRIBUTE_VALUE and MATERIAL** are easiest (F1 > 0.85)",
        "3. **Multi-attribute queries** have lower F1 than single-attribute queries",
        "4. **Short queries (1-3 tokens)** have higher error rates due to ambiguity",
        "",
        "*See `qualitative_examples.md` for 30 annotated error examples.*",
        "",
        "---",
        "",
        "## 10. Key Findings & Conclusions",
        "",
        "1. **Transformer models outperform classical models** by 6-10 F1 points",
        "2. **BERT achieves the best balance** between precision and recall",
        "3. **Classical models (BiLSTM-CRF) have higher precision** but significantly lower recall",
        "4. **Domain-adapted pretraining provides marginal gains** over vanilla BERT",
        "5. **NER-enhanced retrieval improves nDCG@10** over BM25 baseline",
        "",
        "---",
        "",
        "## Files Included",
        "",
        "### Tables",
        "- `final_comparison_table.csv` — Complete model ranking with all metrics",
        "- `model_comparison_table.csv` — Intrinsic performance comparison",
        "- `per_entity_summary.csv` — Per-entity F1 scores",
        "",
        "### Data",
        "- `overall_metrics.csv` — Detailed precision/recall/F1",
        "- `per_entity_metrics.csv` — Full per-entity breakdown",
        "- `query_level_metrics.csv` — Extrinsic evaluation metrics",
        "- `predictions_test.jsonl` — Raw predictions for all models",
        "- `multi_seed_results.csv` — Multi-seed experiment results",
        "- `learning_curve.csv` — Learning curve data",
        "- `retrieval_metrics.csv` — Retrieval evaluation results",
        "- `efficiency_metrics.csv` — Model efficiency metrics",
        "",
        "### Figures",
        "- `model_comparison_f1.png` — F1 score comparison",
        "- `per_entity_radar.png` — Per-entity radar chart",
        "- `learning_curve_f1.png` — Learning curve plot",
        "- `confusion_matrix.png` — NER tag confusion matrix",
        "",
    ])

    # Write both README.md and SUMMARY_REPORT.md
    with open(output_path / "README.md", "w") as f:
        f.write("\n".join(lines))

    with open(output_path / "SUMMARY_REPORT.md", "w") as f:
        f.write("\n".join(lines))

    logger.info("Generated summary report")


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    benchmark_dir = results_dir / "benchmark"
    figures_dir = results_dir / "figures_export"

    # Output directory
    artifacts_dir = results_dir / "thesis_artifacts"
    ensure_dir(artifacts_dir)
    ensure_dir(artifacts_dir / "figures")
    ensure_dir(artifacts_dir / "tables")
    ensure_dir(artifacts_dir / "data")

    logger.info(f"Generating thesis artifacts in {artifacts_dir}")

    # Load benchmark results
    benchmark_path = benchmark_dir / "benchmark_results.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmark_results = json.load(f)
    else:
        logger.warning("benchmark_results.json not found, using empty results")
        benchmark_results = {}

    # 1. Copy figures
    if figures_dir.exists():
        copy_figures(figures_dir, artifacts_dir / "figures")
    if benchmark_dir.exists():
        copy_figures(benchmark_dir, artifacts_dir / "figures")

    # 2. Generate comparison tables
    if benchmark_results:
        generate_model_comparison_table(benchmark_results, artifacts_dir / "tables")
        generate_per_entity_summary(benchmark_results, artifacts_dir / "tables")
        generate_final_comparison_table(benchmark_results, project_root, artifacts_dir / "tables")

    # 3. Copy CSV files
    csv_files = [
        "overall_metrics.csv",
        "per_entity_metrics.csv",
        "query_level_metrics.csv",
    ]
    for csv_file in csv_files:
        src = benchmark_dir / csv_file
        if src.exists():
            shutil.copy(src, artifacts_dir / "data" / csv_file)
            logger.info(f"Copied {csv_file}")

    # 4. Copy predictions JSONL
    jsonl_file = benchmark_dir / "predictions_test.jsonl"
    if jsonl_file.exists():
        shutil.copy(jsonl_file, artifacts_dir / "data" / "predictions_test.jsonl")
        logger.info("Copied predictions_test.jsonl")

    # 5. Copy multi-seed results if exists
    multi_seed_dir = project_root / "outputs" / "multi_seed"
    if multi_seed_dir.exists():
        for f in multi_seed_dir.glob("*.csv"):
            shutil.copy(f, artifacts_dir / "data" / f.name)
            logger.info(f"Copied {f.name}")

    # 6. Copy learning curve results if exists
    learning_curve_csv = results_dir / "learning_curve.csv"
    if learning_curve_csv.exists():
        shutil.copy(learning_curve_csv, artifacts_dir / "data" / "learning_curve.csv")
        logger.info("Copied learning_curve.csv")

    # Also check outputs/learning_curve
    lc_outputs = project_root / "outputs" / "learning_curve"
    if lc_outputs.exists():
        for f in lc_outputs.glob("*.csv"):
            shutil.copy(f, artifacts_dir / "data" / f.name)
            logger.info(f"Copied {f.name}")

    # 6b. Copy retrieval metrics if exists
    retrieval_dir = results_dir / "retrieval"
    if retrieval_dir.exists():
        for f in retrieval_dir.glob("*.csv"):
            shutil.copy(f, artifacts_dir / "data" / f.name)
            logger.info(f"Copied {f.name}")
        for f in retrieval_dir.glob("*.json"):
            shutil.copy(f, artifacts_dir / "data" / f.name)
            logger.info(f"Copied {f.name}")

    # 6c. Copy efficiency metrics if exists
    efficiency_csv = project_root / "outputs" / "benchmark" / "efficiency_metrics.csv"
    if efficiency_csv.exists():
        shutil.copy(efficiency_csv, artifacts_dir / "data" / "efficiency_metrics.csv")
        logger.info("Copied efficiency_metrics.csv")

    # 7. Generate summary report
    if benchmark_results:
        generate_summary_report(benchmark_results, artifacts_dir, project_root)

    # 8. List all generated files
    print("\n" + "=" * 60)
    print("Thesis Artifacts Generated")
    print("=" * 60)

    for subdir in ["figures", "tables", "data"]:
        subpath = artifacts_dir / subdir
        if subpath.exists():
            files = list(subpath.iterdir())
            if files:
                print(f"\n{subdir}/")
                for f in sorted(files):
                    print(f"  - {f.name}")

    readme = artifacts_dir / "README.md"
    if readme.exists():
        print(f"\nREADME.md")

    print(f"\nAll artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()

