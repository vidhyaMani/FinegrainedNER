#!/usr/bin/env python
"""Generate LaTeX tables for gold standard evaluation results.

Usage:
    python scripts/generate_gold_latex.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).parent.parent

    # Load silver_on_gold results (the recommended approach)
    results_path = project_root / "results" / "silver_on_gold_v2_benchmark" / "benchmark_results.json"

    if not results_path.exists():
        print(f"Results not found at {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Output directory
    output_dir = project_root / "results" / "thesis_artifacts_gold"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate main comparison table
    rows = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name.replace("_", "-").title(),
            "Precision": f"{metrics['precision']:.4f}",
            "Recall": f"{metrics['recall']:.4f}",
            "F1": f"{metrics['f1']:.4f}",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("F1", ascending=False)

    # LaTeX table
    latex = r"""
\begin{table}[htbp]
\centering
\caption{NER Model Performance on Gold Standard Test Set}
\label{tab:gold-evaluation}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
\midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['Model']} & {row['Precision']} & {row['Recall']} & {row['F1']} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / "gold_evaluation_table.tex", "w") as f:
        f.write(latex)
    print(f"Saved: {output_dir / 'gold_evaluation_table.tex'}")

    # Per-entity table
    all_entities = set()
    for metrics in results.values():
        per_entity = metrics.get("per_entity_type", {})
        for entity in per_entity.keys():
            if entity not in ("micro avg", "macro avg", "weighted avg"):
                all_entities.add(entity)

    entity_rows = []
    for entity in sorted(all_entities):
        row = {"Entity": entity}
        for model_name, metrics in results.items():
            per_entity = metrics.get("per_entity_type", {})
            entity_metrics = per_entity.get(entity, {})
            f1 = entity_metrics.get("f1-score", 0)
            row[model_name] = f"{f1:.2f}"
        entity_rows.append(row)

    entity_df = pd.DataFrame(entity_rows)

    # Per-entity LaTeX
    entity_latex = r"""
\begin{table}[htbp]
\centering
\caption{Per-Entity F1 Scores on Gold Standard Test Set}
\label{tab:gold-per-entity}
\begin{tabular}{l""" + "c" * len(results) + r"""}
\toprule
\textbf{Entity Type} & """ + " & ".join([f"\\textbf{{{m.replace('_', '-')}}}" for m in results.keys()]) + r""" \\
\midrule
"""

    for _, row in entity_df.iterrows():
        entity_latex += row['Entity']
        for model_name in results.keys():
            entity_latex += f" & {row.get(model_name, '0.00')}"
        entity_latex += " \\\\\n"

    entity_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

    with open(output_dir / "gold_per_entity_table.tex", "w") as f:
        f.write(entity_latex)
    print(f"Saved: {output_dir / 'gold_per_entity_table.tex'}")

    # Summary comparison (gold-trained vs silver-trained)
    gold_results_path = project_root / "results" / "gold_v2_benchmark" / "benchmark_results.json"
    if not gold_results_path.exists():
        # Fall back to original gold benchmark
        gold_results_path = project_root / "results" / "gold_benchmark" / "benchmark_results.json"
    if gold_results_path.exists():
        with open(gold_results_path) as f:
            gold_results = json.load(f)

        comparison_latex = r"""
\begin{table}[htbp]
\centering
\caption{Comparison: Gold-Trained vs Silver-Trained Models (Evaluated on Gold Test Set, n=51)}
\label{tab:gold-vs-silver-training}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \multicolumn{2}{c}{\textbf{Gold-Trained (849)}} & \multicolumn{2}{c}{\textbf{Silver-Trained (52.5K)}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
& Precision & F1 & Precision & F1 \\
\midrule
"""

        model_mapping = {
            "bert_ner": "gold_bert_ner",
            "roberta_ner": "gold_roberta_ner",
            "bilstm_crf": "gold_bilstm_crf",
            "cnn_bilstm": "gold_cnn_bilstm",
        }

        for silver_name, gold_name in model_mapping.items():
            silver_metrics = results.get(silver_name, {})
            gold_metrics = gold_results.get(gold_name, {})

            display_name = silver_name.replace("_", "-").title()
            sp = f"{silver_metrics.get('precision', 0):.4f}"
            sf = f"{silver_metrics.get('f1', 0):.4f}"
            gp = f"{gold_metrics.get('precision', 0):.4f}"
            gf = f"{gold_metrics.get('f1', 0):.4f}"

            comparison_latex += f"{display_name} & {gp} & {gf} & {sp} & {sf} \\\\\n"

        comparison_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

        with open(output_dir / "gold_vs_silver_training_table.tex", "w") as f:
            f.write(comparison_latex)
        print(f"Saved: {output_dir / 'gold_vs_silver_training_table.tex'}")

    # Copy figures
    import shutil
    src_dir = project_root / "results" / "silver_on_gold_v2_benchmark"
    for fig in src_dir.glob("*.png"):
        shutil.copy(fig, output_dir / fig.name)
        print(f"Copied: {fig.name}")

    print(f"\nAll artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()

