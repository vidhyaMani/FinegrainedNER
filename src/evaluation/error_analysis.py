"""Error analysis utilities for NER evaluation.

Provides confusion matrices, common error patterns, and breakdown
analyses by query length, entity rarity, and entity type.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def build_tag_confusion_matrix(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    labels: Optional[List[str]] = None,
    normalize: str = "true",
) -> Tuple[np.ndarray, List[str]]:
    """Build a confusion matrix over IOB2 tags.

    Args:
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.
        labels: Ordered list of label names. If None, auto-derived.
        normalize: 'true' (recall), 'pred' (precision), 'all', or None.

    Returns:
        Tuple of (confusion_matrix, label_names).
    """
    flat_true = [t for seq in true_labels for t in seq]
    flat_pred = [t for seq in pred_labels for t in seq]

    if labels is None:
        labels = sorted(set(flat_true + flat_pred))

    cm = confusion_matrix(flat_true, flat_pred, labels=labels, normalize=normalize)
    return cm, labels


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "NER Tag Confusion Matrix",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """Plot a heatmap confusion matrix.

    Args:
        cm: Confusion matrix array.
        labels: List of label names.
        title: Plot title.
        output_path: Path to save the figure (optional).
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved confusion matrix to {output_path}")
    plt.close()


def analyze_common_errors(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, int]]]:
    """Analyze the most common error patterns.

    Args:
        tokens_list: List of token sequences.
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.
        top_n: Number of top error patterns to return.

    Returns:
        Dict with 'false_positives', 'false_negatives', 'type_confusions'.
    """
    false_positives = Counter()   # Model predicted entity but gold is O
    false_negatives = Counter()   # Gold is entity but model predicted O
    type_confusions = Counter()   # Model predicted wrong entity type

    for tokens, true_seq, pred_seq in zip(tokens_list, true_labels, pred_labels):
        for token, true_tag, pred_tag in zip(tokens, true_seq, pred_seq):
            if true_tag == pred_tag:
                continue

            if true_tag == "O" and pred_tag != "O":
                false_positives[(token, pred_tag)] += 1
            elif true_tag != "O" and pred_tag == "O":
                false_negatives[(token, true_tag)] += 1
            elif true_tag != "O" and pred_tag != "O":
                type_confusions[(token, true_tag, pred_tag)] += 1

    return {
        "false_positives": false_positives.most_common(top_n),
        "false_negatives": false_negatives.most_common(top_n),
        "type_confusions": type_confusions.most_common(top_n),
    }


def analyze_by_query_length(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    bins: List[int] = (1, 3, 5, 8, 12, 20),
) -> Dict[str, Dict[str, float]]:
    """Analyze NER performance by query length.

    Args:
        tokens_list: List of token sequences.
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.
        bins: Length bins for grouping.

    Returns:
        Dict mapping length range → {precision, recall, f1, count}.
    """
    from seqeval.metrics import f1_score, precision_score, recall_score

    length_groups: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))

    for tokens, true_seq, pred_seq in zip(tokens_list, true_labels, pred_labels):
        length = len(tokens)
        # Find the bin
        bin_label = f"{bins[-1]}+"
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                bin_label = f"{bins[i]}-{bins[i + 1] - 1}"
                break

        length_groups[bin_label][0].append(true_seq)
        length_groups[bin_label][1].append(pred_seq)

    results = {}
    for bin_label, (true_seqs, pred_seqs) in sorted(length_groups.items()):
        results[bin_label] = {
            "precision": precision_score(true_seqs, pred_seqs),
            "recall": recall_score(true_seqs, pred_seqs),
            "f1": f1_score(true_seqs, pred_seqs),
            "count": len(true_seqs),
        }

    return results


def analyze_by_entity_rarity(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    rarity_thresholds: Dict[str, int] = None,
) -> Dict[str, Dict[str, float]]:
    """Analyze performance on rare vs. common entities.

    Args:
        tokens_list: List of token sequences.
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.
        rarity_thresholds: Dict with 'rare' and 'common' count thresholds.

    Returns:
        Dict with metrics for 'rare' and 'common' entities.
    """
    if rarity_thresholds is None:
        rarity_thresholds = {"rare": 5, "common": 50}

    # Count entity occurrences
    entity_counts = Counter()
    for true_seq in true_labels:
        for tag in true_seq:
            if tag.startswith("B-"):
                entity_counts[tag[2:]] += 1

    rare_types = {et for et, count in entity_counts.items()
                  if count <= rarity_thresholds["rare"]}
    common_types = {et for et, count in entity_counts.items()
                    if count >= rarity_thresholds["common"]}

    logger.info(f"Rare entity types (≤{rarity_thresholds['rare']}): {rare_types}")
    logger.info(f"Common entity types (≥{rarity_thresholds['common']}): {common_types}")

    results = {
        "entity_counts": dict(entity_counts.most_common()),
        "rare_types": list(rare_types),
        "common_types": list(common_types),
    }
    return results


def generate_error_report(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive error analysis report.

    Args:
        tokens_list: List of token sequences.
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.
        output_dir: Directory to save plots and report (optional).

    Returns:
        Dict with all analysis results.
    """
    from pathlib import Path

    report = {}

    # 1. Confusion matrix
    cm, labels = build_tag_confusion_matrix(true_labels, pred_labels)
    report["confusion_matrix"] = {"matrix": cm.tolist(), "labels": labels}
    if output_dir:
        plot_confusion_matrix(
            cm, labels,
            output_path=str(Path(output_dir) / "confusion_matrix.png"),
        )

    # 2. Common errors
    report["common_errors"] = analyze_common_errors(tokens_list, true_labels, pred_labels)

    # 3. By query length
    report["by_query_length"] = analyze_by_query_length(tokens_list, true_labels, pred_labels)

    # 4. By entity rarity
    report["by_entity_rarity"] = analyze_by_entity_rarity(tokens_list, true_labels, pred_labels)

    # 5. FP/FN counts by entity type
    report["fp_fn_by_entity"] = compute_fp_fn_by_entity(true_labels, pred_labels)

    # 6. Multi-attribute query analysis
    report["multi_attribute_analysis"] = analyze_multi_attribute_queries(
        tokens_list, true_labels, pred_labels
    )

    # 7. COLOR ambiguity analysis
    report["color_ambiguity"] = analyze_color_ambiguity(tokens_list, true_labels, pred_labels)

    logger.info("Error analysis report generated.")
    return report


def compute_fp_fn_by_entity(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, Dict[str, int]]:
    """Compute false positive and false negative counts by entity type.

    Args:
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.

    Returns:
        Dict mapping entity_type -> {fp: count, fn: count, tp: count}.
    """
    entity_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"fp": 0, "fn": 0, "tp": 0})

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            true_type = true_tag[2:] if true_tag != "O" else None
            pred_type = pred_tag[2:] if pred_tag != "O" else None

            if true_type and pred_type:
                if true_type == pred_type:
                    entity_stats[true_type]["tp"] += 1
                else:
                    entity_stats[true_type]["fn"] += 1
                    entity_stats[pred_type]["fp"] += 1
            elif true_type and not pred_type:
                entity_stats[true_type]["fn"] += 1
            elif pred_type and not true_type:
                entity_stats[pred_type]["fp"] += 1

    return dict(entity_stats)


def analyze_multi_attribute_queries(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, Any]:
    """Analyze performance on queries with multiple attributes.

    Args:
        tokens_list: List of token sequences.
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.

    Returns:
        Dict with metrics for single-attr vs multi-attr queries.
    """
    from seqeval.metrics import f1_score, precision_score, recall_score

    single_true, single_pred = [], []
    multi_true, multi_pred = [], []

    for tokens, true_seq, pred_seq in zip(tokens_list, true_labels, pred_labels):
        # Count unique entity types in gold
        entity_types = set()
        for tag in true_seq:
            if tag.startswith("B-"):
                entity_types.add(tag[2:])

        if len(entity_types) <= 1:
            single_true.append(true_seq)
            single_pred.append(pred_seq)
        else:
            multi_true.append(true_seq)
            multi_pred.append(pred_seq)

    results = {
        "single_attribute": {
            "count": len(single_true),
            "precision": precision_score(single_true, single_pred) if single_true else 0,
            "recall": recall_score(single_true, single_pred) if single_true else 0,
            "f1": f1_score(single_true, single_pred) if single_true else 0,
        },
        "multi_attribute": {
            "count": len(multi_true),
            "precision": precision_score(multi_true, multi_pred) if multi_true else 0,
            "recall": recall_score(multi_true, multi_pred) if multi_true else 0,
            "f1": f1_score(multi_true, multi_pred) if multi_true else 0,
        },
    }

    logger.info(
        f"Multi-attr analysis: single={len(single_true)} queries (F1={results['single_attribute']['f1']:.4f}), "
        f"multi={len(multi_true)} queries (F1={results['multi_attribute']['f1']:.4f})"
    )

    return results


def analyze_color_ambiguity(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, Any]:
    """Analyze COLOR entity ambiguity and common errors.

    COLOR is often confused with BRAND (e.g., "blue" in "blue ribbon")
    or PRODUCT_TYPE (e.g., "orange" as fruit vs color).

    Args:
        tokens_list: List of token sequences.
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.

    Returns:
        Dict with COLOR-specific error analysis.
    """
    color_confusions = Counter()  # What COLOR gets confused with
    color_missed = Counter()      # COLOR tokens that were missed (predicted O)
    color_hallucinated = Counter()  # Tokens wrongly predicted as COLOR

    color_tp = 0
    color_fp = 0
    color_fn = 0

    for tokens, true_seq, pred_seq in zip(tokens_list, true_labels, pred_labels):
        for token, true_tag, pred_tag in zip(tokens, true_seq, pred_seq):
            true_is_color = "COLOR" in true_tag
            pred_is_color = "COLOR" in pred_tag

            if true_is_color and pred_is_color:
                color_tp += 1
            elif true_is_color and not pred_is_color:
                color_fn += 1
                if pred_tag == "O":
                    color_missed[token.lower()] += 1
                else:
                    color_confusions[(token.lower(), pred_tag)] += 1
            elif pred_is_color and not true_is_color:
                color_fp += 1
                color_hallucinated[(token.lower(), true_tag)] += 1

    precision = color_tp / (color_tp + color_fp) if (color_tp + color_fp) > 0 else 0
    recall = color_tp / (color_tp + color_fn) if (color_tp + color_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results = {
        "color_metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": color_tp,
            "fp": color_fp,
            "fn": color_fn,
        },
        "color_confusions": color_confusions.most_common(20),
        "color_missed": color_missed.most_common(20),
        "color_hallucinated": color_hallucinated.most_common(20),
    }

    logger.info(
        f"COLOR analysis: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}, "
        f"FP={color_fp}, FN={color_fn}"
    )

    return results


def extract_qualitative_examples(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    num_examples: int = 30,
    include_correct: bool = True,
) -> List[Dict[str, Any]]:
    """Extract qualitative examples for manual inspection.

    Selects diverse examples including:
    - Correct predictions
    - False positives
    - False negatives
    - Type confusions

    Args:
        tokens_list: List of token sequences.
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.
        num_examples: Total number of examples to extract.
        include_correct: Whether to include correct predictions.

    Returns:
        List of example dicts with tokens, gold, pred, and error type.
    """
    examples = []
    correct_examples = []
    error_examples = []

    for i, (tokens, true_seq, pred_seq) in enumerate(zip(tokens_list, true_labels, pred_labels)):
        query = " ".join(tokens)

        # Determine error type
        has_fp = any(t == "O" and p != "O" for t, p in zip(true_seq, pred_seq))
        has_fn = any(t != "O" and p == "O" for t, p in zip(true_seq, pred_seq))
        has_confusion = any(
            t != "O" and p != "O" and t != p
            for t, p in zip(true_seq, pred_seq)
        )
        is_correct = true_seq == pred_seq

        example = {
            "index": i,
            "query": query,
            "tokens": tokens,
            "gold_tags": true_seq,
            "pred_tags": pred_seq,
            "is_correct": is_correct,
            "has_fp": has_fp,
            "has_fn": has_fn,
            "has_type_confusion": has_confusion,
        }

        if is_correct:
            correct_examples.append(example)
        else:
            example["error_types"] = []
            if has_fp:
                example["error_types"].append("FP")
            if has_fn:
                example["error_types"].append("FN")
            if has_confusion:
                example["error_types"].append("TYPE_CONFUSION")
            error_examples.append(example)

    # Select diverse examples
    import random
    random.seed(42)

    n_errors = num_examples - (num_examples // 5 if include_correct else 0)
    n_correct = num_examples // 5 if include_correct else 0

    # Prioritize diverse error types
    fp_examples = [e for e in error_examples if "FP" in e.get("error_types", [])]
    fn_examples = [e for e in error_examples if "FN" in e.get("error_types", [])]
    confusion_examples = [e for e in error_examples if "TYPE_CONFUSION" in e.get("error_types", [])]

    selected = []
    for error_list in [fp_examples, fn_examples, confusion_examples]:
        random.shuffle(error_list)
        selected.extend(error_list[:n_errors // 3])

    # Fill remaining with random errors
    remaining = [e for e in error_examples if e not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[:n_errors - len(selected)])

    # Add correct examples
    if include_correct:
        random.shuffle(correct_examples)
        selected.extend(correct_examples[:n_correct])

    return selected[:num_examples]


def export_qualitative_examples_markdown(
    examples: List[Dict[str, Any]],
    output_path: str,
    model_name: str = "Model",
):
    """Export qualitative examples to a markdown file.

    Args:
        examples: List of example dicts from extract_qualitative_examples.
        output_path: Path to save markdown file.
        model_name: Name of the model for the report.
    """
    lines = [
        f"# Qualitative Error Analysis: {model_name}",
        "",
        f"**Total Examples:** {len(examples)}",
        "",
        "---",
        "",
    ]

    for i, ex in enumerate(examples, 1):
        status = "✅ CORRECT" if ex["is_correct"] else "❌ ERROR"
        error_types = ", ".join(ex.get("error_types", [])) or "N/A"

        lines.extend([
            f"## Example {i}: {status}",
            "",
            f"**Query:** `{ex['query']}`",
            "",
            f"**Error Types:** {error_types}",
            "",
            "| Token | Gold | Predicted |",
            "|-------|------|-----------|",
        ])

        for token, gold, pred in zip(ex["tokens"], ex["gold_tags"], ex["pred_tags"]):
            match_marker = "" if gold == pred else " ⚠️"
            lines.append(f"| {token} | {gold} | {pred}{match_marker} |")

        lines.extend(["", "---", ""])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Exported {len(examples)} qualitative examples to {output_path}")


def analyze_hierarchical_sparsity(
    tokens_list: List[List[str]],
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> Dict[str, Any]:
    """Analyze how hierarchical schema helps with sparse/rare entity types.

    This function demonstrates that:
    1. Rare entity types have lower fine-grained F1
    2. When collapsed to coarse level, recall improves significantly
    3. The hierarchy provides graceful degradation for rare types

    Args:
        tokens_list: List of token sequences.
        true_labels: True fine-grained IOB2 tag sequences.
        pred_labels: Predicted fine-grained IOB2 tag sequences.

    Returns:
        Dict with per-entity-type analysis of hierarchical benefit.
    """
    from src.evaluation.intrinsic import collapse_to_coarse

    # Count entity occurrences
    entity_counts = Counter()
    for true_seq in true_labels:
        for tag in true_seq:
            if tag.startswith("B-"):
                entity_counts[tag[2:]] += 1

    # Compute per-entity fine-grained metrics
    entity_fine_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            if true_tag.startswith("B-") or true_tag.startswith("I-"):
                true_type = true_tag[2:]
                if pred_tag == true_tag:
                    entity_fine_metrics[true_type]["tp"] += 1
                else:
                    entity_fine_metrics[true_type]["fn"] += 1

            if pred_tag.startswith("B-") or pred_tag.startswith("I-"):
                pred_type = pred_tag[2:]
                if true_tag != pred_tag:
                    entity_fine_metrics[pred_type]["fp"] += 1

    # Collapse to coarse and compute coarse metrics
    true_coarse = collapse_to_coarse(true_labels)
    pred_coarse = collapse_to_coarse(pred_labels)

    coarse_metrics = {"tp": 0, "fp": 0, "fn": 0}
    for true_seq, pred_seq in zip(true_coarse, pred_coarse):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            if true_tag.startswith("B-") or true_tag.startswith("I-"):
                if pred_tag == true_tag:
                    coarse_metrics["tp"] += 1
                else:
                    coarse_metrics["fn"] += 1
            if pred_tag.startswith("B-") or pred_tag.startswith("I-"):
                if true_tag != pred_tag:
                    coarse_metrics["fp"] += 1

    def compute_prf(m):
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return {"precision": p, "recall": r, "f1": f}

    # Build results
    results = {
        "entity_counts": dict(entity_counts.most_common()),
        "per_entity_fine_metrics": {},
        "coarse_metrics": compute_prf(coarse_metrics),
    }

    for entity_type, counts in entity_counts.most_common():
        fine_m = entity_fine_metrics[entity_type]
        fine_prf = compute_prf(fine_m)
        results["per_entity_fine_metrics"][entity_type] = {
            "count": counts,
            "fine": fine_prf,
            "is_rare": counts < 1000,  # Threshold for rare
        }

    # Compute average gain for rare vs common
    rare_fine_f1 = []
    common_fine_f1 = []

    for entity_type, data in results["per_entity_fine_metrics"].items():
        if data["is_rare"]:
            rare_fine_f1.append(data["fine"]["f1"])
        else:
            common_fine_f1.append(data["fine"]["f1"])

    results["summary"] = {
        "avg_rare_fine_f1": sum(rare_fine_f1) / len(rare_fine_f1) if rare_fine_f1 else 0,
        "avg_common_fine_f1": sum(common_fine_f1) / len(common_fine_f1) if common_fine_f1 else 0,
        "coarse_f1": results["coarse_metrics"]["f1"],
        "rare_types": [et for et, d in results["per_entity_fine_metrics"].items() if d["is_rare"]],
        "common_types": [et for et, d in results["per_entity_fine_metrics"].items() if not d["is_rare"]],
    }

    logger.info("Hierarchical sparsity analysis:")
    logger.info(f"  Avg rare entity fine F1: {results['summary']['avg_rare_fine_f1']:.4f}")
    logger.info(f"  Avg common entity fine F1: {results['summary']['avg_common_fine_f1']:.4f}")
    logger.info(f"  Coarse-level F1: {results['summary']['coarse_f1']:.4f}")

    return results
