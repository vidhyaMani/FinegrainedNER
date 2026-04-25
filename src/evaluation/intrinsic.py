"""Intrinsic evaluation metrics for NER.

Computes entity-level precision, recall, F1 using seqeval,
with support for per-entity-type breakdown and strict/partial matching.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def collapse_to_coarse(labels: List[List[str]]) -> List[List[str]]:
    """Collapse fine-grained IOB2 labels to coarse-level (PRODUCT only).

    Args:
        labels: List of fine-grained IOB2 tag sequences.

    Returns:
        List of coarse IOB2 tag sequences (O, B-PRODUCT, I-PRODUCT).
    """
    coarse_labels = []
    for seq in labels:
        coarse_seq = []
        for tag in seq:
            if tag == "O":
                coarse_seq.append("O")
            else:
                prefix, _ = tag.split("-", 1)
                coarse_seq.append(f"{prefix}-PRODUCT")
        coarse_labels.append(coarse_seq)
    return coarse_labels


def compute_coarse_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    mode: str = "strict",
) -> Dict[str, Any]:
    """Compute coarse-level NER metrics by collapsing fine labels to PRODUCT.

    This demonstrates hierarchical fallback: even if the model confuses BRAND
    with COLOR, it may still correctly identify the span as a PRODUCT entity.

    Args:
        true_labels: List of true fine-grained IOB2 tag sequences.
        pred_labels: List of predicted fine-grained IOB2 tag sequences.
        mode: 'strict' for exact boundary match.

    Returns:
        Dict with coarse-level precision, recall, f1.
    """
    true_coarse = collapse_to_coarse(true_labels)
    pred_coarse = collapse_to_coarse(pred_labels)

    metrics = compute_ner_metrics(true_coarse, pred_coarse, mode=mode)
    metrics["level"] = "coarse"

    logger.info("Coarse-level (PRODUCT) metrics:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1:        {metrics['f1']:.4f}")

    return metrics


def compute_ner_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    mode: str = "strict",
    scheme: type = IOB2,
) -> Dict[str, Any]:
    """Compute entity-level NER metrics using seqeval.

    Args:
        true_labels: List of true IOB2 tag sequences.
        pred_labels: List of predicted IOB2 tag sequences.
        mode: 'strict' for exact boundary+type match, 'default' for lenient.
        scheme: Tagging scheme (IOB2).

    Returns:
        Dict with precision, recall, f1, and per-entity-type breakdown.
    """
    # Validate sequence lengths match
    assert len(true_labels) == len(pred_labels), (
        f"Sequence count mismatch: true={len(true_labels)}, pred={len(pred_labels)}"
    )
    for i, (true_seq, pred_seq) in enumerate(zip(true_labels, pred_labels)):
        if len(true_seq) != len(pred_seq):
            logger.warning(
                f"Sequence {i} length mismatch: true={len(true_seq)}, pred={len(pred_seq)}. "
                f"Truncating to min length."
            )
            min_len = min(len(true_seq), len(pred_seq))
            true_labels[i] = true_seq[:min_len]
            pred_labels[i] = pred_seq[:min_len]

    # Overall metrics
    use_strict = mode == "strict"
    metrics = {
        "precision": precision_score(
            true_labels, pred_labels, mode="strict" if use_strict else None, scheme=scheme,
        ),
        "recall": recall_score(
            true_labels, pred_labels, mode="strict" if use_strict else None, scheme=scheme,
        ),
        "f1": f1_score(
            true_labels, pred_labels, mode="strict" if use_strict else None, scheme=scheme,
        ),
    }

    # Detailed classification report
    report = classification_report(
        true_labels, pred_labels, mode="strict" if use_strict else None,
        scheme=scheme, output_dict=True,
    )
    metrics["per_entity_type"] = report

    # Pretty-printed report for logging
    report_str = classification_report(
        true_labels, pred_labels, mode="strict" if use_strict else None, scheme=scheme,
    )
    metrics["report_str"] = report_str

    logger.info(f"NER Metrics (mode={mode}):")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1:        {metrics['f1']:.4f}")

    return metrics


def compute_metrics_per_entity(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    entity_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-entity-type metrics.

    Args:
        true_labels: True IOB2 tag sequences.
        pred_labels: Predicted IOB2 tag sequences.
        entity_types: List of entity types to report on (optional).

    Returns:
        Dict mapping entity type → {precision, recall, f1, support}.
    """
    report = classification_report(
        true_labels, pred_labels, output_dict=True, scheme=IOB2, mode="strict",
    )

    results = {}
    for entity_type, metrics in report.items():
        if entity_type in ("micro avg", "macro avg", "weighted avg"):
            results[entity_type] = metrics
        elif entity_types is None or entity_type in entity_types:
            results[entity_type] = metrics

    return results


def compare_models(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Generate a comparison table of multiple models' NER metrics.

    Args:
        results: Dict mapping model name → metrics dict from compute_ner_metrics.

    Returns:
        Formatted comparison string.
    """
    header = f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    separator = "-" * len(header)
    lines = [header, separator]

    for model_name, metrics in sorted(results.items()):
        line = (
            f"{model_name:<30} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )
        lines.append(line)

    table = "\n".join(lines)
    logger.info(f"\nModel Comparison:\n{table}")
    return table

