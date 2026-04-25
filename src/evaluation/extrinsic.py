"""Extrinsic evaluation metrics for NER in e-commerce.

Measures downstream utility of NER predictions:
- Query understanding accuracy (structured attribute extraction)
- Attribute coverage gain
- Reformulation reduction estimation
- Slot-level precision/recall/F1
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def extract_attributes(
    tokens: List[str],
    tags: List[str],
) -> Dict[str, List[str]]:
    """Extract structured attributes from IOB2-tagged tokens.

    Example:
        tokens = ["nike", "red", "running", "shoes"]
        tags   = ["B-BRAND", "B-COLOR", "B-PRODUCT_TYPE", "I-PRODUCT_TYPE"]
        → {"BRAND": ["nike"], "COLOR": ["red"], "PRODUCT_TYPE": ["running shoes"]}

    Args:
        tokens: List of query tokens.
        tags: List of IOB2 tags.

    Returns:
        Dict mapping entity type → list of extracted entity strings.
    """
    attributes: Dict[str, List[str]] = {}
    current_entity = None
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            # Save previous entity if any
            if current_entity and current_tokens:
                attributes.setdefault(current_entity, []).append(" ".join(current_tokens))
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity:
            entity_type = tag[2:]
            if entity_type == current_entity:
                current_tokens.append(token)
            else:
                # Type mismatch — save previous and start new
                if current_tokens:
                    attributes.setdefault(current_entity, []).append(" ".join(current_tokens))
                current_entity = entity_type
                current_tokens = [token]
        else:
            # O tag — save previous entity
            if current_entity and current_tokens:
                attributes.setdefault(current_entity, []).append(" ".join(current_tokens))
            current_entity = None
            current_tokens = []

    # Don't forget the last entity
    if current_entity and current_tokens:
        attributes.setdefault(current_entity, []).append(" ".join(current_tokens))

    return attributes


def extract_slots(
    tokens: List[str],
    tags: List[str],
) -> set:
    """Extract normalized (entity_type, entity_value) slots from IOB2-tagged tokens.

    Args:
        tokens: List of query tokens.
        tags: List of IOB2 tags.

    Returns:
        Set of (entity_type, normalized_value) tuples.
    """
    attrs = extract_attributes(tokens, tags)
    slots = set()
    for entity_type, values in attrs.items():
        for val in values:
            # Normalize: lowercase, strip whitespace
            normalized_val = val.lower().strip()
            slots.add((entity_type, normalized_val))
    return slots


def slot_precision_recall_f1(
    tokens_list: List[List[str]],
    pred_tags_list: List[List[str]],
    gold_tags_list: List[List[str]],
) -> Dict[str, float]:
    """Compute slot-level precision, recall, and F1.

    Treats each (entity_type, entity_value) pair as a slot.
    Computes micro-averaged P/R/F1 over all slots across all queries.

    Args:
        tokens_list: List of token sequences.
        pred_tags_list: Predicted IOB2 tag sequences.
        gold_tags_list: Gold IOB2 tag sequences.

    Returns:
        Dict with slot_precision, slot_recall, slot_f1.
    """
    total_pred_slots = 0
    total_gold_slots = 0
    total_correct_slots = 0

    for tokens, pred_tags, gold_tags in zip(tokens_list, pred_tags_list, gold_tags_list):
        pred_slots = extract_slots(tokens, pred_tags)
        gold_slots = extract_slots(tokens, gold_tags)

        total_pred_slots += len(pred_slots)
        total_gold_slots += len(gold_slots)
        total_correct_slots += len(pred_slots & gold_slots)

    precision = total_correct_slots / total_pred_slots if total_pred_slots > 0 else 0
    recall = total_correct_slots / total_gold_slots if total_gold_slots > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "slot_precision": precision,
        "slot_recall": recall,
        "slot_f1": f1,
        "total_pred_slots": total_pred_slots,
        "total_gold_slots": total_gold_slots,
        "total_correct_slots": total_correct_slots,
    }

    logger.info(
        f"Slot-level metrics: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
    )
    return metrics


def query_understanding_accuracy(
    tokens_list: List[List[str]],
    pred_tags_list: List[List[str]],
    gold_tags_list: List[List[str]],
) -> Dict[str, float]:
    """Compute query understanding accuracy.

    Measures how often the predicted structured attributes exactly match
    the gold attributes.

    Args:
        tokens_list: List of token sequences.
        pred_tags_list: Predicted IOB2 tag sequences.
        gold_tags_list: Gold IOB2 tag sequences.

    Returns:
        Dict with exact_match_accuracy and partial_match_accuracy.
    """
    exact_matches = 0
    partial_matches = 0
    total = len(tokens_list)

    for tokens, pred_tags, gold_tags in zip(tokens_list, pred_tags_list, gold_tags_list):
        pred_attrs = extract_attributes(tokens, pred_tags)
        gold_attrs = extract_attributes(tokens, gold_tags)

        if pred_attrs == gold_attrs:
            exact_matches += 1

        # Partial match: check each entity type separately
        if gold_attrs:
            matching_types = sum(
                1 for et in gold_attrs
                if et in pred_attrs and set(pred_attrs[et]) == set(gold_attrs[et])
            )
            if matching_types > 0:
                partial_matches += 1

    metrics = {
        "exact_match_accuracy": exact_matches / total if total > 0 else 0,
        "partial_match_accuracy": partial_matches / total if total > 0 else 0,
        "total_queries": total,
        "exact_matches": exact_matches,
    }

    logger.info(
        f"Query Understanding: exact_match={metrics['exact_match_accuracy']:.4f}, "
        f"partial_match={metrics['partial_match_accuracy']:.4f}"
    )
    return metrics


def attribute_coverage(
    tokens_list: List[List[str]],
    pred_tags_list: List[List[str]],
    gold_tags_list: List[List[str]],
) -> Dict[str, float]:
    """Compute attribute coverage gain.

    Measures what fraction of expected attributes are recognized by the model.

    Coverage = (attributes_recognized / attributes_expected) × 100

    Note on Attribute Coverage vs Recall:
        This metric is mathematically equivalent to entity-level recall computed
        over normalized (entity_type, entity_value) pairs. The metric is retained
        under the name "attribute coverage" for interpretability in the e-commerce
        domain context, where "coverage" better describes the business goal of
        capturing all product attributes in a query.

    Args:
        tokens_list: List of token sequences.
        pred_tags_list: Predicted IOB2 tag sequences.
        gold_tags_list: Gold IOB2 tag sequences.

    Returns:
        Dict with overall and per-entity-type coverage.
    """
    total_expected = 0
    total_recognized = 0
    type_stats: Dict[str, Dict[str, int]] = {}

    for tokens, pred_tags, gold_tags in zip(tokens_list, pred_tags_list, gold_tags_list):
        pred_attrs = extract_attributes(tokens, pred_tags)
        gold_attrs = extract_attributes(tokens, gold_tags)

        for entity_type, gold_values in gold_attrs.items():
            if entity_type not in type_stats:
                type_stats[entity_type] = {"expected": 0, "recognized": 0}

            for val in gold_values:
                total_expected += 1
                type_stats[entity_type]["expected"] += 1

                pred_values = pred_attrs.get(entity_type, [])
                if val.lower() in [v.lower() for v in pred_values]:
                    total_recognized += 1
                    type_stats[entity_type]["recognized"] += 1

    overall_coverage = (total_recognized / total_expected * 100) if total_expected > 0 else 0

    per_type_coverage = {}
    for et, stats in type_stats.items():
        coverage = (stats["recognized"] / stats["expected"] * 100) if stats["expected"] > 0 else 0
        per_type_coverage[et] = {
            "coverage": coverage,
            "expected": stats["expected"],
            "recognized": stats["recognized"],
        }

    metrics = {
        "overall_coverage": overall_coverage,
        "total_expected": total_expected,
        "total_recognized": total_recognized,
        "per_type_coverage": per_type_coverage,
    }

    logger.info(
        f"Attribute Coverage: {overall_coverage:.1f}% "
        f"({total_recognized}/{total_expected})"
    )
    return metrics


def reformulation_reduction(
    tokens_list: List[List[str]],
    pred_tags_list: List[List[str]],
    gold_metadata: List[Dict[str, str]],
) -> Dict[str, float]:
    """Estimate query reformulation reduction.

    Simulates how often NER output matches ground-truth product metadata
    fields without needing query expansion/reformulation.

    A query "needs reformulation" if the NER model fails to extract
    at least one attribute that exists in the gold metadata.

    Args:
        tokens_list: List of token sequences.
        pred_tags_list: Predicted IOB2 tag sequences.
        gold_metadata: List of dicts with ground-truth product metadata
                       (e.g., {"brand": "nike", "color": "red"}).

    Returns:
        Dict with reformulation reduction metrics.
    """
    needs_reformulation = 0
    total_with_metadata = 0

    for tokens, pred_tags, metadata in zip(tokens_list, pred_tags_list, gold_metadata):
        if not metadata:
            continue

        total_with_metadata += 1
        pred_attrs = extract_attributes(tokens, pred_tags)

        # Check if all metadata attributes are covered by NER
        all_covered = True
        for meta_key, meta_value in metadata.items():
            # Map metadata keys to entity types
            key_to_entity = {
                "brand": "BRAND",
                "color": "COLOR",
                "product_type": "PRODUCT_TYPE",
                "material": "MATERIAL",
                "size": "SIZE_MEASURE",
            }
            entity_type = key_to_entity.get(meta_key.lower())
            if entity_type is None:
                continue

            pred_values = pred_attrs.get(entity_type, [])
            if not any(meta_value.lower() in v.lower() for v in pred_values):
                all_covered = False
                break

        if not all_covered:
            needs_reformulation += 1

    no_reformulation = total_with_metadata - needs_reformulation
    reduction_rate = (no_reformulation / total_with_metadata * 100) if total_with_metadata > 0 else 0

    metrics = {
        "reformulation_reduction_rate": reduction_rate,
        "queries_needing_reformulation": needs_reformulation,
        "queries_not_needing_reformulation": no_reformulation,
        "total_queries_with_metadata": total_with_metadata,
    }

    logger.info(
        f"Reformulation Reduction: {reduction_rate:.1f}% queries don't need reformulation "
        f"({no_reformulation}/{total_with_metadata})"
    )
    return metrics

