"""Annotation quality assurance utilities.

Provides tools for sampling annotated queries for manual review,
computing inter-annotator agreement, and generating QA reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger(__name__)


def sample_for_review(
    df: pd.DataFrame,
    n: int = 1000,
    seed: int = 42,
    stratify_by_annotation: bool = True,
) -> pd.DataFrame:
    """Sample queries for manual QA review.

    Ensures a mix of annotated and unannotated queries for balanced review.
    """
    if stratify_by_annotation and "ner_tags" in df.columns:
        has_entity = df["ner_tags"].apply(
            lambda tags: any(
                t != "O" for t in (tags.tolist() if isinstance(tags, np.ndarray) else tags)
            )
        )
        pos = df[has_entity]
        neg = df[~has_entity]

        n_pos = min(n // 2, len(pos))
        n_neg = min(n - n_pos, len(neg))

        sample = pd.concat([
            pos.sample(n=n_pos, random_state=seed),
            neg.sample(n=n_neg, random_state=seed),
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    logger.info(f"Sampled {len(sample):,} queries for QA review")
    return sample


def export_for_review(
    df: pd.DataFrame,
    output_path: str | Path,
    tokens_col: str = "query_tokens",
    tags_col: str = "ner_tags",
) -> None:
    """Export samples to a TSV file for manual review / correction.

    Format: query_idx  token  silver_tag  corrected_tag (empty for reviewer)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("query_idx\ttoken\tsilver_tag\tcorrected_tag\n")
        for idx, row in df.iterrows():
            tokens = row[tokens_col]
            tags = row[tags_col]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if isinstance(tags, np.ndarray):
                tags = tags.tolist()
            for token, tag in zip(tokens, tags):
                f.write(f"{idx}\t{token}\t{tag}\t\n")
            f.write(f"{idx}\t\t\t\n")  # blank separator

    logger.info(f"Exported {len(df):,} queries for review → {output_path}")


def compute_agreement(
    silver_tags: List[List[str]],
    gold_tags: List[List[str]],
) -> Dict[str, float]:
    """Compute inter-annotator agreement between silver and gold labels."""
    flat_silver = [tag for seq in silver_tags for tag in seq]
    flat_gold = [tag for seq in gold_tags for tag in seq]

    assert len(flat_silver) == len(flat_gold), (
        f"Tag count mismatch: silver={len(flat_silver)}, gold={len(flat_gold)}"
    )

    matches = sum(s == g for s, g in zip(flat_silver, flat_gold))
    accuracy = matches / len(flat_silver)
    kappa = cohen_kappa_score(flat_silver, flat_gold)

    entity_silver = [t for t in flat_silver if t != "O"]
    entity_gold = [t for t in flat_gold if t != "O"]

    metrics = {
        "token_accuracy": accuracy,
        "cohen_kappa": kappa,
        "total_tokens": len(flat_silver),
        "silver_entity_tokens": len(entity_silver),
        "gold_entity_tokens": len(entity_gold),
    }

    logger.info(
        f"Agreement: accuracy={accuracy:.3f}, kappa={kappa:.3f}, "
        f"entity tokens (silver={len(entity_silver)}, gold={len(entity_gold)})"
    )
    return metrics


def generate_qa_report(
    df: pd.DataFrame,
    tags_col: str = "ner_tags",
) -> Dict:
    """Generate a summary report of annotation quality."""
    all_tags = []
    for tags in df[tags_col]:
        if isinstance(tags, np.ndarray):
            tags = tags.tolist()
        all_tags.extend(tags)

    total_tokens = len(all_tags)
    entity_tokens = sum(1 for t in all_tags if t != "O")

    entity_counts: Dict[str, int] = {}
    for tag in all_tags:
        if tag.startswith("B-"):
            etype = tag[2:]
            entity_counts[etype] = entity_counts.get(etype, 0) + 1

    has_entity = sum(
        1 for tags in df[tags_col]
        if any(t != "O" for t in (tags.tolist() if isinstance(tags, np.ndarray) else tags))
    )

    report = {
        "total_queries": len(df),
        "total_tokens": total_tokens,
        "entity_tokens": entity_tokens,
        "entity_token_ratio": entity_tokens / total_tokens if total_tokens > 0 else 0,
        "queries_with_entities": has_entity,
        "entity_counts_by_type": entity_counts,
    }

    logger.info(f"QA Report: {report}")
    return report


def run_qa(
    annotations_dir: str | Path = "data/annotations",
    output_dir: str | Path = "data/annotations",
    config_path: str | Path = "configs/data_config.yaml",
    review_split: str = "train",
) -> None:
    """Run annotation QA: sample queries, export for review, generate report.

    Args:
        annotations_dir: Directory with annotated parquets.
        output_dir: Where to write QA artefacts.
        config_path: YAML config.
        review_split: Which split to sample from for review.
    """
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(config_path)
    if cfg_path.exists():
        cfg = load_yaml_config(cfg_path).get("data", {}).get("annotation", {})
    else:
        cfg = {}
    n_review = cfg.get("qa_sample_size", 1000)

    # Load annotated split
    parquet = annotations_dir / f"{review_split}.parquet"
    if not parquet.exists():
        logger.error(f"{parquet} not found — run annotation first.")
        return
    df = pd.read_parquet(parquet)

    # QA report for all annotated splits
    logger.info("Generating QA reports …")
    for f in sorted(annotations_dir.glob("*.parquet")):
        name = f.stem
        split_df = pd.read_parquet(f)
        if "ner_tags" not in split_df.columns:
            continue
        logger.info(f"\n── {name} ──")
        generate_qa_report(split_df)

    # Sample for manual review
    sample = sample_for_review(df, n=n_review)
    review_path = output_dir / "qa_review_sample.tsv"
    export_for_review(sample, review_path)

    logger.info(f"\n✅ QA complete. Review file: {review_path}")


