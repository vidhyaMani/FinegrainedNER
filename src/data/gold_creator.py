"""Gold dataset creator.

Merges manually corrected annotations back into the silver dataset
to produce gold-standard evaluation and test sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def load_corrections(
    corrections_path: str | Path,
) -> Dict[int, List[Tuple[str, str]]]:
    """Load manually corrected annotations from a TSV review file.

    Expected format: query_idx  token  silver_tag  corrected_tag

    Args:
        corrections_path: Path to the corrected TSV file.

    Returns:
        Dict mapping query_idx → list of (token, corrected_tag) tuples.
    """
    corrections_path = Path(corrections_path)
    logger.info(f"Loading corrections from {corrections_path}...")

    corrections: Dict[int, List[Tuple[str, str]]] = {}
    current_idx = None
    current_tokens = []

    with open(corrections_path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            idx = int(parts[0])
            token = parts[1]
            silver_tag = parts[2]
            corrected_tag = parts[3].strip() if parts[3].strip() else silver_tag

            if not token:  # blank separator between queries
                if current_idx is not None and current_tokens:
                    corrections[current_idx] = current_tokens
                current_tokens = []
                continue

            current_idx = idx
            current_tokens.append((token, corrected_tag))

        # Don't forget the last query
        if current_idx is not None and current_tokens:
            corrections[current_idx] = current_tokens

    logger.info(f"Loaded corrections for {len(corrections):,} queries")
    return corrections


def merge_corrections(
    df: pd.DataFrame,
    corrections: Dict[int, List[Tuple[str, str]]],
    tags_col: str = "ner_tags",
) -> pd.DataFrame:
    """Merge manual corrections into the annotated dataset.

    Args:
        df: Original annotated DataFrame.
        corrections: Dict mapping query_idx → list of (token, corrected_tag).
        tags_col: Column name for NER tags.

    Returns:
        Updated DataFrame with corrected tags.
    """
    df = df.copy()
    corrected_count = 0

    for idx, token_tags in corrections.items():
        if idx not in df.index:
            logger.warning(f"Query index {idx} not found in DataFrame, skipping.")
            continue

        corrected_tags = [tag for _, tag in token_tags]
        original_tags = df.at[idx, tags_col]

        if len(corrected_tags) != len(original_tags):
            logger.warning(
                f"Token count mismatch at index {idx}: "
                f"corrected={len(corrected_tags)}, original={len(original_tags)}. Skipping."
            )
            continue

        # Count actual changes
        changes = sum(o != c for o, c in zip(original_tags, corrected_tags))
        if changes > 0:
            corrected_count += 1
            df.at[idx, tags_col] = corrected_tags

    logger.info(
        f"Merged corrections: {corrected_count:,} queries had tag changes "
        f"out of {len(corrections):,} reviewed"
    )
    return df


def create_gold_set(
    df: pd.DataFrame,
    corrections_path: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    tags_col: str = "ner_tags",
) -> pd.DataFrame:
    """Create a gold-standard dataset from silver annotations + manual corrections.

    Args:
        df: Silver-annotated DataFrame.
        corrections_path: Path to the manual corrections TSV file.
        output_path: Path to save the gold dataset (optional).
        tags_col: Column name for NER tags.

    Returns:
        Gold-standard DataFrame.
    """
    if corrections_path:
        corrections = load_corrections(corrections_path)
        gold_df = merge_corrections(df, corrections, tags_col)
    else:
        logger.warning("No corrections file provided. Gold set = silver set.")
        gold_df = df.copy()

    # Add a flag indicating this is gold-standard
    gold_df["is_gold"] = True

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gold_df.to_parquet(output_path, index=False)
        logger.info(f"Saved gold dataset ({len(gold_df):,} queries) to {output_path}")

    return gold_df

