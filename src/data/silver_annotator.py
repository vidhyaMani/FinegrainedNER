"""Silver annotation pipeline for e-commerce NER.

Generates IOB2-tagged training data by aligning query tokens against
product metadata (brand, color, product type, etc.) using exact and
fuzzy string matching.

Usage::

    from src.data.silver_annotator import run_annotator
    run_annotator()                    # annotates all splits, writes data/annotations/
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.schema.entity_schema import ENTITY_PRIORITY, FineEntity
from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger(__name__)



_MATERIAL_KEYWORDS: set[str] = {
    "leather", "cotton", "polyester", "nylon", "silk", "wool", "linen",
    "denim", "suede", "canvas", "rubber", "plastic", "metal", "aluminum",
    "aluminium", "stainless steel", "steel", "iron", "wood", "wooden",
    "bamboo", "ceramic", "glass", "titanium", "copper", "brass", "bronze",
    "velvet", "satin", "fleece", "foam", "latex", "vinyl", "acrylic",
    "polycarbonate", "silicone", "carbon fiber", "porcelain", "marble",
    "granite", "concrete", "cork", "hemp", "jute", "microfiber",
}

_SIZE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(\d+\.?\d*)\s*(oz|ounce|fl\s*oz|ml|liter|litre|l|gal|gallon)\b", re.I),
    re.compile(r"\b(\d+\.?\d*)\s*(inch|in|inches|ft|feet|foot|cm|mm|m|meter)\b", re.I),
    re.compile(r"\b(\d+\.?\d*)\s*(lb|lbs|pound|pounds|kg|gram|grams|g|mg)\b", re.I),
    re.compile(r"\b(xx?[sl]|small|medium|large|x-?large|xx-?large)\b", re.I),
    re.compile(r"\b(\d+)\s*x\s*(\d+)\b", re.I),                    # 4x6, 8x10
    re.compile(r"\b(\d+\.?\d*)\s*(pack|count|ct|pc|pcs|piece|pieces)\b", re.I),
    re.compile(r"\b(king|queen|twin|full|cal\s*king)\s*size\b", re.I),
]

_ATTRIBUTE_KEYWORDS: set[str] = {
    "waterproof", "water resistant", "wireless", "bluetooth", "rechargeable",
    "portable", "organic", "natural", "vegan", "gluten free", "non gmo",
    "bpa free", "eco friendly", "biodegradable", "solar", "led",
    "usb", "cordless", "adjustable", "foldable", "collapsible",
    "heavy duty", "lightweight", "anti slip", "non stick", "dishwasher safe",
    "machine washable", "hypoallergenic", "unscented", "scented", "fragrance free",
    "indoor", "outdoor", "electric", "manual", "automatic", "digital",
    "magnetic", "insulated", "thermal", "heated", "cooling",
}




def _normalize(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def _to_str_list(val) -> List[str]:
    """Convert a value (str, list, ndarray, None) to a list of non-empty strings."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return [_normalize(str(v)) for v in val if v is not None and str(v).strip()]
    if isinstance(val, list):
        return [_normalize(str(v)) for v in val if v is not None and str(v).strip()]
    if isinstance(val, str) and val.strip():
        return [_normalize(val)]
    return []




def _extract_metadata_values(
    row: pd.Series,
    entity_type: FineEntity,
) -> List[str]:
    """Extract candidate entity values from product metadata for *entity_type*.

    For BRAND and COLOR we pull directly from the corresponding columns.
    For PRODUCT_TYPE, MATERIAL, SIZE_MEASURE, ATTRIBUTE_VALUE we apply
    heuristics on titles / descriptions / bullet points.
    """
    if entity_type == FineEntity.BRAND:
        return _to_str_list(row.get("product_brand"))

    if entity_type == FineEntity.COLOR:
        return _to_str_list(row.get("product_color"))

    if entity_type == FineEntity.PRODUCT_TYPE:
        # Use the last noun-phrase of product titles as proxy for product type.
        # e.g. "Nike Air Max Running Shoes" → candidate "running shoes"
        titles = _to_str_list(row.get("product_title"))
        candidates: list[str] = []
        for title in titles:
            # Take last 1-3 words (often the product noun)
            words = title.split()
            if len(words) >= 2:
                candidates.append(" ".join(words[-2:]))
            if len(words) >= 3:
                candidates.append(" ".join(words[-3:]))
            if len(words) >= 1:
                candidates.append(words[-1])
        return list(set(candidates))

    if entity_type == FineEntity.MATERIAL:
        # Check if any material keyword appears in titles/descriptions
        texts = _to_str_list(row.get("product_title")) + _to_str_list(row.get("product_bullet_point"))
        found: list[str] = []
        combined = " ".join(texts)
        for mat in _MATERIAL_KEYWORDS:
            if mat in combined:
                found.append(mat)
        return found

    if entity_type == FineEntity.SIZE_MEASURE:
        # Regex-based extraction from titles
        texts = _to_str_list(row.get("product_title"))
        found = []
        combined = " ".join(texts)
        for pat in _SIZE_PATTERNS:
            for m in pat.finditer(combined):
                found.append(_normalize(m.group(0)))
        return list(set(found))

    if entity_type == FineEntity.ATTRIBUTE_VALUE:
        texts = _to_str_list(row.get("product_title")) + _to_str_list(row.get("product_bullet_point"))
        found = []
        combined = " ".join(texts)
        for attr in _ATTRIBUTE_KEYWORDS:
            if attr in combined:
                found.append(attr)
        return found

    return []




def _find_span_in_tokens(
    tokens: List[str],
    phrase: str,
    threshold: int = 85,
) -> Optional[Tuple[int, int]]:
    """Find the best matching span of tokens for *phrase*.

    Tries exact match first, then fuzzy matching on n-grams of the same
    length as the phrase.

    Returns:
        ``(start, end)`` inclusive indices, or ``None``.
    """
    phrase_tokens = phrase.lower().split()
    phrase_len = len(phrase_tokens)

    if phrase_len == 0 or phrase_len > len(tokens):
        return None

    # Exact match
    for i in range(len(tokens) - phrase_len + 1):
        window = [t.lower() for t in tokens[i: i + phrase_len]]
        if window == phrase_tokens:
            return (i, i + phrase_len - 1)

    # Fuzzy match on sliding window
    best_score = 0
    best_span = None

    for i in range(len(tokens) - phrase_len + 1):
        window_text = " ".join(tokens[i: i + phrase_len]).lower()
        score = fuzz.ratio(phrase.lower(), window_text)
        if score > best_score and score >= threshold:
            best_score = score
            best_span = (i, i + phrase_len - 1)

    # Single-token fuzzy for single-word entities
    if phrase_len == 1:
        for i, token in enumerate(tokens):
            score = fuzz.ratio(phrase.lower(), token.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_span = (i, i)

    return best_span




def annotate_query(
    tokens: List[str],
    metadata_values: Dict[FineEntity, List[str]],
    threshold: int = 85,
) -> List[str]:
    """Annotate one query's tokens with IOB2 tags.

    Args:
        tokens: Query tokens.
        metadata_values: ``{FineEntity: [candidate_value, …]}``.
        threshold: Fuzzy match threshold.

    Returns:
        List of IOB2 tags, one per token.
    """
    tags = ["O"] * len(tokens)
    annotated: set[int] = set()

    # Process entity types in priority order
    sorted_types = sorted(
        metadata_values.keys(),
        key=lambda et: ENTITY_PRIORITY.get(et, 99),
    )

    for entity_type in sorted_types:
        for value in metadata_values[entity_type]:
            span = _find_span_in_tokens(tokens, value, threshold)
            if span is None:
                continue

            start, end = span
            positions = set(range(start, end + 1))
            if positions & annotated:
                continue  # skip overlap

            tags[start] = f"B-{entity_type.value}"
            for idx in range(start + 1, end + 1):
                tags[idx] = f"I-{entity_type.value}"
            annotated.update(positions)

    return tags




def annotate_dataset(
    df: pd.DataFrame,
    tokens_col: str = "query_tokens",
    threshold: int = 85,
) -> pd.DataFrame:
    """Generate silver IOB2 annotations for every row.

    Args:
        df: Preprocessed DataFrame (one row per unique query).
        tokens_col: Column containing token lists.
        threshold: Fuzzy match threshold.

    Returns:
        DataFrame with ``ner_tags`` column added.
    """
    n = len(df)
    logger.info(f"Generating silver annotations for {n:,} queries …")

    all_tags: list[list[str]] = []
    annotated_count = 0

    for i, (_, row) in enumerate(df.iterrows()):
        tokens = row[tokens_col]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        # Collect metadata candidates
        metadata: Dict[FineEntity, List[str]] = {}
        for etype in FineEntity:
            vals = _extract_metadata_values(row, etype)
            if vals:
                metadata[etype] = vals

        tags = annotate_query(tokens, metadata, threshold)
        all_tags.append(tags)

        if any(t != "O" for t in tags):
            annotated_count += 1

        if (i + 1) % 10_000 == 0:
            logger.info(f"  {i + 1:,} / {n:,} …")

    df = df.copy()
    df["ner_tags"] = all_tags

    pct = annotated_count / n * 100 if n else 0
    logger.info(
        f"Done — {annotated_count:,}/{n:,} queries have ≥1 entity ({pct:.1f}%)"
    )
    return df




def save_conll(
    df: pd.DataFrame,
    output_path: str | Path,
    tokens_col: str = "query_tokens",
    tags_col: str = "ner_tags",
) -> None:
    """Write annotated data in CoNLL format (token TAB tag, blank line between
    queries)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            tokens = row[tokens_col]
            tags = row[tags_col]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if isinstance(tags, np.ndarray):
                tags = tags.tolist()
            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

    logger.info(f"Saved CoNLL → {output_path}  ({len(df):,} sentences)")




def run_annotator(
    processed_dir: str | Path = "data/processed",
    annotations_dir: str | Path = "data/annotations",
    config_path: str | Path = "configs/data_config.yaml",
    splits: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Annotate processed splits and save to ``data/annotations/``.

    For each split (train, val, test, 1k, 2k, 5k) that exists in
    *processed_dir*, generates silver IOB2 tags and writes:
      - ``<split>.parquet`` (full DataFrame with ``ner_tags`` column)
      - ``<split>.conll`` (CoNLL-format text file)

    Args:
        processed_dir: Directory with split parquet files.
        annotations_dir: Output directory.
        config_path: YAML config for annotation settings.
        splits: Which splits to annotate.  ``None`` → all found.

    Returns:
        Dict of split name → annotated DataFrame.
    """
    processed_dir = Path(processed_dir)
    annotations_dir = Path(annotations_dir)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Config
    cfg_path = Path(config_path)
    if cfg_path.exists():
        cfg = load_yaml_config(cfg_path).get("data", {}).get("annotation", {})
    else:
        cfg = {}
    threshold = cfg.get("fuzzy_match_threshold", 85)

    # Discover splits
    if splits is None:
        splits = []
        for f in sorted(processed_dir.glob("*.parquet")):
            name = f.stem
            if name.startswith("queries_"):
                continue  # skip intermediate files
            splits.append(name)

    logger.info(f"Annotating splits: {splits}")

    results: Dict[str, pd.DataFrame] = {}
    for split_name in splits:
        parquet = processed_dir / f"{split_name}.parquet"
        if not parquet.exists():
            logger.warning(f"  {parquet} not found — skipping")
            continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  Annotating '{split_name}' …")
        df = pd.read_parquet(parquet)

        df = annotate_dataset(df, tokens_col="query_tokens", threshold=threshold)

        # Save parquet
        out_parquet = annotations_dir / f"{split_name}.parquet"
        df.to_parquet(out_parquet, index=False)
        logger.info(f"  Saved {out_parquet}")

        # Save CoNLL
        out_conll = annotations_dir / f"{split_name}.conll"
        save_conll(df, out_conll)

        results[split_name] = df

    # Summary
    logger.info(f"\n{'═' * 50}")
    logger.info("Annotation summary:")
    for name, df in results.items():
        n_ent = sum(1 for tags in df["ner_tags"] if any(t != "O" for t in tags))
        logger.info(f"  {name:>6s}: {len(df):>8,} queries, {n_ent:>8,} with entities")

    return results

