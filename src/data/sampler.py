"""Stratified sampling of preprocessed ESCI queries.

Reads the preprocessed parquet file (one row per unique query with
list-valued metadata columns), derives a *dominant label* for
stratification, draws a proportional sample, and writes the result
to ``data/processed/queries_sampled.parquet``.

Usage::

    from src.data.sampler import run_sampler
    df = run_sampler()                         # uses data_config.yaml
    df = run_sampler(sample_size=50_000)       # override sample size
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger(__name__)




def _dominant_value(values) -> str:
    """Return the most common value in a sequence (or the value itself if scalar).

    Handles Python lists, numpy arrays, and scalars.
    Falls back to ``"unknown"`` for empty / null inputs.
    """
    if isinstance(values, (list, np.ndarray)) and len(values) > 0:
        return str(Counter(values).most_common(1)[0][0])
    if isinstance(values, str) and values:
        return values
    return "unknown"


def _assign_stratum(
    df: pd.DataFrame,
    stratify_col: str,
) -> pd.Series:
    """Derive a single categorical stratum value per row.

    Handles list-valued columns (e.g. ``esci_label``, ``product_brand``)
    by picking the dominant (most frequent) element.
    """
    if stratify_col not in df.columns:
        logger.warning(
            f"Column '{stratify_col}' not in DataFrame — "
            "all rows assigned to stratum 'unknown'."
        )
        return pd.Series("unknown", index=df.index)

    return df[stratify_col].apply(_dominant_value)




def sample_queries(
    df: pd.DataFrame,
    n: int = 75_000,
    stratify_col: str = "esci_label",
    seed: int = 42,
) -> pd.DataFrame:
    """Draw a stratified sample of unique queries.

    Proportional allocation: each stratum contributes the same fraction
    of rows it occupies in the full dataset.

    Args:
        df: Preprocessed DataFrame (one row per unique query).
        n: Target sample size (capped at available rows).
        stratify_col: Column (or list-valued column) to stratify by.
        seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with an extra ``_stratum`` column.
    """
    total = len(df)
    n = min(n, total)
    logger.info(
        f"Sampling {n:,} queries from {total:,} "
        f"(stratified by '{stratify_col}') …"
    )

    # Build a flat stratum column
    strata = _assign_stratum(df, stratify_col)
    df = df.copy()
    df["_stratum"] = strata

    # Proportional allocation per stratum
    stratum_counts = df["_stratum"].value_counts()
    stratum_fracs = stratum_counts / stratum_counts.sum()

    sampled_parts: list[pd.DataFrame] = []
    allocated = 0

    items = list(stratum_fracs.items())
    for i, (stratum, frac) in enumerate(items):
        stratum_df = df[df["_stratum"] == stratum]
        if i == len(items) - 1:
            # last stratum gets the remainder to hit exactly n
            stratum_n = n - allocated
        else:
            stratum_n = max(1, round(n * frac))

        stratum_n = min(stratum_n, len(stratum_df))
        allocated += stratum_n

        sampled_parts.append(
            stratum_df.sample(n=stratum_n, random_state=seed)
        )

    result = pd.concat(sampled_parts, ignore_index=True)
    logger.info(
        f"  Sampled {len(result):,} queries across "
        f"{len(stratum_counts)} strata."
    )

    # Log distribution
    for stratum, count in result["_stratum"].value_counts().items():
        logger.info(f"    {stratum}: {count:,}")

    return result




def run_sampler(
    preprocessed_path: str | Path = "data/processed/queries_preprocessed.parquet",
    output_path: str | Path = "data/processed/queries_sampled.parquet",
    config_path: str | Path = "configs/data_config.yaml",
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    """Load preprocessed data, sample, and save.

    Args:
        preprocessed_path: Input parquet from the preprocessing step.
        output_path: Where to write the sampled parquet.
        config_path: YAML config (``data.sample_size``, ``data.stratify_by``,
                     ``data.random_seed``).
        sample_size: Override config's ``sample_size`` if provided.

    Returns:
        The sampled DataFrame.
    """
    # ── Config ─────────────────────────────────────────────────────────────
    config_path = Path(config_path)
    if config_path.exists():
        cfg = load_yaml_config(config_path).get("data", {})
    else:
        logger.warning(f"Config not found at {config_path}, using defaults.")
        cfg = {}

    n = sample_size or cfg.get("sample_size", 75_000)
    stratify_col = cfg.get("stratify_by", "esci_label")
    seed = cfg.get("random_seed", 42)

    # ── Load ───────────────────────────────────────────────────────────────
    preprocessed_path = Path(preprocessed_path)
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found at {preprocessed_path}. "
            "Run the preprocessing step first."
        )
    logger.info(f"Loading preprocessed data from {preprocessed_path} …")
    df = pd.read_parquet(preprocessed_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # ── Sample ─────────────────────────────────────────────────────────────
    sampled = sample_queries(df, n=n, stratify_col=stratify_col, seed=seed)

    # ── Save ───────────────────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_path, index=False)
    logger.info(f"Saved sampled queries → {output_path}  ({len(sampled):,} rows)")

    return sampled

