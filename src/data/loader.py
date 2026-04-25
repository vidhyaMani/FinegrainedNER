"""ESCI dataset loader.

Downloads (if needed) and loads the Amazon ESCI shopping queries dataset,
filters to English (US locale), and joins queries with product metadata.

Supports two file layouts:
  1. Files downloaded directly into ``data/raw/``
     (``shopping_queries_dataset_products.parquet``, etc.)
  2. Cloned repo structure at ``data/raw/esci-data/shopping_queries_dataset/``
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)

ESCI_REPO_URL = "https://github.com/amazon-science/esci-data.git"

# File basenames
_PRODUCTS_BASENAME = "shopping_queries_dataset_products.parquet"
_EXAMPLES_BASENAME = "shopping_queries_dataset_examples.parquet"
_SOURCES_BASENAME = "shopping_queries_dataset_sources.csv"

# Relative path when files live inside a cloned repo
_REPO_SUBDIR = "shopping_queries_dataset"




def _resolve_file(raw_dir: Path, basename: str) -> Path:
    """Locate a dataset file, checking both flat and cloned-repo layouts.

    Search order:
      1. ``<raw_dir>/<basename>``  (flat download)
      2. ``<raw_dir>/esci-data/<_REPO_SUBDIR>/<basename>`` (cloned repo)

    Raises:
        FileNotFoundError: If the file cannot be found in either location.
    """
    # Flat layout: files directly in raw_dir
    flat = raw_dir / basename
    if flat.exists():
        return flat

    # Cloned-repo layout
    cloned = raw_dir / "esci-data" / _REPO_SUBDIR / basename
    if cloned.exists():
        return cloned

    raise FileNotFoundError(
        f"Cannot find '{basename}' in either:\n"
        f"  • {flat}\n"
        f"  • {cloned}\n"
        "Download the ESCI data or run `clone_esci_repo()` first."
    )


def _safe_unique_list(series: pd.Series) -> list:
    """Return a list of unique non-null values from a Series."""
    return list(series.dropna().unique())




def clone_esci_repo(raw_dir: str | Path) -> Path:
    """Clone the ESCI dataset repo into the raw data directory.

    Args:
        raw_dir: Directory to clone into (e.g., ``data/raw``).

    Returns:
        Path to the cloned repo root.
    """
    raw_dir = Path(raw_dir)
    repo_dir = raw_dir / "esci-data"

    if repo_dir.exists():
        logger.info(f"ESCI repo already exists at {repo_dir}, skipping clone.")
        return repo_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning ESCI repo into {repo_dir} …")
    subprocess.run(
        ["git", "clone", "--depth", "1", ESCI_REPO_URL, str(repo_dir)],
        check=True,
    )
    logger.info("Clone complete.")
    return repo_dir




def load_products(
    raw_dir: str | Path,
    locale: str = "us",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load the products parquet file, optionally filtered by locale.

    Args:
        raw_dir: Directory containing the ESCI data files (flat or cloned).
        locale: Language locale to keep (``"us"`` for English). Pass ``None``
                or ``""`` to skip filtering.
        columns: Optional list of columns to read (speeds up I/O for large
                 files). ``None`` reads all columns.

    Returns:
        DataFrame with product metadata.
    """
    raw_dir = Path(raw_dir)
    path = _resolve_file(raw_dir, _PRODUCTS_BASENAME)
    logger.info(f"Loading products from {path} …")
    df = pd.read_parquet(path, columns=columns)
    logger.info(f"  Raw products: {len(df):,}")

    if locale:
        df = df[df["product_locale"] == locale].reset_index(drop=True)
        logger.info(f"  After locale filter ('{locale}'): {len(df):,}")

    return df


def load_examples(
    raw_dir: str | Path,
    locale: str = "us",
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load the query–product examples parquet file, optionally filtered.

    Args:
        raw_dir: Directory containing the ESCI data files.
        locale: Language locale to keep (filters on ``product_locale``).
        columns: Optional subset of columns to read.

    Returns:
        DataFrame with query–product pairs and ESCI relevance labels.
    """
    raw_dir = Path(raw_dir)
    path = _resolve_file(raw_dir, _EXAMPLES_BASENAME)
    logger.info(f"Loading examples from {path} …")
    df = pd.read_parquet(path, columns=columns)
    logger.info(f"  Raw examples: {len(df):,}")

    if locale:
        df = df[df["product_locale"] == locale].reset_index(drop=True)
        logger.info(f"  After locale filter ('{locale}'): {len(df):,}")

    return df


def load_sources(raw_dir: str | Path) -> pd.DataFrame:
    """Load the sources CSV (query ID ↔ source mapping).

    Args:
        raw_dir: Directory containing the ESCI data files.

    Returns:
        DataFrame with source metadata.
    """
    raw_dir = Path(raw_dir)
    path = _resolve_file(raw_dir, _SOURCES_BASENAME)
    logger.info(f"Loading sources from {path} …")
    df = pd.read_csv(str(path))
    logger.info(f"  Sources: {len(df):,}")
    return df




def load_and_join(
    raw_dir: str | Path,
    locale: str = "us",
) -> pd.DataFrame:
    """Load examples and products, then join on ``product_id``.

    Returns a merged DataFrame with query text + product metadata for each
    query–product pair.  This is the starting point for the annotation
    pipeline.

    Args:
        raw_dir: Directory containing the ESCI data files.
        locale: Language locale to keep.

    Returns:
        Merged DataFrame (one row per query–product pair).
    """
    examples = load_examples(raw_dir, locale)
    products = load_products(raw_dir, locale)

    logger.info("Joining examples with product metadata …")
    merged = examples.merge(
        products,
        on=["product_id", "product_locale"],
        how="left",
        suffixes=("", "_product"),
    )
    logger.info(f"  Merged dataset: {len(merged):,} rows")

    n_unmatched = merged["product_title"].isna().sum()
    if n_unmatched > 0:
        logger.warning(
            f"  {n_unmatched:,} examples have no matching product metadata "
            "(product_title is NaN after left join)."
        )

    return merged


def get_unique_queries(
    merged_df: pd.DataFrame,
    query_col: str = "query",
) -> pd.DataFrame:
    """Extract unique queries with aggregated product metadata.

    For each unique query, collects the set of associated brands, colors,
    product types, etc. from its matched products.

    Args:
        merged_df: Output of :func:`load_and_join`.
        query_col: Name of the column containing query text.

    Returns:
        DataFrame indexed by unique query with list-valued metadata columns.
    """
    logger.info("Extracting unique queries with aggregated metadata …")

    # Build aggregation dict – use named functions to avoid late-binding issues
    # with lambdas defined in a loop.
    agg_funcs: dict = {}

    metadata_cols = [
        "product_title",
        "product_brand",
        "product_color",
        "product_bullet_point",
        "product_description",
    ]
    for col in metadata_cols:
        if col in merged_df.columns:
            agg_funcs[col] = _safe_unique_list

    if "esci_label" in merged_df.columns:
        agg_funcs["esci_label"] = list

    if "product_category" in merged_df.columns:
        agg_funcs["product_category"] = _safe_unique_list

    if not agg_funcs:
        logger.warning("No metadata columns found to aggregate — returning deduplicated queries.")
        return merged_df[[query_col]].drop_duplicates().reset_index(drop=True)

    queries = merged_df.groupby(query_col).agg(agg_funcs).reset_index()
    logger.info(f"  Unique queries: {len(queries):,}")
    return queries




def load_dataset(
    raw_dir: str | Path = "data/raw",
    locale: str = "us",
    aggregate: bool = True,
) -> pd.DataFrame:
    """High-level convenience loader.

    Loads, filters, joins, and (optionally) aggregates the ESCI data in one
    call.

    Args:
        raw_dir: Path to raw data directory.
        locale: Locale filter.
        aggregate: If ``True`` (default), return one row per unique query with
                   aggregated product metadata. If ``False``, return the raw
                   merged DataFrame (one row per query–product pair).

    Returns:
        DataFrame ready for downstream processing.
    """
    merged = load_and_join(raw_dir, locale)
    if aggregate:
        return get_unique_queries(merged)
    return merged


