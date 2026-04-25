"""Dataset splitting with stratification and low-resource subset generation.

Reads the sampled parquet file, creates train / val / test splits with
stratification (no query leakage by design — each row is a unique query),
builds low-resource training subsets for few-shot experiments, and writes
everything to ``data/processed/``.

Usage::

    from src.data.splitter import run_splitter
    splits = run_splitter()      # reads data_config.yaml, writes parquets
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.helpers import get_logger, load_yaml_config

logger = get_logger(__name__)




def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Split a DataFrame into train / val / test ensuring no query leakage.

    Because each row is already a *unique query* (from the sampler), simply
    splitting rows guarantees no query appears in more than one partition.

    Args:
        df: DataFrame with one row per unique query.
        train_ratio / val_ratio / test_ratio: Must sum to 1.0.
        stratify_col: Column to stratify splits by (flat string column;
                      use ``_stratum`` produced by the sampler).
        seed: Random seed.

    Returns:
        ``{"train": df, "val": df, "test": df}``
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    logger.info(
        f"Splitting {len(df):,} queries → "
        f"train {train_ratio:.0%} / val {val_ratio:.0%} / test {test_ratio:.0%}"
    )

    # Determine stratify vector (may be None)
    stratify = _safe_stratify(df, stratify_col)

    # Split 1: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, valtest_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify,
    )

    # Split 2: val vs test
    relative_test = test_ratio / val_test_ratio
    stratify_vt = _safe_stratify(valtest_df, stratify_col)
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=relative_test,
        random_state=seed,
        stratify=stratify_vt,
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    for name, s in splits.items():
        logger.info(f"  {name}: {len(s):,}")
    return splits


def _safe_stratify(
    df: pd.DataFrame, col: Optional[str]
) -> Optional[pd.Series]:
    """Return a stratify vector, or ``None`` if the column is missing or has
    strata that are too small for splitting.

    ``train_test_split`` requires every stratum to have ≥ 2 members.
    Rare strata are folded into a catch-all ``"_other"`` bucket.
    """
    if col is None or col not in df.columns:
        return None

    strata = df[col].copy()
    counts = strata.value_counts()
    small = counts[counts < 2].index
    if len(small) > 0:
        logger.info(
            f"  Merging {len(small)} tiny strata (< 2 members) into '_other'"
        )
        strata = strata.where(~strata.isin(small), other="_other")

    # If everything collapsed to one stratum, skip stratification
    if strata.nunique() < 2:
        return None
    return strata




def create_low_resource_subsets(
    train_df: pd.DataFrame,
    sizes: List[int] = (1000, 2000, 5000),
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Create low-resource training subsets for few-shot experiments.

    Args:
        train_df: Full training set.
        sizes: List of subset sizes.
        stratify_col: Column to stratify by.
        seed: Random seed.

    Returns:
        ``{"1k": df, "2k": df, "5k": df}``
    """
    subsets: Dict[str, pd.DataFrame] = {}
    for size in sizes:
        label = f"{size // 1000}k" if size >= 1000 else str(size)

        if size >= len(train_df):
            logger.warning(
                f"Subset {label} ({size}) ≥ train ({len(train_df)}); "
                f"using full training set."
            )
            subsets[label] = train_df.copy()
            continue

        stratify = _safe_stratify(train_df, stratify_col)
        subset, _ = train_test_split(
            train_df,
            train_size=size,
            random_state=seed,
            stratify=stratify,
        )
        subsets[label] = subset.reset_index(drop=True)
        logger.info(f"  Low-resource subset '{label}': {len(subset):,}")

    return subsets




def save_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Write split DataFrames to parquet files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saved {name} ({len(df):,} rows) → {path}")




def run_splitter(
    sampled_path: str | Path = "data/processed/queries_sampled.parquet",
    output_dir: str | Path = "data/processed",
    config_path: str | Path = "configs/data_config.yaml",
) -> Dict[str, pd.DataFrame]:
    """Load sampled data, split, create low-resource subsets, save.

    Args:
        sampled_path: Input parquet from the sampling step.
        output_dir: Directory to write split parquets.
        config_path: YAML config.

    Returns:
        Dict with keys ``train``, ``val``, ``test``, ``1k``, ``2k``, ``5k``.
    """
    # ── Config ─────────────────────────────────────────────────────────────
    config_path = Path(config_path)
    if config_path.exists():
        cfg = load_yaml_config(config_path).get("data", {})
    else:
        logger.warning(f"Config not found at {config_path}, using defaults.")
        cfg = {}

    split_cfg = cfg.get("splits", {})
    train_ratio = split_cfg.get("train", 0.70)
    val_ratio = split_cfg.get("val", 0.15)
    test_ratio = split_cfg.get("test", 0.15)
    seed = cfg.get("random_seed", 42)
    low_resource_sizes = cfg.get("low_resource_sizes", [1000, 2000, 5000])
    stratify_col = "_stratum"  # created by the sampler

    # ── Load ───────────────────────────────────────────────────────────────
    sampled_path = Path(sampled_path)
    if not sampled_path.exists():
        raise FileNotFoundError(
            f"Sampled file not found at {sampled_path}. "
            "Run the sampling step first."
        )
    logger.info(f"Loading sampled data from {sampled_path} …")
    df = pd.read_parquet(sampled_path)
    logger.info(f"  Loaded {len(df):,} rows")

    # ── Split ──────────────────────────────────────────────────────────────
    splits = split_dataset(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_col=stratify_col,
        seed=seed,
    )

    # ── Low-resource subsets ───────────────────────────────────────────────
    low_subsets = create_low_resource_subsets(
        splits["train"],
        sizes=low_resource_sizes,
        stratify_col=stratify_col,
        seed=seed,
    )
    splits.update(low_subsets)

    # ── Save ───────────────────────────────────────────────────────────────
    save_splits(splits, output_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("Split summary:")
    for name, s in splits.items():
        logger.info(f"  {name:>6s}: {len(s):>8,} queries")

    return splits

