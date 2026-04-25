#!/usr/bin/env python
"""Run the full data preparation pipeline: preprocess → sample → split.

Reads settings from ``configs/data_config.yaml`` and writes all artefacts
to ``data/processed/``.

Usage:
    python scripts/prepare_data.py                           # full pipeline
    python scripts/prepare_data.py --skip-preprocess         # if already done
    python scripts/prepare_data.py --sample-size 50000       # override
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def _section(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full data preparation pipeline.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--config", default="configs/data_config.yaml")
    parser.add_argument("--locale", default="us")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Override sample size from config.")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (use existing parquet).")
    args = parser.parse_args()

    processed = Path(args.processed_dir)
    preprocessed_path = processed / "queries_preprocessed.parquet"
    sampled_path = processed / "queries_sampled.parquet"

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    if args.skip_preprocess and preprocessed_path.exists():
        _section("Step 1: Preprocess — SKIPPED (file exists)")
        df_pre = pd.read_parquet(preprocessed_path)
        print(f"  Loaded {len(df_pre):,} preprocessed queries")
    else:
        _section("Step 1: Preprocess")
        from src.data.preprocess import run_pipeline
        df_pre = run_pipeline(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            config_path=args.config,
            locale=args.locale,
        )
        print(f"  ✓ {len(df_pre):,} queries preprocessed → {preprocessed_path}")

    # ── Step 2: Sample ─────────────────────────────────────────────────────
    _section("Step 2: Stratified Sampling")
    from src.data.sampler import run_sampler
    df_sampled = run_sampler(
        preprocessed_path=preprocessed_path,
        output_path=sampled_path,
        config_path=args.config,
        sample_size=args.sample_size,
    )
    print(f"  ✓ {len(df_sampled):,} queries sampled → {sampled_path}")
    print(f"  Stratum distribution:")
    if "_stratum" in df_sampled.columns:
        for stratum, cnt in df_sampled["_stratum"].value_counts().items():
            print(f"    {stratum}: {cnt:,}")

    # ── Step 3: Split ──────────────────────────────────────────────────────
    _section("Step 3: Train / Val / Test Split")
    from src.data.splitter import run_splitter
    splits = run_splitter(
        sampled_path=sampled_path,
        output_dir=args.processed_dir,
        config_path=args.config,
    )
    print(f"\n  Output files in {processed}/:")
    for name, s in splits.items():
        f = processed / f"{name}.parquet"
        print(f"    {f.name:>20s}  — {len(s):>8,} queries")

    # ── Done ───────────────────────────────────────────────────────────────
    _section("✅  Data preparation complete")
    print(f"  All files in: {processed.resolve()}\n")


if __name__ == "__main__":
    main()

