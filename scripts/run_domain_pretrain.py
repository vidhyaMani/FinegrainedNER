#!/usr/bin/env python
"""Domain pretraining script for e-commerce NER.

Extracts product text corpus and runs domain-adaptive pretraining.

Usage:
    python scripts/run_domain_pretrain.py
    python scripts/run_domain_pretrain.py --max-steps 50000 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.models.domain_adapted.pretrain_domain import pretrain_domain
from src.utils.helpers import ensure_dir, get_logger

logger = get_logger(__name__)


def extract_corpus(
    products_path: str = "data/raw/shopping_queries_dataset_products.parquet",
    output_path: str = "data/processed/domain_corpus.txt",
    max_texts: int = 500000,
) -> list:
    """Extract product titles for domain pretraining corpus.

    Args:
        products_path: Path to products parquet file.
        output_path: Path to save corpus text file.
        max_texts: Maximum number of texts to extract.

    Returns:
        List of text strings.
    """
    logger.info(f"Loading products from {products_path}...")
    products = pd.read_parquet(products_path)

    texts = products["product_title"].dropna().tolist()
    texts = texts[:max_texts]

    # Save corpus
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w") as f:
        f.write("\n".join(texts))

    logger.info(f"Extracted {len(texts):,} texts to {output_path}")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Run domain pretraining")
    parser.add_argument(
        "--products-path", type=str,
        default="data/raw/shopping_queries_dataset_products.parquet",
        help="Path to products parquet file",
    )
    parser.add_argument(
        "--corpus-path", type=str,
        default="data/processed/domain_corpus.txt",
        help="Path to save/load corpus",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/esci_bert_dapt",
        help="Output directory for pretrained model",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="bert-base-uncased",
        help="Base model to continue pretraining from",
    )
    parser.add_argument(
        "--max-texts", type=int, default=500000,
        help="Maximum texts to use for pretraining",
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip corpus extraction (use existing corpus)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Step 1: Extract corpus
    if args.skip_extract and Path(args.corpus_path).exists():
        logger.info(f"Loading existing corpus from {args.corpus_path}")
        with open(args.corpus_path) as f:
            texts = f.read().split("\n")
    else:
        texts = extract_corpus(
            products_path=args.products_path,
            output_path=args.corpus_path,
            max_texts=args.max_texts,
        )

    logger.info(f"Corpus size: {len(texts):,} texts")

    # Step 2: Run domain pretraining
    logger.info("Starting domain pretraining...")
    output_path = pretrain_domain(
        base_model_name=args.base_model,
        texts=texts,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    logger.info(f"Domain pretraining complete!")
    logger.info(f"Model saved to: {output_path}")
    logger.info("")
    logger.info("Next step: Fine-tune on NER task:")
    logger.info(f"  python scripts/train_transformer.py --model domain_adapted \\")
    logger.info(f"      --checkpoint {output_path} --output-dir outputs/dapt_bert_ner")


if __name__ == "__main__":
    main()

