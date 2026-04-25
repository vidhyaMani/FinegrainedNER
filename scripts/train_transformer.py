"""Training script for transformer-based NER models (BERT, RoBERTa, Domain-Adapted).

Usage:
    python scripts/train_transformer.py --model bert
    python scripts/train_transformer.py --model roberta
    python scripts/train_transformer.py --model domain_adapted --checkpoint outputs/domain_pretrained/final
"""

from __future__ import annotations

# Fix SSL certificate issues on macOS - MUST be imported first
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import src.utils.ssl_fix  # noqa: F401, E402

import argparse
import json

import pandas as pd

from src.models.domain_adapted.finetune_domain import DomainAdaptedNER
from src.models.transformer.bert_ner import BertNER
from src.models.transformer.roberta_ner import RobertaNER
from src.schema.entity_schema import EntitySchema
from src.utils.helpers import ensure_dir, get_logger, load_yaml_config, set_seed

logger = get_logger(__name__)


def load_split(path: Path):
    """Load a dataset split and extract tokens + tags."""
    df = pd.read_parquet(path)
    # Handle both possible column names
    tokens_col = "query_tokens" if "query_tokens" in df.columns else "tokens"
    tokens_list = df[tokens_col].tolist()
    tags_list = df["ner_tags"].tolist()
    # Convert numpy arrays to Python lists if needed
    tokens_list = [list(t) if hasattr(t, 'tolist') else t for t in tokens_list]
    tags_list = [list(t) if hasattr(t, 'tolist') else t for t in tags_list]
    return tokens_list, tags_list


def main():
    parser = argparse.ArgumentParser(description="Train transformer NER models")
    parser.add_argument(
        "--model", type=str, choices=["bert", "roberta", "domain_adapted", "hierarchical"],
        default="bert", help="Model type to train",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to domain-pretrained checkpoint (for domain_adapted model)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/annotations",
        help="Directory containing train/val/test splits",
    )
    parser.add_argument(
        "--train-file", type=str, default="train.parquet",
        help="Training data file name (for learning curve experiments)",
    )
    parser.add_argument(
        "--val-file", type=str, default="val.parquet",
        help="Validation data file name",
    )
    parser.add_argument(
        "--test-file", type=str, default="test.parquet",
        help="Test data file name",
    )
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="Training configuration file",
    )
    parser.add_argument(
        "--schema", type=str, default="configs/entity_schema.yaml",
        help="Entity schema configuration file",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for model checkpoints",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configs
    config = load_yaml_config(args.config)
    schema = EntitySchema.from_yaml(args.schema)

    output_dir = args.output_dir or f"outputs/{args.model}_ner"
    ensure_dir(output_dir)

    logger.info(f"Training {args.model} NER model")
    logger.info(f"Schema: {schema.num_labels} labels → {schema.labels}")

    # Load data
    data_dir = Path(args.data_dir)
    train_data = load_split(data_dir / args.train_file)
    val_data = load_split(data_dir / args.val_file)

    logger.info(f"Train file: {args.train_file}")
    logger.info(f"Val file: {args.val_file}")
    logger.info(f"Train: {len(train_data[0]):,} samples, Val: {len(val_data[0]):,} samples")

    # Create model based on type
    if args.model == "bert":
        model_config = config["transformer"]["bert"]
        model = BertNER(
            label2id=schema.label2id,
            id2label=schema.id2label,
            model_name=model_config["model_name"],
            max_length=model_config["max_length"],
            lr=float(model_config["lr"]),
            weight_decay=float(model_config["weight_decay"]),
            warmup_ratio=float(model_config["warmup_ratio"]),
            batch_size=model_config["batch_size"],
            max_epochs=model_config["max_epochs"],
            patience=model_config["patience"],
            fp16=model_config["fp16"],
            gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
            label_all_tokens=model_config["label_all_tokens"],
            output_dir=output_dir,
            seed=args.seed,
        )
    elif args.model == "roberta":
        model_config = config["transformer"]["roberta"]
        model = RobertaNER(
            label2id=schema.label2id,
            id2label=schema.id2label,
            model_name=model_config["model_name"],
            max_length=model_config["max_length"],
            lr=float(model_config["lr"]),
            weight_decay=float(model_config["weight_decay"]),
            warmup_ratio=float(model_config["warmup_ratio"]),
            batch_size=model_config["batch_size"],
            max_epochs=model_config["max_epochs"],
            patience=model_config["patience"],
            fp16=model_config["fp16"],
            gradient_accumulation_steps=model_config["gradient_accumulation_steps"],
            label_all_tokens=model_config["label_all_tokens"],
            output_dir=output_dir,
            seed=args.seed,
        )
    elif args.model == "domain_adapted":
        model_config = config["transformer"]["domain_adapted"]
        checkpoint = args.checkpoint
        if not checkpoint:
            logger.error("--checkpoint is required for domain_adapted model")
            sys.exit(1)
        model = DomainAdaptedNER(
            label2id=schema.label2id,
            id2label=schema.id2label,
            model_name=checkpoint,
            max_length=model_config["max_length"],
            lr=float(model_config["lr"]),
            weight_decay=float(model_config["weight_decay"]),
            warmup_ratio=float(model_config["warmup_ratio"]),
            batch_size=model_config["batch_size"],
            max_epochs=model_config["max_epochs"],
            patience=model_config["patience"],
            fp16=model_config["fp16"],
            gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 1),
            output_dir=output_dir,
            seed=args.seed,
        )
    elif args.model == "hierarchical":
        from src.models.hierarchical.hierarchical_ner import HierarchicalNER
        model_config = config["hierarchical"]
        model = HierarchicalNER(
            label2id=schema.label2id,
            id2label=schema.id2label,
            schema=schema,
            model_name=model_config["model_name"],
            max_length=model_config["max_length"],
            coarse_weight=model_config["coarse_weight"],
            fine_weight=model_config["fine_weight"],
            consistency_penalty=model_config["consistency_penalty"],
            lr=model_config["lr"],
            weight_decay=model_config["weight_decay"],
            warmup_ratio=model_config["warmup_ratio"],
            batch_size=model_config["batch_size"],
            max_epochs=model_config["max_epochs"],
            patience=model_config["patience"],
            fp16=model_config["fp16"],
            output_dir=output_dir,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    # Train
    result = model.train_model(train_data, val_data, use_wandb=args.use_wandb)

    # Save
    model.save(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # Evaluate on test set
    test_path = data_dir / args.test_file
    if test_path.exists():
        test_data = load_split(test_path)
        metrics = model.evaluate(test_data)
        logger.info(f"Test file: {args.test_file}")
        logger.info(f"Test metrics: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")

        with open(Path(output_dir) / "test_metrics.json", "w") as f:
            json.dump(
                {k: v for k, v in metrics.items() if k != "report_str"},
                f, indent=2, default=str,
            )


if __name__ == "__main__":
    main()

