"""Training script for classical NER models (BiLSTM-CRF, CNN-BiLSTM).

Usage:
    python scripts/train_classical.py --model bilstm_crf
    python scripts/train_classical.py --model cnn_bilstm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.models.classical.bilstm_crf import BiLSTMCRF
from src.models.classical.cnn_bilstm import CNNBiLSTMCRF
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
    parser = argparse.ArgumentParser(description="Train classical NER models")
    parser.add_argument(
        "--model", type=str, choices=["bilstm_crf", "cnn_bilstm"],
        default="bilstm_crf", help="Model type to train",
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load configs
    config = load_yaml_config(args.config)
    schema = EntitySchema.from_yaml(args.schema)
    model_config = config["classical"][args.model]

    output_dir = args.output_dir or f"outputs/{args.model}"
    ensure_dir(output_dir)

    logger.info(f"Training {args.model} model")
    logger.info(f"Schema: {schema.num_labels} labels")
    logger.info(f"Config: {model_config}")

    # Load data
    data_dir = Path(args.data_dir)
    train_data = load_split(data_dir / args.train_file)
    val_data = load_split(data_dir / args.val_file)

    logger.info(f"Train file: {args.train_file}")
    logger.info(f"Val file: {args.val_file}")
    logger.info(f"Train: {len(train_data[0]):,} samples, Val: {len(val_data[0]):,} samples")

    # Create model
    if args.model == "bilstm_crf":
        model = BiLSTMCRF(
            label2id=schema.label2id,
            id2label=schema.id2label,
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            use_char_lstm=model_config.get("use_char_embeddings", False),
            lr=model_config["lr"],
            weight_decay=float(model_config["weight_decay"]),
            batch_size=model_config["batch_size"],
            max_epochs=model_config["max_epochs"],
            patience=model_config["patience"],
            grad_clip=model_config.get("clip_grad", 5.0),
            seed=args.seed,
        )
    else:
        model = CNNBiLSTMCRF(
            label2id=schema.label2id,
            id2label=schema.id2label,
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            char_embedding_dim=model_config.get("char_embedding_dim", 30),
            cnn_num_filters=tuple(model_config.get("char_cnn_filters", [25, 25, 25])),
            cnn_kernel_sizes=tuple(model_config.get("char_cnn_kernel_sizes", [2, 3, 4])),
            lr=model_config["lr"],
            weight_decay=float(model_config["weight_decay"]),
            batch_size=model_config["batch_size"],
            max_epochs=model_config["max_epochs"],
            patience=model_config["patience"],
            grad_clip=model_config.get("clip_grad", 5.0),
            seed=args.seed,
        )

    # Train
    history = model.train_model(train_data, val_data)

    # Save model and history
    model.save(output_dir)
    with open(Path(output_dir) / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Model saved to {output_dir}")

    # Evaluate on test set if available
    test_path = data_dir / args.test_file
    if test_path.exists():
        test_tokens, test_tags = load_split(test_path)
        pred_tags = model.predict(test_tokens)

        from src.evaluation.intrinsic import compute_ner_metrics
        metrics = compute_ner_metrics(test_tags, pred_tags)
        logger.info(f"Test file: {args.test_file}")
        logger.info(f"Test F1: {metrics['f1']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")

        with open(Path(output_dir) / "test_metrics.json", "w") as f:
            json.dump(
                {k: v for k, v in metrics.items() if k != "report_str"},
                f, indent=2, default=str,
            )


if __name__ == "__main__":
    main()

