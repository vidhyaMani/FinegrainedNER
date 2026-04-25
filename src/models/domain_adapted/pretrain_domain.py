"""Domain-adaptive pretraining for e-commerce NER.

Continues pretraining BERT/RoBERTa on e-commerce product text
(titles, descriptions, bullet points) using the MLM objective,
producing a domain-adapted checkpoint for downstream NER fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.utils.helpers import get_logger, set_seed

logger = get_logger(__name__)


def prepare_pretraining_corpus(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> HFDataset:
    """Prepare a HuggingFace Dataset for MLM pretraining.

    Args:
        texts: List of product text strings (titles, descriptions, etc.).
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token length.

    Returns:
        Tokenized HuggingFace Dataset.
    """
    dataset = HFDataset.from_dict({"text": texts})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing corpus",
    )
    return dataset


def pretrain_domain(
    base_model_name: str = "bert-base-uncased",
    texts: Optional[List[str]] = None,
    output_dir: str = "outputs/domain_pretrained",
    mlm_probability: float = 0.15,
    max_length: int = 128,
    lr: float = 5e-5,
    batch_size: int = 64,
    max_steps: int = 100000,
    warmup_steps: int = 10000,
    save_steps: int = 10000,
    fp16: bool = True,
    seed: int = 42,
) -> Path:
    """Continue pretraining a language model on domain-specific text.

    Args:
        base_model_name: HuggingFace model name (e.g., 'bert-base-uncased').
        texts: List of domain-specific text strings.
        output_dir: Directory to save the pretrained model.
        mlm_probability: Probability of masking tokens.
        max_length: Maximum sequence length.
        lr: Learning rate.
        batch_size: Training batch size.
        max_steps: Maximum training steps.
        warmup_steps: Warmup steps for learning rate scheduler.
        save_steps: Save checkpoint every N steps.
        fp16: Whether to use mixed precision.
        seed: Random seed.

    Returns:
        Path to the saved domain-pretrained model.
    """
    set_seed(seed)
    output_path = Path(output_dir)

    logger.info(f"Starting domain pretraining from '{base_model_name}'...")
    logger.info(f"Corpus size: {len(texts):,} texts")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(base_model_name)

    dataset = prepare_pretraining_corpus(texts, tokenizer, max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=fp16 and torch.cuda.is_available(),
        logging_steps=500,
        seed=seed,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    logger.info("Training started...")
    trainer.train()

    # Save final model
    final_path = output_path / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"Domain pretraining complete. Model saved to {final_path}")
    return final_path


def extract_product_texts(
    products_df,
    text_columns: List[str] = ("product_title", "product_description", "product_bullet_point"),
    max_texts: Optional[int] = None,
) -> List[str]:
    """Extract and concatenate product text fields for pretraining corpus.

    Args:
        products_df: DataFrame with product metadata.
        text_columns: Columns to extract text from.
        max_texts: Maximum number of texts to include.

    Returns:
        List of cleaned text strings.
    """
    texts = []
    for col in text_columns:
        if col in products_df.columns:
            col_texts = products_df[col].dropna().astype(str).tolist()
            texts.extend(col_texts)

    # Deduplicate and filter empty
    seen = set()
    unique_texts = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            unique_texts.append(t)

    if max_texts:
        unique_texts = unique_texts[:max_texts]

    logger.info(f"Extracted {len(unique_texts):,} unique product texts for pretraining")
    return unique_texts

