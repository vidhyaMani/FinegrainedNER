"""BERT-based token classification for NER.

Fine-tunes bert-base-uncased (or any BERT variant) for token-level
NER with proper subword alignment.
"""

from __future__ import annotations

import src.utils.ssl_fix  # noqa: F401

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.models.base import BaseNERModel
from src.utils.helpers import get_logger, set_seed

logger = get_logger(__name__)


def _align_labels_with_tokens(
    labels: List[int],
    word_ids: List[Optional[int]],
    label_all_tokens: bool = False,
) -> List[int]:
    """Align IOB2 label ids to subword tokens.

    For each subword token:
    - If it's the first piece of a word → keep the original label.
    - If it's a continuation piece → use -100 (ignored in loss) or I- tag.
    - Special tokens (None word_id) → -100.

    Args:
        labels: Original word-level label ids.
        word_ids: HuggingFace tokenizer word_ids() output.
        label_all_tokens: If True, label continuation tokens with I- version.

    Returns:
        List of aligned label ids.
    """
    aligned = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != prev_word_id:
            # First subword of a new word
            aligned.append(labels[word_id] if word_id < len(labels) else -100)
        else:
            # Continuation subword
            if label_all_tokens:
                label = labels[word_id] if word_id < len(labels) else -100
                aligned.append(label)
            else:
                aligned.append(-100)
        prev_word_id = word_id

    return aligned


def tokenize_and_align(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    label2id: Dict[str, int],
    max_length: int = 64,
    label_all_tokens: bool = False,
) -> Dict[str, List]:
    """Tokenize and align labels for a batch of examples.

    Expects examples to have 'tokens' and 'ner_tags' keys.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,
        max_length=max_length,
        is_split_into_words=True,
    )

    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        label_ids = [label2id.get(tag, 0) for tag in tags]
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = _align_labels_with_tokens(label_ids, word_ids, label_all_tokens)
        all_labels.append(aligned)

    tokenized["labels"] = all_labels
    return tokenized


class BertNER(BaseNERModel):
    """BERT-based NER model using HuggingFace Transformers."""

    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        batch_size: int = 32,
        max_epochs: int = 10,
        patience: int = 3,
        fp16: bool = True,
        gradient_accumulation_steps: int = 2,
        label_all_tokens: bool = False,
        output_dir: str = "outputs/bert_ner",
        seed: int = 42,
    ):
        super().__init__(label2id, id2label)
        self.model_name = model_name
        self.max_length = max_length
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_all_tokens = label_all_tokens
        self.output_dir = output_dir
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

    def _prepare_dataset(
        self,
        tokens_list: List[List[str]],
        tags_list: List[List[str]],
    ) -> HFDataset:
        """Convert token/tag lists to a HuggingFace Dataset."""
        dataset = HFDataset.from_dict({
            "tokens": tokens_list,
            "ner_tags": tags_list,
        })
        dataset = dataset.map(
            lambda examples: tokenize_and_align(
                examples, self.tokenizer, self.label2id,
                self.max_length, self.label_all_tokens,
            ),
            batched=True,
            remove_columns=dataset.column_names,
        )
        return dataset

    def _compute_metrics(self, eval_pred):
        """Compute seqeval metrics for HuggingFace Trainer."""
        from seqeval.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        pred_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_clean = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                true_seq.append(self.id2label.get(label_id, "O"))
                pred_seq_clean.append(self.id2label.get(pred_id, "O"))
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_clean)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
        }

    def train_model(
        self,
        train_data: Tuple[List[List[str]], List[List[str]]],
        val_data: Tuple[List[List[str]], List[List[str]]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Fine-tune BERT for NER.

        Args:
            train_data: (tokens_list, tags_list).
            val_data: (tokens_list, tags_list).

        Returns:
            Training metrics.
        """
        set_seed(self.seed)

        train_tokens, train_tags = train_data
        val_tokens, val_tags = val_data

        train_dataset = self._prepare_dataset(train_tokens, train_tags)
        val_dataset = self._prepare_dataset(val_tokens, val_tags)

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            warmup_steps=int(len(train_dataset) / self.batch_size * self.max_epochs * self.warmup_ratio),
            fp16=self.fp16 and torch.cuda.is_available(),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            logging_steps=50,
            seed=self.seed,
            report_to="wandb" if kwargs.get("use_wandb", False) else "none",
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)],
        )

        result = trainer.train()
        metrics = trainer.evaluate()

        logger.info(f"Training complete. Metrics: {metrics}")
        return {"train_result": result, "eval_metrics": metrics}

    def predict(
        self,
        texts: List[List[str]],
        **kwargs,
    ) -> List[List[str]]:
        """Predict IOB2 tags for tokenized texts."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        self.model.eval()
        device = next(self.model.parameters()).device

        all_predictions = []
        for tokens in texts:
            inputs = self.tokenizer(
                tokens,
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = outputs.logits.argmax(dim=-1)[0].cpu().numpy()
            word_ids = inputs.word_ids(batch_index=0)

            # Map back to word-level predictions
            word_preds = []
            prev_word_id = None
            for pred_id, word_id in zip(predictions, word_ids):
                if word_id is None:
                    continue
                if word_id != prev_word_id:
                    word_preds.append(self.id2label.get(pred_id, "O"))
                prev_word_id = word_id

            all_predictions.append(word_preds)

        return all_predictions

    def save(self, path: str | Path) -> None:
        """Save model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Saved BERT NER model to {path}")

    def load(self, path: str | Path) -> None:
        """Load model and tokenizer."""
        path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.label2id = self.model.config.label2id
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.num_labels = len(self.label2id)
        logger.info(f"Loaded BERT NER model from {path}")

    def _get_tokens_and_labels(self, data):
        return data[0], data[1]

