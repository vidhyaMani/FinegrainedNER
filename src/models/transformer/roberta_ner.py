"""RoBERTa-based token classification for NER.

Extends BertNER for RoBERTa with appropriate defaults.
RoBERTa differs from BERT in tokenization (byte-level BPE)
and pre-training (no NSP, dynamic masking).
"""

from __future__ import annotations

from typing import Dict

from src.models.transformer.bert_ner import BertNER
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class RobertaNER(BertNER):
    """RoBERTa-based NER model.

    Inherits all logic from BertNER; only changes the default model name
    and adjusts subword handling for RoBERTa's byte-level BPE tokenizer.
    """

    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        model_name: str = "roberta-base",
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
        output_dir: str = "outputs/roberta_ner",
        seed: int = 42,
    ):
        super().__init__(
            label2id=label2id,
            id2label=id2label,
            model_name=model_name,
            max_length=max_length,
            lr=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            fp16=fp16,
            gradient_accumulation_steps=gradient_accumulation_steps,
            label_all_tokens=label_all_tokens,
            output_dir=output_dir,
            seed=seed,
        )
        logger.info(f"Initialized RoBERTa NER model with '{model_name}'")

