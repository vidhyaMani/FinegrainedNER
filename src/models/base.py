"""Abstract base class for all NER models.

Provides a common interface for training, prediction, evaluation,
and serialization across all model families.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class BaseNERModel(ABC):
    """Abstract base class for NER models."""

    def __init__(self, label2id: Dict[str, int], id2label: Dict[int, str], **kwargs):
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(label2id)

    @abstractmethod
    def train_model(
        self,
        train_data: Any,
        val_data: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.

        Returns:
            Dict with training metrics / history.
        """
        ...

    @abstractmethod
    def predict(
        self,
        texts: List[List[str]],
        **kwargs,
    ) -> List[List[str]]:
        """Predict IOB2 tags for a batch of tokenized texts.

        Args:
            texts: List of token lists.

        Returns:
            List of predicted tag lists.
        """
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model weights and config to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model weights and config from disk."""
        ...

    def evaluate(
        self,
        test_data: Any,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate the model on test data.

        Default implementation uses predict + seqeval metrics.
        Subclasses can override for custom evaluation.
        """
        from src.evaluation.intrinsic import compute_ner_metrics

        # Subclasses should implement _get_tokens_and_labels
        tokens_list, true_labels = self._get_tokens_and_labels(test_data)
        pred_labels = self.predict(tokens_list)
        return compute_ner_metrics(true_labels, pred_labels)

    def _get_tokens_and_labels(
        self, data: Any
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Extract token lists and label lists from a dataset.

        Subclasses should override this for their specific data format.
        """
        raise NotImplementedError(
            "Subclasses must implement _get_tokens_and_labels for evaluation."
        )

