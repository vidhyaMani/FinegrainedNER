"""Hierarchical loss functions for coarse-to-fine NER.

Implements losses that jointly optimize coarse-level (PRODUCT vs O) and
fine-grained entity type predictions with consistency constraints.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.utils.helpers import get_logger

logger = get_logger(__name__)


class HierarchicalNERLoss(nn.Module):
    """Hierarchical loss combining coarse and fine-grained NER objectives.

    Loss = coarse_weight * L_coarse + fine_weight * L_fine + consistency_penalty * L_consistency

    Where:
    - L_coarse: CrossEntropy for coarse labels (O, B-PRODUCT, I-PRODUCT)
    - L_fine: CrossEntropy for fine-grained labels (O, B-BRAND, I-BRAND, ...)
    - L_consistency: Penalizes disagreements between coarse and fine predictions
    """

    def __init__(
        self,
        coarse_weight: float = 0.3,
        fine_weight: float = 0.7,
        consistency_penalty: float = 0.1,
        fine_to_coarse_map: Optional[Dict[int, int]] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.consistency_penalty = consistency_penalty
        self.fine_to_coarse_map = fine_to_coarse_map or {}
        self.ignore_index = ignore_index

        self.coarse_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.fine_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def _compute_consistency_loss(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency loss between coarse and fine predictions.

        Penalizes when fine-grained prediction implies a different coarse
        category than the coarse head predicts.
        """
        if not self.fine_to_coarse_map:
            return torch.tensor(0.0, device=coarse_logits.device)

        coarse_preds = coarse_logits.argmax(dim=-1)  # (batch, seq_len)
        fine_preds = fine_logits.argmax(dim=-1)       # (batch, seq_len)

        # Map fine predictions to their expected coarse labels
        expected_coarse = torch.zeros_like(fine_preds)
        for fine_id, coarse_id in self.fine_to_coarse_map.items():
            expected_coarse[fine_preds == fine_id] = coarse_id

        # Count mismatches (only where mask is True)
        mismatches = (coarse_preds != expected_coarse).float() * mask.float()
        consistency_loss = mismatches.sum() / mask.float().sum().clamp(min=1)

        return consistency_loss

    def forward(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        coarse_labels: torch.Tensor,
        fine_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute hierarchical loss.

        Args:
            coarse_logits: (batch, seq_len, num_coarse_labels)
            fine_logits: (batch, seq_len, num_fine_labels)
            coarse_labels: (batch, seq_len) coarse label ids
            fine_labels: (batch, seq_len) fine label ids
            mask: (batch, seq_len) boolean attention mask

        Returns:
            Dict with total_loss, coarse_loss, fine_loss, consistency_loss.
        """
        # Reshape for CrossEntropy: (batch*seq_len, num_labels)
        batch_size, seq_len = fine_labels.shape

        coarse_loss = self.coarse_loss_fn(
            coarse_logits.view(-1, coarse_logits.size(-1)),
            coarse_labels.view(-1),
        )
        fine_loss = self.fine_loss_fn(
            fine_logits.view(-1, fine_logits.size(-1)),
            fine_labels.view(-1),
        )

        if mask is None:
            mask = fine_labels != self.ignore_index

        consistency_loss = self._compute_consistency_loss(coarse_logits, fine_logits, mask)

        total_loss = (
            self.coarse_weight * coarse_loss
            + self.fine_weight * fine_loss
            + self.consistency_penalty * consistency_loss
        )

        return {
            "loss": total_loss,
            "coarse_loss": coarse_loss,
            "fine_loss": fine_loss,
            "consistency_loss": consistency_loss,
        }

