"""CNN-BiLSTM model for sequence labeling NER.

Architecture: Character-level CNN + Word embeddings → BiLSTM → CRF decode.
Adds character-level Conv1D features to capture morphological patterns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
try:
    from torchcrf import CRF
except ImportError:
    from TorchCRF import CRF

from src.models.classical.bilstm_crf import (
    BiLSTMCRF,
    NERDataset,
    collate_fn,
)
from src.utils.helpers import get_logger

logger = get_logger(__name__)


class CharCNN(nn.Module):
    """Character-level CNN for extracting morphological features."""

    def __init__(
        self,
        char_vocab_size: int,
        char_embedding_dim: int = 30,
        num_filters: List[int] = (25, 25, 25),
        kernel_sizes: List[int] = (2, 3, 4),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.char_embeddings = nn.Embedding(
            char_vocab_size, char_embedding_dim, padding_idx=0
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embedding_dim, nf, ks, padding=ks // 2)
            for nf, ks in zip(num_filters, kernel_sizes)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = sum(num_filters)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            char_ids: (batch * seq_len, max_word_len) character indices.

        Returns:
            (batch * seq_len, output_dim) character features.
        """
        # (batch*seq_len, max_word_len, char_embed_dim)
        char_embeds = self.char_embeddings(char_ids)
        # Conv1d expects (batch, channels, length)
        char_embeds = char_embeds.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            out = torch.relu(conv(char_embeds))
            # Max-pool over the sequence length
            out = out.max(dim=2)[0]
            conv_outputs.append(out)

        # (batch*seq_len, total_num_filters)
        features = torch.cat(conv_outputs, dim=1)
        return self.dropout(features)


class CNNBiLSTMCRFModule(nn.Module):
    """PyTorch CNN-BiLSTM-CRF module with character-level CNN."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[np.ndarray] = None,
        char_vocab_size: int = 100,
        char_embedding_dim: int = 30,
        cnn_num_filters: List[int] = (25, 25, 25),
        cnn_kernel_sizes: List[int] = (2, 3, 4),
    ):
        super().__init__()

        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        # Character CNN
        self.char_cnn = CharCNN(
            char_vocab_size=char_vocab_size,
            char_embedding_dim=char_embedding_dim,
            num_filters=cnn_num_filters,
            kernel_sizes=cnn_kernel_sizes,
        )

        lstm_input_dim = embedding_dim + self.char_cnn.output_dim
        self.dropout = nn.Dropout(dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        tag_ids: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_len = word_ids.shape

        word_embeds = self.word_embeddings(word_ids)

        if char_ids is not None:
            max_word_len = char_ids.shape[2]
            char_ids_flat = char_ids.view(batch_size * seq_len, max_word_len)
            char_features = self.char_cnn(char_ids_flat)
            char_features = char_features.view(batch_size, seq_len, -1)
            embeds = torch.cat([word_embeds, char_features], dim=-1)
        else:
            embeds = word_embeds

        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)

        result = {}
        if tag_ids is not None:
            loss = -self.crf(emissions, tag_ids, mask=mask, reduction="mean")
            result["loss"] = loss

        result["predictions"] = self.crf.decode(emissions, mask=mask)
        return result


class CNNBiLSTMCRF(BiLSTMCRF):
    """CNN-BiLSTM-CRF model wrapper.

    Inherits training/inference logic from BiLSTMCRF but uses CNN for
    character features instead of LSTM.
    """

    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        char_embedding_dim: int = 30,
        cnn_num_filters: Tuple[int, ...] = (25, 25, 25),
        cnn_kernel_sizes: Tuple[int, ...] = (2, 3, 4),
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 7,
        grad_clip: float = 5.0,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(
            label2id=label2id,
            id2label=id2label,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            char_embedding_dim=char_embedding_dim,
            use_char_lstm=True,  # We use char features, just via CNN
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            grad_clip=grad_clip,
            device=device,
            seed=seed,
        )
        self.cnn_num_filters = cnn_num_filters
        self.cnn_kernel_sizes = cnn_kernel_sizes

    def _build_model(self, pretrained_embeddings: Optional[np.ndarray] = None):
        """Instantiate the CNN-BiLSTM-CRF module."""
        self.model = CNNBiLSTMCRFModule(
            vocab_size=len(self.word2idx),
            num_labels=self.num_labels,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pretrained_embeddings=pretrained_embeddings,
            char_vocab_size=len(self.char2idx),
            char_embedding_dim=self.char_embedding_dim,
            cnn_num_filters=list(self.cnn_num_filters),
            cnn_kernel_sizes=list(self.cnn_kernel_sizes),
        ).to(self.device)

