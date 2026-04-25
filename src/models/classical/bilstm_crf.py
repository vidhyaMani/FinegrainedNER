"""BiLSTM-CRF model for sequence labeling NER.

Architecture: Pretrained word embeddings → BiLSTM → CRF decode.
Optionally uses character-level embeddings via a separate char LSTM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
try:
    from torchcrf import CRF
except ImportError:
    from TorchCRF import CRF
from tqdm import tqdm

from src.models.base import BaseNERModel
from src.utils.helpers import get_device, get_logger, set_seed

logger = get_logger(__name__)


class NERDataset(Dataset):
    """Dataset for token-level NER with word and character indices."""

    def __init__(
        self,
        tokens_list: List[List[str]],
        tags_list: List[List[str]],
        word2idx: Dict[str, int],
        label2id: Dict[str, int],
        char2idx: Optional[Dict[str, int]] = None,
        max_word_len: int = 30,
    ):
        self.tokens_list = tokens_list
        self.tags_list = tags_list
        self.word2idx = word2idx
        self.label2id = label2id
        self.char2idx = char2idx
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        tags = self.tags_list[idx]

        word_ids = [self.word2idx.get(t.lower(), self.word2idx.get("<UNK>", 1)) for t in tokens]
        tag_ids = [self.label2id.get(t, 0) for t in tags]

        item = {
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "tag_ids": torch.tensor(tag_ids, dtype=torch.long),
            "lengths": len(tokens),
        }

        if self.char2idx is not None:
            char_ids = []
            for token in tokens:
                c_ids = [self.char2idx.get(c, self.char2idx.get("<UNK>", 1))
                         for c in token[:self.max_word_len]]
                # Pad to max_word_len
                c_ids += [0] * (self.max_word_len - len(c_ids))
                char_ids.append(c_ids)
            item["char_ids"] = torch.tensor(char_ids, dtype=torch.long)

        return item


def collate_fn(batch):
    """Collate function with dynamic padding."""
    max_len = max(item["lengths"] for item in batch)

    word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    tag_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    lengths = []

    has_chars = "char_ids" in batch[0]
    if has_chars:
        max_word_len = batch[0]["char_ids"].shape[1]
        char_ids = torch.zeros(len(batch), max_len, max_word_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["lengths"]
        word_ids[i, :seq_len] = item["word_ids"]
        tag_ids[i, :seq_len] = item["tag_ids"]
        mask[i, :seq_len] = True
        lengths.append(seq_len)
        if has_chars:
            char_ids[i, :seq_len] = item["char_ids"]

    result = {
        "word_ids": word_ids,
        "tag_ids": tag_ids,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
    if has_chars:
        result["char_ids"] = char_ids

    return result


class BiLSTMCRFModule(nn.Module):
    """PyTorch BiLSTM-CRF module."""

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[np.ndarray] = None,
        char_vocab_size: int = 0,
        char_embedding_dim: int = 50,
        char_hidden_dim: int = 50,
        use_char_lstm: bool = False,
    ):
        super().__init__()

        self.use_char_lstm = use_char_lstm and char_vocab_size > 0

        # Word embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.word_embeddings.weight.requires_grad = True  # allow fine-tuning

        # Character-level LSTM (optional)
        lstm_input_dim = embedding_dim
        if self.use_char_lstm:
            self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
            self.char_lstm = nn.LSTM(
                char_embedding_dim, char_hidden_dim,
                batch_first=True, bidirectional=True,
            )
            lstm_input_dim += char_hidden_dim * 2

        self.dropout = nn.Dropout(dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Projection to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

    def _get_char_features(self, char_ids: torch.Tensor) -> torch.Tensor:
        """Get character-level features via char LSTM.

        Args:
            char_ids: (batch, seq_len, max_word_len)

        Returns:
            (batch, seq_len, char_hidden_dim * 2)
        """
        batch_size, seq_len, max_word_len = char_ids.shape
        char_ids = char_ids.view(batch_size * seq_len, max_word_len)
        char_embeds = self.char_embeddings(char_ids)
        _, (h_n, _) = self.char_lstm(char_embeds)
        # h_n: (2, batch*seq_len, char_hidden_dim)
        char_features = torch.cat([h_n[0], h_n[1]], dim=-1)
        return char_features.view(batch_size, seq_len, -1)

    def forward(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        tag_ids: Optional[torch.Tensor] = None,
        char_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            word_ids: (batch, seq_len) word indices.
            mask: (batch, seq_len) boolean mask.
            tag_ids: (batch, seq_len) gold tag indices (for training).
            char_ids: (batch, seq_len, max_word_len) character indices.

        Returns:
            Dict with 'loss' (if tag_ids given) and/or 'predictions'.
        """
        embeds = self.word_embeddings(word_ids)

        if self.use_char_lstm and char_ids is not None:
            char_features = self._get_char_features(char_ids)
            embeds = torch.cat([embeds, char_features], dim=-1)

        embeds = self.dropout(embeds)

        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)

        emissions = self.hidden2tag(lstm_out)

        result = {}
        if tag_ids is not None:
            loss = -self.crf(emissions, tag_ids, mask=mask, reduction="mean")
            result["loss"] = loss

        predictions = self.crf.decode(emissions, mask=mask)
        result["predictions"] = predictions

        return result


class BiLSTMCRF(BaseNERModel):
    """BiLSTM-CRF NER model wrapper with training and inference logic."""

    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        char_embedding_dim: int = 50,
        char_hidden_dim: int = 50,
        use_char_lstm: bool = False,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 7,
        grad_clip: float = 5.0,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(label2id, id2label)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.use_char_lstm = use_char_lstm
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = get_device(device)
        self.seed = seed

        self.word2idx: Dict[str, int] = {}
        self.char2idx: Dict[str, int] = {}
        self.model: Optional[BiLSTMCRFModule] = None

    def _build_vocab(
        self,
        tokens_list: List[List[str]],
    ) -> None:
        """Build word and character vocabularies from training data."""
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}

        for tokens in tokens_list:
            for token in tokens:
                lower = token.lower()
                if lower not in self.word2idx:
                    self.word2idx[lower] = len(self.word2idx)
                for char in token:
                    if char not in self.char2idx:
                        self.char2idx[char] = len(self.char2idx)

        logger.info(f"Vocabulary: {len(self.word2idx):,} words, {len(self.char2idx):,} chars")

    def _build_model(self, pretrained_embeddings: Optional[np.ndarray] = None):
        """Instantiate the BiLSTM-CRF module."""
        self.model = BiLSTMCRFModule(
            vocab_size=len(self.word2idx),
            num_labels=self.num_labels,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pretrained_embeddings=pretrained_embeddings,
            char_vocab_size=len(self.char2idx) if self.use_char_lstm else 0,
            char_embedding_dim=self.char_embedding_dim,
            char_hidden_dim=self.char_hidden_dim,
            use_char_lstm=self.use_char_lstm,
        ).to(self.device)

    def train_model(
        self,
        train_data: Tuple[List[List[str]], List[List[str]]],
        val_data: Tuple[List[List[str]], List[List[str]]],
        pretrained_embeddings: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the BiLSTM-CRF model.

        Args:
            train_data: (tokens_list, tags_list) for training.
            val_data: (tokens_list, tags_list) for validation.
            pretrained_embeddings: Optional pretrained word embedding matrix.

        Returns:
            Training history dict.
        """
        set_seed(self.seed)
        train_tokens, train_tags = train_data
        val_tokens, val_tags = val_data

        # Build vocab from training data
        self._build_vocab(train_tokens)
        self._build_model(pretrained_embeddings)

        # Create datasets and dataloaders
        train_dataset = NERDataset(
            train_tokens, train_tags, self.word2idx, self.label2id,
            self.char2idx if self.use_char_lstm else None,
        )
        val_dataset = NERDataset(
            val_tokens, val_tags, self.word2idx, self.label2id,
            self.char2idx if self.use_char_lstm else None,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.max_epochs} [train]"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                output = self.model(
                    word_ids=batch["word_ids"],
                    mask=batch["mask"],
                    tag_ids=batch["tag_ids"],
                    char_ids=batch.get("char_ids"),
                )
                loss = output["loss"]
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{self.max_epochs} [val]"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    output = self.model(
                        word_ids=batch["word_ids"],
                        mask=batch["mask"],
                        tag_ids=batch["tag_ids"],
                        char_ids=batch.get("char_ids"),
                    )
                    val_loss += output["loss"].item()

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            logger.info(
                f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
            self.model.to(self.device)

        return history

    def predict(
        self,
        texts: List[List[str]],
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> List[List[str]]:
        """Predict IOB2 tags for tokenized texts."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        batch_size = batch_size or self.batch_size
        self.model.eval()

        # Create a dummy dataset with "O" tags
        dummy_tags = [["O"] * len(tokens) for tokens in texts]
        dataset = NERDataset(
            texts, dummy_tags, self.word2idx, self.label2id,
            self.char2idx if self.use_char_lstm else None,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )

        all_predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(
                    word_ids=batch["word_ids"],
                    mask=batch["mask"],
                    char_ids=batch.get("char_ids"),
                )
                for pred_ids, length in zip(output["predictions"], batch["lengths"]):
                    tags = [self.id2label.get(pid, "O") for pid in pred_ids[:length]]
                    all_predictions.append(tags)

        return all_predictions

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path / "model.pt")

        config = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "word2idx": self.word2idx,
            "char2idx": self.char2idx,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_char_lstm": self.use_char_lstm,
            "char_embedding_dim": self.char_embedding_dim,
            "char_hidden_dim": self.char_hidden_dim,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved BiLSTM-CRF model to {path}")

    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        self.label2id = config["label2id"]
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        self.word2idx = config["word2idx"]
        self.char2idx = config["char2idx"]
        self.num_labels = len(self.label2id)
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.use_char_lstm = config["use_char_lstm"]

        self._build_model()
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))
        self.model.eval()

        logger.info(f"Loaded BiLSTM-CRF model from {path}")

    def _get_tokens_and_labels(self, data):
        """Extract tokens and labels for evaluation."""
        return data[0], data[1]

