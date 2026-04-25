"""Hierarchical NER model with dual coarse/fine prediction heads.

Architecture: Transformer encoder → Coarse head (PRODUCT vs O)
                                   → Fine head (BRAND, COLOR, etc.)
Uses hierarchical loss for joint optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.models.base import BaseNERModel
from src.models.hierarchical.hierarchical_loss import HierarchicalNERLoss
from src.schema.entity_schema import EntitySchema
from src.utils.helpers import get_device, get_logger, set_seed

logger = get_logger(__name__)


class HierarchicalNERModule(nn.Module):
    """Dual-head transformer for hierarchical NER."""

    def __init__(
        self,
        model_name: str,
        num_coarse_labels: int,
        num_fine_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Coarse-level head: O, B-PRODUCT, I-PRODUCT
        self.coarse_head = nn.Linear(hidden_size, num_coarse_labels)

        # Fine-level head: O, B-BRAND, I-BRAND, B-COLOR, ...
        self.fine_head = nn.Linear(hidden_size, num_fine_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = self.dropout(outputs.last_hidden_state)

        coarse_logits = self.coarse_head(hidden)
        fine_logits = self.fine_head(hidden)

        return {
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
        }


class HierarchicalNERDataset(Dataset):
    """Dataset producing both coarse and fine label sequences."""

    def __init__(
        self,
        tokens_list: List[List[str]],
        tags_list: List[List[str]],
        tokenizer: AutoTokenizer,
        schema: EntitySchema,
        max_length: int = 64,
    ):
        self.tokens_list = tokens_list
        self.tags_list = tags_list
        self.tokenizer = tokenizer
        self.schema = schema
        self.max_length = max_length

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        tags = self.tags_list[idx]

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)

        # Build fine and coarse labels aligned to subword tokens
        fine_label_ids = []
        coarse_label_ids = []
        prev_word_id = None

        for word_id in word_ids:
            if word_id is None:
                fine_label_ids.append(-100)
                coarse_label_ids.append(-100)
            elif word_id != prev_word_id:
                tag = tags[word_id] if word_id < len(tags) else "O"
                fine_label_ids.append(self.schema.label2id.get(tag, 0))
                coarse_tag = self.schema.fine_to_coarse_label(tag)
                coarse_label_ids.append(self.schema.coarse_label2id.get(coarse_tag, 0))
            else:
                fine_label_ids.append(-100)
                coarse_label_ids.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "fine_labels": torch.tensor(fine_label_ids, dtype=torch.long),
            "coarse_labels": torch.tensor(coarse_label_ids, dtype=torch.long),
        }


class HierarchicalNER(BaseNERModel):
    """Hierarchical NER model with joint coarse/fine prediction."""

    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        schema: Optional[EntitySchema] = None,
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
        coarse_weight: float = 0.3,
        fine_weight: float = 0.7,
        consistency_penalty: float = 0.1,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        batch_size: int = 32,
        max_epochs: int = 10,
        patience: int = 3,
        fp16: bool = True,
        output_dir: str = "outputs/hierarchical_ner",
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__(label2id, id2label)
        self.schema = schema or EntitySchema()
        self.model_name = model_name
        self.max_length = max_length
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.consistency_penalty = consistency_penalty
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.fp16 = fp16
        self.output_dir = output_dir
        self.device_pref = device
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Optional[HierarchicalNERModule] = None

        # Build fine→coarse label mapping for consistency loss
        self._fine_to_coarse_id_map = self._build_label_map()

    def _build_label_map(self) -> Dict[int, int]:
        """Map fine-grained label ids to coarse label ids."""
        mapping = {}
        for fine_label, fine_id in self.schema.label2id.items():
            coarse_label = self.schema.fine_to_coarse_label(fine_label)
            coarse_id = self.schema.coarse_label2id.get(coarse_label, 0)
            mapping[fine_id] = coarse_id
        return mapping

    def train_model(
        self,
        train_data: Tuple[List[List[str]], List[List[str]]],
        val_data: Tuple[List[List[str]], List[List[str]]],
        **kwargs,
    ) -> Dict[str, Any]:
        set_seed(self.seed)
        device = get_device(self.device_pref)

        train_tokens, train_tags = train_data
        val_tokens, val_tags = val_data

        train_dataset = HierarchicalNERDataset(
            train_tokens, train_tags, self.tokenizer, self.schema, self.max_length,
        )
        val_dataset = HierarchicalNERDataset(
            val_tokens, val_tags, self.tokenizer, self.schema, self.max_length,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = HierarchicalNERModule(
            model_name=self.model_name,
            num_coarse_labels=len(self.schema.coarse_labels),
            num_fine_labels=self.schema.num_labels,
        ).to(device)

        criterion = HierarchicalNERLoss(
            coarse_weight=self.coarse_weight,
            fine_weight=self.fine_weight,
            consistency_penalty=self.consistency_penalty,
            fine_to_coarse_map=self._fine_to_coarse_id_map,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )

        # Linear warmup scheduler
        total_steps = len(train_loader) * self.max_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_steps,
        )

        history = {"train_loss": [], "val_loss": [], "val_fine_loss": [], "val_coarse_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        use_amp = self.fp16 and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        for epoch in range(1, self.max_epochs + 1):
            # Training
            self.model.train()
            epoch_loss = 0
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    losses = criterion(
                        coarse_logits=outputs["coarse_logits"],
                        fine_logits=outputs["fine_logits"],
                        coarse_labels=batch["coarse_labels"],
                        fine_labels=batch["fine_labels"],
                        mask=batch["attention_mask"].bool(),
                    )
                    loss = losses["loss"]

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    losses = criterion(
                        coarse_logits=outputs["coarse_logits"],
                        fine_logits=outputs["fine_logits"],
                        coarse_labels=batch["coarse_labels"],
                        fine_labels=batch["fine_labels"],
                        mask=batch["attention_mask"].bool(),
                    )
                    val_loss += losses["loss"].item()

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            logger.info(
                f"Epoch {epoch}/{self.max_epochs}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
            self.model.to(device)

        return history

    def predict(
        self,
        texts: List[List[str]],
        return_coarse: bool = False,
        **kwargs,
    ) -> List[List[str]]:
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        device = next(self.model.parameters()).device
        self.model.eval()

        all_predictions = []
        for tokens in texts:
            encoding = self.tokenizer(
                tokens,
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                )

            if return_coarse:
                logits = outputs["coarse_logits"]
                label_map = self.schema.coarse_id2label
            else:
                logits = outputs["fine_logits"]
                label_map = self.schema.id2label

            preds = logits.argmax(dim=-1)[0].cpu().numpy()
            word_ids = encoding.word_ids(batch_index=0)

            word_preds = []
            prev_word_id = None
            for pred_id, word_id in zip(preds, word_ids):
                if word_id is None:
                    continue
                if word_id != prev_word_id:
                    word_preds.append(label_map.get(pred_id, "O"))
                prev_word_id = word_id

            all_predictions.append(word_preds)

        return all_predictions

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        self.tokenizer.save_pretrained(path)

        config = {
            "model_name": self.model_name,
            "num_coarse_labels": len(self.schema.coarse_labels),
            "num_fine_labels": self.schema.num_labels,
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved Hierarchical NER model to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = HierarchicalNERModule(
            model_name=config["model_name"],
            num_coarse_labels=config["num_coarse_labels"],
            num_fine_labels=config["num_fine_labels"],
        )
        self.model.load_state_dict(
            torch.load(path / "model.pt", map_location=get_device(self.device_pref))
        )
        self.label2id = config["label2id"]
        self.id2label = {int(k): v for k, v in config["id2label"].items()}
        logger.info(f"Loaded Hierarchical NER model from {path}")

    def _get_tokens_and_labels(self, data):
        return data[0], data[1]

