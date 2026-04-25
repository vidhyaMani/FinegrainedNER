"""Entity schema definitions for e-commerce NER.

Provides the hierarchical coarse-to-fine entity schema, IOB2 label mappings,
and helper utilities for working with the entity taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class CoarseEntity(str, Enum):
    """Coarse-level entity types."""
    PRODUCT = "PRODUCT"
    O = "O"  # Outside any entity


class FineEntity(str, Enum):
    """Fine-grained entity types under PRODUCT."""
    PRODUCT_TYPE = "PRODUCT_TYPE"
    BRAND = "BRAND"
    MATERIAL = "MATERIAL"
    COLOR = "COLOR"
    SIZE_MEASURE = "SIZE_MEASURE"
    ATTRIBUTE_VALUE = "ATTRIBUTE_VALUE"


# Map fine → coarse
FINE_TO_COARSE: Dict[FineEntity, CoarseEntity] = {
    FineEntity.PRODUCT_TYPE: CoarseEntity.PRODUCT,
    FineEntity.BRAND: CoarseEntity.PRODUCT,
    FineEntity.MATERIAL: CoarseEntity.PRODUCT,
    FineEntity.COLOR: CoarseEntity.PRODUCT,
    FineEntity.SIZE_MEASURE: CoarseEntity.PRODUCT,
    FineEntity.ATTRIBUTE_VALUE: CoarseEntity.PRODUCT,
}

# Priority order for overlap resolution (lower = higher priority)
ENTITY_PRIORITY: Dict[FineEntity, int] = {
    FineEntity.PRODUCT_TYPE: 1,
    FineEntity.BRAND: 2,
    FineEntity.MATERIAL: 3,
    FineEntity.COLOR: 4,
    FineEntity.SIZE_MEASURE: 5,
    FineEntity.ATTRIBUTE_VALUE: 6,
}


@dataclass
class EntitySchema:
    """Complete entity schema with IOB2 label definitions."""

    tagging_scheme: str = "IOB2"
    fine_entities: List[FineEntity] = field(default_factory=lambda: list(FineEntity))

    @property
    def labels(self) -> List[str]:
        """Return the full IOB2 label list including O."""
        tags = ["O"]
        for entity in self.fine_entities:
            tags.append(f"B-{entity.value}")
            tags.append(f"I-{entity.value}")
        return tags

    @property
    def label2id(self) -> Dict[str, int]:
        """Map label string → integer id."""
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id2label(self) -> Dict[int, str]:
        """Map integer id → label string."""
        return {idx: label for idx, label in enumerate(self.labels)}

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    @property
    def coarse_labels(self) -> List[str]:
        """Coarse-level labels: O, B-PRODUCT, I-PRODUCT."""
        return ["O", "B-PRODUCT", "I-PRODUCT"]

    @property
    def coarse_label2id(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.coarse_labels)}

    @property
    def coarse_id2label(self) -> Dict[int, str]:
        return {idx: label for idx, label in enumerate(self.coarse_labels)}

    def fine_to_coarse_label(self, fine_label: str) -> str:
        """Convert a fine-grained IOB2 label to its coarse equivalent."""
        if fine_label == "O":
            return "O"
        prefix, entity_type = fine_label.split("-", 1)
        return f"{prefix}-PRODUCT"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EntitySchema":
        """Load schema from a YAML config file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        schema_cfg = config.get("schema", config)
        fine_names = list(schema_cfg.get("fine_entities", {}).keys())
        fine_entities = [FineEntity(name) for name in fine_names]
        return cls(
            tagging_scheme=schema_cfg.get("tagging_scheme", "IOB2"),
            fine_entities=fine_entities,
        )


# Default global schema instance
DEFAULT_SCHEMA = EntitySchema()

