"""Baseline NER using spaCy's pretrained models.

Runs spaCy's general-purpose NER and maps its entity types
to the e-commerce schema to quantify the domain gap.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import spacy
from spacy.language import Language

from src.models.base import BaseNERModel
from src.utils.helpers import get_logger

logger = get_logger(__name__)

# Mapping from spaCy's generic entity types to our e-commerce schema
# Only partial mapping is possible — most spaCy types don't align well
SPACY_TO_ECOMMERCE = {
    "ORG": "BRAND",         # Organizations often correspond to brands
    "PRODUCT": "PRODUCT_TYPE",
    "QUANTITY": "SIZE_MEASURE",
    "CARDINAL": "SIZE_MEASURE",
    "MONEY": "ATTRIBUTE_VALUE",
}


def _load_spacy_model(model_name: str = "en_core_web_sm") -> Language:
    """Load a spaCy model, downloading if needed."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.info(f"Downloading spaCy model '{model_name}'...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)


class SpacyBaseline(BaseNERModel):
    """spaCy-based NER baseline for comparison.

    Uses spaCy's pretrained NER pipeline and maps recognized entities
    to the e-commerce entity schema.
    """

    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        model_name: str = "en_core_web_sm",
        entity_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__(label2id, id2label)
        self.model_name = model_name
        self.entity_mapping = entity_mapping or SPACY_TO_ECOMMERCE
        self.nlp = _load_spacy_model(model_name)
        logger.info(f"Initialized spaCy baseline with '{model_name}'")

    def train_model(self, train_data, val_data, **kwargs):
        """No training needed — uses pretrained spaCy model."""
        logger.info("spaCy baseline uses pretrained model. No training needed.")
        return {"note": "pretrained model, no training performed"}

    def predict(
        self,
        texts: List[List[str]],
        **kwargs,
    ) -> List[List[str]]:
        """Predict IOB2 tags using spaCy NER with entity type mapping.

        Args:
            texts: List of token lists.

        Returns:
            List of IOB2 tag lists mapped to e-commerce schema.
        """
        all_predictions = []

        for tokens in texts:
            text = " ".join(tokens)
            doc = self.nlp(text)

            # Initialize all tags as O
            tags = ["O"] * len(tokens)

            # Map spaCy entities to our schema
            for ent in doc.ents:
                mapped_type = self.entity_mapping.get(ent.label_)
                if mapped_type is None:
                    continue

                # Align spaCy char-level spans to our word tokens
                ent_tokens = ent.text.lower().split()
                for i in range(len(tokens)):
                    window = [t.lower() for t in tokens[i: i + len(ent_tokens)]]
                    if window == ent_tokens:
                        full_tag = f"B-{mapped_type}"
                        if full_tag in self.label2id:
                            tags[i] = full_tag
                            for j in range(1, len(ent_tokens)):
                                i_tag = f"I-{mapped_type}"
                                if i + j < len(tags) and i_tag in self.label2id:
                                    tags[i + j] = i_tag
                        break

            all_predictions.append(tags)

        return all_predictions

    def save(self, path):
        """No custom save needed for pretrained spaCy model."""
        logger.info("spaCy baseline uses pretrained model. Nothing to save.")

    def load(self, path):
        """No custom load needed for pretrained spaCy model."""
        logger.info("spaCy baseline uses pretrained model. Nothing to load.")

    def _get_tokens_and_labels(self, data):
        return data[0], data[1]


def run_spacy_baseline(
    tokens_list: List[List[str]],
    tags_list: List[List[str]],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    model_name: str = "en_core_web_sm",
) -> Dict[str, any]:
    """Convenience function to run spaCy baseline and get metrics.

    Args:
        tokens_list: Test set token sequences.
        tags_list: Test set gold tag sequences.
        label2id: Label to ID mapping.
        id2label: ID to label mapping.
        model_name: spaCy model to use.

    Returns:
        Evaluation metrics dict.
    """
    from src.evaluation.intrinsic import compute_ner_metrics

    baseline = SpacyBaseline(label2id, id2label, model_name)
    predictions = baseline.predict(tokens_list)
    metrics = compute_ner_metrics(tags_list, predictions)

    logger.info(f"spaCy Baseline ({model_name}) — F1: {metrics['f1']:.4f}")
    return metrics

