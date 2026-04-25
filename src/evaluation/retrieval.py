"""Retrieval evaluation for NER in e-commerce.

Implements downstream retrieval evaluation:
- BM25/TF-IDF baseline over product text
- Attribute-weighted reranking using predicted entities
- Metrics: nDCG@K, Recall@K, MRR@K

This module evaluates how well NER predictions improve product search.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.utils.helpers import get_logger

logger = get_logger(__name__)


def compute_ndcg_at_k(
    relevance_scores: List[float],
    k: int = 10,
) -> float:
    """Compute normalized Discounted Cumulative Gain at K.

    Args:
        relevance_scores: List of relevance scores for ranked results.
        k: Cutoff rank.

    Returns:
        nDCG@K score.
    """
    relevance_scores = relevance_scores[:k]

    # DCG
    dcg = sum(
        rel / math.log2(i + 2)  # +2 because log2(1) = 0
        for i, rel in enumerate(relevance_scores)
    )

    # Ideal DCG (sorted relevance)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal_relevance)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: set,
    k: int = 10,
) -> float:
    """Compute Recall at K.

    Args:
        retrieved_ids: List of retrieved product IDs in ranked order.
        relevant_ids: Set of relevant product IDs.
        k: Cutoff rank.

    Returns:
        Recall@K score.
    """
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k & relevant_ids)

    return hits / len(relevant_ids)


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: set,
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank at K.

    Args:
        retrieved_ids: List of retrieved product IDs in ranked order.
        relevant_ids: Set of relevant product IDs.
        k: Cutoff rank.

    Returns:
        MRR@K score.
    """
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)

    return 0.0


class BM25Retriever:
    """BM25-based product retriever."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Initialize BM25 retriever.

        Args:
            k1: BM25 k1 parameter.
            b: BM25 b parameter.
        """
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0.0
        self.corpus_size = 0
        self.inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.doc_ids: List[str] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def index(
        self,
        documents: Dict[str, str],
    ):
        """Index documents for retrieval.

        Args:
            documents: Dict mapping doc_id -> text.
        """
        self.doc_ids = list(documents.keys())
        self.corpus_size = len(documents)

        total_length = 0
        for doc_id, text in documents.items():
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Count term frequencies
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            # Update inverted index and doc freqs
            for term, freq in term_freqs.items():
                self.inverted_index[term][doc_id] = freq
                self.doc_freqs[term] += 1

        self.avg_doc_length = total_length / self.corpus_size if self.corpus_size > 0 else 0
        logger.info(f"Indexed {self.corpus_size} documents")

    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term."""
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def _score_document(
        self,
        query_terms: List[str],
        doc_id: str,
    ) -> float:
        """Compute BM25 score for a document."""
        doc_length = self.doc_lengths.get(doc_id, 0)
        if doc_length == 0:
            return 0.0

        score = 0.0
        for term in query_terms:
            if term not in self.inverted_index:
                continue

            tf = self.inverted_index[term].get(doc_id, 0)
            if tf == 0:
                continue

            idf = self._compute_idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

            score += idf * numerator / denominator

        return score

    def retrieve(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Retrieve top-k documents for a query.

        Args:
            query: Query string.
            k: Number of documents to retrieve.

        Returns:
            List of (doc_id, score) tuples.
        """
        query_terms = self._tokenize(query)

        # Get candidate documents (those containing at least one query term)
        candidates = set()
        for term in query_terms:
            candidates.update(self.inverted_index.get(term, {}).keys())

        # Score candidates
        scores = []
        for doc_id in candidates:
            score = self._score_document(query_terms, doc_id)
            scores.append((doc_id, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]


class AttributeWeightedReranker:
    """Reranker that boosts documents matching NER-extracted attributes."""

    def __init__(
        self,
        attribute_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize reranker.

        Args:
            attribute_weights: Dict mapping entity_type -> weight.
        """
        self.attribute_weights = attribute_weights or {
            "BRAND": 2.0,
            "PRODUCT_TYPE": 1.5,
            "COLOR": 1.2,
            "MATERIAL": 1.2,
            "SIZE_MEASURE": 1.0,
            "ATTRIBUTE_VALUE": 1.0,
        }
        self.product_attributes: Dict[str, Dict[str, set]] = {}

    def index_product_attributes(
        self,
        products: Dict[str, Dict[str, Any]],
    ):
        """Index product attributes for matching.

        Args:
            products: Dict mapping product_id -> {attribute: value}.
        """
        for prod_id, attrs in products.items():
            self.product_attributes[prod_id] = {}
            for attr_name, attr_value in attrs.items():
                if attr_value:
                    # Normalize and store as set of values
                    if isinstance(attr_value, str):
                        values = {attr_value.lower().strip()}
                    elif isinstance(attr_value, (list, set)):
                        values = {str(v).lower().strip() for v in attr_value}
                    else:
                        values = {str(attr_value).lower().strip()}

                    # Map to entity types
                    attr_to_entity = {
                        "brand": "BRAND",
                        "product_type": "PRODUCT_TYPE",
                        "color": "COLOR",
                        "material": "MATERIAL",
                        "size": "SIZE_MEASURE",
                    }
                    entity_type = attr_to_entity.get(attr_name.lower(), attr_name.upper())
                    self.product_attributes[prod_id][entity_type] = values

        logger.info(f"Indexed attributes for {len(products)} products")

    def compute_attribute_boost(
        self,
        extracted_attrs: Dict[str, List[str]],
        product_id: str,
    ) -> float:
        """Compute attribute match boost for a product.

        Args:
            extracted_attrs: Dict mapping entity_type -> list of values (from NER).
            product_id: Product ID to score.

        Returns:
            Boost factor (>= 1.0).
        """
        prod_attrs = self.product_attributes.get(product_id, {})
        if not prod_attrs:
            return 1.0

        boost = 1.0
        for entity_type, extracted_values in extracted_attrs.items():
            weight = self.attribute_weights.get(entity_type, 1.0)
            prod_values = prod_attrs.get(entity_type, set())

            for val in extracted_values:
                val_normalized = val.lower().strip()
                if val_normalized in prod_values or any(val_normalized in pv for pv in prod_values):
                    boost *= weight

        return boost

    def rerank(
        self,
        initial_results: List[Tuple[str, float]],
        extracted_attrs: Dict[str, List[str]],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Rerank results using attribute matching.

        Args:
            initial_results: List of (doc_id, score) from initial retrieval.
            extracted_attrs: Dict mapping entity_type -> list of values (from NER).
            k: Number of results to return.

        Returns:
            Reranked list of (doc_id, score).
        """
        reranked = []
        for doc_id, score in initial_results:
            boost = self.compute_attribute_boost(extracted_attrs, doc_id)
            reranked.append((doc_id, score * boost))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k]


def extract_attributes_from_ner(
    tokens: List[str],
    tags: List[str],
) -> Dict[str, List[str]]:
    """Extract structured attributes from NER predictions.

    Args:
        tokens: List of tokens.
        tags: List of IOB2 tags.

    Returns:
        Dict mapping entity_type -> list of values.
    """
    from src.evaluation.extrinsic import extract_attributes
    return extract_attributes(tokens, tags)


def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    retriever: BM25Retriever,
    reranker: Optional[AttributeWeightedReranker] = None,
    ner_predictions: Optional[List[Tuple[List[str], List[str]]]] = None,
    k: int = 10,
) -> Dict[str, float]:
    """Evaluate retrieval performance with optional NER-based reranking.

    Args:
        queries: List of query dicts with 'query', 'relevant_products', etc.
        retriever: BM25 retriever.
        reranker: Optional attribute-weighted reranker.
        ner_predictions: Optional list of (tokens, tags) for each query.
        k: Cutoff rank.

    Returns:
        Dict with nDCG@K, Recall@K, MRR@K metrics.
    """
    ndcg_scores = []
    recall_scores = []
    mrr_scores = []

    for i, query_info in enumerate(queries):
        query_text = query_info["query"]
        relevant_ids = set(query_info.get("relevant_products", []))

        # Initial retrieval
        results = retriever.retrieve(query_text, k=k * 2)  # Get more for reranking

        # Optional reranking with NER
        if reranker and ner_predictions and i < len(ner_predictions):
            tokens, tags = ner_predictions[i]
            extracted_attrs = extract_attributes_from_ner(tokens, tags)
            results = reranker.rerank(results, extracted_attrs, k=k)
        else:
            results = results[:k]

        retrieved_ids = [doc_id for doc_id, _ in results]

        # Compute relevance scores (binary: 1 if relevant, 0 otherwise)
        relevance = [1.0 if doc_id in relevant_ids else 0.0 for doc_id in retrieved_ids]

        ndcg_scores.append(compute_ndcg_at_k(relevance, k))
        recall_scores.append(compute_recall_at_k(retrieved_ids, relevant_ids, k))
        mrr_scores.append(compute_mrr(retrieved_ids, relevant_ids, k))

    metrics = {
        f"nDCG@{k}": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f"Recall@{k}": float(np.mean(recall_scores)) if recall_scores else 0.0,
        f"MRR@{k}": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "num_queries": len(queries),
    }

    logger.info(
        f"Retrieval metrics: nDCG@{k}={metrics[f'nDCG@{k}']:.4f}, "
        f"Recall@{k}={metrics[f'Recall@{k}']:.4f}, MRR@{k}={metrics[f'MRR@{k}']:.4f}"
    )

    return metrics


def tune_attribute_weights(
    val_queries: List[Dict[str, Any]],
    retriever: BM25Retriever,
    reranker: AttributeWeightedReranker,
    ner_predictions: List[Tuple[List[str], List[str]]],
    k: int = 10,
    weight_grid: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, float]:
    """Tune attribute weights on validation set.

    Args:
        val_queries: Validation queries.
        retriever: BM25 retriever.
        reranker: Reranker to tune.
        ner_predictions: NER predictions for validation queries.
        k: Cutoff rank.
        weight_grid: Dict mapping entity_type -> list of weights to try.

    Returns:
        Best weights dict.
    """
    if weight_grid is None:
        weight_grid = {
            "BRAND": [1.0, 1.5, 2.0, 2.5],
            "PRODUCT_TYPE": [1.0, 1.5, 2.0],
            "COLOR": [1.0, 1.2, 1.5],
        }

    best_ndcg = 0.0
    best_weights = reranker.attribute_weights.copy()

    # Simple grid search (can be extended to more sophisticated methods)
    for brand_w in weight_grid.get("BRAND", [2.0]):
        for pt_w in weight_grid.get("PRODUCT_TYPE", [1.5]):
            for color_w in weight_grid.get("COLOR", [1.2]):
                reranker.attribute_weights["BRAND"] = brand_w
                reranker.attribute_weights["PRODUCT_TYPE"] = pt_w
                reranker.attribute_weights["COLOR"] = color_w

                metrics = evaluate_retrieval(
                    val_queries, retriever, reranker, ner_predictions, k
                )

                if metrics[f"nDCG@{k}"] > best_ndcg:
                    best_ndcg = metrics[f"nDCG@{k}"]
                    best_weights = reranker.attribute_weights.copy()

    logger.info(f"Best weights (nDCG@{k}={best_ndcg:.4f}): {best_weights}")
    reranker.attribute_weights = best_weights

    return best_weights

