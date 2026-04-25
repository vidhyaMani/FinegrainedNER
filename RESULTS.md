# Experiment Results

## Overview

This document summarizes the results of fine-grained NER experiments on e-commerce product queries using the Amazon ESCI dataset.

**Experiment Date:** March 15, 2026

## Models Evaluated

| Model | Type | Description |
|-------|------|-------------|
| BiLSTM-CRF | Classical | Word embeddings + BiLSTM + CRF |
| CNN-BiLSTM-CRF | Classical | Char CNN + BiLSTM + CRF |
| BERT-NER | Transformer | Fine-tuned bert-base-uncased |
| RoBERTa-NER | Transformer | Fine-tuned roberta-base |

## Main Results

### Overall Performance

| Model | Precision | Recall | F1 | Rank |
|-------|-----------|--------|-----|------|
| **BERT-NER** | 0.5544 | 0.4718 | **0.5098** | |
| RoBERTa-NER | 0.5080 | 0.4749 | 0.4909 | |
| BiLSTM-CRF | 0.6091 | 0.3530 | 0.4469 | |
| CNN-BiLSTM-CRF | 0.6630 | 0.2922 | 0.4056 | 4 |

**Best Model: BERT-NER with F1 = 0.5098**

### Per-Entity Performance (F1 Score)

| Entity Type | BiLSTM-CRF | CNN-BiLSTM | BERT | RoBERTa |
|-------------|------------|------------|------|---------|
| ATTRIBUTE_VALUE | 0.93 | - | 0.92 | 0.92 |
| MATERIAL | 0.85 | - | 0.85 | 0.85 |
| SIZE_MEASURE | 0.56 | - | 0.67 | 0.63 |
| BRAND | 0.63 | - | 0.67 | 0.62 |
| PRODUCT_TYPE | 0.31 | - | 0.41 | 0.41 |
| COLOR | 0.20 | - | 0.10 | 0.17 |

### Extrinsic Evaluation

| Model | Query Understanding (Exact) | Attribute Coverage |
|-------|----------------------------|-------------------|
| BERT-NER | 37.88% | 47.2% |
| RoBERTa-NER | 35.52% | 47.6% |
| BiLSTM-CRF | 37.40% | 35.3% |
| CNN-BiLSTM-CRF | 37.11% | 29.2% |

## Key Findings

### 1. Model Architecture Comparison

- **BERT achieves the best overall F1 (0.5098)**, outperforming classical models by 6-10 percentage points
- Classical models have higher precision but significantly lower recall
- Transformer models provide better balance between precision and recall

### 2. Entity-Specific Insights

- **ATTRIBUTE_VALUE and MATERIAL** are easiest to recognize (F1 > 0.85)
- **COLOR is the hardest entity type** (F1 < 0.20) - likely due to ambiguity with brand names and product types
- **PRODUCT_TYPE** performance is moderate (~0.41 F1) despite being the most common entity

### 3. Classical vs Transformer

- BiLSTM-CRF: High precision (0.61) but low recall (0.35) - conservative predictions
- BERT: Better recall (0.47) with reasonable precision (0.55) - more balanced
- Transformers benefit from pretrained representations for entity boundary detection

## Data Statistics

| Split | Samples |
|-------|---------|
| Train | 52,500 |
| Val | 11,250 |
| Test | 11,250 |

## Visualizations

Generated charts in `results/benchmark/`:
- `model_comparison_f1.png` - F1 score comparison bar chart
- `model_comparison_precision.png` - Precision comparison
- `model_comparison_recall.png` - Recall comparison  
- `per_entity_radar.png` - Radar chart by entity type

## Reproducibility

### Environment

```
Python 3.11
PyTorch 2.1+
Transformers 4.36+
```

### Running Experiments

```bash
# Train all models
python scripts/train_classical.py --model bilstm_crf
python scripts/train_classical.py --model cnn_bilstm
python scripts/train_transformer.py --model bert
python scripts/train_transformer.py --model roberta

# Run benchmark
python scripts/benchmark.py --models-dir outputs/ --output-dir results/benchmark
```

## Files

- Model checkpoints: `outputs/*/`
- Benchmark results: `results/benchmark/`
- Predictions: `results/benchmark/*_predictions.json`
