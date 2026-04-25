# Fine-Grained E-Commerce NER

NER system for extracting product attributes from e-commerce search queries, built on the Amazon ESCI dataset. This was developed as part of my MSc thesis investigating how data quality (silver vs gold labels) and model architecture (classical vs transformer) affect entity recognition in product search.

## Entity Schema

| Entity | Example |
|--------|---------|
| `PRODUCT_TYPE` | running shoes, laptop, shampoo |
| `BRAND` | nike, apple, samsung |
| `MATERIAL` | leather, cotton, stainless steel |
| `COLOR` | red, navy blue, matte black |
| `SIZE_MEASURE` | XL, 32oz, 15-inch |
| `ATTRIBUTE_VALUE` | waterproof, organic, wireless |

Tagging follows IOB2 format. A coarse-grained grouping (PRODUCT / BRAND / ATTRIBUTE) is also evaluated — see `configs/entity_schema.yaml`.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

Or with the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
  data/          # loading, preprocessing, silver annotation, gold creation
  models/        # BiLSTM-CRF, CNN-BiLSTM, BERT, RoBERTa, hierarchical, domain-adapted
  evaluation/    # intrinsic (seqeval), extrinsic, error analysis, baselines
  schema/        # entity schema definition and label mappings
  utils/         # shared helpers

scripts/
  prepare_data.py              # end-to-end data pipeline
  train_classical.py           # train BiLSTM-CRF / CNN-BiLSTM
  train_transformer.py         # train BERT / RoBERTa / hierarchical
  run_domain_pretrain.py       # continued pretraining on product text
  benchmark.py                 # evaluate all models, produce tables and charts
  eval_flat_vs_hierarchical.py # coarse vs fine entity comparison
  run_learning_curve.py        # training-size ablation
  run_multi_seed.py            # variance across random seeds
  generate_thesis_artifacts.py # export figures and LaTeX tables
  generate_gold_latex.py       # gold-standard result tables

configs/
  entity_schema.yaml   # entity type definitions and IOB2 label list
  data_config.yaml     # dataset paths and preprocessing settings
  train_config.yaml    # hyperparameters for all models
```

## Reproducing Experiments

### Data preparation

```bash
python scripts/prepare_data.py
```

### Training

```bash
python scripts/train_classical.py --model bilstm_crf
python scripts/train_classical.py --model cnn_bilstm
python scripts/train_transformer.py --model bert
python scripts/train_transformer.py --model roberta
```

### Evaluation

```bash
python scripts/benchmark.py --models-dir outputs/ --output-dir results/benchmark
```

### Additional experiments

```bash
python scripts/run_domain_pretrain.py --max-steps 10000 --batch-size 32
python scripts/run_learning_curve.py
python scripts/run_multi_seed.py
python scripts/eval_flat_vs_hierarchical.py
```

## Results

Best overall F1 on silver test set: **BERT-NER 0.5098**

| Model | P | R | F1 |
|-------|---|---|-----|
| BERT-NER | 0.554 | 0.472 | 0.510 |
| RoBERTa-NER | 0.508 | 0.475 | 0.491 |
| BiLSTM-CRF | 0.609 | 0.353 | 0.447 |
| CNN-BiLSTM-CRF | 0.663 | 0.292 | 0.406 |

Full results, per-entity breakdowns, and gold-standard comparisons are in [RESULTS.md](RESULTS.md). Figures are under `results/`.

## Data

The raw Amazon ESCI data is not included (see [amazon-science/esci-data](https://github.com/amazon-science/esci-data)). Annotation files in `data/annotations/` are gitignored due to size — contact me or run `prepare_data.py` to regenerate.

