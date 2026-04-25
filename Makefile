.PHONY: setup data train-classical train-transformer train-hierarchical train-domain train-all benchmark baseline run-all clean test

PYTHON = .venv/bin/python
PIP = .venv/bin/pip
setup:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PYTHON) -m spacy download en_core_web_sm
data: data-prepare data-annotate

data-download:
	$(PYTHON) -c "\
		from src.data.loader import clone_esci_repo; \
		clone_esci_repo('data/raw')"

data-load:
	$(PYTHON) scripts/load_data.py --raw-dir data/raw --locale us

data-preprocess:
	$(PYTHON) scripts/preprocess_data.py --raw-dir data/raw --config configs/data_config.yaml

data-sample:
	$(PYTHON) -c "\
		from src.data.sampler import run_sampler; \
		run_sampler()"

data-split:
	$(PYTHON) -c "\
		from src.data.splitter import run_splitter; \
		run_splitter()"

data-prepare:
	$(PYTHON) scripts/prepare_data.py

data-annotate:
	@echo "Run annotation pipeline after data preparation"
train-classical:
	$(PYTHON) scripts/train_classical.py --model bilstm_crf
	$(PYTHON) scripts/train_classical.py --model cnn_bilstm

train-transformer:
	$(PYTHON) scripts/train_transformer.py --model bert
	$(PYTHON) scripts/train_transformer.py --model roberta

train-hierarchical:
	$(PYTHON) scripts/train_transformer.py --model hierarchical

train-domain:
	@echo "Running domain pretraining (this may take 12-24 hours on GPU)..."
	$(PYTHON) scripts/run_domain_pretrain.py --max-steps 10000 --batch-size 32
	@echo "Fine-tuning on NER..."
	$(PYTHON) scripts/train_transformer.py --model domain_adapted \
		--checkpoint outputs/esci_bert_dapt/final --output-dir outputs/dapt_bert_ner

train-all: train-classical train-transformer train-hierarchical
run-all:
	./scripts/run_all_experiments.sh
baseline:
	$(PYTHON) -c "\
		from src.evaluation.baseline import SpacyBaseline; \
		from src.schema.entity_schema import EntitySchema; \
		import pandas as pd; \
		schema = EntitySchema.from_yaml('configs/entity_schema.yaml'); \
		baseline = SpacyBaseline(label2id=schema.label2id, id2label=schema.id2label); \
		test_df = pd.read_parquet('data/annotations/test.parquet'); \
		pred = baseline.predict(test_df['tokens'].tolist()); \
		m = baseline.evaluate(test_df['ner_tags'].tolist(), pred); \
		print(f'spaCy Baseline - P: {m[\"precision\"]:.4f}, R: {m[\"recall\"]:.4f}, F1: {m[\"f1\"]:.4f}')"

benchmark:
	$(PYTHON) scripts/benchmark.py --models-dir outputs/ --data-dir data/annotations
flat-vs-hierarchical:
	$(PYTHON) scripts/eval_flat_vs_hierarchical.py \
		--test-data data/annotations/test.parquet \
		--predictions-dir results/benchmark \
		--output-dir results/flat_vs_hierarchical
retrieval-eval:
	$(PYTHON) scripts/run_retrieval_eval.py
multi-seed:
	$(PYTHON) scripts/run_multi_seed.py --models bert_ner bilstm_crf --seeds 42 123 456

multi-seed-aggregate:
	$(PYTHON) scripts/run_multi_seed.py --aggregate-only
learning-curve:
	$(PYTHON) scripts/run_learning_curve.py --models bert_ner bilstm_crf

learning-curve-plot:
	$(PYTHON) scripts/run_learning_curve.py --plot-only
thesis-artifacts:
	$(PYTHON) scripts/generate_thesis_artifacts.py
test:
	$(PYTHON) -m pytest tests/ -v --tb=short
clean:
	rm -rf outputs/ data/processed/ data/annotations/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf data/raw/ .venv/
