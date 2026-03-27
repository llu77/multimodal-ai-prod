# ============================================
# Multimodal AI Production — Makefile
# أوامر مختصرة للتشغيل والتطوير
# ============================================

.PHONY: setup download train train-grpo serve test eval clean

# ── Setup ──
setup:
	bash scripts/setup.sh

download:
	python scripts/download_model.py --all

# ── Training ──
train:
	python -m src.training.train --config config/config.yaml --stage sft

train-grpo:
	python -m src.training.train --config config/config.yaml --stage grpo

train-all:
	python -m src.training.train --config config/config.yaml --stage all

# ── Serving ──
serve:
	python -m src.api.server

serve-dev:
	uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# ── RAG ──
index:
	python -m src.rag.engine --config config/config.yaml --index-dir ./data/raw

index-query:
	@read -p "Query: " q; python -m src.rag.engine --config config/config.yaml --query "$$q"

# ── Evaluation ──
eval-create:
	python -m src.evaluation.evaluator --create-sample

eval:
	python -m src.evaluation.evaluator --config config/config.yaml --eval-file data/eval/eval_set.jsonl --output reports/eval_report

# ── Testing ──
test:
	python -m pytest tests/ -v --tb=short

test-fast:
	python -m pytest tests/ -v --tb=short -m "not slow and not gpu"

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing

# ── Docker ──
docker-build:
	docker build -t multimodal-ai .

docker-run:
	docker compose up -d

docker-stop:
	docker compose down

# ── Cleanup ──
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache logs/*.log

# ── Info ──
info:
	@echo "=== Multimodal AI Production System ==="
	@echo "Python files: $$(find src -name '*.py' | wc -l)"
	@echo "Test files:   $$(find tests -name 'test_*.py' | wc -l)"
	@echo "Total lines:  $$(find src tests -name '*.py' -exec cat {} + | wc -l)"
	@python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null || echo "GPU: torch not installed"
