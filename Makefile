# ── FraudGuard ML — Makefile ──────────────────────────────────────────────────
#
# Common commands:
#   make install        Install all Python dependencies
#   make train          Train the model (runs full pipeline)
#   make api            Start the FastAPI inference server
#   make dashboard      Start the Streamlit dashboard
#   make simulate       Run the real-time transaction simulator
#   make mlflow         Start the MLflow tracking UI
#   make test           Run the full test suite with coverage
#   make lint           Lint and format-check the codebase
#   make docker-up      Start all services via Docker Compose
#   make clean          Remove generated artifacts
# ─────────────────────────────────────────────────────────────────────────────

PYTHON      := python
PIP         := pip
PYTEST      := pytest
UVICORN     := uvicorn
STREAMLIT   := streamlit
MLFLOW      := mlflow

CONFIG      := config/config.yaml
MODEL_DIR   := models
REPORT_DIR  := reports
LOG_DIR     := logs

API_HOST    := 0.0.0.0
API_PORT    := 8000
DASH_PORT   := 8501
MLFLOW_PORT := 5000

.PHONY: all install train api dashboard simulate mlflow test lint \
        docker-build docker-up docker-down clean help drift retrain

# ── Default ───────────────────────────────────────────────────────────────────
all: help

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed."

install-dev: install
	@echo "📦 Installing dev tools..."
	$(PIP) install ruff mypy bandit pytest-cov
	@echo "✅ Dev tools installed."

# ── Training ──────────────────────────────────────────────────────────────────
train:
	@echo "🏋️  Training models..."
	@mkdir -p $(MODEL_DIR) $(REPORT_DIR)/figures $(LOG_DIR)
	$(PYTHON) main.py --config $(CONFIG)
	@echo "✅ Training complete. Models saved to $(MODEL_DIR)/"

train-tuned:
	@echo "🔬 Training with Optuna hyperparameter tuning (slow)..."
	$(PYTHON) -c "\
		import yaml; \
		cfg = yaml.safe_load(open('$(CONFIG)')); \
		cfg['tuning']['enabled'] = True; \
		yaml.dump(cfg, open('/tmp/config_tuned.yaml','w')); \
	"
	$(PYTHON) main.py --config /tmp/config_tuned.yaml

# ── Inference API ─────────────────────────────────────────────────────────────
api:
	@echo "🚀 Starting FastAPI inference server on http://$(API_HOST):$(API_PORT)"
	@echo "   Swagger UI → http://localhost:$(API_PORT)/docs"
	$(UVICORN) api.main:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--reload \
		--log-level info

api-prod:
	@echo "🚀 Starting API (production mode, 4 workers)..."
	$(UVICORN) api.main:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--workers 4 \
		--log-level warning

# ── Dashboard ─────────────────────────────────────────────────────────────────
dashboard:
	@echo "📊 Starting Streamlit dashboard on http://localhost:$(DASH_PORT)"
	$(STREAMLIT) run dashboard/app.py \
		--server.port $(DASH_PORT) \
		--server.address localhost \
		--server.headless false

# ── Simulation ────────────────────────────────────────────────────────────────
simulate:
	@echo "💳 Starting transaction simulator → API at http://localhost:$(API_PORT)"
	$(PYTHON) simulation/real_time_transactions.py \
		--api-url http://localhost:$(API_PORT) \
		--tps 2 \
		--duration 120 \
		--fraud-rate 0.05

simulate-fast:
	@echo "💳 Fast simulation (10 TPS, 30s)..."
	$(PYTHON) simulation/real_time_transactions.py \
		--api-url http://localhost:$(API_PORT) \
		--tps 10 \
		--duration 30 \
		--fraud-rate 0.10

simulate-infinite:
	@echo "💳 Infinite simulation (Ctrl+C to stop)..."
	$(PYTHON) simulation/real_time_transactions.py \
		--api-url http://localhost:$(API_PORT) \
		--tps 2 \
		--duration 0

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow:
	@echo "🔬 Starting MLflow UI on http://localhost:$(MLFLOW_PORT)"
	$(MLFLOW) ui \
		--host 0.0.0.0 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri mlruns

# ── Monitoring ────────────────────────────────────────────────────────────────
monitor-alerts:
	@echo "🚨 Tailing fraud alerts from logs/fraud_alerts.jsonl..."
	$(PYTHON) monitoring/fraud_alerts.py \
		--log-file logs/fraud_alerts.jsonl

drift:
	@echo "📊 Running drift detection..."
	$(PYTHON) src/monitoring/drift_detector.py \
		--baseline data/processed/X_train.csv \
		--current  data/processed/X_val.csv \
		--report   reports/drift_report.json

evaluate:
	@echo "📊 Running model evaluation..."
	$(PYTHON) scripts/evaluate.py --business-impact

evaluate-custom:
	@echo "📊 Running evaluation on custom CSV..."
	@[ -n "$(CSV)" ] || (echo "Usage: make evaluate-custom CSV=path/to/data.csv" && exit 1)
	$(PYTHON) scripts/evaluate.py --data $(CSV) --business-impact

# ── Retraining ────────────────────────────────────────────────────────────────
retrain:
	@echo "🔄 Running automated retraining pipeline..."
	$(PYTHON) scripts/retrain.py \
		--check-drift \
		--min-improvement 0.005

retrain-force:
	@echo "🔄 Force retraining (skip drift check)..."
	$(PYTHON) scripts/retrain.py

retrain-dry:
	@echo "🔄 Dry-run retraining (no model promotion)..."
	$(PYTHON) scripts/retrain.py --dry-run

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Running test suite..."
	$(PYTEST) tests/ -v \
		--cov=src \
		--cov=api \
		--cov=simulation \
		--cov=monitoring \
		--cov-report=term-missing \
		--cov-report=html:reports/coverage_html \
		--cov-fail-under=60 \
		-x

test-fast:
	@echo "🧪 Running fast tests (no coverage)..."
	$(PYTEST) tests/ -v -x -q

test-api:
	@echo "🧪 Testing API endpoints only..."
	$(PYTEST) tests/test_api.py -v

test-simulation:
	@echo "🧪 Testing simulation & monitoring..."
	$(PYTEST) tests/test_simulation.py tests/test_drift_detector.py -v

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	@echo "🔍 Linting with ruff..."
	ruff check src/ api/ dashboard/ simulation/ monitoring/ tests/ scripts/ main.py
	@echo "✅ Lint passed."

format:
	@echo "✨ Auto-formatting with ruff..."
	ruff format src/ api/ dashboard/ simulation/ monitoring/ tests/ scripts/ main.py

typecheck:
	@echo "🔍 Type-checking with mypy..."
	mypy src/ api/ --ignore-missing-imports --no-strict-optional

security:
	@echo "🔒 Running Bandit security scan..."
	@mkdir -p reports
	bandit -r src/ api/ simulation/ monitoring/ -ll \
		-f json -o reports/bandit_report.json || true
	@echo "Security report → reports/bandit_report.json"

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t fraudguard-ml:latest .

docker-up:
	@echo "🐳 Starting all services..."
	docker compose up --build -d
	@echo ""
	@echo "Services running:"
	@echo "  API       → http://localhost:8000/docs"
	@echo "  Dashboard → http://localhost:8501"
	@echo "  MLflow    → http://localhost:5000"

docker-down:
	@echo "🐳 Stopping all services..."
	docker compose down

docker-logs:
	docker compose logs -f --tail=50

docker-train:
	@echo "🐳 Running training job in Docker..."
	docker compose run --rm train

# ── Utilities ─────────────────────────────────────────────────────────────────
health:
	@echo "🏥 Checking API health..."
	@curl -s http://localhost:$(API_PORT)/health | python -m json.tool || \
		echo "❌ API not reachable. Start with: make api"

metrics:
	@echo "📊 Fetching live metrics..."
	@curl -s "http://localhost:$(API_PORT)/metrics" | python -m json.tool || \
		echo "❌ API not reachable."

predict-test:
	@echo "🔮 Sending test prediction..."
	@curl -s -X POST http://localhost:$(API_PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"V1":-1.3598,"V2":-0.0728,"V3":2.5364,"V4":1.3782,"V5":-0.3383,\
"V6":0.4624,"V7":0.2396,"V8":0.0987,"V9":0.3638,"V10":-0.0902,\
"V11":-0.5516,"V12":-0.6178,"V13":-0.9914,"V14":-0.3114,"V15":1.4682,\
"V16":-0.4704,"V17":0.2079,"V18":0.0258,"V19":0.4039,"V20":0.2514,\
"V21":-0.0183,"V22":0.2778,"V23":-0.1105,"V24":0.0669,"V25":0.1285,\
"V26":-0.1891,"V27":0.1336,"V28":-0.0211,"Amount":149.62,"Time":0.0}' \
	| python -m json.tool

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.xml
	@echo "✅ Clean."

clean-models:
	@echo "🗑️  Removing trained models (keep backups)..."
	rm -f $(MODEL_DIR)/*.pkl $(MODEL_DIR)/*.json $(MODEL_DIR)/*.png
	@echo "✅ Models removed."

clean-all: clean clean-models
	rm -rf $(REPORT_DIR)/figures reports/coverage_html
	rm -rf mlruns
	@echo "✅ Full clean complete."

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║        FraudGuard ML — Available Commands                ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Setup         make install        Install deps          ║"
	@echo "  ║                make install-dev    + dev tools           ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Pipeline      make train          Full training run     ║"
	@echo "  ║                make train-tuned    With Optuna tuning    ║"
	@echo "  ║                make evaluate       Evaluate saved model  ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Services      make api            Start inference API   ║"
	@echo "  ║                make dashboard      Start Streamlit UI    ║"
	@echo "  ║                make mlflow         Start MLflow UI       ║"
	@echo "  ║                make simulate       Run tx simulator      ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Monitoring    make monitor-alerts Tail fraud alert log  ║"
	@echo "  ║                make drift          Run drift detection   ║"
	@echo "  ║                make retrain        Auto retrain pipeline ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Quality       make test           Full test + coverage  ║"
	@echo "  ║                make lint           Ruff lint check       ║"
	@echo "  ║                make format         Auto-format code      ║"
	@echo "  ║                make typecheck      mypy type check       ║"
	@echo "  ║                make security       Bandit SAST scan      ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Docker        make docker-up      Start all containers  ║"
	@echo "  ║                make docker-down    Stop all containers   ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Utilities     make health         API health check      ║"
	@echo "  ║                make metrics        Live API metrics      ║"
	@echo "  ║                make predict-test   Test prediction call  ║"
	@echo "  ╠══════════════════════════════════════════════════════════╣"
	@echo "  ║  Cleanup       make clean          Remove .pyc, cache    ║"
	@echo "  ║                make clean-all      Remove models+mlruns  ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo ""
