# ── FraudGuard ML — Multi-stage Dockerfile ────────────────────────────────────
#
# Stage 1: builder  — install all Python deps into a venv
# Stage 2: runtime  — lean image that copies only the venv + source
#
# Build:
#   docker build -t fraudguard-ml .
#
# Run API:
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models fraudguard-ml
#
# Run dashboard:
#   docker run -p 8501:8501 fraudguard-ml \
#     streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
#
# Run simulator:
#   docker run fraudguard-ml \
#     python simulation/real_time_transactions.py --api-url http://host.docker.internal:8000
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtualenv for clean layer caching
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install Python deps first (separate layer = cached unless requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="FraudGuard ML Platform"
LABEL description="Credit Card Fraud Detection — FastAPI + XGBoost"
LABEL version="2.1.0"

WORKDIR /app

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY src/         ./src/
COPY api/         ./api/
COPY dashboard/   ./dashboard/
COPY simulation/  ./simulation/
COPY monitoring/  ./monitoring/
COPY scripts/     ./scripts/
COPY config/      ./config/
COPY main.py      .

# Runtime directories (populated via volume mounts in production)
RUN mkdir -p \
    models \
    data/raw \
    data/processed \
    reports/figures \
    logs \
    mlruns \
    models/backup

# Non-root user (security best practice)
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

# Health check for Kubernetes liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8501

# Default: start the inference API
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
