# ── Fraud Triage Engine — Docker Image ────────────────────────────────────────
#
# Build:  docker build -t fraud-triage-api .
# Run:    docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-triage-api
# Docs:   http://localhost:8000/docs
#
# Two-stage build:
#   Stage 1 (builder) — install heavy ML dependencies into a venv
#   Stage 2 (runtime) — copy only the venv and source, no build tools
#
# This keeps the final image lean (~1.2 GB vs ~2.5 GB single-stage).
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed to compile some ML wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies — copy requirements first to leverage Docker layer cache
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        fastapi>=0.111.0 \
        uvicorn[standard]>=0.29.0 \
        pydantic>=2.7.0 \
        joblib>=1.4.0


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libgomp1 needed at runtime by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy source
COPY src/           ./src/
COPY api/           ./api/
COPY outputs/       ./outputs/

# models/ is volume-mounted at runtime — not baked into the image.
# This means you can update the model without rebuilding the image.
# The directory is created here so the container starts cleanly if
# no volume is mounted (API starts in degraded mode, /health still returns 200).
RUN mkdir -p models outputs/reports

# Transfer ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Environment
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED="1"
ENV MODELS_DIR="/app/models"

# Health check — Docker will mark the container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Start the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]