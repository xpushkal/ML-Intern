# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.9-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.9-slim

LABEL maintainer="mlops-task" \
      description="MLOps batch job — rolling mean signal pipeline"

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application files
COPY run.py        .
COPY config.yaml   .
COPY data.csv      .

# Non-root user for security best-practice
RUN useradd --create-home appuser
USER appuser

# Default command: run pipeline and emit metrics to stdout
# Exit code propagates automatically (0 = success, non-zero = failure)
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
