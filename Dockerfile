# Dockerfile — Olist Delivery Delay Prediction Inference Service
# Lean, multi-stage or slim python container (no notebooks, no dev tools)
FROM python:3.11-slim as base

# Prevents Python from writing pyc files to disk and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000

WORKDIR /app

# Install system dependencies needed for compiling or curl health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, config, and trained model artifacts
COPY config/ config/
COPY src/ src/
COPY app/ app/
COPY artifacts/models/ artifacts/models/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Healthcheck to ensure container is responding
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
