# ── ClinicalTriageEnv Docker Image ──────────────────────────────────────────
# Base: Python 3.11-slim (matches constraint: vcpu=2, memory=8GB)
FROM python:3.11-slim

# Metadata
LABEL maintainer="ClinicalTriageEnv"
LABEL description="OpenEnv hospital emergency department triage environment"
LABEL version="1.0.0"

# Non-root user for HF Spaces security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY env/ ./env/
COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Create empty __init__.py for server package
RUN touch server/__init__.py

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose HF Spaces default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
