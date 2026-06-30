# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# FinRisk Copilot — production image
# CPU-only Python service. Exposes the FastAPI app on port 8000.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Prevent Python from buffering stdout/stderr (logs appear in real time)
# and writing .pyc files inside the image (smaller, cleaner).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create a non-root user. Containers should never run as root in production —
# if the app gets exploited, the attacker is sandboxed to a low-privilege user.
RUN groupadd --system app && \
    useradd --system --gid app --home /home/app --create-home app

WORKDIR /app

# ---- System libraries -----------------------------------------------------
# libgomp1 is OpenMP runtime, required by LightGBM and some other ML libs.
# We use --no-install-recommends to keep the image smaller.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# ---- Python dependency layer (cached unless requirements.txt changes) -----
# Copying requirements FIRST means the slow pip install layer is cached
# across rebuilds whenever only source code changes.
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# ---- App layer (invalidates only on source / model / data changes) ---------
COPY src/        /app/src/
COPY scripts/    /app/scripts/
COPY monitoring/ /app/monitoring/
COPY data/rag/   /app/data/rag/
COPY data/interim/ /app/data/interim/
COPY models/     /app/models/

# Make sure runtime-writable dirs exist and are owned by the non-root user.
# Make sure runtime-writable dirs exist and are owned by the non-root user.
# /app/hf_cache is for sentence-transformers / huggingface model downloads.
RUN mkdir -p /app/logs /app/artifacts /app/hf_cache && \
    chown -R app:app /app /home/app

# Point HuggingFace + sentence-transformers caches at a writable dir.
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/hf_cache

USER app

EXPOSE 8000

# Health check — Docker / Kubernetes use this to know if the container is alive.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Default command: run uvicorn binding to all interfaces inside the container.
# Do NOT use --reload here (that's a dev flag and re-spawns workers on file changes).
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]