# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# FinRisk Copilot — Hugging Face Spaces demo image.
#
# Differs from docker/Dockerfile (the API-only production image) in three ways:
#   1. Trains the LightGBM model at build time, because models/ and mlflow.db
#      are gitignored and Spaces builds from a clean checkout.
#   2. Ships the Streamlit UI as well as the FastAPI service.
#   3. Serves Streamlit on 7860 (the port Spaces expects) and keeps FastAPI
#      on localhost:8000 as an internal dependency.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is the OpenMP runtime required by LightGBM.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src/             /app/src/
COPY scripts/         /app/scripts/
COPY data/rag/        /app/data/rag/
COPY data/interim/    /app/data/interim/
COPY streamlit_app.py /app/streamlit_app.py
COPY start.sh         /app/start.sh

# Caches must be writable by the non-root user and live somewhere persistent
# inside the image, or HuggingFace downloads fail at runtime.
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/hf_cache \
    MPLCONFIGDIR=/app/.mpl

RUN mkdir -p /app/models /app/logs /app/artifacts /app/hf_cache /app/.mpl && \
    chmod +x /app/start.sh && \
    chown -R app:app /app

USER app

# Train at build time: produces models/credit_risk_model.pkl and mlflow.db
# inside the image, so the API has a model to serve on first request.
RUN python src/training/train_model.py

# Streamlit is the public face of the Space.
ENV FINRISK_API_URL=http://localhost:8000
EXPOSE 7860

CMD ["/app/start.sh"]
