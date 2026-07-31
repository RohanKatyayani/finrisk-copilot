#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start both processes for the Hugging Face Space.
#
# FastAPI runs in the background on localhost:8000 (internal only).
# Streamlit runs in the foreground on 0.0.0.0:7860 (the public port).
#
# Streamlit must be the foreground process: when it exits, the container
# should exit too, so the Space restarts cleanly rather than hanging.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "[start] launching FastAPI on :8000 ..."
uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --log-level info &
API_PID=$!

# Wait for the API to answer /health before starting the UI, so the first
# thing a visitor sees isn't a connection error.
echo "[start] waiting for API to become healthy ..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "[start] API healthy after ${i}s"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "[start] ERROR: API process died during startup" >&2
        exit 1
    fi
    sleep 1
done

echo "[start] launching Streamlit on :7860 ..."
exec streamlit run streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
