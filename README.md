# FinRisk Copilot
 
A production-style banking AI system that combines tabular ML, a LoRA-fine-tuned LLM, and retrieval-augmented generation behind a single FastAPI service — tracked with MLflow, monitored with Evidently, containerized, and wired with CI.
 
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking%20%2B%20registry-0194E2)](https://mlflow.org/)
[![CI](https://github.com/RohanKatyayani/finrisk-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/RohanKatyayani/finrisk-copilot/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
 
---
 
## What it does
 
FinRisk Copilot scores credit-risk applications, explains its decisions in plain English, and answers banking-policy questions from source documents. Three complementary components sit behind one API:
 
- **Risk Scorer** — A LightGBM classifier trained on the German Credit dataset. Handles class imbalance, logs experiments to MLflow, returns calibrated probabilities, and is served from the MLflow Model Registry.
- **LLM Explainer** — TinyLlama-1.1B fine-tuned with LoRA on synthetic bank-tone explanations. Generates regulator-style reasoning for each decision. Hosted on the [Hugging Face Hub](https://huggingface.co/rohankatyayani/tinyllama-credit-explainer).
- **Policy Assistant (RAG)** — Retrieval-augmented answers over public banking-policy PDFs (Basel / FATF). Documents are chunked and embedded with `sentence-transformers` (all-MiniLM-L6-v2) into a FAISS index; retrieval is grounded and generation runs on Groq (Llama 3.1 8B). Answers include citations, and out-of-scope questions are refused rather than hallucinated.
All three are served behind one FastAPI app with a small set of clean endpoints.
 
---
 
## Live demo
 
_A hosted, interactive UI (four tabs: Predict / Explain / Combined / Ask Policy) ships with the deployment milestone._
 
<!-- TODO after deploy: paste the live app URL here, e.g. https://huggingface.co/spaces/rohankatyayani/finrisk-copilot -->
<!-- TODO after recording: embed or link the 60-second demo video -->
 
---
 
## Quickstart
 
Requires Python 3.11. The first `/explain` call downloads several GB of model weights from the Hugging Face Hub and caches them locally.
 
```bash
# 1. Clone and set up
git clone https://github.com/RohanKatyayani/finrisk-copilot.git
cd finrisk-copilot
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
 
# 2. Train the LightGBM model (logs to MLflow, saves models/credit_risk_model.pkl)
python src/training/train_model.py
 
# 3. Start the API
uvicorn src.service.app:app --port 8000
```
 
### API endpoints
 
| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/predict` | POST | Credit-risk score from LightGBM |
| `/explain` | POST | Plain-English explanation from fine-tuned TinyLlama |
| `/predict_and_explain` | POST | Score + explanation in one call |
| `/ask_policy` | POST | Grounded answer over banking-policy PDFs, with citations |
 
**Try `/predict_and_explain`:**
 
```bash
curl -X POST http://localhost:8000/predict_and_explain \
  -H "Content-Type: application/json" \
  -d '{
    "status":"A11","duration":24,"credit_history":"A34","purpose":"A43",
    "amount":3500,"savings":"A61","employment_duration":"A73",
    "installment_rate":2,"personal_status_sex":"A93","other_debtors":"A101",
    "present_residence":2,"property":"A121","age":30,
    "other_installment_plans":"A143","housing":"A152","number_credits":1,
    "job":"A173","people_liable":1,"telephone":"A192","foreign_worker":"A201"
  }'
```
 
Sample response (explanation text illustrative):
```json
{
  "prediction": 1,
  "probabilities": [0.163, 0.837],
  "explanation": "The applicant's credit history and existing obligations suggest elevated repayment risk ..."
}
```
 
**Try `/ask_policy`:**
 
```bash
curl -X POST http://localhost:8000/ask_policy \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Basel require for operational risk capital?"}'
```
 
---
 
## Architecture
 
```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Service                        │
│                                                             │
│  POST /predict              → LightGBM (German Credit)      │
│  POST /explain              → TinyLlama (LoRA fine-tuned)   │
│  POST /predict_and_explain  → Both, combined                │
│  POST /ask_policy           → RAG over policy PDFs          │
└─────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
  ┌──────────┐       ┌─────────────┐      ┌───────────────┐
  │ LightGBM │       │  TinyLlama  │      │ FAISS + MiniLM│
  │ pipeline │       │   + LoRA    │      │  + Groq LLM   │
  └──────────┘       └─────────────┘      └───────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   ┌────────────────────────────────────────────────┐
   │     MLflow tracking + registry  ·  Evidently    │
   └────────────────────────────────────────────────┘
```
 
A note on serving the LLM: PyTorch deadlocks when loaded inside a forked uvicorn worker on macOS. `src/models/lora_infer.py` works around this by running each inference in a fresh Python subprocess. In production this would be a separate microservice on GPU hardware.
 
---
 
## MLOps
 
This project is built to show end-to-end operational ownership, not just modeling:
 
- **Experiment tracking** — every training run is logged to MLflow (params, metrics, artifacts).
- **Model Registry** — the LightGBM model is versioned through a `None → Staging → Production` lifecycle in an MLflow registry (SQLite backend). A promotion CLI (`scripts/promote_model.py`) moves versions between stages, and the API loads the current Production model from the registry with a pickle fallback.
- **Drift monitoring** — Evidently runs Kolmogorov–Smirnov (numeric) and chi-square (categorical) tests to compare live inputs against the training distribution.
- **Containerization** — a production Dockerfile (non-root user, pinned system libs, healthcheck) runs the full service; verified end-to-end inside the container.
- **CI** — GitHub Actions runs a single pipeline on every push: lint (ruff) → format check (black) → train the model → pytest → Docker build.
- **Reproducible builds** — all Python dependencies are pinned to exact versions, so local and CI environments resolve identically.
---
 
## Tech stack
 
- **ML:** scikit-learn, LightGBM, SHAP, imbalanced-learn
- **LLM:** Hugging Face Transformers, PEFT (LoRA), TinyLlama-1.1B
- **RAG:** sentence-transformers, FAISS, Groq (Llama 3.1 8B)
- **Serving:** FastAPI, uvicorn, pydantic
- **MLOps:** MLflow (tracking + registry), Evidently (monitoring), Docker, GitHub Actions
---
 
## Project structure
 
```
finrisk-copilot/
├── src/
│   ├── service/app.py           # FastAPI app + endpoints
│   ├── models/lora_infer.py     # Subprocess-isolated LLM inference
│   ├── training/
│   │   ├── train_model.py        # LightGBM training + MLflow logging
│   │   └── make_explanations.py  # Synthetic explanation dataset generator
│   └── rag/                      # RAG pipeline: chunk, embed, index, retrieve
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_generate_explanations.ipynb
│   └── llama_finetune.ipynb      # LoRA fine-tuning on Colab
├── data/                         # German Credit dataset (small training CSV tracked)
├── tests/                        # pytest suite
├── scripts/promote_model.py      # MLflow Registry stage-promotion CLI
├── docker/                       # Container image
├── .github/workflows/ci.yml      # CI pipeline
└── requirements.txt              # Pinned dependencies
```
 
---
 
## Status
 
- [x] Risk-scoring pipeline (LightGBM + MLflow)
- [x] Synthetic explanation dataset generation
- [x] LoRA fine-tuning of TinyLlama (Colab + Hugging Face Hub)
- [x] LLM-backed explanation endpoint
- [x] RAG over banking-policy PDFs with citations
- [x] MLflow Model Registry + promotion CLI
- [x] Evidently drift monitoring
- [x] Dockerized deployment + GitHub Actions CI
- [x] Pinned dependencies for reproducible builds
- [ ] Hosted interactive UI (Streamlit / HF Spaces)
- [ ] Model and data cards
- [ ] Demo recording
---
 
## Links
 
- **Fine-tuned model:** [rohankatyayani/tinyllama-credit-explainer](https://huggingface.co/rohankatyayani/tinyllama-credit-explainer)
- **Training notebook:** [`notebooks/llama_finetune.ipynb`](notebooks/llama_finetune.ipynb)
---
 
## License
 
MIT © Rohan Katyayani