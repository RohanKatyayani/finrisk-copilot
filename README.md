# FinRisk Copilot

A production-style banking AI system that combines tabular ML, a LoRA-fine-tuned LLM, and retrieval-augmented generation behind a single FastAPI service.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

FinRisk Copilot scores credit-risk applications and explains its decisions in plain English. It's structured around three complementary components:

- **Risk Scorer** — A LightGBM classifier trained on the German Credit dataset. Handles class imbalance, logs experiments to MLflow, returns calibrated probabilities.
- **LLM Explainer** — TinyLlama-1.1B fine-tuned with LoRA on synthetic bank-tone explanations. Generates regulator-style reasoning for each decision. Hosted on [Hugging Face Hub](https://huggingface.co/rohankatyayani/tinyllama-credit-explainer) and demoable live on [Hugging Face Spaces](https://huggingface.co/spaces/rohankatyayani/tinyllama-credit-explainer).
- **Policy Assistant (RAG)** — *Coming in next milestone.* Retrieves from public banking policy PDFs and answers questions with citations.

All three are served behind one FastAPI app with a small set of clean endpoints.

---

## Quickstart

Requires Python 3.11. The first `/explain` call downloads ~4GB of model weights from Hugging Face Hub and caches them locally.

```bash
# 1. Clone and set up
git clone https://github.com/RohanKatyayani/finrisk-copilot.git
cd finrisk-copilot
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Train the LightGBM model (logs to MLflow, saves models/credit_risk_model.pkl)
python src/training/train_model.py

# 3. Start the API
uvicorn src.service.app:app --port 8000
```

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/predict` | POST | Credit risk score from LightGBM |
| `/explain` | POST | Plain-English explanation from fine-tuned TinyLlama |
| `/predict_and_explain` | POST | Score + explanation in one call |

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

Sample response:
```json
{
  "prediction": 1,
  "probabilities": [0.163, 0.837],
  "explanation": "The applicant has a history of late payments and high-interest loans..."
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Service                          │
│                                                             │
│  POST /predict              → LightGBM (German Credit)      │
│  POST /explain              → TinyLlama (LoRA fine-tuned)   │
│  POST /predict_and_explain  → Both, combined                │
│  POST /ask_policy           → RAG over policy PDFs (WIP)    │
└─────────────────────────────────────────────────────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
  ┌──────────┐       ┌─────────────┐      ┌──────────┐
  │ LightGBM │       │  TinyLlama  │      │   FAISS  │
  │ pipeline │       │   + LoRA    │      │  + PDFs  │
  └──────────┘       └─────────────┘      └──────────┘
       │                    │                    │
       ▼                    ▼                    ▼
   ┌────────────────────────────────────────────────┐
   │              MLflow tracking                   │
   └────────────────────────────────────────────────┘
```

A note on serving the LLM: PyTorch deadlocks when loaded inside a forked uvicorn worker on macOS. `src/models/lora_infer.py` works around this by running each inference in a fresh Python subprocess. In production this would be a separate microservice.

---

## Tech stack

- **ML:** scikit-learn, LightGBM, SHAP, imbalanced-learn
- **LLM:** Hugging Face Transformers, PEFT (LoRA), TinyLlama-1.1B
- **Serving:** FastAPI, uvicorn, pydantic
- **MLOps:** MLflow (tracking + registry), Evidently (monitoring), Docker, GitHub Actions
- **RAG:** sentence-transformers, FAISS

---

## Project structure

```
finrisk-copilot/
├── src/
│   ├── service/app.py          # FastAPI app + endpoints
│   ├── models/lora_infer.py    # Subprocess-isolated LLM inference
│   ├── training/
│   │   ├── train_model.py      # LightGBM training + MLflow logging
│   │   └── make_explanations.py# Synthetic explanation dataset generator
│   └── rag/                    # RAG pipeline (in progress)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_generate_explanations.ipynb
│   └── llama_finetune.ipynb    # LoRA fine-tuning on Colab
├── data/                       # German Credit dataset
├── tests/                      # pytest test suite
├── docker/Dockerfile           # Container image
├── scripts/                    # Data prep utilities
└── requirements.txt
```

---

## Status

- [x] Risk scoring pipeline (LightGBM + MLflow)
- [x] Synthetic explanation dataset generation
- [x] LoRA fine-tuning of TinyLlama (Colab + Hugging Face Hub)
- [x] LLM-backed explanation endpoint
- [x] RAG over banking policy PDFs
- [x] Evidently drift monitoring
- [x] Dockerized deployment + GitHub Actions CI
- [ ] Model and data cards

---

## Links

- **Live demo:** [TinyLlama Credit Explainer on Hugging Face Spaces](https://huggingface.co/spaces/rohankatyayani/tinyllama-credit-explainer)
- **Fine-tuned model:** [rohankatyayani/tinyllama-credit-explainer](https://huggingface.co/rohankatyayani/tinyllama-credit-explainer)
- **Training notebook:** [`notebooks/llama_finetune.ipynb`](notebooks/llama_finetune.ipynb)

---

## License

MIT © Rohan Katyayani
