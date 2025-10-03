# FinRisk Copilot — End‑to‑End Banking AI (LoRA/PEFT + RAG + MLOps)

**Elevator pitch:** A production‑style system that detects suspicious transactions (tabular ML),
generates bank‑tone explanations using a **LoRA‑fine‑tuned** LLM, and answers policy questions via **RAG**.
Everything is tracked with MLflow, served via FastAPI, dockerized, and wired with CI + monitoring.

> Designed to show **senior‑level ownership** across DS, LLMs (PEFT), and MLOps for a bank context.

---

## Architecture (three components)

- **A) Fraud Detector (Tabular ML)** — LightGBM/XGBoost, class imbalance handling, calibration, SHAP.
- **B) Reason‑Code Explainer (LLM + LoRA/QLoRA)** — PEFT adapters to write short regulator‑style narratives.
- **C) Policy/Risk Assistant (RAG)** — PDF ingestion → embeddings → vector index → grounded answers with citations.

Glue: MLflow tracking/registry, FastAPI service, Docker, GitHub Actions, Evidently monitoring.

```
Data → Features → Train (tabular) → PR‑AUC/Threshold → SHAP
           ↘ signals → LoRA (LLM) → case narration
Docs/PDFs → chunk/embed → vector store → RAG answers
All artifacts tracked in MLflow → served via FastAPI → monitored
```

---

## Quickstart (Milestone 1 — Repo & Environment)

> **Python version:** 3.11 recommended (widest library compatibility).

1. **Create virtual environment**
   ```bash
   python3.11 -m venv .venv           # ensure Python 3.11 installed
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Run the API locally (smoke test)**
   ```bash
   uvicorn src.service.app:app --reload --port 8000
   # open http://127.0.0.1:8000/health → {"status":"ok"}
   ```

3. **Run tests**
   ```bash
   pytest -q
   ```

4. **Pre‑commit hooks (format & lint on commit)**
   ```bash
   pre-commit install
   ```

5. **Git & GitHub (first push)**
   ```bash
   git init
   git branch -M main
   git add .
   git commit -m "init: scaffold repo, README, CI, FastAPI smoke test"
   git remote add origin <YOUR_GITHUB_REPO_URL>
   git push -u origin main
   ```

> **Mac with Apple Silicon (M‑series):** Some packages (e.g., FAISS, bitsandbytes) are optional and platform‑specific. 
> Start with the current `requirements.txt`. We’ll add GPU‑only extras later from a Linux/CUDA machine or Colab.

---

## Project Structure

```
finrisk-copilot/
├─ README.md
├─ requirements.txt
├─ .pre-commit-config.yaml
├─ .gitignore
├─ LICENSE
├─ .github/workflows/ci.yml
├─ docker/Dockerfile
├─ src/
│  ├─ service/app.py            # FastAPI health check + placeholder endpoints
│  ├─ __init__.py
│  ├─ config/__init__.py
│  ├─ data/__init__.py
│  ├─ models/__init__.py
│  ├─ rag/__init__.py
│  ├─ utils/__init__.py
├─ data/                        # (ignored) raw/interim/processed go here
│  ├─ .gitignore
│  ├─ raw/.gitkeep
│  ├─ interim/.gitkeep
│  └─ processed/.gitkeep
└─ tests/
   ├─ test_api.py
   └─ __init__.py
```

## 🧠 Fine-Tuning LLM for Credit Risk Explanations

This notebook, `llama_finetune.ipynb`, demonstrates how I fine-tuned **TinyLlama**, a lightweight open LLM, on a dataset of German Credit Risk profiles. The goal is to make the model generate **interpretable explanations** for credit approval or denial decisions.

### ⚙️ What it does
- Uses **Hugging Face Transformers**, **PEFT**, and **LoRA** for parameter-efficient fine-tuning.
- Quantized the model to 8-bit using `bitsandbytes` to fit within Colab GPU memory.
- Fine-tuned on `german_credit_explanations.jsonl`, a dataset containing financial attributes and human-readable risk explanations.
- The final model (`tinyllama-credit-explainer`) is uploaded to my [Hugging Face Hub](https://huggingface.co/rohankatyayani/tinyllama-credit-explainer).

### 🧩 Tech Stack
- **Model:** TinyLlama-1.1B
- **Libraries:** Hugging Face Transformers, Datasets, PEFT (LoRA), BitsAndBytes
- **Notebook:** Google Colab (with CUDA)
- **Repo:** [finrisk-copilot](https://github.com/RohanKatyayani/finrisk-copilot)
- **Goal:** Explain model decisions in financial risk applications — improving transparency and trust in AI-driven lending.

---

💬 *Next step:* Integrate this fine-tuned model into the FinRisk Copilot pipeline for explainable credit assessments.

---

## Roadmap (you’ll tick these off)

- [x] **Milestone 1:** Repo ready, env & CI pass, API health OK
- [x] **Milestone 2:** Tabular fraud dataset prepared (train/valid split), baseline LightGBM with PR‑AUC
- [x] **Milestone 3:** SHAP explanations + threshold tuning + calibration
- [x] **Milestone 4:** Synthetic “reason‑code” dataset generation
- [x] **Milestone 5:** LoRA/QLoRA training (GPU/Colab) + eval
- [ ] **Milestone 6:** RAG over policy PDFs + citations
- [ ] **Milestone 7:** Unified FastAPI service endpoints
- [ ] **Milestone 8:** Docker + GitHub Actions CI
- [ ] **Milestone 9:** Monitoring (Evidently) + model/data cards
- [ ] **Milestone 10:** Demo script + interview notes

---

## Interview soundbite for this milestone

> “I started by pinning Python 3.11, setting up a clean repo with pre‑commit (black/ruff), CI on PRs, and a FastAPI health endpoint. 
> This gives me reproducibility and guardrails before I add heavy ML/LLM pieces.”

---

## License

MIT © 2025 Rohan Katyayani
