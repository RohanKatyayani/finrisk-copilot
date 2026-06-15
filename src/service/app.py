"""
src/service/app.py

FinRisk Copilot — FastAPI service
Endpoints:
  GET  /health              — liveness check
  POST /predict             — LightGBM credit risk score
  POST /explain             — TinyLlama plain-English explanation
  POST /predict_and_explain — combined (score + explanation in one call)
  POST /ask_policy          — RAG over banking policy PDFs (Groq Llama 3.1)
"""

import os
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinRisk Copilot",
    description="Credit risk scoring + plain-English explanations + policy QA via RAG",
    version="3.0",
)

# ---------------------------------------------------------------------------
# Load LightGBM pipeline at startup
# ---------------------------------------------------------------------------
def _find_model(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

_model_path = _find_model([
    "/app/models/credit_risk_model.pkl",                          # Docker
    os.path.abspath("models/credit_risk_model.pkl"),              # local
])

if _model_path:
    lgbm_pipeline = joblib.load(_model_path)
    model_loaded = True
    print(f"✅ LightGBM pipeline loaded from: {_model_path}")
else:
    lgbm_pipeline = None
    model_loaded = False
    print("❌ LightGBM model not found — /predict will return 503")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    status: str
    duration: int = Field(..., gt=0)
    credit_history: str
    purpose: str
    amount: int = Field(..., gt=0)
    savings: str
    employment_duration: str
    installment_rate: int = Field(..., ge=1, le=4)
    personal_status_sex: str
    other_debtors: str
    present_residence: int
    property: str
    age: int = Field(..., ge=18)
    other_installment_plans: str
    housing: str
    number_credits: int
    job: str
    people_liable: int
    telephone: str
    foreign_worker: str


class ExplainRequest(BaseModel):
    features: dict = Field(..., description="Same keys as /predict body")
    prediction: int = Field(..., ge=0, le=1, description="0=good, 1=bad credit")


class AskPolicyRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question about banking policy")
    k: int = Field(4, ge=1, le=10, description="Number of chunks to retrieve")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_lgbm(req: PredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([req.model_dump()])
        pred = int(lgbm_pipeline.predict(df)[0])
        proba = lgbm_pipeline.predict_proba(df)[0].tolist()
        return pred, proba
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
def predict(req: PredictionRequest):
    pred, proba = _run_lgbm(req)
    logger.info(f"predict | pred={pred} proba={proba}")
    return {"prediction": pred, "probabilities": proba}


@app.post("/explain")
def explain(req: ExplainRequest):
    """
    Generate a plain-English explanation for a credit decision.
    Pass the same feature dict you'd send to /predict, plus the prediction (0 or 1).
    Note: First call downloads/loads the model (~30s). Subsequent calls are faster.
    """
    try:
        from src.models.lora_infer import generate_explanation
        explanation = generate_explanation(req.features, req.prediction)
        logger.info(f"explain | pred={req.prediction} | explanation={explanation[:80]}")
        return {"explanation": explanation, "prediction": req.prediction}
    except Exception as e:
        logger.error(f"Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_and_explain")
def predict_and_explain(req: PredictionRequest):
    """
    Convenience endpoint: run LightGBM prediction then generate explanation.
    Returns score + probabilities + plain-English reasoning in one call.
    """
    pred, proba = _run_lgbm(req)
    try:
        from src.models.lora_infer import generate_explanation
        explanation = generate_explanation(req.model_dump(), pred)
    except Exception as e:
        logger.warning(f"Explanation failed, returning score only: {e}")
        explanation = "Explanation unavailable."

    logger.info(f"predict_and_explain | pred={pred} | explanation={explanation[:80]}")
    return {
        "prediction": pred,
        "probabilities": proba,
        "explanation": explanation,
    }


@app.post("/ask_policy")
def ask_policy(req: AskPolicyRequest):
    """
    Answer banking policy questions using retrieval-augmented generation
    over Basel + FATF source documents. Returns a grounded answer with
    inline citations [1], [2]... and a separate sources array.
    """
    try:
        from src.rag.qa import answer_question
        result = answer_question(req.question, k=req.k)
        logger.info(f"ask_policy | q={req.question[:60]!r} | n_sources={len(result['sources'])}")
        return result
    except FileNotFoundError as e:
        logger.error(f"RAG index missing: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"ask_policy error: {e}")
        raise HTTPException(status_code=500, detail=str(e))