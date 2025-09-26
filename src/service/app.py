from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
import os

# --- Ensure logs directory exists ---
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI
app = FastAPI(title="Credit Risk API", version="1.0")

# --- Load model ---
# --- Load model ---
try:
    # Try absolute path inside Docker
    model_path = os.path.join("/app", "models", "credit_risk_model.pkl")

    if not os.path.exists(model_path):
        # fallback for local runs
        model_path = os.path.join(os.path.dirname(__file__), "../../models/credit_risk_model.pkl")
        model_path = os.path.abspath(model_path)

    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    model_loaded = True
except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    model_loaded = False


# --- Request Schema ---
class PredictionRequest(BaseModel):
    status: str = Field(..., description="Account status")
    duration: int = Field(..., gt=0, description="Duration in months (must be >0)")
    credit_history: str
    purpose: str
    amount: int = Field(..., gt=0, description="Credit amount")
    savings: str
    employment_duration: str
    installment_rate: int = Field(..., ge=1, le=4, description="Installment rate (1-4)")
    personal_status_sex: str
    other_debtors: str
    present_residence: int
    property: str
    age: int = Field(..., ge=18, description="Age must be >= 18")
    other_installment_plans: str
    housing: str
    number_credits: int
    job: str
    people_liable: int
    telephone: str
    foreign_worker: str


# --- Endpoints ---
@app.get("/")
def root():
    return {"message": "Credit Risk API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict")
def predict(request: PredictionRequest):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.model_dump()])

        # Predict
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0].tolist()

        # Log request & result
        logging.info(f"Input: {request.model_dump()} | Prediction: {prediction} | Prob: {probabilities}")

        return {
            "prediction": int(prediction),
            "probabilities": probabilities
        }

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))