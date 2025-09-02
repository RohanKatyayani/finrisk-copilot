from fastapi import FastAPI
import joblib
import pandas as pd

# Load model at startup
model = joblib.load("models/credit_risk_model.pkl")

app = FastAPI(title="Credit Risk API", version="1.0")


@app.get("/")
def read_root():
    return {"message": "Credit Risk API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    # Convert input JSON to DataFrame
    df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": proba
    }