from fastapi.testclient import TestClient
from src.service.app import app

# client = TestClient(app)
#
#
# def test_health():
#     r = client.get("/health")
#     assert r.status_code == 200
#     assert r.json().get("status") == "ok"

# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from src.service.app import app

client = TestClient(app)


def test_health():
    """Test the /health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_predict():
    """Test the /predict endpoint with a sample payload"""
    sample_request = {
        "status": "A11",
        "duration": 12,
        "credit_history": "A34",
        "purpose": "A43",
        "amount": 1500,
        "savings": "A65",
        "employment_duration": "A75",
        "installment_rate": 2,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "present_residence": 2,
        "property": "A121",
        "age": 35,
        "other_installment_plans": "A143",
        "housing": "A152",
        "number_credits": 1,
        "job": "A173",
        "people_liable": 1,
        "telephone": "A192",
        "foreign_worker": "A201"
    }

    response = client.post("/predict", json=sample_request)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "probabilities" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) == 2