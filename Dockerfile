FROM python:3.11-slim

WORKDIR /app

#COPY models/credit_risk_model.pkl models/credit_risk_model.pkl
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure logs directory exists
RUN mkdir -p /app/models

EXPOSE 8000
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]