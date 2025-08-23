from fastapi import FastAPI

app = FastAPI(title="FinRisk Copilot")

@app.get("/health")
def health():
    return {"status": "ok"}
