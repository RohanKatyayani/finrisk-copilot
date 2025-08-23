from fastapi import FastAPI

app = FastAPI()

@app.get("/")   # already made
def read_root():
    return {"message": "FinRisk CoPilot is running 🚀"}

@app.get("/hello")   # NEW ROUTE
def say_hello():
    return {"greeting": "Hello Rohan! 👋"}