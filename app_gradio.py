"""
FinRisk Copilot — Gradio UI for Hugging Face Spaces.

Unlike streamlit_app.py (which is an HTTP client for the FastAPI service), this
app runs as a single process and calls the project's Python functions directly.
Spaces serves one process on one port, so there is no separate API to talk to.

The LightGBM model is trained at startup because models/ is gitignored.
"""

import os
import subprocess
import sys
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

from src.rag.qa import answer_question

# --------------------------------------------------------------------------
# ZeroGPU support. The `spaces` package only exists on Hugging Face hardware;
# locally we fall back to a no-op decorator so the same file runs anywhere.
# --------------------------------------------------------------------------
try:
    import spaces

    GPU = spaces.GPU(duration=60)
except Exception:  # not on Spaces

    def GPU(fn):
        return fn


MODEL_PATH = Path("models/credit_risk_model.pkl")
EXPLAINER_ID = "rohankatyayani/tinyllama-credit-explainer"

# --------------------------------------------------------------------------
# Startup: make sure a trained model exists.
# --------------------------------------------------------------------------
if not MODEL_PATH.exists():
    print("[startup] No model found — training LightGBM pipeline ...", flush=True)
    subprocess.run([sys.executable, "src/training/train_model.py"], check=True)

pipeline = joblib.load(MODEL_PATH)
print(f"[startup] Loaded model from {MODEL_PATH}", flush=True)

# --------------------------------------------------------------------------
# Explainer: loaded lazily, kept on CPU, moved to GPU inside the GPU call.
# --------------------------------------------------------------------------
_tok = None
_llm = None


def _load_explainer():
    global _tok, _llm
    if _llm is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("[explainer] loading weights ...", flush=True)
        _tok = AutoTokenizer.from_pretrained(EXPLAINER_ID)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token
        _llm = AutoModelForCausalLM.from_pretrained(
            EXPLAINER_ID, dtype=torch.float32, low_cpu_mem_usage=True
        )
        _llm.eval()
    return _tok, _llm


@GPU
def generate_explanation(features: dict, prediction: int, max_new_tokens: int = 120) -> str:
    """Generate a bank-tone explanation for a decision. 1 = good/approved."""
    import torch

    tok, llm = _load_explainer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm.to(device)

    decision = "approved" if prediction == 1 else "denied"
    feat_str = ", ".join(f"{k}={v}" for k, v in features.items())
    prompt = (
        "Explain the credit risk decision for the following applicant profile.\n"
        f"Input: {feat_str}\nDecision: {decision}.\nExplanation:"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.inference_mode():
        out = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1] :]
    text = tok.decode(new_tokens, skip_special_tokens=True).strip()
    if not text or len(text) < 10:
        text = f"Application {decision} based on the provided financial profile."
    return text


# --------------------------------------------------------------------------
# German Credit field mappings: human label -> dataset code.
# --------------------------------------------------------------------------
STATUS = {
    "No checking account": "A14",
    "< 0 DM": "A11",
    "0-200 DM": "A12",
    ">= 200 DM / salary assigned": "A13",
}
CREDIT_HISTORY = {
    "No credits / all paid duly": "A30",
    "All paid at this bank": "A31",
    "Existing credits paid duly": "A32",
    "Past delay in paying": "A33",
    "Critical account / other credits": "A34",
}
PURPOSE = {
    "Car (new)": "A40",
    "Car (used)": "A41",
    "Furniture/equipment": "A42",
    "Radio/TV": "A43",
    "Domestic appliances": "A44",
    "Repairs": "A45",
    "Education": "A46",
    "Retraining": "A48",
    "Business": "A49",
    "Other": "A410",
}
SAVINGS = {
    "< 100 DM": "A61",
    "100-500 DM": "A62",
    "500-1000 DM": "A63",
    ">= 1000 DM": "A64",
    "Unknown / none": "A65",
}
EMPLOYMENT = {
    "Unemployed": "A71",
    "< 1 year": "A72",
    "1-4 years": "A73",
    "4-7 years": "A74",
    ">= 7 years": "A75",
}
PERSONAL = {
    "Male: single": "A93",
    "Male: divorced/separated": "A91",
    "Male: married/widowed": "A94",
    "Female: divorced/separated/married": "A92",
    "Female: single": "A95",
}
DEBTORS = {"None": "A101", "Co-applicant": "A102", "Guarantor": "A103"}
PROPERTY = {
    "Real estate": "A121",
    "Life insurance / savings agreement": "A122",
    "Car or other": "A123",
    "Unknown / none": "A124",
}
PLANS = {"None": "A143", "Bank": "A141", "Stores": "A142"}
HOUSING = {"Own": "A152", "Rent": "A151", "For free": "A153"}
JOB = {
    "Skilled employee / official": "A173",
    "Unskilled - resident": "A172",
    "Unemployed / unskilled - non-resident": "A171",
    "Management / self-employed / highly qualified": "A174",
}
TELEPHONE = {"Yes, registered": "A192", "None": "A191"}
FOREIGN = {"Yes": "A201", "No": "A202"}

CLASS_LABELS = {0: "Bad credit", 1: "Good credit"}

FIELD_ORDER = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]


def build_form():
    """Render the 20-field application form; return the component list."""
    with gr.Row():
        with gr.Column():
            status = gr.Dropdown(
                list(STATUS), value="No checking account", label="Checking account status"
            )
            credit_history = gr.Dropdown(
                list(CREDIT_HISTORY), value="Existing credits paid duly", label="Credit history"
            )
            purpose = gr.Dropdown(list(PURPOSE), value="Radio/TV", label="Purpose")
            savings = gr.Dropdown(list(SAVINGS), value="< 100 DM", label="Savings")
            employment = gr.Dropdown(
                list(EMPLOYMENT), value="1-4 years", label="Employment duration"
            )
            personal = gr.Dropdown(
                list(PERSONAL), value="Male: single", label="Personal status / sex"
            )
            debtors = gr.Dropdown(list(DEBTORS), value="None", label="Other debtors")
            prop = gr.Dropdown(list(PROPERTY), value="Real estate", label="Property")
            plans = gr.Dropdown(list(PLANS), value="None", label="Other installment plans")
            housing = gr.Dropdown(list(HOUSING), value="Own", label="Housing")
        with gr.Column():
            job = gr.Dropdown(list(JOB), value="Skilled employee / official", label="Job")
            telephone = gr.Dropdown(list(TELEPHONE), value="Yes, registered", label="Telephone")
            foreign = gr.Dropdown(list(FOREIGN), value="Yes", label="Foreign worker")
            duration = gr.Number(value=24, label="Duration (months)", precision=0)
            amount = gr.Number(value=3500, label="Credit amount (DM)", precision=0)
            age = gr.Number(value=30, label="Age", precision=0)
            rate = gr.Slider(1, 4, value=2, step=1, label="Installment rate (% of income)")
            residence = gr.Number(value=2, label="Present residence since (years)", precision=0)
            ncred = gr.Number(value=1, label="Existing credits at this bank", precision=0)
            liable = gr.Number(value=1, label="People liable to maintain", precision=0)

    return [
        status,
        credit_history,
        purpose,
        savings,
        employment,
        personal,
        debtors,
        prop,
        plans,
        housing,
        job,
        telephone,
        foreign,
        duration,
        amount,
        age,
        rate,
        residence,
        ncred,
        liable,
    ]


def to_payload(
    status,
    credit_history,
    purpose,
    savings,
    employment,
    personal,
    debtors,
    prop,
    plans,
    housing,
    job,
    telephone,
    foreign,
    duration,
    amount,
    age,
    rate,
    residence,
    ncred,
    liable,
):
    """Map form values to the dataset's coded feature dict."""
    return {
        "status": STATUS[status],
        "duration": int(duration),
        "credit_history": CREDIT_HISTORY[credit_history],
        "purpose": PURPOSE[purpose],
        "amount": int(amount),
        "savings": SAVINGS[savings],
        "employment_duration": EMPLOYMENT[employment],
        "installment_rate": int(rate),
        "personal_status_sex": PERSONAL[personal],
        "other_debtors": DEBTORS[debtors],
        "present_residence": int(residence),
        "property": PROPERTY[prop],
        "age": int(age),
        "other_installment_plans": PLANS[plans],
        "housing": HOUSING[housing],
        "number_credits": int(ncred),
        "job": JOB[job],
        "people_liable": int(liable),
        "telephone": TELEPHONE[telephone],
        "foreign_worker": FOREIGN[foreign],
    }


def score(features: dict):
    df = pd.DataFrame([features])[FIELD_ORDER]
    pred = int(pipeline.predict(df)[0])
    proba = pipeline.predict_proba(df)[0].tolist()
    return pred, proba


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------
def do_predict(*vals):
    feats = to_payload(*vals)
    pred, proba = score(feats)
    md = f"### {CLASS_LABELS[pred]}\n**Confidence:** {max(proba):.1%}"
    return md, {"prediction": pred, "probabilities": proba}


def do_explain(decision_label, *vals):
    feats = to_payload(*vals)
    pred = 1 if decision_label == "Good credit" else 0
    text = generate_explanation(feats, pred)
    return f"### Explanation for: {CLASS_LABELS[pred]}\n\n{text}"


def do_combined(*vals):
    feats = to_payload(*vals)
    pred, proba = score(feats)
    text = generate_explanation(feats, pred)
    md = (
        f"### {CLASS_LABELS[pred]}\n"
        f"**Confidence:** {max(proba):.1%}\n\n"
        f"**Explanation**\n\n{text}"
    )
    return md, {"prediction": pred, "probabilities": proba, "explanation": text}


def do_policy(question, k):
    if not question or len(question.strip()) < 3:
        return "Please enter a question.", [], {}
    try:
        data = answer_question(question.strip(), int(k))
    except Exception as e:
        return f"**Request failed:** {e}", [], {}
    rows = [
        [s["rank"], s["source"], s["chunk_id"], round(s["score"], 3)]
        for s in data.get("sources", [])
    ]
    return f"### Answer\n\n{data.get('answer', '')}", rows, data


EXAMPLES = [
    "What is a risk-based approach to money laundering risk?",
    "What does customer due diligence require when establishing a business relationship?",
    "How should a bank identify the beneficial owner of a legal entity?",
    "What enhanced due diligence applies to politically exposed persons?",
    "What is the best pizza topping?",
]

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
with gr.Blocks(title="FinRisk Copilot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🏦 FinRisk Copilot\n"
        "Credit-risk scoring, plain-English explanations, and AML/KYC policy Q&A.\n\n"
        "[GitHub](https://github.com/RohanKatyayani/finrisk-copilot) · "
        "[Model card](https://github.com/RohanKatyayani/finrisk-copilot/blob/main/docs/MODEL_CARD.md) · "
        "*Portfolio demo — not for real credit decisions.*"
    )

    with gr.Tab("Predict"):
        gr.Markdown("Score an application with the LightGBM model.")
        p_inputs = build_form()
        p_btn = gr.Button("Score application", variant="primary")
        p_out = gr.Markdown()
        p_json = gr.JSON(label="Raw response")
        p_btn.click(do_predict, inputs=p_inputs, outputs=[p_out, p_json])

    with gr.Tab("Explain"):
        gr.Markdown(
            "Generate the bank's reasoning for a decision **you** choose — useful for "
            "comparing how the same profile is justified either way."
        )
        e_decision = gr.Radio(
            ["Good credit", "Bad credit"], value="Good credit", label="Decision to explain"
        )
        e_inputs = build_form()
        e_btn = gr.Button("Generate explanation", variant="primary")
        e_out = gr.Markdown()
        e_btn.click(do_explain, inputs=[e_decision] + e_inputs, outputs=e_out)

    with gr.Tab("Combined"):
        gr.Markdown("Score **and** explain in one step.")
        c_inputs = build_form()
        c_btn = gr.Button("Score and explain", variant="primary")
        c_out = gr.Markdown()
        c_json = gr.JSON(label="Raw response")
        c_btn.click(do_combined, inputs=c_inputs, outputs=[c_out, c_json])

    with gr.Tab("Ask Policy"):
        gr.Markdown(
            "Retrieval-augmented answers grounded in three Basel and FATF documents on "
            "anti-money-laundering, CFT, and customer due diligence (561 chunks). "
            "Answers cite passages inline as [1], [2]. Questions outside that scope — "
            "including capital-adequacy topics — are refused by design."
        )
        q = gr.Textbox(label="Question", lines=2, placeholder="Ask about AML, CFT, KYC or CDD ...")
        gr.Examples(EXAMPLES, inputs=q, label="Try an example (the last one is out of scope)")
        k = gr.Slider(1, 10, value=4, step=1, label="Passages to retrieve (k)")
        a_btn = gr.Button("Ask", variant="primary")
        a_out = gr.Markdown()
        a_src = gr.Dataframe(
            headers=["#", "Document", "Chunk", "Similarity"], label="Sources", wrap=True
        )
        a_json = gr.JSON(label="Raw response")
        a_btn.click(do_policy, inputs=[q, k], outputs=[a_out, a_src, a_json])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
