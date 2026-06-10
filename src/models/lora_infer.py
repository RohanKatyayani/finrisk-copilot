import json, subprocess, sys, logging
logger = logging.getLogger(__name__)

_INFER_SCRIPT = """
import sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
data  = json.loads(sys.argv[1])
feats, pred, max_t = data["features"], data["prediction"], data.get("max_new_tokens", 120)
MODEL_ID = "rohankatyayani/tinyllama-credit-explainer"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
decision = "denied" if pred == 1 else "approved"
feat_str = ", ".join(f"{k}={v}" for k, v in feats.items())
prompt = f"Explain the credit risk decision for the following applicant profile.\\nInput: {feat_str}\\nDecision: {decision}.\\nExplanation:"
inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=max_t, do_sample=False, repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
new_tokens = out[0][inputs["input_ids"].shape[-1]:]
explanation = tok.decode(new_tokens, skip_special_tokens=True).strip()
if not explanation or len(explanation) < 10:
    explanation = f"Application {decision} based on the provided financial profile."
print(json.dumps({"explanation": explanation}))
"""

def generate_explanation(features: dict, prediction: int, max_new_tokens: int = 120) -> str:
    payload = json.dumps({"features": features, "prediction": prediction, "max_new_tokens": max_new_tokens})
    try:
        result = subprocess.run(
            [sys.executable, "-c", _INFER_SCRIPT, payload],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed: {result.stderr[-200:]}")
        output_line = [l for l in result.stdout.strip().splitlines() if l.startswith("{")]
        if not output_line:
            raise RuntimeError("No JSON output from subprocess")
        return json.loads(output_line[-1])["explanation"]
    except subprocess.TimeoutExpired:
        decision_word = "denied" if prediction == 1 else "approved"
        return f"Application {decision_word} based on the provided financial profile."
