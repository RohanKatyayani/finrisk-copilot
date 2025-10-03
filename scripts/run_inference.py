import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose your repo name on HF
MODEL_ID = "rohankatyayani/tinyllama-credit-explainer"

# Detect device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"💡 Using device: {device}")

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

# Example prompt
prompt = "Explain why this applicant with profile: status=no checking account, duration=36, savings=unknown/no savings account, employment=1 <= ... < 4 years, amount=9055, age=35 is labeled as bad."

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("\n🔹 Model response:")
print(response)