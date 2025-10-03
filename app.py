import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load fine-tuned model from Hugging Face Hub
model_id = "rohankatyayani/tinyllama-credit-explainer"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Build pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def explain_credit(profile):
    prompt = f"Explain why this applicant with profile: {profile} is labeled as bad."
    response = generator(
        prompt,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return response[0]["generated_text"]

# Gradio UI
demo = gr.Interface(
    fn=explain_credit,
    inputs=gr.Textbox(lines=3, label="Applicant Profile"),
    outputs=gr.Textbox(label="Model Explanation"),
    title="TinyLlama Credit Risk Explainer",
    description="Paste an applicant's financial profile and get a simple explanation for the risk label."
)

if __name__ == "__main__":
    demo.launch()