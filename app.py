import gradio as gr
from huggingface_hub import InferenceClient
import traceback, pkg_resources

print("Gradio version:", pkg_resources.get_distribution("gradio").version)

# Initialize Hugging Face Inference Client
MODEL_ID = "rohankatyayani/tinyllama-credit-explainer"
client = InferenceClient(MODEL_ID)

def explain_credit(profile):
    """
    Generate an explanation for the given credit applicant profile.
    Includes detailed debug info to identify failures.
    """
    try:
        prompt = f"Explain why this applicant with profile: {profile} is labeled as good or bad."
        response = client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
        # If response is a dict or object, convert to string
        return str(response)
    except Exception as e:
        tb = traceback.format_exc()
        return f"⚠️ **Error calling model**:\n```\n{e}\n```\n**Traceback:**\n```\n{tb}\n```"

with gr.Blocks(title="TinyLlama Credit Explainer 💳") as demo:
    gr.Markdown("# 💳 TinyLlama Credit Explainer (Debug Mode)")
    input_box = gr.Textbox(
        label="Applicant Profile",
        placeholder="status=no checking account, duration=36, savings=unknown/no savings account, employment=1 <= ... < 4 years, amount=9055, age=35"
    )
    output_box = gr.Textbox(label="Response", lines=12)
    btn = gr.Button("Explain")
    btn.click(explain_credit, inputs=input_box, outputs=output_box)

demo.launch()