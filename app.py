import gradio as gr
from huggingface_hub import InferenceClient
import pkg_resources

print("Gradio version:", pkg_resources.get_distribution("gradio").version)

# Initialize Hugging Face Inference Client
client = InferenceClient("rohankatyayani/tinyllama-credit-explainer")

def explain_credit(profile):
    """
    Generate an explanation for the given credit applicant profile.
    """
    try:
        prompt = f"Explain why this applicant with profile: {profile} is labeled as good or bad."
        response = client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- Gradio UI ---
with gr.Blocks(title="TinyLlama Credit Explainer 💳") as demo:
    gr.Markdown(
        """
        # 💳 TinyLlama Credit Explainer
        Analyze and explain credit risk labels (good/bad) based on applicant data.
        Fine-tuned using the TinyLlama model on FICO-style datasets.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            profile_input = gr.Textbox(
                label="Applicant Profile",
                placeholder="Example: status=no checking account, duration=24, savings=unknown, employment=1 <= ... < 4 years, amount=3200, age=29",
                lines=3,
            )
            submit_btn = gr.Button("Explain")

        with gr.Column(scale=1):
            explanation_output = gr.Textbox(
                label="Model Explanation",
                lines=8,
                interactive=False
            )

    submit_btn.click(explain_credit, inputs=profile_input, outputs=explanation_output)

demo.launch()