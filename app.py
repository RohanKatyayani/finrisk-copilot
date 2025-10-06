import gradio as gr
import pkg_resources
from huggingface_hub import InferenceClient

print("Gradio version:", pkg_resources.get_distribution("gradio").version)

# Initialize Hugging Face model client
client = InferenceClient("rohankatyayani/tinyllama-credit-explainer")

def respond(message, system_message):
    """Handles the chat logic with TinyLlama credit explainer."""
    prompt = f"{system_message}\n\nApplicant profile: {message}"
    response = client.text_generation(prompt, max_new_tokens=256, temperature=0.7)
    return response.strip()

with gr.Blocks(title="TinyLlama Credit Explainer") as demo:
    gr.Markdown(
        """
        # 💳 TinyLlama Credit Explainer
        Explain **credit risk decisions** made by your fine-tuned TinyLlama model on FICO data.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            system_message = gr.Textbox(
                label="System Instruction",
                value="Explain why a credit applicant is labeled as good or bad based on their financial profile.",
            )
            user_input = gr.Textbox(
                label="Applicant Profile",
                placeholder="status=no checking account, duration=36, savings=unknown, employment=1 <= ... < 4 years, amount=9055, age=35",
            )
            submit = gr.Button("Explain")

        with gr.Column(scale=3):
            output = gr.Textbox(label="Model Explanation", lines=6)

    submit.click(fn=respond, inputs=[user_input, system_message], outputs=output)

demo.launch()