### CONFIG ###
CRYSTAL_URL = "http://localhost:6657/v1"
CRYSTAL_MODEL = "oylan_a_v_t_2_5"
CRYSTAL_LORA_MODEL = "oylan_2_5_vision_lora"  # New model config for Vision
CHERRY_URL = "http://localhost:6697/v1"
CHERRY_MODEL = "oylan_a_v_t_3_0"
MAX_TOKENS = 16000
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
PRESENCE_PENALTY = 1.5
GRADIO_TEMP_DIR = "./gradio_cache"

### IMPORTS ###
import os
import base64
import gradio as gr
import openai
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

### SETUP ###
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR

### FUNCTIONS ###
def encode_image_base64(image: Image.Image) -> str:
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def parse_thinking(text: str) -> tuple:
    if "</think>" in text:
        parts = text.split("</think>", 1)
        thinking = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
        return thinking, answer
    return "", text

def call_model(base_url: str, model: str, messages: list) -> tuple:
    client = openai.Client(base_url=base_url, api_key="EMPTY")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
        extra_body={
            "top_p": TOP_P,
            "top_k": TOP_K,
            "presence_penalty": PRESENCE_PENALTY
        }
    )
    content = response.choices[0].message.content
    return parse_thinking(content)

def generate_text(query: str):
    messages = [
        {"role": "system", "content": "You are helpful AI Assistant."},
        {"role": "user", "content": query}
    ]
    
    with ThreadPoolExecutor(max_workers=2) as ex:
        # Text requests use the standard CRYSTAL_MODEL
        crystal_future = ex.submit(call_model, CRYSTAL_URL, CRYSTAL_MODEL, messages)
        cherry_future = ex.submit(call_model, CHERRY_URL, CHERRY_MODEL, messages)
        
        crystal_thinking, crystal_answer = crystal_future.result()
        cherry_thinking, cherry_answer = cherry_future.result()
    
    return crystal_thinking, crystal_answer, cherry_thinking, cherry_answer

def generate_image(query: str, image: Image.Image):
    if image is None:
        return "", "Please upload an image.", "", "Please upload an image."
    
    base64_image = encode_image_base64(image)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        }
    ]
    
    with ThreadPoolExecutor(max_workers=2) as ex:
        # Vision requests use the CRYSTAL_LORA_MODEL
        crystal_future = ex.submit(call_model, CRYSTAL_URL, CRYSTAL_LORA_MODEL, messages)
        cherry_future = ex.submit(call_model, CHERRY_URL, CHERRY_MODEL, messages)
        
        crystal_thinking, crystal_answer = crystal_future.result()
        cherry_thinking, cherry_answer = cherry_future.result()
    
    return crystal_thinking, crystal_answer, cherry_thinking, cherry_answer

### MAIN EXECUTION ###
css = """
#main-title h1 {
    font-size: 2.3em !important;
    text-align: center;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎉 CRYSTAL vs CHERRY Model Comparison 🎉", elem_id="main-title")
    
    # --- INPUT SECTION (Top) ---
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Text Inference"):
                    text_query = gr.Textbox(
                        label="Query Input",
                        placeholder="Enter your query here...",
                        lines=3
                    )
                    text_submit = gr.Button("Submit Text", variant="primary")
                
                with gr.TabItem("Image Inference"):
                    with gr.Row():
                        image_query = gr.Textbox(
                            label="Query Input",
                            placeholder="Describe this image in detail",
                            lines=3,
                            scale=3
                        )
                        image_upload = gr.Image(type="pil", label="Upload Image", height=150, scale=1)
                    image_submit = gr.Button("Submit Image", variant="primary")

    # --- OUTPUT SECTION (Bottom) ---
    with gr.Row():
        # Crystal Output Block
        with gr.Column():
            gr.Markdown("## 💎 CRYSTAL")
            # Thinking block exists but is hidden
            crystal_thinking = gr.Textbox(visible=False) 
            crystal_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=25,
                show_copy_button=True
            )
        
        # Cherry Output Block
        with gr.Column():
            gr.Markdown("## 🍒 CHERRY")
            # Thinking block exists but is hidden
            cherry_thinking = gr.Textbox(visible=False)
            cherry_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=25,
                show_copy_button=True
            )
    
    # --- EVENT HANDLERS ---
    text_submit.click(
        fn=generate_text,
        inputs=[text_query],
        outputs=[crystal_thinking, crystal_output, cherry_thinking, cherry_output]
    )
    
    image_submit.click(
        fn=generate_image,
        inputs=[image_query, image_upload],
        outputs=[crystal_thinking, crystal_output, cherry_thinking, cherry_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(show_error=True, share=True)