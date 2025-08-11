import gradio as gr
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import faiss
from datasets import load_dataset
import requests # <-- Add new imports
import io

# --- Configuration ---
MODEL_PATH = "clip_finetuned" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FAISS_INDEX_PATH = "gallery.index"

# --- Load Model, Processor, and FAISS Index ---
print("Loading model and processor...")
model = CLIPModel.from_pretrained(MODEL_PATH).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_PATH)

print("Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# --- Connect to the COCO dataset on the Hub ---
print("Connecting to COCO dataset on the Hub...")
val_dataset = load_dataset("phiyodr/coco2017", split="validation", trust_remote_code=True)
    
print(f"Successfully connected to dataset with {len(val_dataset)} images.")

# --- The Search Function (Corrected) ---
def image_search(query_text: str, top_k: int):
    with torch.no_grad():
        inputs = processor(text=query_text, return_tensors="pt").to(DEVICE)
        text_embedding = model.get_text_features(**inputs)
        text_embedding /= text_embedding.norm(p=2, dim=-1, keepdim=True)

    distances, indices = faiss_index.search(text_embedding.cpu().numpy(), int(top_k))

    # --- CORRECTED LOGIC ---
    # For each result, get its URL from the dataset, download it, and open it as an image.
    results = []
    for i in indices[0]:
        item = val_dataset[int(i)]
        image_url = item['coco_url']
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        results.append(image)
        
    return results

# --- Gradio Interface (No changes needed here) ---
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🖼️ CLIP-Powered Image Search Engine")
    gr.Markdown("Enter a text description to search for matching images.")
    
    with gr.Row():
        query_input = gr.Textbox(label="Search Query", placeholder="e.g., a red car parked near a building", scale=4)
        k_slider = gr.Slider(minimum=1, maximum=12, value=4, step=1, label="Number of Results")
        submit_btn = gr.Button("Search", variant="primary")

    gallery_output = gr.Gallery(label="Search Results", show_label=False, columns=4, height="auto")

    submit_btn.click(fn=image_search, inputs=[query_input, k_slider], outputs=gallery_output)
    
    gr.Examples(
        examples=[["a dog catching a frisbee", 4], ["two people eating pizza", 8]],
        inputs=[query_input, k_slider]
    )

iface.launch(share=True)