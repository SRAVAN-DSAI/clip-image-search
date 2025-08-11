# CLIP-Powered Image Search Engine

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sravan837/clip-image-search)

This project is an end-to-end demonstration of building a powerful text-to-image search engine. It involves fine-tuning an OpenAI CLIP model on the COCO 2017 dataset for enhanced domain-specific understanding and deploying the final application as an interactive Gradio web demo.

## Features
- **Fine-Tuned AI Model**: Utilizes a CLIP model fine-tuned on the COCO dataset for more accurate image-text matching.
- **Efficient Vector Search**: Employs FAISS (Facebook AI Similarity Search) to create a highly efficient index for retrieving similar images from thousands of vector embeddings.
- **Interactive Web UI**: Features an intuitive and user-friendly web interface built with Gradio.
- **Live Deployment**: The application is publicly accessible and deployed on Hugging Face Spaces.

---

## Tech Stack
- **Backend**: Python
- **ML Framework**: PyTorch
- **Core Libraries**:
  - `Hugging Face Transformers` (for CLIP model)
  - `Hugging Face Datasets` (for data handling)
  - `FAISS` (for vector search)
  - `Gradio` (for the web interface)

---

## Project Workflow

The project is divided into two main stages: offline processing and live inference.

### Offline: Model Training and Indexing
1.  **Fine-Tuning**: The base `openai/clip-vit-base-patch32` model is fine-tuned on the COCO 2017 dataset. To maintain most of the pre-trained knowledge while adapting to the new data, only the model's projection heads are trained.
2.  **Indexing**: The fine-tuned vision encoder is used to process all 5,000 images from the COCO validation set, converting each one into a vector embedding. These embeddings are then stored in a FAISS index file for fast lookup.

### Live: Inference with Gradio
1.  **User Input**: A user enters a text query (e.g., "a person riding a horse") into the Gradio web application.
2.  **Text Encoding**: The fine-tuned CLIP text encoder converts the user's query into a vector embedding.
3.  **Similarity Search**: This text embedding is used to search the FAISS index, which quickly finds the indices of the most similar image vectors.
4.  **Image Retrieval**: The application fetches the corresponding images for the retrieved indices directly from the public COCO dataset on the Hugging Face Hub.
5.  **Display**: The final images are displayed to the user in the gallery.



---

## Deployment

The live application is deployed on Hugging Face Spaces and can be accessed at the link below. The Space runs the `app.py` script, which loads the fine-tuned model, the FAISS index, and handles all user interactions.

** [Live Demo Link](https://huggingface.co/spaces/sravan837/clip-image-search)**