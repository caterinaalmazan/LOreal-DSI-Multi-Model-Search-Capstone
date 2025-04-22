# app.py

import torch
# Prevent Streamlit’s file watcher from iterating torch.classes.__path__
torch.classes.__path__ = []


import streamlit as st
import numpy as np
import pandas as pd
import faiss
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer 
from PIL import Image
import requests
from openai import OpenAI

# --- Cache the CLIP model & processor once per session ---
@st.cache_resource(show_spinner=False)
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

@st.cache_resource(show_spinner=False)
def load_strans():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# --- Cache data & FAISS indices once per session ---
@st.cache_data(show_spinner=False)
def load_data_and_indices(model):
    df = pd.read_csv("data/Amazon2023DS_partial_NLP.csv")
    if model == "Clip-Clip":
        text_index = faiss.read_index("data/clip_text_vector_index_w_NLP_2.faiss")
        image_index = faiss.read_index("data/clip_image_vector_index.faiss")
    elif model == "Clip-STrans":
        text_index = faiss.read_index("data/SBERT_text_vector_index_w_NLP.faiss")
        image_index = faiss.read_index("data/clip_image_vector_index.faiss")
    else:
        text_index = faiss.read_index("data/clip_text_vector_index_w_NLP_2.faiss")
        image_index = faiss.read_index("data/clip_image_vector_index.faiss")
    return df, text_index, image_index



def get_clip_text_embedding(text: str, model, processor) -> np.ndarray:
    """Generate a normalized CLIP text embedding."""
    inputs = processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    arr = feats.squeeze().cpu().detach().numpy().astype("float32").reshape(1, -1)
    faiss.normalize_L2(arr)
    return arr

def get_sbert_text_embedding(text: str, model):
    """Generate sbert text embedding for a new query."""
    embeddings = model.encode(text, normalize_embeddings=True)

    return embeddings.reshape(1, -1).astype("float32")

def retrieve_similar_products(query: str, k: int = 5, alpha: float = 0.5, model: str = "Clip-Clip") -> pd.DataFrame:
    """Combine text & image FAISS searches, then rank by weighted score."""
    df, text_idx, img_idx = load_data_and_indices(model)
    clip_model, clip_processor = load_clip()
    strans_model = load_strans()
    multiplier = 50
    if model == "Clip-Clip":
        emb = get_clip_text_embedding(query, clip_model, clip_processor)
        # search both indices
        td, ti = text_idx.search(emb, k * multiplier)
        id, ii = img_idx.search(emb, k * multiplier)
    elif model == "Clip-STrans":
        # Compute CLIP text embedding for query (512D)
        query_text_emb_clip = get_clip_text_embedding(query, clip_model, clip_processor)
        # Compute SBERT text embedding for query (384D)
        query_text_emb_sbert = get_sbert_text_embedding(query, strans_model)
        # Step 1: Search Text FAISS
        td, ti = text_idx.search(query_text_emb_sbert, k * multiplier)
        # Step 2: Search Image FAISS Using Same Text Query
        id, ii = img_idx.search(query_text_emb_clip, k * multiplier)
    else:
        emb = get_clip_text_embedding(query, clip_model, clip_processor)
        # search both indices
        td, ti = text_idx.search(emb, k * multiplier)
        id, ii = img_idx.search(emb, k * multiplier)


    candidates = set(ti[0]) | set(ii[0])
    scores = {cand: 0.0 for cand in candidates}

    # accumulate weighted distances
    for rank, idx in enumerate(ti[0]):
        scores[idx] += alpha * td.flatten()[rank]
    for rank, idx in enumerate(ii[0]):
        scores[idx] += (1 - alpha) * id.flatten()[rank]

    topk = sorted(scores, key=scores.get, reverse=True)[:k]
    return df.iloc[topk].reset_index(drop=True)


# --- Streamlit UI ---
st.set_page_config(page_title="Gènie", layout="centered")
st.title("💄💋🪞 Hi, I'm Gènie - Your personal beauty assistant!")
st.subheader("What can I help you with today?")

user_query = st.text_input("Type your question here...", key="input")
col1, col2 = st.columns([0.7, 0.3])
with col1:
    alpha = st.slider("Alpha:", 0.0, 1.0, 0.5)
with col2:
    model = st.radio(
        "Model:",
        ["Clip-Clip", "Clip-STrans", "Blip-Blip"],
    )

if st.button("Enter") and user_query:
    results = retrieve_similar_products(user_query, k=5, alpha=alpha, model=model)

    st.markdown("### You asked:")
    st.write(user_query)

    st.markdown("### 💡 Recommendations:")
    for i, row in results.iterrows():
        st.image(row["image"], width=200)
        st.markdown(
            f"**{row['title']}**  \n"
            f"Store: {row['store']}  \n"
            f"Price: {row['price']}  \n"
            f"Rating: {row['average_rating']}"
        )

else:
    st.write("Awaiting your question...")
