import gc
import torch
from torch.quantization import quantize_dynamic
import streamlit as st
import numpy as np
import pandas as pd
import faiss
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from openai import OpenAI

# --- 1. Quantized CLIP & lazy‑loaded SBERT ---
torch.backends.quantized.engine = "qnnpack"


@st.cache_resource(show_spinner=False)
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to("cpu")
    from torch.quantization import quantize_dynamic
    # this will now use QNNPACK under the hood
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# we'll only call this if the user picks Clip‑STrans
def load_strans():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- 2. Memory‑map FAISS + lean Parquet data + downcast dtypes ---
@st.cache_data(show_spinner=False)
def load_data_and_indices(model_choice):
    # ——— Load only needed CSV columns, memory‑map, and downcast dtypes ———
    df = pd.read_csv(
        "data/Amazon2023DS_partial_NLP.csv",
        usecols=["image", "title", "store", "price", "average_rating"],
        dtype={
            "store": "category",
            "price": "float32",
            "average_rating": "float32",
        },
        memory_map=True,
        low_memory=False,
    )

    # ——— Memory‑map FAISS indexes ———
    io_flag = faiss.IO_FLAG_MMAP
    if model_choice == "Clip-Clip":
        text_idx = faiss.read_index("data/clip_text_vector_index_w_NLP_2.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)
    elif model_choice == "Clip-STrans":
        text_idx = faiss.read_index("data/SBERT_text_vector_index_w_NLP.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)
    else:  # Blip‑Blip fallback
        text_idx = faiss.read_index("data/clip_text_vector_index_w_NLP_2.faiss", io_flag)
        img_idx  = faiss.read_index("data/clip_image_vector_index.faiss", io_flag)

    return df, text_idx, img_idx


def get_clip_text_embedding(text: str, clip_model, clip_processor) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    arr = feats.squeeze().cpu().detach().numpy().astype("float32").reshape(1, -1)
    faiss.normalize_L2(arr)
    return arr

def get_sbert_text_embedding(text: str, strans_model) -> np.ndarray:
    emb = strans_model.encode(text, normalize_embeddings=True)
    return emb.reshape(1, -1).astype("float32")

def retrieve_similar_products(
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    model_choice: str = "Clip-Clip",
    multiplier: int = 20,  # reduced from 50 → 20
) -> pd.DataFrame:
    df, text_idx, img_idx = load_data_and_indices(model_choice)
    clip_model, clip_processor = load_clip()

    # lazy‑load SBERT only if needed
    strans_model = load_strans() if model_choice == "Clip-STrans" else None

    # embed + search
    if model_choice == "Clip-Clip":
        emb = get_clip_text_embedding(query, clip_model, clip_processor)
        td, ti = text_idx.search(emb, k * multiplier)
        id_, ii = img_idx.search(emb, k * multiplier)

    elif model_choice == "Clip-STrans":
        emb_clip = get_clip_text_embedding(query, clip_model, clip_processor)
        emb_sbert = get_sbert_text_embedding(query, strans_model)
        td, ti = text_idx.search(emb_sbert, k * multiplier)
        id_, ii = img_idx.search(emb_clip, k * multiplier)

    else:  # Blip‑Blip fallback to Clip‑Clip
        emb = get_clip_text_embedding(query, clip_model, clip_processor)
        td, ti = text_idx.search(emb, k * multiplier)
        id_, ii = img_idx.search(emb, k * multiplier)

    # aggregate scores
    candidates = set(ti[0]) | set(ii[0])
    scores = {idx: 0.0 for idx in candidates}
    for rank, idx in enumerate(ti[0]):
        scores[idx] += alpha * td.flatten()[rank]
    for rank, idx in enumerate(ii[0]):
        scores[idx] += (1 - alpha) * id_.flatten()[rank]

    topk = sorted(scores, key=scores.get, reverse=True)[:k]
    results = df.iloc[topk].reset_index(drop=True)

    # --- cleanup to free memory between runs ---
    for name in ('emb', 'emb_clip', 'emb_sbert','td', 'ti', 'id_', 'ii'):
        if name in locals():
            del locals()[name]
    gc.collect()

    return results

# --- Streamlit UI remains largely unchanged ---
st.set_page_config(page_title="Gènie", layout="centered")
st.title("💄💋🪞 Hi, I'm Gènie - Your personal beauty assistant!")
st.subheader("What can I help you with today?")

query = st.text_input("Type your question here…", key="input")
col1, col2 = st.columns([0.7, 0.3])
with col1:
    alpha = st.slider("Alpha:", 0.0, 1.0, 0.5)
with col2:
    model_choice = st.radio("Model:", ["Clip-Clip", "Clip-STrans", "Blip-Blip"])

if st.button("Enter") and query:
    results = retrieve_similar_products(query, k=5, alpha=alpha, model_choice=model_choice)
    st.markdown("### You asked:")
    st.write(query)
    st.markdown("### 💡 Recommendations:")
    for _, row in results.iterrows():
        st.image(row["image"], width=200)
        st.markdown(
            f"**{row['title']}**  \n"
            f"Store: {row['store']}  \n"
            f"Price: ${row['price']:.2f}  \n"
            f"Rating: {row['average_rating']:.1f}/5"
        )
else:
    st.write("Awaiting your question…")
