import streamlit as st
from sentence_transformers import SentenceTransformer

# Cache the model to avoid reloading it on every interaction
@st.cache_resource
def get_model(model_name="nomic-ai/modernbert-embed-base"):
    return SentenceTransformer(model_name)

def embed_texts(model, texts, batch_size=16):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
