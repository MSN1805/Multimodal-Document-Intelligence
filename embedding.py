import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


def generate_embeddings(text_chunks):
    return model.encode(
        text_chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )
