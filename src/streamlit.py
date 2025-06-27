import streamlit as st
import torch
from model import LanguageClassifier
from feature_extraction import word_to_ngram_features
from data_loader import load_vocab_dict
import numpy as np
import os
from huggingface_hub import hf_hub_download



MODEL_PATH = "https://huggingface.co/ethoms29/romance-classifier/resolve/main/best_model.pth"
LOCAL_MODEL_PATH = "../models/best_model.pth"
if os.path.exists("src"):
    VOCAB_PATH = "models/vocab.json"
else:
    VOCAB_PATH = "../models/vocab.json"

@st.cache_resource
def load_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        st.info("Loading model from local file...")
        model_path = LOCAL_MODEL_PATH
    else:
        st.info("Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="ethoms29/romance-classifier",
            filename="best_model.pth",
            cache_dir="./hf_cache"  
        )
    return model_path

model_path = load_model()
st.write("model path", model_path)
vocab = load_vocab_dict()
st.write("Trying path:", VOCAB_PATH)

st.sidebar.write(f"Model Path: {MODEL_PATH}")
st.sidebar.write(f"Vocab Path: {VOCAB_PATH}")
st.sidebar.write(f"Model exists: {os.path.exists(MODEL_PATH)}")
st.sidebar.write(f"Vocab exists: {os.path.exists(VOCAB_PATH)}")
model = LanguageClassifier(len(vocab))
model.load_state_dict(torch.load(model_path))
model.eval()

label_to_language = {
    0: "English",
    1: "Spanish",
    2: "French",
    3: "Portuguese",
    4: "Italian",
    5: "Romanian",
    6: "Unknown"
}

st.title("🧠 Romance Language Classifier")
st.write("Type a sentence in a Romance language and see which language it's in.")

sentence = st.text_input("Your sentence:")

if sentence:
    features = word_to_ngram_features(sentence, vocab)
    tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = probs[0][pred_idx].item()

    st.write(f"**Predicted Language:** {label_to_language[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2%}")
