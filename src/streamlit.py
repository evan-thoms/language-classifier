import streamlit as st
import torch
from model import LanguageClassifier
from feature_extraction import word_to_ngram_features
from data_loader import load_vocab_dict
import numpy as np


vocab = load_vocab_dict()
model = LanguageClassifier(len(vocab))
model.load_state_dict(torch.load("../models/best_model.pth"))
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
