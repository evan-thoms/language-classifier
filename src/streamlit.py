import streamlit as st
import torch
import numpy as np
import os


from feature_extraction import word_to_ngram_features
from huggingface_hub import hf_hub_download
from data_loader import load_vocab_dict
from model import LanguageClassifier


if os.path.exists("src"):
    VOCAB_PATH = "models/vocab.json"
else:
    VOCAB_PATH = "../models/vocab.json"
LOCAL_MODEL_PATH = "../models/best_model.pth"

#Loads classifier model from hugging face or locally
@st.cache_resource
def load_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print("Loading model from local file...")
        model_path = LOCAL_MODEL_PATH
    else:
        print("Downloading model from Hugging Face...")
        model_path = hf_hub_download(
            repo_id="ethoms29/romance-classifier",
            filename="best_model.pth",
            cache_dir="./hf_cache"  
        )
    return model_path


MODEL_PATH = load_model()
vocab = load_vocab_dict()
model = LanguageClassifier(len(vocab))
model.load_state_dict(torch.load(MODEL_PATH))
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
st.write("Type a sentence in a Romance language (+English) and see which language it's in")
st.write("Languages include English, Spanish, French, Italian, Portuguese, and Romanian")

with st.expander("ℹ️ About the Model"):
    st.markdown("""
    This model was trained on a mixture of:
    - **Wikipedia articles** (across multiple topics like history, earth science, and computer science)
    - **Parallel TED Talk translations**
    
    It classifies input into one of 5 Romance languages: Spanish, French, Portuguese, Italian, or Romanian with English, and with a 7th fallback category of **Unknown**.

    **Note:** This is a basic n-gram based model and not a full transformer or deep language model. It's fast and lightweight, but **may not be accurate on short, slangy, or ambiguous sentences**.
    """)

sentence = st.text_input("Your sentence:")


#Puts sentence through model and outputs most likely language
if sentence:
    features = word_to_ngram_features(sentence, vocab)
    tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        sorted_indices = np.argsort(probs)

        top_idx = probs[len(sorted_indices)-1]
        

        pred_idx = torch.argmax(probs).item()
       
        confidence = probs[0][pred_idx].item()

        st.write(f"**Predicted Language:** {label_to_language[pred_idx]}")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        confidence2 = probs[0][pred_idx].item()


        second_pred = pred_idx

        with st.expander("🔍 See second guess"):
            st.write(f"Second-highest prediction: **{label_to_language[second_pred]}**")
            st.write(f"Confidence: {confidence2:.2%}")
