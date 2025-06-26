import streamlit as st
import torch
from model import LanguageClassifier
from feature_extraction import word_to_ngram_features
from data_loader import load_vocab_dict
import numpy as np
import os

def get_project_paths():
    """Get project paths with fallback options for different environments"""
    try:
        if __file__:
            print("first")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(base_dir)
        else:
            raise AttributeError("__file__ not available")
    except (AttributeError, TypeError):
        current_dir = os.getcwd()
        if current_dir.endswith('/src') or current_dir.endswith('\\src'):
            print("endswithsrc")
            project_dir = os.path.dirname(current_dir)
        elif os.path.exists(os.path.join(current_dir, 'src')):
            print("second")
            project_dir = current_dir
        else:
            print("looking in project root")
            # Try to find the project root
            search_dir = current_dir
            while search_dir != os.path.dirname(search_dir):
                if (os.path.exists(os.path.join(search_dir, 'src')) and 
                    os.path.exists(os.path.join(search_dir, 'models'))):
                    project_dir = search_dir
                    break
                search_dir = os.path.dirname(search_dir)
            else:
                print("looking here")
                project_dir = current_dir
    print("making modelpaths ", project_dir)
    model_path = os.path.join(project_dir, "models", "best_model.pth")
    vocab_path = os.path.join(project_dir, "models", "vocab.json")
    
    return project_dir, model_path, vocab_path

PROJECT_DIR, MODEL_PATH, VOCAB_PATH = get_project_paths()
                      

vocab = load_vocab_dict()
st.write("Trying path:", VOCAB_PATH)
st.sidebar.write(f"Project Dir: {PROJECT_DIR}")
st.sidebar.write(f"Model Path: {MODEL_PATH}")
st.sidebar.write(f"Vocab Path: {VOCAB_PATH}")
st.sidebar.write(f"Model exists: {os.path.exists(MODEL_PATH)}")
st.sidebar.write(f"Vocab exists: {os.path.exists(VOCAB_PATH)}")
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
