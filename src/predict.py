import torch
from sklearn.metrics import classification_report, confusion_matrix
from model import LanguageClassifier
from data_loader import load_data, load_vocab_dict
from feature_extraction import word_to_ngram_features
import argparse

BEST_MODEL_PATH = "../models/best_model.pth"
VOCAB_PATH = "../models/vocab.json"

def predict_lang(sentence, model, vocab_dict, threshold=0.35):
    model.eval()
    features = word_to_ngram_features(sentence, vocab_dict)
    tensor = torch.tensor([features], dtype=torch.float32)

    with torch.no_grad():
        pred = model(tensor)
        probs = torch.softmax(pred, dim=1)
        max_prob, lang = torch.max(probs, dim=1)

        if max_prob.item()<threshold:
            return 6, max_prob.item()
        else:
            return lang.item(), max_prob.item()

def main():
    print("Predicting Language")
    parser = argparse.ArgumentParser(description="Predict language of example sentence")
    parser.add_argument("--input_sentence", type=str, default="", help="Example sentence to be predicted")
    args = parser.parse_args()

    if args.input_sentence.strip():
        print("Using inputted sentence")
        test_sentences=[args.input_sentence]
    else:
        print("Using exapmle sentences")
        test_sentences = [
        "The moon is made of rock.",
        "La computadora es muy rápida.",
        "Les enfants jouent dans le jardin.",
        "As crianças estão brincando no jardim",
        "I bambini stanno giocando in giardino.",
        "Copiii se joacă în grădină.",
        "A laila, aia ke kilokilo o nā hōkū.",
        "pláž je teplá"
        
    ]

    print("Evaluating")
    vocab_dict = load_vocab_dict()
    model = LanguageClassifier(len(vocab_dict))

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    label_to_language = {
        0: "English",
        1: "Spanish",
        2: "French",
        3: "Portuguese",
        4: "Italian",
        5: "Romanian",
        6: "Unknown Language"
    }

    

    for sent in test_sentences:
        label, conf = predict_lang(sent, model, vocab_dict)
        print(f"'{sent}' -> {label_to_language[label]} Confidence: {conf:.2f}")
if __name__ == "__main__":
    main()