import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from model import LanguageClassifier
from feature_extraction import word_to_ngram_features
from data_loader import load_data
from data_loader import load_vocab_dict
from train import calculate_accuracy

MODEL_PATH = "models/best_model.pth"
VOCAB_PATH = "models/vocab.json"

label_to_language = {
    0: "English",
    1: "Spanish",
    2: "French",
    3: "Portuguese",
    4: "Italian",
    5: "Romanian",
    6: "Unknown"
}

def main(model_path=MODEL_PATH):
    print('Evaluating model')
    vocab_dict = load_vocab_dict()
    model = LanguageClassifier(len(vocab_dict))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    _, _, _, _, test_sents, test_labels = load_data()

    test_features = [word_to_ngram_features(s, vocab_dict) for s in test_sents]
    test_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_targets = torch.tensor(test_labels, dtype=torch.long)



    with torch.no_grad():
        pred = model(test_tensor)
        _, predicted = torch.max(pred, dim=1)
        acc = calculate_accuracy(pred, test_targets)
    print(f"Test Accuracy: {100*acc:.2f}%")

    print("\nClassification Report:")
    report = classification_report(test_targets, predicted, target_names=label_to_language.values())
    print(report)
    with open("/eval_report.txt", "w") as f:
        f.write(report)
        f.write(f"Test Accuracy: {100*acc:.2f}%")

    cm = confusion_matrix(test_targets, predicted)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_to_language.values(),
                yticklabels=label_to_language.values())
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    main()