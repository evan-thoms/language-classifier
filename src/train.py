import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import datetime
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import TensorDataset, DataLoader
from feature_extraction import word_to_ngram_features
from data_loader import create_vocab_dict
from model import LanguageClassifier
from data_loader import load_data


LOCAL_MODEL_PATH = "../models/best_model.pth"
if os.path.exists("src"):
    VOCAB_PATH = "models/vocab.json"
    TRAINING_METADATA_PATH = "models/training_metadata.json"
    BEST_MODEL_PATH = "models/best_model.pth"
    RECENT_MODEL_PATH = "models/recent_model.pth"
else:
    VOCAB_PATH = "../models/vocab.json"
    TRAINING_METADATA_PATH = "../models/training_metadata.json"
    BEST_MODEL_PATH = "../models/best_model.pth"
    RECENT_MODEL_PATH = "../models/recent_model.pth"


#Calculates accuracy of predictions against target labels
def calculate_accuracy(predictions, targets):
    _, predicted = torch.max(predictions,1)
    correct = (predicted == targets).sum().item()
    return correct/len(targets)

#Saves training metadata to a file in /models
def save_training_metadata(val_acc, test_acc, epoch, time, model_type="basic"):
    print("saving json data")
    data = {
        "best_val_acc": val_acc,
        "test_acc": test_acc,
        "epoch": epoch,
        "timestamp": time,
        "model_type": model_type,
        "training_duration": time
    }
    with open(TRAINING_METADATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
        
#Gets the best accuracy from the current best_model
def load_best_accuracy():
    print("Getting previous best accuracy")
    if os.path.exists(TRAINING_METADATA_PATH):
        with open(TRAINING_METADATA_PATH, "r") as f:
            meta = json.load(f)
            acc = meta.get("best_val_acc", 0.0)
            print("returning previous best accuracy: ", acc)
            return acc

def main():
    print("Starting Training")
    
    train_sents, train_labels, val_sents, val_labels, test_sents, test_labels = load_data()
    print(f"Training samples: {len(train_sents)}")
    print(f"Validation samples: {len(val_sents)}")
    print(f"Test samples: {len(test_sents)}")
    
    vocab_dict = create_vocab_dict(train_sents+val_sents+test_sents)
    print(f"Vocabulary size: {len(vocab_dict)}")

    print("Converting sentences to ngram features")
    train_features = [word_to_ngram_features(s, vocab_dict) for s in train_sents]
    val_features = [word_to_ngram_features(s, vocab_dict) for s in val_sents]          
    test_features = [word_to_ngram_features(s, vocab_dict) for s in test_sents]

    train_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_targets = torch.tensor(train_labels, dtype=torch.long)

    val_tensor = torch.tensor(val_features, dtype=torch.float32)
    val_targets = torch.tensor(val_labels, dtype=torch.long)

    test_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_targets = torch.tensor(test_labels, dtype=torch.long)

    batch_size = 32
    train_dataset = TensorDataset(train_tensor, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_tensor, val_targets)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    num_epochs = 100

    # Use original model
    model = LanguageClassifier(len(vocab_dict))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Beginning training loop")
    start_time = datetime.datetime.now()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            pred = model(batch_inputs)
            loss = loss_fn(pred, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(pred, batch_targets)
            
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                val_pred = model(batch_inputs)
                v_loss = loss_fn(val_pred, batch_targets)

                val_loss += v_loss.item()
                val_acc += calculate_accuracy(val_pred, batch_targets)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        if val_acc > best_val_acc:
            print(f"New best validation accuracy: {val_acc:.4f}")
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            print(f"Epoch: {epoch:3d} - Train Loss: {train_loss:.4f} Acc: {100*train_acc:.2f}% "
                  f"--- Val Loss: {val_loss:.4f} Acc: {100*val_acc:.2f}%")

        if patience_counter >= patience:
            print("Stopping early - Preventing overfitting")
            break

    end_time = datetime.datetime.now()
    training_duration = str(end_time - start_time)

    # Save the most recent model
    torch.save(model.state_dict(), RECENT_MODEL_PATH)

    print("Testing with test set")
    model.eval()
    with torch.no_grad():
        test_preds = model(test_tensor)
        test_acc = calculate_accuracy(test_preds, test_targets)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Save training metadata
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_training_metadata(val_acc, test_acc, epoch, time, "basic")
    
    print(f"Training completed in {training_duration}")

if __name__ == "__main__":
    main()

