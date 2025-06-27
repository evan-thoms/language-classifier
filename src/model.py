import torch
import torch.nn as nn

#Defines model
class LanguageClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128,64)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(64,32)
        self.dropout3 = nn.Dropout(0.2)
        self.layer4 = nn.Linear(32,7)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = self.layer4(x)
        return x