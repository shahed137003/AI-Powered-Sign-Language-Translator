import torch
import torch.nn as nn
import numpy as np
from .config import MODEL_PATH, LABEL_PATH, DEVICE #get the paths
import sys
from pathlib import Path

# add preprocessing path
from .config import PREPROCESSING_PATH
sys.path.append(str(PREPROCESSING_PATH))

from preprocessing.constants import FEATURE_DIM


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y[..., :x.size(2)]
        return y + self.res(x)


class TCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        channels = [128, 128, 128, 128]
        layers = []

        for i, c in enumerate(channels):
            in_dim = input_dim if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_dim, c, dilation=2**i))

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class ModelWrapper: #This is the inference interface
    def __init__(self):
        self.labels = np.load(LABEL_PATH, allow_pickle=True) #loads class names from label encoder
        self.model = TCN(FEATURE_DIM, len(self.labels)).to(DEVICE)

        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE) #loads trained weights from model
        self.model.load_state_dict(checkpoint['model_state_dict']) #restores model
        self.model.eval()

    def predict(self, sequence): #function to predict and returns the label and the confidence
        x = torch.from_numpy(sequence).float().transpose(0, 1).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        return self.labels[idx.item()], conf.item() * 100