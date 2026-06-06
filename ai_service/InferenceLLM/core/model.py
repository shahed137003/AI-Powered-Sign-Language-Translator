import torch
import torch.nn as nn
import numpy as np
from .config import DEVICE

class TemporalBlock(nn.Module):
    def __init__(self, inp, out, dil, drop):
        super().__init__()
        p = dil
        self.net = nn.Sequential(
            nn.Conv1d(inp, out, 3, padding=p, dilation=dil),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(out, out, 3, padding=p, dilation=dil),
            nn.BatchNorm1d(out),
            nn.ReLU(),
            nn.Dropout(drop)
        )
        self.res = nn.Identity() if inp == out else nn.Conv1d(inp, out, 1)

    def forward(self, x):
        y = self.net(x)
        if y.size(-1) != x.size(-1):
            y = y[..., :x.size(-1)]
        return y + self.res(x)


class TCN(nn.Module):
    def __init__(self, input_dim, num_classes, ch=128, layers=4, drop=0.3):
        super().__init__()
        blocks = []
        for i in range(layers):
            ind = input_dim if i == 0 else ch
            blocks.append(TemporalBlock(ind, ch, 2**i, drop))
        self.tcn = nn.Sequential(*blocks)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(ch, num_classes)

    def masked_pool(self, x, m):
        m = m.unsqueeze(1)
        x = x * m
        s = x.sum(-1)
        d = m.sum(-1).clamp(min=1)
        return s / d

    def forward(self, x, m):
        x = self.tcn(x)
        x = self.masked_pool(x, m)
        x = self.drop(x)
        return self.fc(x)


class ModelWrapper:
    def __init__(self, model, labels):
        """Receives model and labels from main.py"""
        self.model = model
        self.labels = labels
        self.model.eval()

    def predict(self, sequence):
        """Returns (label_str, confidence_percentage)"""
        x = torch.from_numpy(sequence).float().transpose(0, 1).unsqueeze(0)
        m = (sequence.sum(axis=1) != 0).astype(np.float32)
        m = torch.from_numpy(m).float().unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x, m)
            probs = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)

        label = self.labels[idx.item()]
        confidence = conf.item() * 100
        print(f"  Model predict: {label} ({confidence:.1f}%)")
        return label, confidence