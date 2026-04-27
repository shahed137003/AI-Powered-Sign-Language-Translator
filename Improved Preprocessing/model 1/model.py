import os
import time
import gc
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# -----------------------
# CONFIG (LOCAL FIX)
# -----------------------

DATA_DIR = r"balanced_dataset"   
OUTPUT_DIR = r"training_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 120
LR = 2e-4

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------
# LOAD DATA
# -----------------------

def load_split(split):
    X = np.load(os.path.join(DATA_DIR, split, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, split, "y.npy"))
    m = np.load(os.path.join(DATA_DIR, split, "mask.npy"))
    return X, y, m


X_train, y_train, m_train = load_split("train")
X_val, y_val, m_val = load_split("val")
X_test, y_test, m_test = load_split("test")

NUM_CLASSES = len(np.unique(y_train))
INPUT_DIM = X_train.shape[2]

print("\n--- DATASET SUMMARY ---")
print(f"Classes: {NUM_CLASSES}")
print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"Input dim: {INPUT_DIM}")
print(f"Using Device: {DEVICE}")
print("-----------------------\n")

# -----------------------
# SAVE LABEL ENCODER
# -----------------------

label_map = {int(i): int(i) for i in np.unique(y_train)}
np.save(os.path.join(OUTPUT_DIR, "label_encoder.npy"), label_map)

# -----------------------
# DATASET
# -----------------------

class SignDataset(Dataset):
    def __init__(self, X, mask, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        m = self.mask[idx]
        y = self.y[idx]

        if self.augment and torch.rand(1) < 0.3:
            x = x + torch.randn_like(x) * 0.002

        return x, m, y


train_loader = DataLoader(
    SignDataset(X_train, m_train, y_train, augment=True),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SignDataset(X_val, m_val, y_val),
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    SignDataset(X_test, m_test, y_test),
    batch_size=BATCH_SIZE
)

# -----------------------
# MODEL (UNCHANGED)
# -----------------------

class ST_GCN_Block(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()

        self.spatial = nn.Conv1d(in_ch, in_ch, 1, groups=in_ch)

        self.temporal = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )

        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout1d(0.25)

    def forward(self, x):
        r = self.res(x)
        x = self.spatial(x)
        x = self.temporal(x)
        return self.drop(x) + r


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([
            ST_GCN_Block(256, 256, 1),
            ST_GCN_Block(256, 384, 2),
            ST_GCN_Block(384, 384, 4),
            ST_GCN_Block(384, 512, 8),
        ])

        self.mha = nn.MultiheadAttention(512, 8, batch_first=True)

        self.pool = nn.Linear(512, 1)

        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x, mask):

        x = self.stem(x)
        x = x.transpose(1, 2)

        for b in self.blocks:
            x = b(x)

        x = x.transpose(1, 2)

        key_padding_mask = (mask == 0).bool()

        attn, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = x + attn

        score = self.pool(x).squeeze(-1)
        score = score.masked_fill(key_padding_mask, -1e4)

        w = torch.softmax(score, dim=1).unsqueeze(-1)
        x = (x * w).sum(1)

        return self.head(x)


model = Model().to(DEVICE)

# -----------------------
# TRAIN SETUP (UNCHANGED)
# -----------------------

criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.02)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=len(train_loader)
)

scaler = torch.cuda.amp.GradScaler()

# -----------------------
# TRAIN LOOP (UNCHANGED)
# -----------------------

train_accs, val_accs, train_losses = [], [], []
best_acc = 0

for epoch in range(EPOCHS):

    model.train()
    correct, total, loss_sum = 0, 0, 0

    for x, m, y in train_loader:
        x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x, m)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_sum += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    train_loss = loss_sum / len(train_loader)

    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for x, m, y in val_loader:
            x, m = x.to(DEVICE), m.to(DEVICE)
            out = model(x, m)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.numpy())

    val_acc = accuracy_score(targets, preds)

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))
        tag = "⭐"
    else:
        tag = ""

    print(f"E{epoch+1:03d} | Train acc: {train_acc:.4f} | Loss: {train_loss:.4f} | Val acc: {val_acc:.4f} {tag}")

# -----------------------
# TEST + SAVE
# -----------------------

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "model.pth")))
model.eval()

preds, targets = [], []

with torch.no_grad():
    for x, m, y in test_loader:
        x, m = x.to(DEVICE), m.to(DEVICE)
        out = model(x, m)
        preds.extend(out.argmax(1).cpu().numpy())
        targets.extend(y.numpy())

wrong_rows = []
for p, t in zip(preds, targets):
    if p != t:
        wrong_rows.append({"prediction": int(p), "truth": int(t)})

pd.DataFrame(wrong_rows).to_csv(
    os.path.join(OUTPUT_DIR, "wrong_predictions.csv"), index=False
)

acc = accuracy_score(targets, preds)
print(f"\n🎯 FINAL TEST ACC: {acc:.4f}")
print(f"📁 Saved everything to: {OUTPUT_DIR}")