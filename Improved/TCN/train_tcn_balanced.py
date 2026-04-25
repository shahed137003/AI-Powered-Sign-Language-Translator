import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = r"D:\Additionalpreprocessing\balanced_dataset"
OUTPUT_DIR = r"D:\Additionalpreprocessing\training_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 
EPOCHS = 100
PATIENCE = 15
LR = 1e-3

# -----------------------
# LOAD & WEIGHTING
# -----------------------
def load_split(split):
    X = np.load(f"{DATA_DIR}/{split}/X.npy")
    y = np.load(f"{DATA_DIR}/{split}/y.npy")
    m = np.load(f"{DATA_DIR}/{split}/mask.npy")
    return X, y, m

X_train, y_train, m_train = load_split("train")
X_val, y_val, m_val       = load_split("val")
X_test, y_test, m_test    = load_split("test")

class_map = np.load(f"{DATA_DIR}/class_map.npy", allow_pickle=True).item()
NUM_CLASSES = len(class_map)

weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# -----------------------
# MODEL
# -----------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.45),
            nn.Conv1d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.45),
        )
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res(x)

class TCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(input_dim, 64, 1),
            TemporalBlock(64, 128, 2),
            TemporalBlock(128, 256, 4),
            TemporalBlock(256, 128, 8),
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def masked_pool(self, x, mask):
        mask = mask.unsqueeze(1)
        x = x * mask
        return x.sum(dim=2) / mask.sum(dim=2).clamp(min=1)

    def forward(self, x, mask):
        x = self.tcn(x)
        x = self.masked_pool(x, mask)
        x = self.dropout(x)
        return self.fc(x)

# -----------------------
# DATASET
# -----------------------
class SignDataset(Dataset):
    def __init__(self, X, mask, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx].transpose(0, 1), self.mask[idx], self.y[idx]

train_loader = DataLoader(SignDataset(X_train, m_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SignDataset(X_val, m_val, y_val), batch_size=BATCH_SIZE)
test_loader = DataLoader(SignDataset(X_test, m_test, y_test), batch_size=BATCH_SIZE)

model = TCN(X_train.shape[2], NUM_CLASSES).to(DEVICE)

# -----------------------
# TRAIN SETUP
# -----------------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.15, weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# FIXED: Removed 'verbose=True' to fix the TypeError
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# -----------------------
# ENGINE
# -----------------------
def run_epoch(loader, is_train):
    model.train() if is_train else model.eval()
    t_loss, t_acc, t_count = 0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, m, y in loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            out = model(x, m)
            loss = criterion(out, y)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item() * y.size(0)
            t_acc += (out.argmax(1) == y).sum().item()
            t_count += y.size(0)
    return t_loss / t_count, t_acc / t_count

# -----------------------
# LOOP
# -----------------------
best_acc, patience_count = 0, 0
history = {"t_loss": [], "v_loss": [], "t_acc": [], "v_acc": []}

print(f"\n🚀 Training on {DEVICE}...")

for epoch in range(EPOCHS):
    # Track current LR for display
    current_lr = optimizer.param_groups[0]['lr']
    
    train_loss, train_acc = run_epoch(train_loader, True)
    val_loss, val_acc = run_epoch(val_loader, False)
    
    # Update scheduler
    scheduler.step(val_acc) 
    
    # Check if LR changed (since verbose is gone)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < current_lr:
        print(f"📉 Learning Rate reduced to {new_lr}")

    history["t_loss"].append(train_loss); history["v_loss"].append(val_loss)
    history["t_acc"].append(train_acc); history["v_acc"].append(val_acc)

    print(f"E{epoch+1:03d} | LR: {current_lr:.6f} | Train: {train_acc:.3f} | Val: {val_acc:.3f} | Loss: {val_loss:.3f}")

    if val_acc > best_acc:
        best_acc = val_acc
        patience_count = 0
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/best_model.pth")
        print("⭐ New Best Model Saved")
    else:
        patience_count += 1
        if patience_count >= PATIENCE: 
            print("⏹️ Early Stopping")
            break

# -----------------------
# EVAL & SAVE
# -----------------------
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/best_model.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x, m, y in test_loader:
        out = model(x.to(DEVICE), m.to(DEVICE))
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(y.numpy())

print(f"\n🎯 FINAL TEST ACCURACY: {accuracy_score(y_true, y_pred):.4f}")

with open(f"{OUTPUT_DIR}/class_map.json", "w") as f: json.dump(class_map, f)
report = classification_report(y_true, y_pred, target_names=list(class_map.keys()), zero_division=0)
with open(f"{OUTPUT_DIR}/final_report.txt", "w") as f: f.write(report)

print("✅ Training assets saved successfully.")