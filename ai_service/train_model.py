import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ============================================================
# 1. CONFIGURATION (Global)
# ============================================================
DATA_DIR = Path(r"F:\Demiana Ayman\small data\landmarks with sw mask")
DEVICE = "cpu"
TARGET_FRAMES = 157
FEATURE_DIM = 438
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 12
GRAD_CLIP = 1.0
LABEL_SMOOTH = 0.1

MODEL_SAVE_PATH = "tcn_best_cpu.pth"
LABEL_ENCODER_PATH = "label_encoder.npy"

# ============================================================
# 2. MODEL DEFINITION (Global - Safe to import)
# ============================================================
class TemporalBlock(nn.Module):
    def __init__(self, ic, oc, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, 3, padding=d, dilation=d),
            nn.BatchNorm1d(oc),
            nn.ReLU(),
            nn.Conv1d(oc, oc, 3, padding=d, dilation=d),
            nn.BatchNorm1d(oc),
            nn.ReLU(),
        )
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        y = y[..., :x.size(2)]
        return y + self.res(x)

class TCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        chans = [192, 192, 192, 192]
        layers = []
        for i, c in enumerate(chans):
            layers.append(TemporalBlock(FEATURE_DIM if i == 0 else chans[i-1], c, 2 ** i))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(chans[-1], num_classes)

    def masked_pool(self, x, m):
        m = m.unsqueeze(1)
        return (x * m).sum(2) / (m.sum(2) + 1e-6)

    def forward(self, x, m):
        x = self.tcn(x)
        x = self.masked_pool(x, m)
        return self.fc(x)

# ============================================================
# 3. TRAINING LOOP (Protected - Will NOT run when imported)
# ============================================================
if __name__ == "__main__":
    print(f"Scanning directory: {DATA_DIR}")
    files, masks, labels = [], [], []

    for f in DATA_DIR.glob("*.npy"):
        if f.name.endswith("_mask.npy"):
            continue

        mask_f = f.with_name(f.stem + "_mask.npy")
        if not mask_f.exists():
            continue

        arr = np.load(f)
        if arr.shape != (TARGET_FRAMES, FEATURE_DIM):
            continue

        files.append(str(f))
        masks.append(str(mask_f))
        labels.append(f.stem.split("_")[0])

    print(f"Found {len(files)} valid sequences.")

    cnt = Counter(labels)
    keep = [i for i, y in enumerate(labels) if cnt[y] >= 2]
    files = [files[i] for i in keep]
    masks = [masks[i] for i in keep]
    labels = [labels[i] for i in keep]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    np.save(LABEL_ENCODER_PATH, le.classes_)
    num_classes = len(le.classes_)

    print(f"Training on {num_classes} classes.")

    X_tr, X_tmp, y_tr, y_tmp, m_tr, m_tmp = train_test_split(files, y, masks, test_size=0.2, stratify=y, random_state=42)
    X_val, X_te, y_val, y_te, m_val, m_te = train_test_split(X_tmp, y_tmp, m_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    class ASLDataset(Dataset):
        def __init__(self, files, masks, labels):
            self.files, self.masks, self.labels = files, masks, labels
        def __len__(self):
            return len(self.files)
        def __getitem__(self, idx):
            x = torch.from_numpy(np.load(self.files[idx])).float().transpose(0, 1)
            m = torch.from_numpy(np.load(self.masks[idx])).float()
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, m, y

    train_loader = DataLoader(ASLDataset(X_tr, m_tr, y_tr), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ASLDataset(X_val, m_val, y_val), BATCH_SIZE)
    test_loader  = DataLoader(ASLDataset(X_te, m_te, y_te), BATCH_SIZE)

    model = TCN(num_classes).to(DEVICE)

    class SmoothCE(nn.Module):
        def __init__(self, eps=0.1):
            super().__init__()
            self.eps = eps
        def forward(self, logits, target):
            n = logits.size(1)
            logp = torch.log_softmax(logits, 1)
            y = torch.zeros_like(logp).fill_(self.eps / n)
            y.scatter_(1, target.unsqueeze(1), 1 - self.eps)
            return -(y * logp).sum(1).mean()

    criterion = SmoothCE(LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, EPOCHS)

    def run(loader, train=True):
        model.train() if train else model.eval()
        loss_sum, correct, total = 0, 0, 0
        with torch.set_grad_enabled(train):
            for x, m, y in loader:
                x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                out = model(x, m)
                loss = criterion(out, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                loss_sum += loss.item() * y.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        return loss_sum / total, correct / total

    print("Starting Training...")
    best, patience = -1e9, 0
    for e in range(EPOCHS):
        tr_l, tr_a = run(train_loader, True)
        va_l, va_a = run(val_loader, False)
        scheduler.step()
        metric = va_a - va_l

        print(f"Epoch {e+1:03d} | Train Loss: {tr_l:.3f} Acc: {tr_a:.3f} | Val Loss: {va_l:.3f} Val Acc: {va_a:.3f}")

        if metric > best:
            best = metric
            patience = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping triggered!")
                break

    print("\nLoading best model for Final Testing...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    test_loss, test_acc = run(test_loader, train=False)
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}%")