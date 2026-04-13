import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import time

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_DIR = Path(r"D:\CSE-Senior 2\GP\all data\After preprocessing\landmarks_preprocssed_with_SW_and_mask")
MODEL_SAVE_PATH = "strong_asl_model.pth"
LABEL_ENCODER_PATH = "label_encoder_large.npy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FRAMES = 158  # Updated for your new data
FEATURE_DIM = 438
BATCH_SIZE = 16      # Adjust to 32 if you get "Out of Memory"
EPOCHS = 30
MAX_LR = 1e-3
WEIGHT_DECAY = 0.01

# ============================================================
# 2. MODEL ARCHITECTURE
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        key_padding_mask = (mask == 0) if mask is not None else None
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)
        return x.transpose(1, 2)

class TemporalBlock(nn.Module):
    def __init__(self, ic, oc, d, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, 3, padding=d, dilation=d),
            nn.BatchNorm1d(oc),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(oc, oc, 3, padding=d, dilation=d),
            nn.BatchNorm1d(oc),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.se = SEBlock(oc)
        self.res = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()
    def forward(self, x):
        return self.se(self.net(x) + self.res(x))

class StrongASLModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        chans = [64, 128, 128, 256]
        layers = []
        for i, c in enumerate(chans):
            layers.append(TemporalBlock(FEATURE_DIM if i == 0 else chans[i-1], c, 2**i))
        self.tcn = nn.Sequential(*layers)
        self.attention = MultiHeadAttention(chans[-1])
        self.classifier = nn.Sequential(
            nn.Linear(chans[-1], 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def masked_pool(self, x, m):
        m = m.unsqueeze(1)
        return (x * m).sum(2) / (m.sum(2) + 1e-6)

    def forward(self, x, m):
        x = self.tcn(x)
        x = self.attention(x, m)
        x = self.masked_pool(x, m)
        return self.classifier(x)

# ============================================================
# 3. DATASET & LOADING
# ============================================================

class ASLDataset(Dataset):
    def __init__(self, files, masks, labels):
        self.files, self.masks, self.labels = files, masks, labels
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        # Load landmarks and transpose for Conv1D: (Frames, Feats) -> (Feats, Frames)
        x = torch.from_numpy(np.load(self.files[idx])).float().transpose(0, 1)
        m = torch.from_numpy(np.load(self.masks[idx])).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, m, y

def prepare_data():
    print(f"Scanning: {DATA_DIR}")
    files, masks, labels = [], [], []
    # Using glob to find all non-mask files
    all_files = list(DATA_DIR.glob("*.npy"))
    
    for f in tqdm(all_files, desc="Filtering Data"):
        if f.name.endswith("_mask.npy"): continue
        mask_f = f.with_name(f.stem + "_mask.npy")
        if mask_f.exists():
            files.append(str(f))
            masks.append(str(mask_f))
            # Extract class name (assumes filename starts with class name)
            labels.append(f.stem.split("_")[0])

    le = LabelEncoder()
    y = le.fit_transform(labels)
    np.save(LABEL_ENCODER_PATH, le.classes_)
    
    # Stratified split to handle small classes
    X_tr, X_val, y_tr, y_val, m_tr, m_val = train_test_split(
        files, y, masks, test_size=0.15, stratify=y, random_state=42
    )
    
    return X_tr, X_val, y_tr, y_val, m_tr, m_val, len(le.classes_)

# ============================================================
# 4. TRAINING ENGINE
# =============================-==============================

if __name__ == "__main__":
    X_tr, X_val, y_tr, y_val, m_tr, m_val, num_classes = prepare_data()
    print(f"Classes: {num_classes} | Training samples: {len(X_tr)}")

    train_loader = DataLoader(ASLDataset(X_tr, m_tr, y_tr), BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ASLDataset(X_val, m_val, y_val), BATCH_SIZE, num_workers=4)

    model = StrongASLModel(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, 
                           steps_per_epoch=len(train_loader), 
                           epochs=EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, m, y in pbar:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x, m)
            loss = criterion(outputs, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2f}")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                out = model(x, m)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Saved best model with {val_acc:.4f} accuracy")

    print(f"Training Complete. Best Accuracy: {best_acc:.4f}")