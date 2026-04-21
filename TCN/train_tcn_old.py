# ============================================================
# APPLY TCN MODEL TO YOUR PROCESSED DATA (FIXED)
# ============================================================

import os
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from pathlib import Path

# ============================================================
# LOAD FROM YOUR CSV SPLIT (ONLY CHANGE HERE)
# ============================================================

DATA_DIR = Path(r"D:\500 words\data\splits_with_mask")
CSV_PATH = DATA_DIR / "splits.csv"

df = pd.read_csv(CSV_PATH)

def load_split(name):
    sub = df[df["split"] == name]

    data, masks, labels = [], [], []

    for _, r in sub.iterrows():
        x = np.load(DATA_DIR / r["filepath"])

        if r["maskpath"] and str(r["maskpath"]) != "":
            m = np.load(DATA_DIR / r["maskpath"])
        else:
            m = np.ones(x.shape[0])

        if x.shape[1] == 439:
            m = x[:, -1]
            x = x[:, :-1]

        data.append(x)
        masks.append(m)
        labels.append(r["label"])

    return np.array(data), np.array(masks), np.array(labels)

processed_data, processed_masks, processed_labels = load_split("train")
val_data, val_masks, val_labels = load_split("val")
test_data, test_masks, test_labels = load_split("test")

# merge splits like your original pipeline expects
processed_data = processed_data
processed_masks = processed_masks
processed_labels = processed_labels

# ============================================================
# CONFIGURATION - ADJUST FOR YOUR DATA
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

FEATURE_DIM = processed_data.shape[2]  # Should be 438
TARGET_FRAMES = processed_data.shape[1]  # Max frames from your data

print(f"📊 Data Information:")
print(f"   Feature dimension: {FEATURE_DIM}")
print(f"   Target frames: {TARGET_FRAMES}")
print(f"   Total samples: {len(processed_data)}")

# ============================================================
# ENCODE LABELS AND CHECK CLASS DISTRIBUTION
# ============================================================

le = LabelEncoder()
all_labels_encoded = le.fit_transform(processed_labels)
num_classes = len(le.classes_)

print(f"\n🏷️ Label encoding complete: {num_classes} classes")

label_counts = Counter(processed_labels)
print(f"\n📊 Class distribution:")
print(f"   Total classes: {num_classes}")
print(f"   Min samples per class: {min(label_counts.values())}")
print(f"   Max samples per class: {max(label_counts.values())}")
print(f"   Mean samples per class: {np.mean(list(label_counts.values())):.2f}")

rare_classes = {label: count for label, count in label_counts.items() if count < 3}
if rare_classes:
    print(f"\n⚠️ Classes with <3 samples ({len(rare_classes)} classes):")
    for label, count in list(rare_classes.items())[:10]:
        print(f"   {label}: {count} sample(s)")

# ============================================================
# FILTER RARE CLASSES (OPTIONAL)
# ============================================================

MIN_SAMPLES_PER_CLASS = 2

if min(label_counts.values()) < MIN_SAMPLES_PER_CLASS:
    print(f"\n⚠️ Filtering classes with < {MIN_SAMPLES_PER_CLASS} samples...")
    
    keep_classes = [label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_CLASS]
    keep_mask = [label in keep_classes for label in processed_labels]

    processed_data = processed_data[keep_mask]
    processed_masks = processed_masks[keep_mask]
    processed_labels = processed_labels[keep_mask]

    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(processed_labels)
    num_classes = len(le.classes_)

# ============================================================
# CREATE STRATIFIED SPLIT (WITH FALLBACK)
# ============================================================

print(f"\n📊 Creating train/val/test splits...")

try:
    X_train, X_temp, y_train, y_temp, m_train, m_temp = train_test_split(
        processed_data, all_labels_encoded, processed_masks,
        test_size=0.2, stratify=all_labels_encoded, random_state=42
    )

    X_val, X_test, y_val, y_test, m_val, m_test = train_test_split(
        X_temp, y_temp, m_temp,
        test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"✅ Stratified split successful")

except ValueError as e:
    print(f"⚠️ Stratified split failed: {e}")

    X_train, X_temp, y_train, y_temp, m_train, m_temp = train_test_split(
        processed_data, all_labels_encoded, processed_masks,
        test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test, m_val, m_test = train_test_split(
        X_temp, y_temp, m_temp,
        test_size=0.5, random_state=42
    )

# ============================================================
# DATASET CLASS
# ============================================================

class ASLDataset(Dataset):
    def __init__(self, data, masks, labels, augment=False):
        self.data = torch.from_numpy(data).float()
        self.masks = torch.from_numpy(masks).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        m = self.masks[idx]
        y = self.labels[idx]

        if self.augment:
            if torch.rand(1) > 0.5:
                x = x + torch.randn_like(x) * 0.01
            if torch.rand(1) > 0.5:
                x = x * (torch.rand(1) * 0.2 + 0.9)

        x = x.transpose(0, 1)
        return x, m, y

# ============================================================
# DATALOADERS
# ============================================================

BATCH_SIZE = min(8, len(X_train))
NUM_WORKERS = 0

train_loader = DataLoader(ASLDataset(X_train, m_train, y_train, True),
                          batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(ASLDataset(X_val, m_val, y_val, False),
                        batch_size=BATCH_SIZE, shuffle=False)

test_loader = DataLoader(ASLDataset(X_test, m_test, y_test, False),
                         batch_size=BATCH_SIZE, shuffle=False)

print(f"\n✅ DataLoaders created:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# ============================================================
# (REST OF YOUR CODE UNCHANGED)
# ============================================================


# ============================================================
# TCN MODEL DEFINITION
# ============================================================

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        y = self.net(x)
        if y.size(2) != x.size(2):
            y = y[..., :x.size(2)]
        return y + self.res(x)

class TCN(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=64, num_layers=3):
        super().__init__()
        # Adjust number of layers based on dataset size
        actual_layers = min(num_layers, len(X_train) // 10)  # Don't use too many layers for small dataset
        channels = [num_channels] * max(1, actual_layers)
        
        layers = []
        for i, c in enumerate(channels):
            in_dim = input_dim if i == 0 else channels[i-1]
            layers.append(TemporalBlock(in_dim, c, dilation=2**i))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], num_classes)
        self.dropout = nn.Dropout(0.4)
        
    def masked_pool(self, x, mask):
        """Masked global average pooling"""
        mask = mask.unsqueeze(1)  # (batch, 1, time)
        masked_x = x * mask
        summed = masked_x.sum(dim=2)
        valid_counts = mask.sum(dim=2).clamp(min=1)
        return summed / valid_counts
    
    def forward(self, x, mask):
        x = self.tcn(x)
        x = self.masked_pool(x, mask)
        x = self.dropout(x)
        return self.fc(x)

# Adjust model size based on dataset
if len(X_train) < 50:
    num_channels = 32
    num_layers = 2
    print(f"\n📐 Small dataset detected. Using smaller model: {num_layers} layers, {num_channels} channels")
elif len(X_train) < 200:
    num_channels = 64
    num_layers = 3
    print(f"\n📐 Medium dataset detected. Using medium model: {num_layers} layers, {num_channels} channels")
else:
    num_channels = 128
    num_layers = 4
    print(f"\n📐 Large dataset detected. Using larger model: {num_layers} layers, {num_channels} channels")

# Initialize model
model = TCN(
    input_dim=FEATURE_DIM, 
    num_classes=num_classes,
    num_channels=num_channels,
    num_layers=num_layers
).to(DEVICE)

# Calculate model size
total_params = sum(p.numel() for p in model.parameters())
params_per_sample = total_params / len(X_train)

print(f"\n📐 MODEL ARCHITECTURE:")
print(f"   Input: {FEATURE_DIM} x {TARGET_FRAMES}")
print(f"   Architecture: TCN with {num_layers} layers, {num_channels} channels")
print(f"   Total parameters: {total_params:,}")
print(f"   Parameters per sample: {params_per_sample:.1f}")

if params_per_sample > 500:
    print(f"   ⚠️ WARNING: Model may overfit! Consider reducing channels or layers.")
elif params_per_sample < 50:
    print(f"   ✅ Model size appropriate for dataset")
else:
    print(f"   ⚠️ Model size acceptable")

# ============================================================
# LOSS FUNCTION AND OPTIMIZER
# ============================================================

# Compute class weights for imbalance (handle case with single class)
if num_classes > 1:
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print(f"\n⚖️ Class weights computed for {num_classes} classes")
else:
    class_weights = None
    print(f"\n⚠️ Only one class found. Using standard cross-entropy.")

class SmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, logits, targets):
        n_classes = logits.size(1)
        if n_classes == 1:
            # Binary case
            return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), targets.float())
        
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        log_probs = torch.log_softmax(logits, dim=1)
        
        if self.weight is not None:
            weight = self.weight[targets]
            loss = -(true_dist * log_probs).sum(dim=1) * weight
        else:
            loss = -(true_dist * log_probs).sum(dim=1)
            
        return loss.mean()

# Training configuration
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
GRAD_CLIP = 1.0
LABEL_SMOOTH = 0.1

criterion = SmoothCrossEntropy(smoothing=LABEL_SMOOTH, weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, m, y in loader:
        x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(x, m)
        loss = criterion(outputs, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total

def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, m, y in loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            outputs = model(x, m)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * y.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    return total_loss / total, correct / total

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "="*60)
print("🚀 STARTING TRAINING")
print("="*60)
print(f"   Device: {DEVICE}")
print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LR}")
print(f"   Epochs: {EPOCHS}")
print("="*60)

best_val_acc = 0
patience_counter = 0
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    # Train
    train_loss, train_acc = train_one_epoch(train_loader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validate
    val_loss, val_acc = evaluate(val_loader)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Update scheduler
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print progress
    print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
          f"LR: {current_lr:.2e} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
        }, 'best_tcn_model.pth')
        print(f"  ✅ Best model saved! (Val Acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

# Load best model
checkpoint = torch.load('best_tcn_model.pth', map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
print(f"   Validation Accuracy: {checkpoint['val_acc']:.4f}")

# Test evaluation
test_loss, test_acc = evaluate(test_loader)
print(f"\n🎯 TEST SET RESULTS:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ============================================================
# SAVE RESULTS
# ============================================================

# Save label encoder
np.save('label_encoder.npy', le.classes_)

# Save training history
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accs': train_accs,
    'val_accs': val_accs,
    'best_val_acc': best_val_acc,
    'test_acc': test_acc,
    'test_loss': test_loss,
    'num_classes': num_classes,
    'target_frames': TARGET_FRAMES,
    'feature_dim': FEATURE_DIM,
    'model_params': total_params,
}

with open('training_history.pkl', 'wb') as f:
    pickle.dump(history, f)

print(f"\n💾 Saved files:")
print(f"   Model: best_tcn_model.pth")
print(f"   Label encoder: label_encoder.npy")
print(f"   Training history: training_history.pkl")

# ============================================================
# PLOT RESULTS
# ============================================================

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].axhline(y=test_loss, color='r', linestyle='--', label=f'Test Loss: {test_loss:.3f}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[1].plot(val_accs, label='Val Acc', linewidth=2)
    axes[1].axhline(y=test_acc, color='r', linestyle='--', label=f'Test Acc: {test_acc:.3f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()
    print("📊 Training plots saved to 'training_results.png'")
    
except Exception as e:
    print(f"⚠️ Could not plot: {e}")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)