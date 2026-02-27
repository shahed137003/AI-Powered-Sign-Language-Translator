import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from scipy import interpolate

# ============================================================
# CONFIGURATION
# ============================================================
RAW_DATA_DIR = Path(r"F:\Demiana Ayman\small data\Landmarks before")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_FRAMES = 157
FEATURE_DIM = 438  # Matches your model's expected input
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4

# ============================================================
# DATASET: MINIMAL TEMPORAL RESIZING ONLY
# ============================================================
class ASLRawDataset(Dataset):
    def __init__(self, files, labels, target_frames=157):
        self.files = files
        self.labels = labels
        self.target_frames = target_frames

    def __len__(self):
        return len(self.files)

    def _temporal_resize(self, seq):
        T, D = seq.shape
        # Linear interpolation to force all sequences to TARGET_FRAMES
        new_seq = np.zeros((self.target_frames, D), dtype=np.float32)
        x_orig = np.arange(T)
        x_target = np.linspace(0, T - 1, self.target_frames)
        
        for d in range(D):
            if np.any(np.isfinite(seq[:, d])):
                f = interpolate.interp1d(x_orig, seq[:, d], kind='linear', 
                                         bounds_error=False, fill_value="extrapolate")
                new_seq[:, d] = f(x_target)
        return new_seq

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        # Ensure feature dimension matches (stripping extra columns like visibility if needed)
        if data.shape[1] > FEATURE_DIM:
            data = data[:, :FEATURE_DIM]
            
        x_raw = self._temporal_resize(data)
        # Convert to (Channels, Time) for TCN
        x = torch.from_numpy(x_raw).float().transpose(0, 1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        # Dummy mask (all 1s) because we aren't using the preprocessing mask here
        m = torch.ones(self.target_frames, dtype=torch.float32)
        return x, m, y

# ============================================================
# MODEL: TCN ARCHITECTURE (Identical to your processed version)
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
        layers = [TemporalBlock(FEATURE_DIM, chans[0], 1)]
        for i in range(1, len(chans)):
            layers.append(TemporalBlock(chans[i-1], chans[i], 2 ** i))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(chans[-1], num_classes)
    def forward(self, x, m):
        x = self.tcn(x)
        # Global Average Pooling across time
        return self.fc(x.mean(2))

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print(f"Loading raw data from {RAW_DATA_DIR}...")
    all_files = list(RAW_DATA_DIR.glob("*.npy"))
    raw_labels = [f.stem.split(" ")[0] for f in all_files]
    
    # Filter classes with >= 2 samples
    cnt = Counter(raw_labels)
    valid_idx = [i for i, lbl in enumerate(raw_labels) if cnt[lbl] >= 2]
    files = [str(all_files[i]) for i in valid_idx]
    labels = [raw_labels[i] for i in valid_idx]
    
    le = LabelEncoder()
    y = le.fit_transform(labels)
    num_classes = len(le.classes_)

    X_train, X_val, y_train, y_val = train_test_split(files, y, test_size=0.2, stratify=y)
    
    train_loader = DataLoader(ASLRawDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ASLRawDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = TCN(num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"Baseline Test: {num_classes} classes | Device: {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        for x, m, target in train_loader:
            x, target = x.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x, m), target)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, m, target in val_loader:
                x, target = x.to(DEVICE), target.to(DEVICE)
                correct += (model(x, m).argmax(1) == target).sum().item()
        
        print(f"Epoch {epoch+1} | Accuracy: {correct/len(X_val):.4f}")