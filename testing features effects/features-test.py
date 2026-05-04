import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import pandas as pd
import random

# =========================
# CONFIG
# =========================
DATA_DIR = r"D:\Additionalpreprocessing 4\150_words\final_dataset"
OUTPUT_DIR = r"D:\Additionalpreprocessing 4\150_words\testing features effects\feature_search"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
GRAD_CLIP = 1.0
LABEL_SMOOTH = 0.1

SEED = 42

# =========================
# DETERMINISM
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# FEATURE GROUPS
# =========================
feature_slices = {
    "x": (0, 438),
    "v": (438, 876),
    "a": (876, 1314),
    "v2": (1314, 1752),
    "direction": (1752, 2190),
    "pose_bones": (2190, 2208),
    "lh_bones": (2208, 2268),
    "rh_bones": (2268, 2328),
    "lh_angles": (2328, 2343),
    "rh_angles": (2343, 2358),
    "relative": (2358, 2673),
    "distances": (2673, 2676),
    "handshape": (2676, 2680)
}

# =========================
# EXPERIMENTS
# =========================
experiments = [
    ("ALL", []),
    ("NO_VEL", ["v"]),
    ("NO_ACC", ["a"]),
    ("NO_VEL_ACC", ["v", "a"]),
    ("NO_V2", ["v2"]),
    ("NO_DIRECTION", ["direction"]),
    ("NO_MOTION", ["v", "a", "v2", "direction"]),
    ("NO_BONES", ["pose_bones", "lh_bones", "rh_bones"]),
    ("NO_ANGLES", ["lh_angles", "rh_angles"]),
    ("NO_GEOMETRY", ["pose_bones", "lh_bones", "rh_bones", "lh_angles", "rh_angles"]),
    ("NO_RELATIVE", ["relative"]),
    ("NO_DISTANCE", ["distances"]),
    ("NO_HANDSHAPE", ["handshape"]),
    ("ONLY_X", ["v","a","v2","direction","pose_bones","lh_bones","rh_bones","lh_angles","rh_angles","relative","distances","handshape"]),
    # Remove weak features together
    ("NO_WEAK_FEATURES", ["v","a","relative","distances","handshape"]),
    # Core model (no motion, no weak stuff)
    ("CORE_GEOMETRY_ONLY", ["v","a","v2","direction","relative","distances","handshape"]),
    # Add ONLY best motion candidate
    ("CORE_PLUS_V2", ["v","a","direction","relative","distances","handshape"]),
    # Angles-only geometry (important test)
    ("X_PLUS_ANGLES_ONLY", ["v","a","v2","direction","pose_bones","lh_bones","rh_bones","relative","distances","handshape"]),
    #  Minimal realistic deploy model, keep only hand bones + angles
    ("MINIMAL_STRONG", ["v","a","v2","direction","relative","distances","handshape","pose_bones"]),
    # geometry only (no raw, no motion)
    ("RELATIVE_PLUS_ANGLES", ["x","v","a","v2","direction","pose_bones","lh_bones","rh_bones","distances","handshape"]),
    # Remove raw → test if features alone are enough
    ("NO_X", ["x"]),
    # Remove noisy motion (keep only v2)
    ("NO_NOISY_MOTION", ["v","a","direction"]),
    # Only v2 (clean temporal signal)
    ("ONLY_V2", ["x","v","a","direction","pose_bones","lh_bones","rh_bones","lh_angles","rh_angles","relative","distances","handshape"]),
    # Remove bones → are angles enough?
    ("ANGLES_ONLY_GEOMETRY", ["pose_bones","lh_bones","rh_bones"]),
    # Remove weak helpers
    ("NO_SMALL_FEATURES", ["distances","handshape"]),
    # Keep strong core only
    ("MINIMAL_CLEAN", ["v","a","direction","distances","handshape"]),
    # Raw + angles only
    ("X_PLUS_ANGLES_ONLY_CLEAN", ["v","a","v2","direction","pose_bones","lh_bones","rh_bones","relative","distances","handshape"]),
    ]

# =========================
# DATASET
# =========================
class NPZDataset(Dataset):
    def __init__(self, folder, remove_groups=None):
        self.files = sorted(list(Path(folder).glob("*.npz")))
        self.remove_groups = remove_groups or []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        x = data["x"].astype(np.float32)
        y = int(data["y"])
        m = data["mask"].astype(np.float32)

        for g in self.remove_groups:
            s, e = feature_slices[g]
            x[:, s:e] = 0.0

        return torch.from_numpy(x), torch.from_numpy(m), y

# =========================
# MODEL
# =========================
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
    def __init__(self, input_dim, num_classes):
        super().__init__()
        layers = []
        for i in range(4):
            in_dim = input_dim if i == 0 else 128
            layers.append(TemporalBlock(in_dim, 128, dilation=2**i))

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.4)

    def masked_pool(self, x, mask):
        mask = mask.unsqueeze(1)
        x = x * mask
        summed = x.sum(dim=2)
        valid = mask.sum(dim=2).clamp(min=1)
        return summed / valid

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.masked_pool(x, mask)
        x = self.dropout(x)
        return self.fc(x)

# =========================
# RUN EXPERIMENT
# =========================
def run_experiment(name, remove_groups):

    set_seed(SEED)

    trial_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(trial_dir, exist_ok=True)

    print("\n==============================")
    print("Experiment:", name)
    print("Removing:", remove_groups)
    print("==============================")

    train_ds = NPZDataset(os.path.join(DATA_DIR, "train"), remove_groups)
    val_ds   = NPZDataset(os.path.join(DATA_DIR, "val"), remove_groups)
    test_ds  = NPZDataset(os.path.join(DATA_DIR, "test"), remove_groups)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    sample_x, _, _ = train_ds[0]
    FEATURE_DIM = sample_x.shape[1]

    labels = [train_ds[i][2] for i in range(len(train_ds))]
    unique_labels = sorted(set(labels))
    num_classes = len(unique_labels)
    # =========================
    # DATA STATS PRINTING
    # =========================
    print("\n📊 DATASET STATS")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    print(f"Total samples: {len(train_ds) + len(val_ds) + len(test_ds)}")

    print(f"\nNumber of classes: {num_classes}")
  
    print(f"\nFeature dimension: {FEATURE_DIM}")
    # SAVE LABEL ENCODER
    label_map = {int(i): int(i) for i in unique_labels}
    np.save(os.path.join(trial_dir, "label_encoder.npy"), label_map)

    model = TCN(FEATURE_DIM, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val = float("inf")
    patience = 0

    logs = []

    for epoch in range(EPOCHS):

        model.train()
        tl, tc, tt = 0, 0, 0

        for x, m, y in train_loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            out = model(x, m)
            loss = criterion(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            tl += loss.item() * y.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tt += y.size(0)

        train_loss = tl / tt
        train_acc = tc / tt

        model.eval()
        vl, vc, vt = 0, 0, 0

        with torch.no_grad():
            for x, m, y in val_loader:
                x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                out = model(x, m)
                loss = criterion(out, y)

                vl += loss.item() * y.size(0)
                vc += (out.argmax(1) == y).sum().item()
                vt += y.size(0)

        val_loss = vl / vt
        val_acc = vc / vt

        scheduler.step(val_loss)

        print(f"E{epoch+1:02d} | TL {train_loss:.4f} | TA {train_acc:.4f} | VL {val_loss:.4f} | VA {val_acc:.4f}")

        logs.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(trial_dir, "best_model.pth"))
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break

    pd.DataFrame(logs).to_csv(os.path.join(trial_dir, "epoch_logs.csv"), index=False)

    # TEST
    model.load_state_dict(torch.load(os.path.join(trial_dir, "best_model.pth")))
    model.eval()

    tl, tc, tt = 0, 0, 0
    preds, targets = [], []

    with torch.no_grad():
        for x, m, y in test_loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            out = model(x, m)
            loss = criterion(out, y)

            tl += loss.item() * y.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tt += y.size(0)

            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    test_loss = tl / tt
    test_acc = tc / tt
    print("Test accuracy: ",test_acc," and test loss: ",test_loss)
    wrong = [{"pred": int(p), "true": int(t)} for p, t in zip(preds, targets) if p != t]
    pd.DataFrame(wrong).to_csv(os.path.join(trial_dir, "wrong_predictions.csv"), index=False)
    print("-----------------------------------------------------------------")
    return {
        "experiment": name,
        "removed": ",".join(remove_groups),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    }

# =========================
# RUN ALL
# =========================
results = []

for name, remove in experiments:
    res = run_experiment(name, remove)
    results.append(res)

pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "feature_ablation_results.csv"), index=False)

print("\nAll experiments completed and saved.")