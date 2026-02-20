import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import ASLDataset
from src.configs.tcn_config import *

def build_dataloaders():

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

    # remove rare classes
    cnt = Counter(labels)
    keep = [i for i, y in enumerate(labels) if cnt[y] >= 2]

    files = [files[i] for i in keep]
    masks = [masks[i] for i in keep]
    labels = [labels[i] for i in keep]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    np.save(LABEL_ENCODER_PATH, le.classes_)

    X_tr, X_tmp, y_tr, y_tmp, m_tr, m_tmp = train_test_split(
        files, y, masks, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_te, y_val, y_te, m_val, m_te = train_test_split(
        X_tmp, y_tmp, m_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    train_loader = DataLoader(ASLDataset(X_tr, m_tr, y_tr), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ASLDataset(X_val, m_val, y_val), BATCH_SIZE)
    test_loader  = DataLoader(ASLDataset(X_te, m_te, y_te), BATCH_SIZE)

    return train_loader, val_loader, test_loader, len(le.classes_)