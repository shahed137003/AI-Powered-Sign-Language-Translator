from __future__ import annotations

from pathlib import Path
import numpy as np

from .constants import FEATURE_DIM


def iter_npy_files(root: Path):
    """Yield .npz files recursively."""
    return (
        p for p in root.rglob("*.npz")
        if p.is_file()
    )


def load_keypoints_npy(path: Path) -> np.ndarray:
    """Load x from .npz file."""
    
    data = np.load(path, allow_pickle=True)

    if "x" not in data:
        raise ValueError(f"No 'x' key in {path}")

    x = data["x"]

    if x.ndim != 2 or x.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected shape (T,{FEATURE_DIM}), got {x.shape}")

    return x.astype(np.float32)

def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
